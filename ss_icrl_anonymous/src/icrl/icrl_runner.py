# =============================================================
# File: src/icrl/icrl0_runner.py          (rev 2025‑08‑30‑m)
# =============================================================
"""
ICRL‑0 – compressed, English‑only  (稳定版 + DEBUG).

修复日志
--------
1. 摘要为空不写入历史，并截断历史长度 (MAX_KEEP=5)。
2. 用 Markdown 列表代替 `{ ... }` 区块，避免模型复制花括号。
3. generation 统一加 repetition_penalty=1.1。
4. _extract_numeric 先从最后一行反扫只含数字/分数的行。
5. 保留 DEBUG 日志；DEBUG=False 时完全关闭。
"""

from __future__ import annotations
import argparse, datetime, json, csv, random, re, logging
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Union
import math

import sys
import os
import torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from vllm import LLM, SamplingParams
from icrl.prompt_cache import PromptCache
from icrl.multiview_answer_adapter import (
    extract_boxed_answer,
    extract_current_assistant_span,
    truncate_after_boxed_answer,
)
from icrl.utility import *
from icrl.utility import _normalize
from icrl.setting import setting

_cfg: Dict[str, Any] = yaml.safe_load(CONFIG_YML.open()) if CONFIG_YML.is_file() else {}
import subprocess


def get_total_gpu_mem():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    mem_list = [int(x) for x in result.stdout.strip().split("\n")]
    return sum(mem_list) / 1024  # MB -> GB


def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "on")


def _overlay_from_env(base: Dict[str, Any]) -> Dict[str, Any]:
    env_map = {
        "summary_length": ("ENV_SUMMARY_LEN", int),
        "context_len": ("ENV_CONTEXT_LEN", int),
        "k": ("ENV_K", int),
        "rounds": ("ENV_ROUNDS", int),
        "entropy_k": ("ENV_ENTROPY_K", int),
        "test_sample": ("ENV_TEST_SAMPLE", int),
        "seed": ("ENV_SEED", int),
        "entropy": ("ENV_ENTROPY", str2bool),
        "start_idx": ("ENV_START_IDX", int),
        "end_idx": ("ENV_END_IDX", int),
        "use_gt_reward": ("ENV_USE_GT_REWARD", str2bool),
        "epsilon": ("ENV_EPSILON", float)
    }

    out = dict(base)
    for cfg_key, (env_key, caster) in env_map.items():
        val = os.environ.get(env_key)

        if val is not None and val != "":
            try:
                out[cfg_key] = caster(val)
            except Exception:
                print(f"invalid val: {val}")
                sys.exit(1)
    return out


if ALLOW_OVERWRITE:
    _cfg = _overlay_from_env(_cfg)


def cfg(k, d): return _cfg.get(k, d)


# ---------- regex ----------
_THINK_TAGS = re.compile(r"</?think>", re.I)
_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.I)
_NUMBER = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")
_LETTER = re.compile(r"\b([A-E])\b", re.I)
_SENT_SPLIT = re.compile(r"[.!?]")

ENABLE_PENALTY = None
SUMMARY_LEN = 500
USE_REWARD = True
BASE_MODEL = False
TOTAL_TOKEN = 0
COUNT_GPU_MEM = False
PEAK_GPU_GB = 0
EPSILON = 0
BOXED_ANSWER_POSITION = "last"


def debug_jsonl(filename: Union[str, Path], data: dict):
    """
    将单个 dict 写入 JSONL 文件，每行一个 JSON 对象。
    如果需要写多个 dict，可以先循环调用本函数。

    Args:
        filename: 输出文件路径
        data: 单个 dict
    """
    out_file = Path(filename)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")


def _generation_for_parsing(value: Any):
    text = str(value)
    source = (
        text
        if BOXED_ANSWER_POSITION == "last"
        else extract_current_assistant_span(text)
    )
    if extract_boxed_answer(source, position=BOXED_ANSWER_POSITION) is None:
        return None
    if BOXED_ANSWER_POSITION == "last":
        return source
    return truncate_after_boxed_answer(source, position=BOXED_ANSWER_POSITION)


def _normalize_generation(value: Any):
    source = _generation_for_parsing(value)
    return None if source is None else _normalize(source)


def _majority_raw(lst: List[str]) -> str:
    cnt, raw = {}, {}
    for a in lst:
        k = _normalize_generation(a)
        if not k:
            continue
        cnt[k] = cnt.get(k, 0) + 1
        raw.setdefault(k, a)
    if not cnt:
        return random.choice(lst)
    top = max(cnt.values())
    winners = [k for k, v in cnt.items() if v == top]
    choice = random.choice(winners)
    return raw[choice]


def _get_entropy(lst: List[str], max_classes: int) -> float | None:
    """
    计算给定字符串列表的归一化答案熵。
    对无效答案按比例加权惩罚，惩罚权重基于最大类别数对应的最大熵。

    Args:
        lst: 候选答案列表
        max_classes: 预估最大类别数，用于计算最大熵作为惩罚基数
        enable_penalty: 是否启用无效样本惩罚

    Returns:
        float: 熵值（含惩罚）
        None: 如果列表为空或无有效答案
    """
    global ENABLE_PENALTY
    total_count = len(lst)
    if total_count == 0:
        return None

    cnt = {}
    valid_count = 0
    penalty_weight = math.log2(max_classes)  # 最大可能的熵，惩罚基数

    for a in lst:
        norm_a = _normalize_generation(a)
        if not norm_a:
            continue
        cnt[norm_a] = cnt.get(norm_a, 0) + 1
        valid_count += 1

    if valid_count == 0:
        # 全无效样本，返回较大惩罚值
        return penalty_weight * 5

    entropy = -sum((v / valid_count) * math.log2(v / valid_count) for v in cnt.values())
    if entropy == 0 and DEBUG:
        logging.info(f"[eval] Entropy=0 ,values = {cnt}")
        logging.info(f"[eval] Entropy=0 ,lst = {lst}")
    if ENABLE_PENALTY:

        invalid_ratio = (total_count - valid_count) / total_count
        if DEBUG:
            logging.info(f"[eval] ENABLE_PENALTY,invalid_rato = {invalid_ratio}")
        entropy += penalty_weight * invalid_ratio

    return entropy


def strip_reasoning(txt: str) -> str:
    if "</think>" in txt.lower():
        txt = txt.lower().split("</think>", 1)[-1]
    return txt.strip()


# ---------- prompt builders ----------


def _render_prompt(
        lines: List[str],
        tok=None,
        prompt_format: str = "raw",
        enable_thinking: bool | None = None,
) -> str:
    raw_prompt = "\n".join(lines).rstrip()
    if prompt_format == "raw":
        return raw_prompt
    if prompt_format != "chat_template":
        raise ValueError(f"Unknown prompt_format={prompt_format!r}")
    if tok is None:
        raise ValueError("A tokenizer is required for prompt_format='chat_template'")

    system_prompt = lines[0].rstrip()
    user_prompt = "\n".join(lines[2:]).rstrip()
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    return tok.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def build_prompt(
        q: str,
        hist: Dict[str, List[str]],
        tok=None,
        prompt_format: str = "raw",
        enable_thinking: bool | None = None,
) -> str:
    global USE_REWARD
    global BASE_MODEL
    lines = [SYS_PROMPT.rstrip(), "", q.rstrip(), ""]
    if USE_REWARD == False:
        lines.append("previous attempts:")
        lines += [f"[{i}]- {s}" for i, s in enumerate(hist["good"])]
        return _render_prompt(lines, tok, prompt_format, enable_thinking)
    if BASE_MODEL == True:
        return _render_prompt(lines, tok, prompt_format, enable_thinking)

    lines.append("bad ideas (reward 0):")
    lines += [f"[{i}]- {s}" for i, s in enumerate(hist["bad"])]
    lines.append("")
    lines.append("good ideas (reward 1):")
    lines += [f"[{i}]- {s}" for i, s in enumerate(hist["good"])]

    # lines.append("The following answer may be a bad idea:")
    # lines += [f"[{i}]- {s}" for i, s in enumerate(hist["bad"])]
    # lines.append("")
    # # lines.append("good ideas (reward 1):")
    # lines.append("The following answer may be a good idea:")
    # lines += [f"[{i}]- {s}" for i, s in enumerate(hist["good"])]

    return _render_prompt(lines, tok, prompt_format, enable_thinking)


def delete_with_entropy(q: str, h: Dict[str, List[str]], model, tok, args, max_new):
    if len(h['good']) > MAX_KEEP:
        to_delete = h['good']
        to_keep = h['bad']
        delete_key = 'good'
        keep_key = 'bad'
    else:
        to_delete = h['bad']
        to_keep = h['good']
        delete_key = 'bad'
        keep_key = 'good'

    prompts = []
    for i in range(len(to_delete)):
        new_delete_list = to_delete[:i] + to_delete[i + 1:]
        new_h = {
            delete_key: new_delete_list,
            keep_key: to_keep.copy()
        }
        prompts.append(build_prompt(
            q,
            new_h,
            tok,
            args.prompt_format,
            args.enable_thinking,
        ))
    new_batch = generate_batch(model, tok, prompts, args.entropy_k, max_new, args.temp, args.top_p)
    entropies = [_get_entropy(lst, args.entropy_k) for lst in new_batch]
    if DEBUG:
        logging.info(f"[eval] delete entropy  = {entropies}")
    if any(e is not None for e in entropies):
        min_idx = min(
            [(i, e) for i, e in enumerate(entropies) if e is not None],
            key=lambda x: x[1]
        )[0]
    else:
        min_idx = 0  # 如果entropy都算不了 就删第一个
    del to_delete[min_idx]


# ---------- update history ----------
def update_history(h: Dict[str, List[str]], summary: str, reward: bool, q: str, model, tok, args, max_new):
    global USE_REWARD
    """写入摘要到对应桶，并截断长度。summary 已保证非空。"""
    bucket = h["good"] if reward else h["bad"]
    if USE_REWARD == False:
        bucket = h['good']  # ignore the pseudo reward
    if summary == None: return False
    bucket.append(summary)
    if len(bucket) > MAX_KEEP and args.entropy == True:  # 此时根据entropy来删多余任务
        delete_with_entropy(q, h, model, tok, args, max_new)
    del bucket[:-MAX_KEEP]  # 只保留最后 MAX_KEEP 条
    return True


# ---------- LLM wrappers ----------
def _gen(model, tok, prompt: str, max_new: int,
         temp: float, top_p: float) -> str:
    global TOTAL_TOKEN
    sampling_params = SamplingParams(
        n=1,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_new,
        min_tokens=8,
        repetition_penalty=1.1,
        stop_token_ids=[tok.eos_token_id],
    )
    outputs = model.generate([prompt], sampling_params)
    out = outputs[0].outputs[0]
    num_tokens = len(out.token_ids)
    TOTAL_TOKEN += num_tokens
    return outputs[0].outputs[0].text.strip()


def generate_batch(model, tok, prompts: List[str], n: int, max_new: int,
                   temp: float, top_p: float) -> List[List[str]]:
    if DEBUG and prompts:
        logging.info(f"[generate_batch] first prompt len={len(prompts[0])}")
    global TOTAL_TOKEN, PEAK_GPU_GB, COUNT_GPU_MEM

    sampling_params = SamplingParams(
        n=n,  # 等价于 num_return_sequences
        max_tokens=max_new,  # 新生成 token 数
        min_tokens=8,
        temperature=temp,
        top_p=top_p,
        repetition_penalty=1.1,

        stop_token_ids=[tok.eos_token_id],  # 防止跑太长
    )

    outputs = model.generate(prompts, sampling_params)

    res: List[List[str]] = []
    for i, out in enumerate(outputs):
        gens = [
            strip_reasoning(o.text)
            for o in out.outputs
        ]
        num_tokens = sum([len(o.token_ids) for o in out.outputs])
        TOTAL_TOKEN += num_tokens
        res.append(gens)
        if DEBUG and i == 0:
            logging.info(f"[generate_batch] sample gens={gens[:3]}")
    if COUNT_GPU_MEM:
        PEAK_GPU_GB = max(PEAK_GPU_GB, get_total_gpu_mem())
    return res


def _fallback_summary(ans_raw: str) -> str:
    first = _SENT_SPLIT.split(ans_raw.strip())[0]
    first = re.sub(r"\$.*?\$", " ", first)
    first = re.sub(r"[^A-Za-z\s]", " ", first).strip()
    return first or "reasoning unavailable"


def _compress_answer(ans_raw: str, model, tok, output_dir=None) -> str:
    global SUMMARY_LEN
    if not ans_raw.strip():
        return ""
    parsed_source = _generation_for_parsing(ans_raw)
    if parsed_source is None:
        return ""
    source = parsed_source.strip()
    prompt = _SUMMARY_PROMPT.format(source)

    model_name = model.llm_engine.model_config.model

    summary = _gen(model, tok, prompt, max_new=SUMMARY_LEN, temp=0.05, top_p=0.9).strip()
    summary = summary if summary else _fallback_summary(ans_raw)
    if output_dir:
        debug_jsonl(output_dir / "debug.jsonl", {"ans_raw": ans_raw, "summary": summary})
    return summary


# ---------- metric ----------
def mean_at_k(preds, refs):
    return sum(sum(_normalize_generation(c) == _normalize(r) for c in cand) / len(cand)
               for cand, r in zip(preds, refs)) / len(preds)


def maj_at_k(preds, refs):
    ls = []
    for cand, r in zip(preds, refs):
        dic = {}
        for c in cand:
            num = _normalize_generation(c)
            count = dic.get(num, 0)
            dic[num] = count + 1
        sorted_ans = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        if sorted_ans[0][0] == _normalize(r):
            ls.append(1.)
        else:
            ls.append(0.)
    return sum(ls) / len(ls)
    # return sum(sum(_normalize(c) == _normalize(r) for c in cand)/len(cand)
    #            for cand, r in zip(preds, refs)) / len(preds)


def _metric_answer(value, generated=False):
    normalized = _normalize_generation(value) if generated else _normalize(value)
    return None if normalized is None or normalized == "" else normalized


def _strict_unique_majority(candidates, include_invalid):
    answers = [_metric_answer(candidate, generated=True) for candidate in candidates]
    counts = Counter(
        answer if answer is not None else "__INVALID__"
        for answer in answers
        if include_invalid or answer is not None
    )
    if not counts:
        return None
    max_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == max_count]
    return winners[0] if len(winners) == 1 else None


def metric_snapshot(preds, refs):
    """Compute strict evaluation metrics for one rollout snapshot."""
    if len(preds) != len(refs):
        raise ValueError(
            f"Prediction/reference length mismatch: {len(preds)} != {len(refs)}"
        )
    if not preds:
        return {
            "mean@k": 0.0,
            "maj_s@k": 0.0,
            "acc_s": 0.0,
            "maj_in@k": 0.0,
            "acc_in": 0.0,
            "pass@k": 0.0,
        }

    mean_scores = []
    strict_majority_scores = []
    inclusive_majority_scores = []
    pass_scores = []
    for candidates, ref in zip(preds, refs):
        reference = _metric_answer(ref)
        answers = [
            _metric_answer(candidate, generated=True)
            for candidate in candidates
        ]
        denominator = len(answers)
        mean_scores.append(
            sum(answer is not None and answer == reference for answer in answers)
            / denominator
            if denominator
            else 0.0
        )
        strict_answer = _strict_unique_majority(candidates, include_invalid=False)
        inclusive_answer = _strict_unique_majority(candidates, include_invalid=True)
        strict_majority_scores.append(
            float(reference is not None and strict_answer == reference)
        )
        inclusive_majority_scores.append(
            float(reference is not None and inclusive_answer == reference)
        )
        pass_scores.append(
            float(reference is not None and reference in set(answers))
        )

    size = len(preds)
    strict_accuracy = sum(strict_majority_scores) / size
    inclusive_accuracy = sum(inclusive_majority_scores) / size
    return {
        "mean@k": sum(mean_scores) / size,
        "maj_s@k": strict_accuracy,
        # ICPO uses majority as its only answer-selection policy.
        "acc_s": strict_accuracy,
        "maj_in@k": inclusive_accuracy,
        "acc_in": inclusive_accuracy,
        "pass@k": sum(pass_scores) / size,
    }


def metric_series(round_preds, round_refs, final_preds, final_refs):
    snapshots = [
        metric_snapshot(predictions, references)
        for predictions, references in zip(round_preds, round_refs)
    ]
    snapshots.append(metric_snapshot(final_preds, final_refs))
    metric_names = (
        "mean@k",
        "maj_s@k",
        "acc_s",
        "maj_in@k",
        "acc_in",
        "pass@k",
    )
    return {
        name: [round(snapshot[name] * 100, 2) for snapshot in snapshots]
        for name in metric_names
    }


def compute_dynamics(round_preds, final_preds, refs):
    """Track strict-majority correctness from the first round to final evaluation."""
    if not round_preds:
        return {"w2c": 0, "c2w": 0, "net_correction": 0, "wrong_lock": 0}
    if any(len(predictions) != len(refs) for predictions in round_preds):
        raise ValueError("Round prediction/reference length mismatch")
    if len(final_preds) != len(refs):
        raise ValueError("Final prediction/reference length mismatch")

    w2c = 0
    c2w = 0
    wrong_lock = 0
    for question_idx, ref in enumerate(refs):
        reference = _metric_answer(ref)
        selected_answers = [
            _strict_unique_majority(
                predictions[question_idx], include_invalid=False
            )
            for predictions in round_preds
        ]
        selected_answers.append(
            _strict_unique_majority(
                final_preds[question_idx], include_invalid=False
            )
        )
        initial_answer = selected_answers[0]
        final_answer = selected_answers[-1]
        initial_correct = reference is not None and initial_answer == reference
        final_correct = reference is not None and final_answer == reference
        if not initial_correct and final_correct:
            w2c += 1
        if initial_correct and not final_correct:
            c2w += 1
        if (
                not initial_correct
                and initial_answer is not None
                and all(answer == initial_answer for answer in selected_answers)
        ):
            wrong_lock += 1
    return {
        "w2c": w2c,
        "c2w": c2w,
        "net_correction": w2c - c2w,
        "wrong_lock": wrong_lock,
    }


# =============================================================


def choose_entropy(pseudo, k_batch, qs, hist, model, tok, args, cache, max_new, out_dir, round):
    import copy
    from dataclasses import dataclass
    import random
    global EPSILON

    @dataclass
    class Candidate:
        idx: int
        summary: str
        reward: bool
        hist: list
        question: str

    selected_samples = []

    for i, (h, anss_raw, pseudo_raw) in enumerate(zip(hist, k_batch, pseudo)):
        candidates = []

        for j, ans_raw in enumerate(anss_raw):
            ans_nom = _normalize_generation(ans_raw)
            pseudo_nom = (
                _normalize(pseudo_raw)
                if args.use_gt_reward
                else _normalize_generation(pseudo_raw)
            )
            reward = ans_nom and ans_nom == pseudo_nom

            if args.only_reward_context:
                if ans_nom is not None:
                    label = "good" if reward else "bad"
                    summary = f"Answer {{{ans_nom}}}"
                    new_hist = copy.deepcopy(h)
                    update_history(new_hist, summary, reward, qs[i], model, tok, args, max_new)
                    candidates.append(Candidate(j, summary, reward, new_hist, qs[i]))
            else:
                summary = _compress_answer(ans_raw, model, tok,
                                           out_dir).strip()  # 对每个candidates summary后加入到临时hist中 跑并计算entropy

                if summary and ans_nom is not None:
                    summary += f"\nThus, the answer is boxed{{{ans_nom}}}"
                    new_hist = copy.deepcopy(h)
                    update_history(new_hist, summary, reward, qs[i], model, tok, args, max_new)
                    candidates.append(Candidate(j, summary, reward, new_hist, qs[i]))

        if DEBUG:
            logging.info(f"[eval] len candidates sample = {len(candidates)}")
        if len(candidates) == 0:
            selected_idx = random.randrange(args.k)
            selected_samples.append((None, 0, qs[i]))  # summary 是空的 此时所有candidates都不行，直接跳过这个question
            continue

        if len(candidates) == 1:
            selected_idx = candidates[0].idx
            selected_samples.append(
                (candidates[0].summary, candidates[0].reward, qs[i]))  # summary 只有一个 就选这个了，也不用算entropy
            continue

        # 多个候选摘要时，用 entropy 选择
        prompts = [
            build_prompt(
                c.question,
                c.hist,
                tok,
                args.prompt_format,
                args.enable_thinking,
            )
            for c in candidates
        ]
        new_batch = generate_batch(model, tok, prompts, args.entropy_k, max_new, args.temp, args.top_p)
        entropies = [_get_entropy(lst, args.entropy_k) for lst in new_batch]
        #############
        final_out_file = out_dir / "entropy.jsonl"
        with open(final_out_file, "a", encoding="utf-8") as f:
            json_line = json.dumps({"round": round, "entropies": entropies}, ensure_ascii=False)
            f.write(json_line + "\n")
        ############
        if DEBUG:
            logging.info(f"[eval] entropy sample = {entropies}")
        if any(e is not None for e in entropies):
            valid_idx = [(i, e) for i, e in enumerate(entropies) if e is not None]
            if random.random() < EPSILON:
                min_idx = random.choice(valid_idx)[0]
            else:
                min_idx = min(
                    valid_idx,
                    key=lambda x: x[1]
                )[0]  # 如果有能计算entropy的，选择argmin
        else:
            min_idx = random.randint(0, len(candidates) - 1)  # 没有能计算entropy的 随便选一个

        selected = candidates[min_idx]
        if DEBUG:
            logging.info(f"[eval] min_idx = {min_idx}")
        selected_samples.append((selected.summary, selected.reward, qs[i]))
    history_entries_written = 0
    for h, new_ans in zip(hist, selected_samples):
        summary, reward, q = new_ans
        if summary:
            history_entries_written += int(
                update_history(h, summary, reward, q, model, tok, args, max_new)
            )
    return history_entries_written


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_global_variable(args):
    global ENABLE_PENALTY
    global SUMMARY_LEN
    global SYS_PROMPT
    global _SUMMARY_PROMPT
    global USE_REWARD
    global BASE_MODEL
    global EPSILON
    global BOXED_ANSWER_POSITION
    SUMMARY_LEN = args.summary_length
    ENABLE_PENALTY = args.enable_penalty
    USE_REWARD = args.use_reward
    BOXED_ANSWER_POSITION = args.boxed_answer_position
    if "AIME" in args.task_dir:
        setting.BENCHMARK = "AIME"
    elif "AMC" in args.task_dir:
        setting.BENCHMARK = "AMC"
    elif "GPQA" in args.task_dir:
        setting.BENCHMARK = "GPQA"
    elif "MATH" in args.task_dir:
        setting.BENCHMARK = "MATH"
    else:
        print("Not a valid benchmark")
        sys.exit(1)
    if args.rounds == 0:
        BASE_MODEL = True
    SYS_PROMPT, _SUMMARY_PROMPT = get_prompt(setting.BENCHMARK, USE_REWARD, args.rounds)

    set_seed(args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--csv_path", default=f"{ROOT_DIR}/results/icrl0/metrics.csv")
    ap.add_argument("--k", type=int, default=cfg("k", 16))
    ap.add_argument("--batch", type=int, default=cfg("batch", 4))
    ap.add_argument("--temp", type=float, default=cfg("temperature", 0.6))
    ap.add_argument("--top_p", type=float, default=cfg("top_p", 0.95))
    ap.add_argument("--ctx", type=int, default=cfg("context_len", 3072))
    ap.add_argument("--rounds", type=int, default=cfg("rounds", 3))
    ap.add_argument("--entropy", type=bool, default=cfg("entropy", True))
    ap.add_argument("--entropy_k", type=int, default=cfg("entropy_k", 16))
    ap.add_argument("--enable_penalty", type=bool, default=cfg("entropy_penalty", True))
    ap.add_argument("--summary_length", type=int, default=cfg("summary_length", 500))
    ap.add_argument("--answer_length", type=int, default=cfg("answer_length", 5000))
    ap.add_argument(
        "--prompt_format",
        choices=("raw", "chat_template"),
        default=cfg("prompt_format", "raw"),
        help="Render reasoning prompts as original raw text or tokenizer chat messages.",
    )
    ap.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=cfg("enable_thinking", None),
        help="Enable or disable thinking through a tokenizer chat template.",
    )
    ap.add_argument(
        "--boxed_answer_position",
        choices=("first", "last"),
        default=cfg("boxed_answer_position", "last"),
        help="Parse the first boxed answer after truncating continuations, or retain legacy last-boxed parsing.",
    )
    ap.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=cfg("gpu_memory_utilization", 0.9),
        help="Fraction of GPU memory reserved by this vLLM instance.",
    )
    ap.add_argument("--final_gen_k", type=int, default=cfg("final_gen_k", 64))
    ap.add_argument("--test_sample", type=int, default=cfg("test_sample", None))
    ap.add_argument("--start_idx", type=int, default=cfg("start_idx", None))
    ap.add_argument("--end_idx", type=int, default=cfg("end_idx", None))
    ap.add_argument("--seed", type=int, default=cfg("seed", 42))
    ap.add_argument("--use_reward", type=bool, default=True)  # For ablation study
    ap.add_argument("--use_gt_reward", type=bool, default=cfg("use_gt_reward", False))  # For ablation study
    ap.add_argument("--epsilon", type=float, default=cfg("epsilon", 0.0))
    ap.add_argument("--only_reward_context", type=bool, default=False)  # For ablation study
    ap.add_argument_group
    args = ap.parse_args()
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")

    set_global_variable(args)

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(f"{args.output_dir}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    # ---- logging ----
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(out_dir / "debug.log", "w", "utf-8"),
                logging.StreamHandler()
            ]
        )
        logging.info("DEBUG MODE ON")

    model = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tokenizer_mode="auto",
        tensor_parallel_size=TENSOR_PARALLEL,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=args.ctx,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tok = model.get_tokenizer()

    # ---------- load dataset ----------
    ds_path = Path(args.task_dir)
    file = "test.parquet" if (ds_path / "test.parquet").exists() else "test.json"
    dataset = load_dataset(
        "parquet" if file.endswith(".parquet") else "json",
        data_files=str(ds_path / file), split="train"
    )
    if args.test_sample is not None:
        dataset = dataset.select(range(args.test_sample))
    start_idx = 0 if args.start_idx is None else args.start_idx
    end_idx = len(dataset) if args.end_idx is None else args.end_idx
    dataset = dataset.select(range(start_idx, end_idx))

    cache = PromptCache(out_dir / ".prompt_cache")
    preds, refs, ans_records = [], [], []
    round_preds = [[] for _ in range(args.rounds)]
    round_refs = [[] for _ in range(args.rounds)]
    history_entries_written_by_round = [0 for _ in range(args.rounds)]
    wrote_example = False

    pbar = tqdm(range(0, len(dataset), args.batch), unit="batch", desc="ICRL‑0‑m")

    for st in pbar:
        ed = min(st + args.batch, len(dataset))
        sub = dataset.select(range(st, ed))
        qs = [ex.get("prompt") or ex["problem"] for ex in sub]
        refs_batch = [ex.get("answer") or ex.get("solution") or "" for ex in sub]
        refs += refs_batch

        hist = [{"good": [], "bad": []} for _ in sub]

        # ---------- bootstrap ----------
        for r in range(1, args.rounds + 1):
            prompts = []
            for q, h in zip(qs, hist):
                key = f"{hash(q)}-r{r}-g{len(h['good'])}-b{len(h['bad'])}"
                p = cache.get(key) or build_prompt(
                    q,
                    h,
                    tok,
                    args.prompt_format,
                    args.enable_thinking,
                )
                cache.put(key, p);
                prompts.append(p)

            max_new = min(args.answer_length, args.ctx - max(len(tok(p).input_ids) for p in prompts))
            max_new = max(max_new, 1)
            k_batch = generate_batch(model, tok, prompts, args.k,
                                     max_new, args.temp, args.top_p)
            round_preds[r - 1].extend(k_batch)
            round_refs[r - 1].extend(refs_batch)

            pseudo = [_majority_raw(lst) for lst in k_batch]

            # ===== 新增：记录每轮多数投票正确率 =====
            for qi, (q, pseudo_ans, ref_raw, klist) in enumerate(
                    zip(qs, pseudo, refs_batch, k_batch)
            ):
                ref_norm = _normalize(ref_raw)
                pseudo_norm = _normalize_generation(pseudo_ans)
                # 统计k个候选中正确答案的个数
                n_correct_in_k = sum(
                    1 for a in klist
                    if _normalize_generation(a)
                    and _normalize_generation(a) == ref_norm
                )
                # 统计答案分布
                from collections import Counter
                answer_dist = dict(Counter(
                    _normalize_generation(a)
                    for a in klist
                    if _normalize_generation(a) is not None
                ))
                debug_jsonl(out_dir / "majority_trace.jsonl", {
                    "round": r,
                    "question_idx": st + qi,
                    "ref": ref_norm,
                    "majority_answer": pseudo_norm,
                    "majority_correct": bool(pseudo_norm and pseudo_norm == ref_norm),
                    "n_correct_in_k": n_correct_in_k,
                    "answer_distribution": answer_dist,
                })
            # ===== 新增结束 =====

            if args.use_gt_reward == True:
                pseudo = refs_batch
            if args.entropy == False:
                rand_idx = [random.randrange(args.k) for _ in prompts]
                picked = [lst[i] for lst, i in zip(k_batch, rand_idx)]

                for q, h, ans_raw, pseudo_raw in zip(q, hist, picked, pseudo):
                    ans_nom = _normalize_generation(ans_raw)
                    pseudo_nom = (
                        _normalize(pseudo_raw)
                        if args.use_gt_reward
                        else _normalize_generation(pseudo_raw)
                    )
                    reward = ans_nom and ans_nom == pseudo_nom
                    if args.only_reward_context:
                        if ans_nom is not None:
                            label = "good" if reward else "bad"
                            summary = f"Answer {{{ans_nom}}}"
                            history_entries_written_by_round[r - 1] += int(
                                update_history(
                                    h, summary, reward, q, model, tok, args, max_new
                                )
                            )
                    else:
                        summary = _compress_answer(ans_raw, model, tok, out_dir).strip()
                        if summary and ans_nom is not None:
                            summary += f"\nThus, the answer is boxed{{{ans_nom}}}"  # 只记录非空摘要
                            history_entries_written_by_round[r - 1] += int(
                                update_history(
                                    h, summary, reward, q, model, tok, args, max_new
                                )
                            )
            else:  # 判断是否需要计算entropy
                history_entries_written_by_round[r - 1] += choose_entropy(
                    pseudo=pseudo,
                    k_batch=k_batch,
                    qs=qs,
                    hist=hist,
                    model=model,
                    tok=tok,
                    args=args,
                    cache=cache,
                    max_new=max_new,
                    out_dir=out_dir,
                    round=r
                )

        # ---------- evaluation ----------
        final_prompts = [
            build_prompt(
                q,
                h,
                tok,
                args.prompt_format,
                args.enable_thinking,
            )
            for q, h in zip(qs, hist)
        ]
        max_new = min(args.answer_length, args.ctx - max(len(tok(p).input_ids) for p in final_prompts))
        max_new = max(max_new, 1)
        final_k = generate_batch(model, tok, final_prompts, args.final_gen_k,
                                 max_new, args.temp, args.top_p)
        preds += final_k

        if DEBUG:
            logging.info(
                f"[eval] normalized k list sample = "
                f"{[_normalize_generation(x) for x in final_k[0]]}"
            )

        if not wrote_example:
            (out_dir / "example.txt").write_text(
                f"PROMPT:\n{final_prompts[0]}\n\nANSWER:\n{final_k[0][0]}",
                encoding="utf-8"
            )
            wrote_example = True

        # ----- final output and answer for debugging--------
        final_out_file = out_dir / "prompts_and_preds.jsonl"
        with open(final_out_file, "a", encoding="utf-8") as f:
            for prompt, gens in zip(final_prompts, final_k):
                # 每条写入一行 JSON
                json_line = json.dumps({"prompt": prompt, "gens": gens}, ensure_ascii=False)
                f.write(json_line + "\n")
        # --------- record answers ---------
        for idx, (klist, ex, ref_raw) in enumerate(zip(final_k, sub, refs_batch)):
            ref_norm = _normalize(ref_raw)

            cand_list = [{"raw": a,
                          "numeric": _normalize_generation(a),
                          "correct": bool(
                              _normalize_generation(a)
                              and _normalize_generation(a) == ref_norm
                          )}
                         for a in klist]
            ans_records.append({
                "id": ex.get("id", start_idx + st + idx),
                "ref": ref_norm,
                "candidates": cand_list,
                "normalized_candidates": [c["numeric"] for c in cand_list]
            })
            (out_dir / "answer.json").write_text(
                json.dumps(ans_records, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            # ===== 新增：记录最终评估轮到 majority_trace =====
            from collections import Counter
            n_correct_in_k = sum(1 for c in cand_list if c["correct"])
            answer_dist = dict(Counter(
                c["numeric"] for c in cand_list if c["numeric"] is not None
            ))
            # 从 cand_list 算多数投票
            valid_answers = [c["numeric"] for c in cand_list if c["numeric"] is not None]
            majority_answer = Counter(valid_answers).most_common(1)[0][0] if valid_answers else None

            debug_jsonl(out_dir / "majority_trace.jsonl", {
                "round": "final",
                "question_idx": st + idx,
                "ref": ref_norm,
                "majority_answer": majority_answer,
                "majority_correct": bool(majority_answer and majority_answer == ref_norm),
                "n_correct_in_k": n_correct_in_k,
                "answer_distribution": answer_dist,
            })
            # ===== 新增结束 =====

    # ---------- write outputs ----------
    mean_k = round(mean_at_k(preds, refs) * 100, 2)
    maj_k = round(maj_at_k(preds, refs) * 100, 2)
    final_metric_snapshot = metric_snapshot(preds, refs)
    metrics_by_round = metric_series(round_preds, round_refs, preds, refs)
    dynamics = compute_dynamics(round_preds, preds, refs)
    metric_steps = [
        *(f"round{round_idx}" for round_idx in range(1, args.rounds + 1)),
        "final",
    ]
    history_entries_written = sum(history_entries_written_by_round)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps({"gens": g}, ensure_ascii=False) for g in preds),
        encoding="utf-8"
    )
    (out_dir / "answer.json").write_text(
        json.dumps(ans_records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    metric = {"model": Path(args.model_path).name, "task": ds_path.name,
              "mean@k": mean_k,
              "maj@k": maj_k,
              "maj_s@k": round(final_metric_snapshot["maj_s@k"] * 100, 2),
              "acc_s": round(final_metric_snapshot["acc_s"] * 100, 2),
              "maj_in@k": round(final_metric_snapshot["maj_in@k"] * 100, 2),
              "acc_in": round(final_metric_snapshot["acc_in"] * 100, 2),
              "pass@k": round(final_metric_snapshot["pass@k"] * 100, 2),
              "metric_steps": metric_steps,
              "mean@k_by_round": metrics_by_round["mean@k"],
              "maj_s@k_by_round": metrics_by_round["maj_s@k"],
              "acc_s_by_round": metrics_by_round["acc_s"],
              "maj_in@k_by_round": metrics_by_round["maj_in@k"],
              "acc_in_by_round": metrics_by_round["acc_in"],
              "pass@k_by_round": metrics_by_round["pass@k"],
              "feedback_write_attempts": len(dataset) * args.rounds,
              "feedback_write_accepted": history_entries_written,
              "feedback_write_acceptance_rate": (
                  history_entries_written / (len(dataset) * args.rounds)
                  if len(dataset) and args.rounds
                  else 0.0
              ),
              "history_entries_written": history_entries_written,
              "history_entries_written_by_round": history_entries_written_by_round,
              "answer_selection_policy": "majority",
              "dynamics_selection_protocol": "strict_valid_majority",
              **dynamics,
              "k": args.k, "rounds": args.rounds,
              "batch": args.batch, "temp": args.temp, "top_p": args.top_p,
              "context_len": args.ctx,
              "prompt_format": args.prompt_format,
              "enable_thinking": args.enable_thinking,
              "boxed_answer_position": args.boxed_answer_position,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "entropy_k": args.entropy_k,
              "final_gen_k": args.final_gen_k,
              "task_number": len(dataset),
              "seed": args.seed,
              "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
              "total_time": (datetime.datetime.now() - start_time).total_seconds(),
              "Peak_GPU": round(PEAK_GPU_GB, 2),
              "Tokens": TOTAL_TOKEN,
              "total_tokens": int(TOTAL_TOKEN)}
    (out_dir / "metrics.json").write_text(json.dumps(metric, indent=2),
                                          encoding="utf-8")

    csv_path = Path(args.csv_path);
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    head = not csv_path.exists()
    with csv_path.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(metric.keys()))
        if head: w.writeheader()
        w.writerow(metric)

    print(f"\n[✓] {metric['model']} on {metric['task']} → mean@k {metric['mean@k']}%")


if __name__ == "__main__":
    main()
