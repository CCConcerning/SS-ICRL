from __future__ import annotations

"""Strategy-Structured In-Context Reinforcement Learning (SS-ICRL).

The paper method follows one explicit path: fixed reasoning strategies,
Reliability-Hill pseudo-label selection, strategy-specific trajectory feedback,
and multi-round in-context updates. Ground-truth answers are used only for
evaluation.
"""

import argparse
import csv
import datetime
import html
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

import icrl.icrl_runner as icpo
from icrl.multiview_answer_adapter import (
    canonicalize_generation_answer,
    canonicalize_reference_answer,
    extract_current_assistant_span,
    truncate_after_boxed_answer,
)
from icrl.setting import setting


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_YML = Path(
    os.environ.get(
        "SS_ICRL_CONFIG",
        ROOT_DIR / "configs" / "ss_icrl.yaml",
    )
)

HistoryEntry = Dict[str, Any]
PrivateHistory = Dict[str, List[HistoryEntry]]


_SYSTEM_PROMPTS = {
    "AIME": (
        "You are an AI mathematician. All content you output MUST be in English.\n"
        "**You are only allowed to provide explanations in plain English. Do NOT write any code, "
        "pseudocode, or technical snippets.**\n"
        "**Finish all your reasoning, then on a NEW line output exactly one number "
        "(the answer) and nothing else.**\n"
        "Your final output MUST be in the format boxed{<number>}, where <number> is the "
        "final numeric answer only (no expressions, variables, or additional text)."
        "The content inside boxed{ } must be a decimal number, not a fraction or any other form."
    ),
    "AMC": (
        "You are an AI mathematician. All content you output MUST be in English.\n"
        "**You are only allowed to provide explanations in plain English. Do NOT write any code, "
        "pseudocode, or technical snippets.**\n"
        "**Finish all your reasoning, then on a NEW line output exactly one number "
        "(the answer) and nothing else.**\n"
        "Your final output MUST be in the format boxed{<number>}, where <number> is the "
        "final numeric answer only (no expressions, variables, or additional text)."
        "The content inside boxed{ } must be a decimal number, not a fraction or any other form."
    ),
    "MATH": (
        "You are an AI mathematician. All content you output MUST be in English.\n"
        "**You are only allowed to provide explanations in plain English. Do NOT write any code, "
        "pseudocode, or technical snippets.**\n"
        "**Finish all your reasoning, then on a NEW line output exactly one answer "
        "(the answer) and nothing else.**\n"
        "Your final output MUST be in the format boxed{<answer>}, where <answer> is the "
        "final answer in any form (number, fraction, expression, or text), without extra explanation or text."
    ),
    "GPQA": (
        "You are an AI mathematician. All content you output MUST be in English.\n"
        "**You are only allowed to provide explanations in plain English. Do NOT write any code, "
        "pseudocode, or technical snippets.**\n"
        "**Finish all your reasoning, then on a NEW line output exactly one letter "
        "(the answer) and nothing else.**\n"
        "Your final output MUST be in the format boxed{<letter>}, where <letter> is exactly one of A, B, C, D."
    ),
}

_HISTORY_INSTRUCTIONS = {
    "AIME": "Use the question and these ideas to deduce the correct numeric answer.",
    "AMC": "Use the question and these ideas to deduce the correct numeric answer.",
    "MATH": "Use the question and these ideas to deduce the correct answer.",
    "GPQA": "Use the question and these ideas to deduce the correct choice.",
}


def load_yaml_defaults() -> Dict[str, Any]:
    if not CONFIG_YML.is_file():
        return {}
    with CONFIG_YML.open(encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {CONFIG_YML}")
    return data


_YAML_DEFAULTS = load_yaml_defaults()


def cfg(key: str, fallback: Any) -> Any:
    value = _YAML_DEFAULTS.get(key)
    return fallback if value is None else value


def parse_views(view_arg: Optional[str]) -> List[Tuple[str, str]]:
    configured = _YAML_DEFAULTS.get("views")
    if not isinstance(configured, dict) or not configured:
        raise ValueError(f"{CONFIG_YML} must define a non-empty 'views' mapping")
    names = list(configured) if not view_arg else [x.strip() for x in view_arg.split(",") if x.strip()]
    unknown = [name for name in names if name not in configured]
    if unknown:
        raise ValueError(f"Unknown strategies {unknown}; configured strategies are {list(configured)}")
    return [(name, str(configured[name])) for name in names]


def infer_benchmark(task_dir: str) -> str:
    name = Path(task_dir).name.upper()
    for benchmark in ("AIME", "AMC", "GPQA", "MATH"):
        if benchmark in name:
            return benchmark
    raise ValueError(f"Cannot infer benchmark from task_dir={task_dir!r}")


def append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_seeded_vllm(model_path: str, seed: int, gpu_memory_utilization: float) -> LLM:
    kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "tokenizer_mode": "auto",
        "tensor_parallel_size": icpo.TENSOR_PARALLEL,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "seed": int(seed),
        "gpu_memory_utilization": float(gpu_memory_utilization),
    }
    if os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN") == "1":
        kwargs["max_model_len"] = 8192
    return LLM(**kwargs)


def answer_key(value: Any, include_invalid: bool = False) -> Optional[str]:
    normalized = icpo._normalize(str(value)) if value is not None else None
    if normalized is None:
        return "__INVALID__" if include_invalid else None
    if isinstance(normalized, float):
        return str(int(normalized)) if normalized.is_integer() else repr(normalized)
    return str(normalized)


def generation_answer(
    raw: Any,
    use_answer_adapter: bool,
    boxed_answer_position: str,
) -> Optional[str]:
    current_turn = extract_current_assistant_span(raw)
    if use_answer_adapter:
        value = canonicalize_generation_answer(
            current_turn,
            setting.BENCHMARK,
            boxed_answer_position=boxed_answer_position,
        )
    elif boxed_answer_position == "first":
        value = truncate_after_boxed_answer(current_turn, position="first")
    else:
        value = current_turn
    return answer_key(value)


def reference_answer(raw: Any, use_answer_adapter: bool) -> Optional[str]:
    value = (
        canonicalize_reference_answer(raw, setting.BENCHMARK)
        if use_answer_adapter
        else raw
    )
    return answer_key(value)


def reliability_hill_scores(
    view_answers: Dict[str, List[Optional[str]]],
    view_budgets: Dict[str, int],
    include_invalid: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Compute the Reliability-Hill score from strategy-level answer counts."""
    view_names = list(view_answers)
    candidate_order: List[str] = []
    counts_by_view: Dict[str, Counter] = {}
    for view_name in view_names:
        keys = [
            answer if answer is not None else "__INVALID__"
            for answer in view_answers[view_name]
            if include_invalid or answer is not None
        ]
        counts_by_view[view_name] = Counter(keys)
        for answer in keys:
            if answer not in candidate_order:
                candidate_order.append(answer)

    if not candidate_order:
        return {}, []

    answer_space = sorted(candidate_order)

    view_probs = {
        view_name: {
            answer: counts_by_view[view_name].get(answer, 0) / view_budgets[view_name]
            for answer in answer_space
        }
        for view_name in view_names
    }

    view_weights: Dict[str, float] = {}
    view_hill_numbers: Dict[str, float] = {}
    for view_name in view_names:
        valid_mass = sum(view_probs[view_name].values())
        if valid_mass <= 0.0:
            view_weights[view_name] = 0.0
            view_hill_numbers[view_name] = 0.0
            continue
        collision = sum(
            (view_probs[view_name][answer] / valid_mass) ** 2
            for answer in answer_space
        )
        view_weights[view_name] = collision
        view_hill_numbers[view_name] = 1.0 / collision

    total_weight = sum(view_weights.values())
    scores: Dict[str, Dict[str, Any]] = {}
    for answer in answer_space:
        weighted_support_by_view = {
            view_name: view_weights[view_name] * view_probs[view_name][answer]
            for view_name in view_names
        }
        weighted_mass = sum(weighted_support_by_view.values())
        reliability = weighted_mass / total_weight if total_weight > 0.0 else 0.0
        q = {
            view_name: value / weighted_mass
            for view_name, value in weighted_support_by_view.items()
            if value > 0.0 and weighted_mass > 0.0
        }
        if q:
            effective_view_count = 1.0 / sum(probability**2 for probability in q.values())
            effective_view_coverage = effective_view_count / len(view_names)
        else:
            effective_view_count = 0.0
            effective_view_coverage = 0.0
        score = reliability * (1.0 + effective_view_coverage) / 2.0
        scores[answer] = {
            "view_probs": {
                view_name: view_probs[view_name][answer] for view_name in view_names
            },
            "view_internal_hill_numbers": dict(view_hill_numbers),
            "view_reliabilities": dict(view_weights),
            "reliability_weighted_support": reliability,
            "reliability_adjusted_view_distribution": q,
            "reliability_adjusted_effective_view_count": effective_view_count,
            "reliability_adjusted_effective_view_coverage": effective_view_coverage,
            "reliability_hill_score": score,
            "score": score,
            "raw_score": score,
            "feedback_reward": score,
        }
    return scores, candidate_order


def select_pseudo_label(
    view_answers: Dict[str, List[Optional[str]]],
    view_budgets: Dict[str, int],
    *,
    include_invalid: bool = False,
    random_ties: bool = True,
) -> Dict[str, Any]:
    scores, candidate_order = reliability_hill_scores(
        view_answers,
        view_budgets,
        include_invalid=include_invalid,
    )
    if not scores:
        return {
            "policy": "reliability_hill",
            "scores": {},
            "selected_answer": None,
            "selected_answer_key": None,
            "selected_tied_answers": [],
            "tie_break": "seeded_random" if random_ties else "first_occurrence",
            "selection_include_invalid": include_invalid,
            "reason": "empty_answer_space",
        }
    best_score = max(item["raw_score"] for item in scores.values())
    tie_order = sorted(scores) if random_ties else candidate_order
    tied = [answer for answer in tie_order if scores[answer]["raw_score"] == best_score]
    selected = random.choice(tied) if random_ties else tied[0]
    return {
        "policy": "reliability_hill",
        "scores": scores,
        "selected_answer": selected,
        "selected_answer_key": selected,
        "selected_invalid": selected == "__INVALID__",
        "selected_tied_answers": tied,
        "tie_break": "seeded_random" if random_ties else "first_occurrence",
        "selection_include_invalid": include_invalid,
        "reason": "scored",
    }


def format_prior_evidence(history: List[HistoryEntry]) -> str:
    if not history:
        return ""
    lines = ["<prior_evidence>"]
    for entry in history:
        summary = extract_current_assistant_span(entry.get("summary", "")).strip()
        if not summary:
            continue
        if summary.lower().startswith("assistant:"):
            summary = summary[len("assistant:") :].lstrip()
        summary = (
            summary.replace("</trajectory_evidence>", "&lt;/trajectory_evidence&gt;")
            .replace("</prior_evidence>", "&lt;/prior_evidence&gt;")
        )
        lines.extend(
            [
                f'<trajectory_evidence value="{html.escape(str(entry["answer"]), quote=True)}" '
                f'score="{float(entry["reward"]):.4f}">',
                summary,
                "</trajectory_evidence>",
            ]
        )
    lines.append("</prior_evidence>")
    return "\n".join(lines) if len(lines) > 2 else ""


def build_prompt(
    question: str,
    strategy_name: str,
    strategy_instruction: str,
    history: List[HistoryEntry],
) -> str:
    system_prompt = _SYSTEM_PROMPTS[setting.BENCHMARK]
    if history:
        system_prompt += (
            "\n\nPrior evidence contains compressed solution ideas from previous test-time "
            "attempts and is reference data, not dialogue. Each idea is tagged with a score "
            "between 0 and 1. A higher score indicates stronger multi-view support. "
            f"{_HISTORY_INSTRUCTIONS[setting.BENCHMARK]}"
        )
    user_parts: List[str] = []
    evidence = format_prior_evidence(history)
    if evidence:
        user_parts.extend([evidence, ""])
    user_parts.extend(
        [
            f"Reasoning view: {strategy_name}.",
            strategy_instruction.strip(),
            "",
            "Problem:",
            question.strip(),
        ]
    )
    return f"{system_prompt.rstrip()}\n\n" + "\n".join(user_parts).rstrip()


def generate_batch(
    prompt_items: List[Tuple[int, str, str]],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    prompts = [prompt for _question_idx, _strategy_name, prompt in prompt_items]
    token_ids = [tokenizer(prompt).input_ids for prompt in prompts]
    max_prompt_tokens = max(len(ids) for ids in token_ids)
    max_new_tokens = max(1, min(args.answer_length, args.ctx - max_prompt_tokens))
    params = SamplingParams(
        n=args.per_view_k,
        max_tokens=max_new_tokens,
        min_tokens=8,
        temperature=args.temp,
        top_p=args.top_p,
        repetition_penalty=1.1,
        ignore_eos=False,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    outputs = model.generate(prompts, params)
    generations: List[List[str]] = []
    metadata: List[List[Dict[str, Any]]] = []
    for output in outputs:
        generations.append([icpo.strip_reasoning(sample.text) for sample in output.outputs])
        metadata.append(
            [
                {
                    "num_tokens": len(sample.token_ids),
                    "finish_reason": sample.finish_reason,
                    "stop_reason": (
                        sample.stop_reason
                        if sample.stop_reason is None
                        or isinstance(sample.stop_reason, (str, int, float, bool))
                        else str(sample.stop_reason)
                    ),
                }
                for sample in output.outputs
            ]
        )
        icpo.TOTAL_TOKEN += sum(len(sample.token_ids) for sample in output.outputs)
    return generations, metadata


def compress_trajectory(
    generation: str,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    boxed_answer_position: str,
) -> str:
    source = extract_current_assistant_span(generation)
    if boxed_answer_position == "first":
        source = truncate_after_boxed_answer(source, position="first")
    if not source.strip():
        return ""
    summary = icpo._compress_answer(source.strip(), model, tokenizer, output_dir)
    return extract_current_assistant_span(summary).strip()


def initialize_histories(
    num_questions: int,
    views: List[Tuple[str, str]],
) -> List[PrivateHistory]:
    return [
        {strategy_name: [] for strategy_name, _instruction in views}
        for _ in range(num_questions)
    ]


def generate_phase(
    questions: List[str],
    references: List[str],
    histories: List[PrivateHistory],
    views: List[Tuple[str, str]],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    phase: str,
    round_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    prompt_items: List[Tuple[int, str, str]] = []
    for question_idx, question in enumerate(questions):
        for strategy_name, instruction in views:
            prompt_items.append(
                (
                    question_idx,
                    strategy_name,
                    build_prompt(
                        question,
                        strategy_name,
                        instruction,
                        histories[question_idx][strategy_name],
                    ),
                )
            )

    generated, generation_metadata = generate_batch(
        prompt_items,
        model,
        tokenizer,
        args,
    )
    records: List[Dict[str, Any]] = [
        {
            "phase": phase,
            **({"round": round_idx} if round_idx is not None else {}),
            "views": {
                strategy_name: {
                    "instruction": instruction,
                    "gens": [],
                    "generation_metadata": [],
                    "answers": [],
                    **({"prompt": None} if args.save_prompts else {}),
                }
                for strategy_name, instruction in views
            },
            "history_lengths_before": {
                strategy_name: len(histories[question_idx][strategy_name])
                for strategy_name, _instruction in views
            },
        }
        for question_idx in range(len(questions))
    ]

    for (question_idx, strategy_name, prompt), generations, metadata in zip(
        prompt_items,
        generated,
        generation_metadata,
    ):
        view_record = records[question_idx]["views"][strategy_name]
        view_record["gens"] = generations
        view_record["generation_metadata"] = metadata
        view_record["answers"] = [
            generation_answer(
                generation,
                args.use_answer_adapter,
                args.boxed_answer_position,
            )
            for generation in generations
        ]
        if args.save_prompts:
            view_record["prompt"] = prompt

    view_budgets = {strategy_name: args.per_view_k for strategy_name, _ in views}
    for question_idx, record in enumerate(records):
        view_answers = {
            strategy_name: record["views"][strategy_name]["answers"]
            for strategy_name, _instruction in views
        }
        operational = select_pseudo_label(
            view_answers,
            view_budgets,
            random_ties=True,
        )
        evaluation = select_pseudo_label(
            view_answers,
            view_budgets,
            random_ties=False,
        )
        inclusive_evaluation = select_pseudo_label(
            view_answers,
            view_budgets,
            include_invalid=True,
            random_ties=False,
        )
        ref = reference_answer(references[question_idx], args.use_answer_adapter)
        operational["selected_correct"] = bool(
            ref is not None and operational["selected_answer"] == ref
        )
        evaluation["selected_correct"] = bool(
            ref is not None and evaluation["selected_answer"] == ref
        )
        inclusive_evaluation["selected_correct"] = bool(
            ref is not None and inclusive_evaluation["selected_answer"] == ref
        )
        record.update(
            {
                "ref": ref,
                "answer_selection": operational,
                "global_answer": operational["selected_answer"],
                "global_correct": operational["selected_correct"],
                "evaluation_answer_selection": evaluation,
                "evaluation_answer": evaluation["selected_answer"],
                "evaluation_correct": evaluation["selected_correct"],
                "inclusive_evaluation_answer_selection": inclusive_evaluation,
                "flat_generations": [
                    generation
                    for strategy_name, _instruction in views
                    for generation in record["views"][strategy_name]["gens"]
                ],
                "flat_answers": [
                    answer
                    for strategy_name, _instruction in views
                    for answer in record["views"][strategy_name]["answers"]
                ],
            }
        )
    return records


def update_strategy_histories(
    records: List[Dict[str, Any]],
    histories: List[PrivateHistory],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    round_idx: int,
    boxed_answer_position: str,
) -> int:
    """Write one supporting trajectory to each strategy-specific history."""
    pending_updates: List[Dict[str, Optional[HistoryEntry]]] = []
    for record in records:
        pseudo_label = record["global_answer"]
        scores = record["answer_selection"]["scores"]
        updates: Dict[str, Optional[HistoryEntry]] = {
            strategy_name: None for strategy_name in record["views"]
        }
        if pseudo_label is not None:
            reward = float(scores[pseudo_label]["feedback_reward"])
            for strategy_name, view_record in record["views"].items():
                supporting_indices = [
                    index
                    for index, answer in enumerate(view_record["answers"])
                    if answer == pseudo_label
                ]
                if not supporting_indices:
                    continue
                selected_index = supporting_indices[0]
                summary = compress_trajectory(
                    view_record["gens"][selected_index],
                    model,
                    tokenizer,
                    output_dir,
                    boxed_answer_position,
                )
                if not summary:
                    continue
                updates[strategy_name] = {
                    "round": round_idx,
                    "answer": pseudo_label,
                    "reward": reward,
                    "summary": summary,
                    "selected_index": selected_index,
                }
        pending_updates.append(updates)

    written = 0
    for question_idx, updates in enumerate(pending_updates):
        for strategy_name, entry in updates.items():
            if entry is None:
                continue
            histories[question_idx][strategy_name].append(entry)
            written += 1
        records[question_idx]["feedback_updates"] = updates
        records[question_idx]["history_lengths_after"] = {
            strategy_name: len(items)
            for strategy_name, items in histories[question_idx].items()
        }
    return written


def generation_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_counts: List[int] = []
    finish_reasons: Counter = Counter()
    total = boxed = multi_boxed = valid = 0
    for record in records:
        for view_record in record["views"].values():
            for generation, answer, metadata in zip(
                view_record["gens"],
                view_record["answers"],
                view_record["generation_metadata"],
            ):
                total += 1
                box_count = generation.count(r"\boxed")
                boxed += int(box_count > 0)
                multi_boxed += int(box_count > 1)
                valid += int(answer is not None)
                token_counts.append(int(metadata["num_tokens"]))
                finish_reasons[str(metadata["finish_reason"])] += 1
    return {
        "num_generations": total,
        "mean_response_tokens": mean(token_counts),
        "max_response_tokens": max(token_counts) if token_counts else 0,
        "finish_reason_counts": dict(finish_reasons),
        "boxed_rate": boxed / total if total else 0.0,
        "multi_boxed_rate": multi_boxed / total if total else 0.0,
        "valid_answer_rate": valid / total if total else 0.0,
    }


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def mean_at_k(
    predictions: List[List[Optional[str]]],
    references: List[Optional[str]],
) -> float:
    return mean(
        mean(float(answer is not None and answer == reference) for answer in answers)
        for answers, reference in zip(predictions, references)
    )


def first_majority_answer(
    answers: List[Optional[str]],
    *,
    include_invalid: bool,
) -> Optional[str]:
    keys = [
        answer if answer is not None else "__INVALID__"
        for answer in answers
        if include_invalid or answer is not None
    ]
    if not keys:
        return None
    counts = Counter(keys)
    maximum = max(counts.values())
    return next(answer for answer in keys if counts[answer] == maximum)


def strict_majority_answer(
    answers: List[Optional[str]],
    *,
    include_invalid: bool,
) -> Optional[str]:
    counts = Counter(
        answer if answer is not None else "__INVALID__"
        for answer in answers
        if include_invalid or answer is not None
    )
    if not counts:
        return None
    maximum = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == maximum]
    return winners[0] if len(winners) == 1 else None


def strict_rh_answer(selection: Dict[str, Any]) -> Optional[str]:
    scores = {
        answer: float(details["raw_score"])
        for answer, details in selection.get("scores", {}).items()
    }
    if not scores:
        return None
    maximum = max(scores.values())
    winners = [answer for answer, score in scores.items() if score == maximum]
    return winners[0] if len(winners) == 1 else None


def metric_snapshot(records: List[Dict[str, Any]]) -> Dict[str, float]:
    predictions = [record["flat_answers"] for record in records]
    references = [record["ref"] for record in records]
    return {
        "mean@k": mean_at_k(predictions, references),
        "maj@k": mean(
            float(first_majority_answer(answers, include_invalid=True) == reference)
            for answers, reference in zip(predictions, references)
        ),
        "maj_s@k": mean(
            float(strict_majority_answer(answers, include_invalid=False) == reference)
            for answers, reference in zip(predictions, references)
        ),
        "maj_in@k": mean(
            float(strict_majority_answer(answers, include_invalid=True) == reference)
            for answers, reference in zip(predictions, references)
        ),
        "acc_s": mean(
            float(strict_rh_answer(record["evaluation_answer_selection"]) == record["ref"])
            for record in records
        ),
        "acc_in": mean(
            float(
                strict_rh_answer(record["inclusive_evaluation_answer_selection"])
                == record["ref"]
            )
            for record in records
        ),
        "pass@k": mean(
            float(reference is not None and reference in set(answers))
            for answers, reference in zip(predictions, references)
        ),
    }


def metric_series(
    round_records: List[Dict[str, Any]],
    final_records: List[Dict[str, Any]],
    rounds: int,
) -> Tuple[List[str], Dict[str, List[float]]]:
    steps = [
        (
            f"round{round_idx}",
            [record for record in round_records if record.get("round") == round_idx],
        )
        for round_idx in range(1, rounds + 1)
    ]
    steps.append(("final", final_records))
    snapshots = [metric_snapshot(records) for _name, records in steps]
    names = [name for name, _records in steps]
    series = {
        metric: [round(snapshot[metric] * 100, 2) for snapshot in snapshots]
        for metric in snapshots[0] if snapshots
    }
    return names, series


def compute_dynamics(
    round_records: List[Dict[str, Any]],
    final_records: List[Dict[str, Any]],
    rounds: int,
) -> Dict[str, int]:
    if rounds == 0:
        return {"w2c": 0, "c2w": 0, "net_correction": 0, "wrong_lock": 0}
    by_round = {
        round_idx: [record for record in round_records if record["round"] == round_idx]
        for round_idx in range(1, rounds + 1)
    }
    w2c = c2w = wrong_lock = 0
    for question_idx, final_record in enumerate(final_records):
        sequence = [by_round[index][question_idx] for index in range(1, rounds + 1)]
        initial_correct = bool(sequence[0]["global_correct"])
        final_correct = bool(final_record["global_correct"])
        w2c += int(not initial_correct and final_correct)
        c2w += int(initial_correct and not final_correct)
        initial_answer = sequence[0]["global_answer"]
        answers = [record["global_answer"] for record in sequence] + [
            final_record["global_answer"]
        ]
        wrong_lock += int(
            not initial_correct
            and initial_answer is not None
            and all(answer == initial_answer for answer in answers)
        )
    return {
        "w2c": w2c,
        "c2w": c2w,
        "net_correction": w2c - c2w,
        "wrong_lock": wrong_lock,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SS-ICRL with Reliability-Hill feedback.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--csv_path",
        default=str(ROOT_DIR / "results" / "ss_icrl" / "metrics.csv"),
    )
    parser.add_argument("--k", type=int, default=cfg("k", 32))
    parser.add_argument("--per_view_k", type=int, default=cfg("per_view_k", 8))
    parser.add_argument("--batch", type=int, default=cfg("batch", 1))
    parser.add_argument("--temp", type=float, default=cfg("temperature", 0.6))
    parser.add_argument("--top_p", type=float, default=cfg("top_p", 0.95))
    parser.add_argument("--ctx", type=int, default=cfg("context_len", 8192))
    parser.add_argument("--rounds", type=int, default=cfg("rounds", 5))
    parser.add_argument("--summary_length", type=int, default=cfg("summary_length", 500))
    parser.add_argument("--answer_length", type=int, default=cfg("answer_length", 5000))
    parser.add_argument("--test_sample", type=int, default=cfg("test_sample", None))
    parser.add_argument("--start_idx", type=int, default=cfg("start_idx", None))
    parser.add_argument("--end_idx", type=int, default=cfg("end_idx", None))
    parser.add_argument("--seed", type=int, default=cfg("seed", 42))
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=cfg("gpu_memory_utilization", 0.7),
    )
    parser.add_argument(
        "--views",
        default=None,
        help="Optional comma-separated subset of strategies defined in the configuration.",
    )
    parser.add_argument(
        "--use_answer_adapter",
        action=argparse.BooleanOptionalAction,
        default=cfg("use_answer_adapter", False),
    )
    parser.add_argument(
        "--boxed_answer_position",
        choices=("first", "last"),
        default=cfg("boxed_answer_position", "first"),
    )
    parser.add_argument(
        "--save_prompts",
        action=argparse.BooleanOptionalAction,
        default=cfg("save_prompts", True),
    )
    parser.add_argument(
        "--save_round_dynamics",
        action=argparse.BooleanOptionalAction,
        default=cfg("save_round_dynamics", True),
    )
    # Required by the ICPO compression utility.
    parser.add_argument("--enable_penalty", action="store_false", default=False)
    parser.add_argument("--use_reward", action="store_true", default=True)
    parser.add_argument("--epsilon", type=float, default=0.0)
    return parser


def validate_args(args: argparse.Namespace, views: List[Tuple[str, str]]) -> None:
    if args.rounds < 0:
        raise ValueError("rounds must be non-negative")
    if args.per_view_k <= 0:
        raise ValueError("per_view_k must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    expected_k = args.per_view_k * len(views)
    if args.k != expected_k:
        raise ValueError(
            f"k must equal per_view_k * number_of_strategies ({expected_k}), got {args.k}"
        )


def attach_record_metadata(
    records: List[Dict[str, Any]],
    subset: Any,
    batch_start: int,
    start_idx: int,
    questions: List[str],
) -> None:
    for question_idx, record in enumerate(records):
        record["id"] = subset[question_idx].get(
            "id",
            start_idx + batch_start + question_idx,
        )
        record["question"] = questions[question_idx]


def main() -> None:
    args = build_arg_parser().parse_args()
    views = parse_views(args.views)
    validate_args(args, views)

    benchmark = infer_benchmark(args.task_dir)
    setting.BENCHMARK = benchmark
    icpo.set_global_variable(args)
    np.random.seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(f"{args.output_dir}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "views.json").write_text(
        json.dumps(dict(views), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    model = load_seeded_vllm(
        args.model_path,
        args.seed,
        args.gpu_memory_utilization,
    )
    tokenizer = model.get_tokenizer()

    task_dir = Path(args.task_dir)
    file_name = "test.parquet" if (task_dir / "test.parquet").exists() else "test.json"
    dataset = load_dataset(
        "parquet" if file_name.endswith(".parquet") else "json",
        data_files=str(task_dir / file_name),
        split="train",
    )
    if args.test_sample is not None:
        dataset = dataset.select(range(min(args.test_sample, len(dataset))))
    start_idx = 0 if args.start_idx is None else args.start_idx
    end_idx = len(dataset) if args.end_idx is None else min(args.end_idx, len(dataset))
    dataset = dataset.select(range(start_idx, end_idx))

    all_round_records: List[Dict[str, Any]] = []
    all_final_records: List[Dict[str, Any]] = []
    histories_written = 0

    progress = tqdm(range(0, len(dataset), args.batch), desc="SS-ICRL", unit="batch")
    for batch_start in progress:
        batch_end = min(batch_start + args.batch, len(dataset))
        subset = dataset.select(range(batch_start, batch_end))
        questions = [example.get("prompt") or example["problem"] for example in subset]
        references = [example.get("answer") or example.get("solution") or "" for example in subset]
        histories = initialize_histories(len(questions), views)

        for round_idx in range(1, args.rounds + 1):
            records = generate_phase(
                questions,
                references,
                histories,
                views,
                model,
                tokenizer,
                args,
                phase="round",
                round_idx=round_idx,
            )
            attach_record_metadata(records, subset, batch_start, start_idx, questions)
            histories_written += update_strategy_histories(
                records,
                histories,
                model,
                tokenizer,
                output_dir,
                round_idx,
                args.boxed_answer_position,
            )
            all_round_records.extend(records)

        final_records = generate_phase(
            questions,
            references,
            histories,
            views,
            model,
            tokenizer,
            args,
            phase="final",
        )
        attach_record_metadata(final_records, subset, batch_start, start_idx, questions)
        for question_idx, record in enumerate(final_records):
            record["history"] = histories[question_idx]
        all_final_records.extend(final_records)
        progress.set_postfix(done=len(all_final_records))

    if args.save_round_dynamics:
        (output_dir / "round_dynamics.jsonl").write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=False) for record in all_round_records
            ),
            encoding="utf-8",
        )
    (output_dir / "final_dynamics.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in all_final_records),
        encoding="utf-8",
    )
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(
            json.dumps({"id": record["id"], "gens": record["flat_generations"]}, ensure_ascii=False)
            for record in all_final_records
        ),
        encoding="utf-8",
    )
    (output_dir / "answer.json").write_text(
        json.dumps(
            [
                {
                    "id": record["id"],
                    "ref": record["ref"],
                    "selected_answer": record["evaluation_answer"],
                    "selected_correct": record["evaluation_correct"],
                    "operational_selected_answer": record["global_answer"],
                    "operational_selected_correct": record["global_correct"],
                    "normalized_candidates": record["flat_answers"],
                    "answer_selection": record["evaluation_answer_selection"],
                    "operational_answer_selection": record["answer_selection"],
                }
                for record in all_final_records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    diagnostics = {
        "rounds": generation_diagnostics(all_round_records),
        "final": generation_diagnostics(all_final_records),
        "overall": generation_diagnostics([*all_round_records, *all_final_records]),
    }
    (output_dir / "generation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    final_snapshot = metric_snapshot(all_final_records)
    metric_steps, metrics_by_round = metric_series(
        all_round_records,
        all_final_records,
        args.rounds,
    )
    dynamics = compute_dynamics(
        all_round_records,
        all_final_records,
        args.rounds,
    )
    num_examples = len(all_final_records)
    metrics = {
        "model": Path(args.model_path).name,
        "task": task_dir.name,
        "benchmark": benchmark,
        "num_examples": num_examples,
        "method": "SS-ICRL",
        "answer_selection_policy": "reliability_hill",
        "feedback_target": "global_answer",
        "feedback_content": "trajectory",
        "feedback_reward_mode": "policy_score",
        "feedback_context_scope": "strategy_specific",
        "views": ",".join(name for name, _instruction in views),
        "answer_selection_acc": round(
            mean(float(record["evaluation_correct"]) for record in all_final_records) * 100,
            2,
        ),
        **{name: round(value * 100, 2) for name, value in final_snapshot.items()},
        "metric_steps": metric_steps,
        **{f"{name}_by_round": values for name, values in metrics_by_round.items()},
        "rounds": args.rounds,
        "per_view_k": args.per_view_k,
        "k": args.k,
        "history_entries_written": histories_written,
        "answer_generations_per_question": args.k * (args.rounds + 1),
        "summary_generations_per_question": (
            histories_written / num_examples if num_examples else 0.0
        ),
        "total_model_generations_per_question": (
            args.k * (args.rounds + 1)
            + (histories_written / num_examples if num_examples else 0.0)
        ),
        "Tokens": int(icpo.TOTAL_TOKEN),
        "temperature": args.temp,
        "top_p": args.top_p,
        "seed": args.seed,
        **dynamics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    append_csv(Path(args.csv_path), metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
