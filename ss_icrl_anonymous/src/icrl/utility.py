
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, datetime,json, csv, random, re, logging
from pathlib import Path
from typing import List, Dict, Any,Union
import math
import torch, yaml
import sys
from icrl.setting import setting
# ---------------- Config & Debug ----------------
DEBUG = True      # ⇦ 如无需日志，改为 False
MAX_KEEP = 10000      # 每类历史最多保留条数
TENSOR_PARALLEL = torch.cuda.device_count()
ALLOW_OVERWRITE= True


ROOT_DIR   = Path(__file__).resolve().parents[2]
CONFIG_YML = ROOT_DIR / "configs" / "icrl.yaml"
SYS_PROMPT = None
_SUMMARY_PROMPT = None

AIME_SYS_PROMPT = (
    "You are an AI mathematician. All content you output MUST be in English.\n"
    "**You are only allowed to provide explanations in plain English. Do NOT write any code, pseudocode, or technical snippets. Explain the concept of XYZ in detail.**"
    "Below are compressed solution ideas from previous attempts; each idea is tagged "
    "with reward 1 (correct) or reward 0 (incorrect). Use the question and these ideas "
    "to deduce the correct numeric answer.\n"
    "**Finish all your reasoning, then on a NEW line output exactly one number "
    "(the answer) and nothing else.**\n"
    "Your final output MUST be in the format boxed{<number>}, where <number> is the "
    "final numeric answer only (no expressions, variables, or additional text)."
    "The content inside boxed{ } must be a decimal number, not a fraction or any other form."
)
# AIME_SYS_PROMPT = (
#     "You are an AI mathematician. All content you output MUST be in English.\n"
#     "**You are only allowed to provide explanations in plain English. Do NOT write any code, pseudocode, or technical snippets.**\n\n"
#     "Below are candidate answers from previous attempts, labeled as potentially correct (good) "
#     "or potentially incorrect (bad). Use these hints along with the question to deduce the correct answer.\n\n"
#     "**Finish all your reasoning, then on a NEW line output exactly one number "
#     "(the answer) and nothing else.**\n"
#     "Your final output MUST be in the format boxed{<number>}, where <number> is the "
#     "final numeric answer only (no expressions, variables, or additional text). "
#     "The content inside boxed{ } must be a decimal number, not a fraction or any other form."
# )
AIME_SYS_PROMPT_NO_REWARD = (
    "You are an AI mathematician. All content you output MUST be in English.\n"
    "**You are only allowed to provide explanations in plain English. Do NOT write any code, pseudocode, or technical snippets. Explain the concept of XYZ in detail.**"
    "Below are compressed solution ideas from previous attempts; each idea is tagged "
    "with reward 1 (correct) or reward 0 (incorrect). Use the question and these ideas "
    "to deduce the correct numeric answer.\n"
    "**Finish all your reasoning, then on a NEW line output exactly one number "
    "(the answer) and nothing else.**\n"
    "Your final output MUST be in the format boxed{<number>}, where <number> is the "
    "final numeric answer only (no expressions, variables, or additional text)."
    "The content inside boxed{ } must be a decimal number, not a fraction or any other form."
    
)
AIME_SUMMARY_PROMPT = (
    "Provide a concise summary of the reasoning in the answer below. "
    "Do NOT add any introductory phrases or extra explanations. "
    "Omit all numerical calculations. "
    "The summary must be self-contained, no more than 100 tokens.\n\n"
    "If there is a final numeric result, include it at the end in the format boxed{{<number>}} "
    "(decimal only, no fractions, variables, or extra text). "
    "If there is no numeric answer, do not output boxed{{}}.\n\n"
    "[Answer start]\n{}\n[Answer end]\n\n"
    "Summary:"
)
AMC_SYS_PROMPT=AIME_SYS_PROMPT
AMC_SUMMARY_PROMPT=AIME_SUMMARY_PROMPT


GPQA_SYS_PROMPT=(
    "You are an AI mathematician. All content you output MUST be in English.\n"
    "**You are only allowed to provide explanations in plain English. Do NOT write any code, pseudocode, or technical snippets. Explain the concept of XYZ in detail.**"
    "Below are compressed solution ideas from previous attempts; each idea is tagged "
    "with reward 1 (correct) or reward 0 (incorrect). Use the question and these ideas "
    "to deduce the correct choice.\n"
    "**Finish all your reasoning, then on a NEW line output exactly one letter "
    "(the answer) and nothing else.**\n"
    "Your final output MUST be in the format boxed{<letter>}, where <letter> is exactly one of A, B, C, D."
)
GPQA_SUMMARY_PROMPT= (
    "Provide a concise summary of the reasoning in the answer below. "
    "Do NOT add any introductory phrases or extra explanations. "
    "Omit all numerical calculations. "
    "The summary must be self-contained, no more than 100 tokens.\n\n"
    "If there is a final answer choice, include it at the end in the format boxed{{<letter>}} "
    "(must be exactly one of A, B, C, D, no extra text). "
    "If there is no final answer, do not output boxed{{}}.\n\n"
    "[Answer start]\n{}\n[Answer end]\n\n"
    "Summary:"
)

MATH_SYS_PROMPT = (
    "You are an AI mathematician. All content you output MUST be in English.\n"
    "**You are only allowed to provide explanations in plain English. Do NOT write any code, pseudocode, or technical snippets. Explain the concept of XYZ in detail.** "
    "Below are compressed solution ideas from previous attempts; each idea is tagged "
    "with reward 1 (correct) or reward 0 (incorrect). Use the question and these ideas "
    "to deduce the correct answer.\n"
    "**Finish all your reasoning, then on a NEW line output exactly one answer "
    "(the answer) and nothing else.**\n"
    "Your final output MUST be in the format boxed{<answer>}, where <answer> is the "
    "final answer in any form (number, fraction, expression, or text), without extra explanation or text."
)
MATH_SUMMARY_PROMPT = (
    "Provide a concise summary of the reasoning in the answer below. "
    "Do NOT add any introductory phrases or extra explanations. "
    "Omit all calculations unless essential to understanding. "
    "The summary must be self-contained, no more than 100 tokens.\n\n"
    "If there is a final answer, include it at the end in the format boxed{{<answer>}} "
    "(any form is allowed: number, fraction, expression, or text). "
    "If there is no final answer, do not output boxed{{}}.\n\n"
    "[Answer start]\n{}\n[Answer end]\n\n"
    "Summary:"
)
def get_prompt(bench_name,reward=True,rounds =5):
    mapping ={
        "AIME":(AIME_SYS_PROMPT,AIME_SUMMARY_PROMPT),
        "AMC":(AMC_SYS_PROMPT,AMC_SUMMARY_PROMPT),
        "GPQA":(GPQA_SYS_PROMPT,GPQA_SUMMARY_PROMPT),
        "MATH":(MATH_SYS_PROMPT,MATH_SUMMARY_PROMPT)
    }
    sys_prompt,summary_prompt = mapping[bench_name]
    if reward ==False:
        sys_prompt=sys_prompt.replace("each idea is tagged with reward 1 (correct) or reward 0 (incorrect).","")
    if rounds ==0:
        sys_prompt=sys_prompt.replace("**Below are compressed solution ideas from previous attempts; each idea is tagged with reward 1 (correct) or reward 0 (incorrect). Use the question and these ideas to deduce the correct numeric answer.","")

    return sys_prompt,summary_prompt


import re
import logging

DEBUG = False


import re
import logging

DEBUG = False
BENCHMARK = "AIME"  # 默认，可根据 args.task_dir 设置

def _normalize(solution_str: str):
    """
    Extract and normalize final answer from a raw string (LLM output or ref answer).
    Handles benchmarks: AIME, AMC, GPQA, MATH.
    
    Returns:
        float / str / None:
            - AIME/AMC: float
            - GPQA: uppercase letter
            - MATH: string
    """
    benchmark =setting.BENCHMARK
    solution_str = solution_str.strip()
    raw_ans = None
    # ---------------- 数字型答案 ----------------
    if benchmark in ["AIME", "AMC"]:
        # 优先尝试 boxed{}
        match = re.findall(r"boxed\{(\-?[0-9\.\,]+)\}", solution_str)
        if match:
            raw_ans = match[-1]
        else:
            s = solution_str.replace(",", "").replace("$", "").replace(" ", "")
            if re.fullmatch(r"\-?\d+(\.\d+)?", s):
                raw_ans = s

        if raw_ans is None:
            return None

        # 归一化为 float
        ans_clean = raw_ans.replace("$", "").replace(",", "").replace(" ", "")
        if ans_clean.startswith("+"):
            ans_clean = ans_clean[1:]
        if ans_clean.endswith("."):
            ans_clean = ans_clean[:-1]
        try:
            ans_float = float(ans_clean)
            if DEBUG:
                logging.debug(f"[normalize] '{solution_str[:40]}' -> {ans_float}")
            return ans_float
        except ValueError:
            return None

    # ---------------- 选择题答案 ----------------
    elif benchmark == "GPQA":
        # 尝试 boxed{}
        match = re.findall(r"boxed\{([A-D])\}", solution_str, re.IGNORECASE)
        if match:
            raw_ans = match[-1].upper()
        else:
            # 直接匹配 A-D
            match = re.findall(r"\b([A-D])\b", solution_str, re.IGNORECASE)
            if match:
                raw_ans = match[-1].upper()
        return raw_ans

    # ---------------- 通用答案 ----------------
    elif benchmark == "MATH":
        # match = re.findall(r"boxed\{([^\}]+)\}", solution_str)
        # if match:
        #     raw_ans = match[-1].strip()
        # else:
        #     # fallback: 最后一行非空字符串
        #     lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
        #     if lines:
        #         raw_ans = lines[-1]
        #     else:
        #         raw_ans = solution_str.strip()
        # return raw_ans
        from icrl.math_norm import mathd_normalize_answer
        return mathd_normalize_answer(solution_str)

    else:
        raise ValueError(f"{benchmark} is not a valid benchmark.")

    
    
#     def _extract_numeric(solution_str:str, method="strict")->str:
#     assert method in ["strict", "flexible"]
#     #from https://github.com/PRIME-RL/TTRL/blob/main/verl/verl/utils/reward_score/gsm8k.py, modify the search patteren into boxed{}
#     if method == "strict":
#         # match \boxed{number}
#         solution = re.findall(r"\\boxed\{(\-?[0-9\.\,]+)\}", solution_str)
#         if not solution: 
#             final_answer = None
#         else:
#             final_answer = solution[-1].replace(",", "").replace("$", "")
#     elif method == "flexible":
#         answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
#         final_answer = None
#         if len(answer) == 0:
#             # no reward is there is no answer
#             pass
#         else:
#             invalid_str = ["", "."]
#             # find the last number that is not '.'
#             for final_answer in reversed(answer):
#                 if final_answer not in invalid_str:
#                     break
#     return final_answer

# def _normalize(ans: str) -> float:
#     raw = ans
#     if re.fullmatch(r"-?\d+(\.\d+)?", ans.strip()):  # if input is a str of number
#         ans = ans.strip()
#     else:
#         ans = _extract_numeric(ans)
#     if ans is None:
#         return None
#     ans = ans.replace("$", "").replace(",", "").replace(" ", "")
#     if ans.startswith("+"):
#         ans = ans[1:]
#     if ans.endswith("."):
#         ans = ans[:-1]
    
#     try:
#         ans_float = float(ans)
#     except ValueError:
#         return None
    
#     if DEBUG:
#         logging.debug(f"[normalize] '{raw[:40]}' -> {ans_float}")
    
#     return ans_float