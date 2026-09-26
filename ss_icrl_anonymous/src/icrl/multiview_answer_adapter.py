from __future__ import annotations

"""Answer extraction and canonicalization for the multi-view ICRL runner.

Model generations and dataset references deliberately use different entry
points.  A generation is valid only when it contains a recoverable ``\boxed``
answer; a reference is already an answer and therefore needs no such marker.
Both are then mapped to the same deterministic representation so equivalent
surface forms vote for the same answer hypothesis.
"""

import ast
import re
from fractions import Fraction
from typing import Any, List, Optional, Tuple


_MATRIX_RE = re.compile(
    r"^\\begin\{(?P<kind>p|b|B|v|V|small)?matrix\}"
    r"(?P<body>.*)"
    r"\\end\{(?P=kind)?matrix\}$",
    re.DOTALL,
)

_NEXT_TURN_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:human|user|system)[ \t]*:"
    r"|<\|im_start\|>[ \t]*(?:user|system)\b"
    r"|<\|start_header_id\|>[ \t]*(?:user|system)[ \t]*<\|end_header_id\|>"
)

_BOXED_MARKER_RE = re.compile(r"\\?boxed\b", re.IGNORECASE)


def extract_current_assistant_span(raw: Any) -> str:
    """Return only the generated text before a simulated next user turn.

    A literal ``Assistant:`` is deliberately not a boundary. Some models first
    continue the prompt's evidence format and then emit ``Assistant:`` before
    producing their actual solution.
    """
    if raw is None:
        return ""
    text = str(raw)
    boundary = _NEXT_TURN_RE.search(text)
    return text[: boundary.start()].rstrip() if boundary else text


def _extract_boxed_answers(raw: Any) -> List[Tuple[str, int]]:
    """Return complete boxed-answer bodies and their exclusive end offsets."""
    if raw is None:
        return []
    text = str(raw)
    answers: List[Tuple[str, int]] = []
    start = 0
    while True:
        marker_match = _BOXED_MARKER_RE.search(text, start)
        if marker_match is None:
            break
        brace_index = marker_match.end()
        while brace_index < len(text) and text[brace_index].isspace():
            brace_index += 1
        if brace_index >= len(text) or text[brace_index] != "{":
            start = marker_match.end()
            continue

        depth = 1
        cursor = brace_index + 1
        while cursor < len(text) and depth:
            if text[cursor] == "{" and not _is_escaped(text, cursor):
                depth += 1
            elif text[cursor] == "}" and not _is_escaped(text, cursor):
                depth -= 1
            cursor += 1
        if depth == 0:
            body = text[brace_index + 1 : cursor - 1].strip()
            if body:
                answers.append((body, cursor))
            start = cursor
        else:
            start = marker_match.end()
    return answers


def extract_boxed_answer(raw: Any, position: str = "last") -> Optional[str]:
    """Return the first or last complete boxed-answer body, with no fallback."""
    if position not in {"first", "last"}:
        raise ValueError(f"Unknown boxed answer position: {position!r}")
    answers = _extract_boxed_answers(raw)
    if not answers:
        return None
    index = 0 if position == "first" else -1
    return answers[index][0]


def extract_last_boxed_answer(raw: Any) -> Optional[str]:
    """Return the final complete ``\boxed{...}`` body, with no fallback."""
    return extract_boxed_answer(raw, position="last")


def truncate_after_boxed_answer(raw: Any, position: str = "last") -> str:
    """Drop text after the selected complete boxed answer when one exists."""
    if position not in {"first", "last"}:
        raise ValueError(f"Unknown boxed answer position: {position!r}")
    text = "" if raw is None else str(raw)
    answers = _extract_boxed_answers(text)
    if not answers:
        return text
    index = 0 if position == "first" else -1
    return text[: answers[index][1]].rstrip()


def canonicalize_generation_answer(
    raw: Any,
    benchmark: str,
    boxed_answer_position: str = "last",
) -> Optional[str]:
    boxed = extract_boxed_answer(
        extract_current_assistant_span(raw),
        position=boxed_answer_position,
    )
    if boxed is None:
        return None
    return canonicalize_answer_content(boxed, benchmark)


def canonicalize_reference_answer(raw: Any, benchmark: str) -> Optional[str]:
    """Canonicalize a dataset answer without requiring generation syntax."""
    if raw is None:
        return None
    text = str(raw).strip()
    boxed = extract_last_boxed_answer(text)
    return canonicalize_answer_content(boxed if boxed is not None else text, benchmark)


def canonicalize_answer_content(raw: Any, benchmark: str) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    benchmark = str(benchmark).upper()
    if benchmark == "GPQA":
        return _canonicalize_choice(text)
    if benchmark in {"AIME", "AMC"}:
        normalized = _canonicalize_math(text)
        value = _numeric_fraction(normalized) if normalized is not None else None
        return _format_fraction(value) if value is not None else None
    return _canonicalize_math(text)


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def _clean_math_surface(text: str) -> str:
    text = text.strip().replace("\n", "")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\!", "").replace(r"\,", "")
    text = text.replace(r"\;", "").replace(r"\:", "")
    text = text.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    text = text.replace(r"\$", "").replace("$", "").strip()
    while text.endswith(".") and not re.fullmatch(r"[+-]?\d+\.", text):
        text = text[:-1].rstrip()
    text = re.sub(r"\s+", "", text)
    return text


def _canonicalize_choice(text: str) -> Optional[str]:
    text = _unwrap_text_command(_clean_math_surface(text))
    match = re.fullmatch(r"\(?([A-Da-d])\)?", text)
    return match.group(1).upper() if match else None


def _canonicalize_math(text: str) -> Optional[str]:
    text = _clean_math_surface(text)
    if not text:
        return None

    text = _unwrap_text_command(text)
    text = text.replace(r"^{\circ}", "").replace(r"^\circ", "")
    text = text.replace(r"\%", "")
    equation_parts = text.split("=")
    if len(equation_parts) == 2 and len(equation_parts[0]) <= 2:
        text = equation_parts[1]
    choice = re.fullmatch(r"\(?([A-Da-d])\)?", text)
    if choice:
        return choice.group(1).upper()

    matrix = _canonicalize_matrix(text)
    if matrix is not None:
        return matrix

    sequence = _canonicalize_sequence(text)
    if sequence is not None:
        return sequence

    base_number = re.fullmatch(r"([+-]?[0-9A-Za-z]+)_\{?(\d+)\}?", text)
    if base_number:
        return f"{base_number.group(1)}_{{{base_number.group(2)}}}"

    numeric = _numeric_fraction(text)
    if numeric is not None:
        return _format_fraction(numeric)

    text = _strip_balanced_outer_braces(text)
    text = re.sub(r"\\sqrt([^\{])", r"\\sqrt{\1}", text)
    text = re.sub(r"\\{2,}", lambda _match: "\\", text)
    return text or None


def _unwrap_text_command(text: str) -> str:
    for command in (r"\text", r"\mathrm", r"\operatorname"):
        prefix = command + "{"
        if text.startswith(prefix) and text.endswith("}"):
            end = _matching_brace(text, len(command))
            if end == len(text) - 1:
                return text[len(prefix) : -1].strip()
    return text


def _matching_brace(text: str, open_index: int) -> Optional[int]:
    if open_index >= len(text) or text[open_index] != "{":
        return None
    depth = 1
    for index in range(open_index + 1, len(text)):
        if text[index] == "{" and not _is_escaped(text, index):
            depth += 1
        elif text[index] == "}" and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return index
    return None


def _strip_balanced_outer_braces(text: str) -> str:
    while text.startswith("{") and _matching_brace(text, 0) == len(text) - 1:
        text = text[1:-1]
    return text


def _canonicalize_matrix(text: str) -> Optional[str]:
    match = _MATRIX_RE.fullmatch(text)
    if match is None:
        return None
    rows = [row for row in re.split(r"\\\\", match.group("body")) if row]
    cells_by_row = [[cell for cell in row.split("&")] for row in rows]
    if not cells_by_row or any(not cells for cells in cells_by_row):
        return None
    canonical_rows = [
        [_canonicalize_math(cell) for cell in cells]
        for cells in cells_by_row
    ]
    if any(cell is None for row in canonical_rows for cell in row):
        return None
    flat = [str(cell) for row in canonical_rows for cell in row]
    if len(canonical_rows) == 1 or all(len(row) == 1 for row in canonical_rows):
        return f"({','.join(flat)})"
    widths = {len(row) for row in canonical_rows}
    if len(widths) != 1:
        return None
    return f"matrix[{len(canonical_rows)}x{len(canonical_rows[0])}]({','.join(flat)})"


def _canonicalize_sequence(text: str) -> Optional[str]:
    if len(text) < 3 or (text[0], text[-1]) not in {("(", ")"), ("[", "]")}:
        return None
    parts = _split_top_level(text[1:-1], ",")
    if len(parts) <= 1:
        return None
    canonical = [_canonicalize_math(part) for part in parts]
    if any(part is None for part in canonical):
        return None
    return f"{text[0]}{','.join(str(part) for part in canonical)}{text[-1]}"


def _split_top_level(text: str, delimiter: str) -> List[str]:
    parts: List[str] = []
    start = 0
    braces = brackets = parentheses = 0
    for index, char in enumerate(text):
        if char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char == "(":
            parentheses += 1
        elif char == ")":
            parentheses -= 1
        elif char == delimiter and braces == brackets == parentheses == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _numeric_fraction(text: str) -> Optional[Fraction]:
    expression = _latex_numeric_expression(text)
    if expression is None:
        return None
    if "," in expression:
        if not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", expression):
            return None
        expression = expression.replace(",", "")
    if not re.fullmatch(r"[0-9eE+\-*/().]+", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_fraction_ast(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _latex_numeric_expression(text: str) -> Optional[str]:
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace("^", "**")
    output: List[str] = []
    cursor = 0
    marker = r"\frac"
    while cursor < len(text):
        if not text.startswith(marker, cursor):
            output.append(text[cursor])
            cursor += 1
            continue
        numerator_open = cursor + len(marker)
        if numerator_open >= len(text) or text[numerator_open] != "{":
            return None
        numerator_close = _matching_brace(text, numerator_open)
        if numerator_close is None:
            return None
        denominator_open = numerator_close + 1
        if denominator_open >= len(text) or text[denominator_open] != "{":
            return None
        denominator_close = _matching_brace(text, denominator_open)
        if denominator_close is None:
            return None
        numerator = _latex_numeric_expression(text[numerator_open + 1 : numerator_close])
        denominator = _latex_numeric_expression(text[denominator_open + 1 : denominator_close])
        if numerator is None or denominator is None:
            return None
        output.append(f"(({numerator})/({denominator}))")
        cursor = denominator_close + 1
    return "".join(output)


def _eval_fraction_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_fraction_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_fraction_ast(node.left)
        right = _eval_fraction_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow) and right.denominator == 1:
            return left ** int(right)
    raise ValueError("not a numeric expression")


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"\\frac{{{value.numerator}}}{{{value.denominator}}}"
