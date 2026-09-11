"""Response format assessment and source-ordered consumption; no repairs or scoring.

Source: SOURCE_AUDIT.md sections 4.7, 6.6, 7.6 and the approved Stage 2
clarifications. Each recovery branch examines only its first candidate.
"""

import json
import re
from typing import Literal

from pydantic import JsonValue

from .schemas import BenchmarkError, ParseResult, StrictModel


class StrictJSONAssessment(StrictModel):
    valid: bool
    format_valid: bool
    top_level_type: Literal["object", "array", "string", "number", "boolean", "null"] | None
    diagnostics: tuple[BenchmarkError, ...] = ()


def _diagnostic(code: str, message: str) -> BenchmarkError:
    return BenchmarkError(stage="parser", code=code, message=message)


def _validate_raw(raw: str | None) -> None:
    if raw is not None and not isinstance(raw, str):
        raise TypeError("raw response must be a string or None")


def _reject_constant(value: str) -> None:
    raise ValueError("non-standard JSON numeric constant")


_STRICT_DECODER = json.JSONDecoder(parse_constant=_reject_constant)
_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


def _decode(text: str) -> JsonValue:
    # Match production, including Python's non-finite numeric extensions.
    # Strict RFC-format assessment is independent of this decoder.
    return json.loads(text)


def _first_fence(raw: str) -> str | None:
    match = _FENCE.search(raw)
    return match.group(1) if match is not None else None


def _first_balanced(raw: str, opening: str, closing: str) -> tuple[int, int] | None:
    """Scan from the first opening delimiter, ignoring delimiters in strings.

    An unclosed first candidate does not restart at a later opening delimiter.
    A balanced but malformed candidate is returned once for the caller to reject.
    """
    start = raw.find(opening)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(raw)):
        char = raw[index]
        # Match production ordering, including escapes outside JSON strings.
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return start, index + 1
    return None


def _top_level_type(value: JsonValue) -> str:
    if value is None:
        return "null"
    return {dict: "object", list: "array", str: "string", bool: "boolean",
            int: "number", float: "number"}[type(value)]


def check_strict_json(raw: str | None, *, object_required: bool = False) -> StrictJSONAssessment:
    """Assess the entire response, allowing surrounding JSON whitespace only."""
    _validate_raw(raw)
    if raw is None or not raw.strip():
        return StrictJSONAssessment(valid=False, format_valid=False, top_level_type=None,
                                    diagnostics=(_diagnostic("blank_output", "No JSON output"),))
    diagnostics = []
    try:
        value = _STRICT_DECODER.decode(raw)
    except ValueError:
        diagnostics.append(_diagnostic("invalid_json", "Whole response is not valid JSON"))
        if _first_fence(raw) is not None:
            diagnostics.append(_diagnostic("markdown_wrapping", "Response contains a fenced code block"))
        # These are diagnostics only; they never change strict success or drive
        # production candidate selection.
        try:
            _, end = _STRICT_DECODER.raw_decode(raw.lstrip())
            if raw.lstrip()[end:].strip():
                diagnostics.append(_diagnostic("trailing_prose", "Content follows a JSON value"))
        except ValueError:
            pass
        for opening, closing in (("{", "}"), ("[", "]")):
            span = _first_balanced(raw, opening, closing)
            if span is None:
                continue
            try:
                _STRICT_DECODER.decode(raw[span[0]:span[1]])
            except ValueError:
                continue
            if raw[:span[0]].strip():
                diagnostics.append(_diagnostic("leading_prose", "Content precedes embedded JSON"))
            if raw[span[1]:].strip() and not any(d.code == "trailing_prose" for d in diagnostics):
                diagnostics.append(_diagnostic("trailing_prose", "Content follows embedded JSON"))
            break
        return StrictJSONAssessment(valid=False, format_valid=False, top_level_type=None,
                                    diagnostics=tuple(diagnostics))
    if object_required and not isinstance(value, dict):
        diagnostics.append(_diagnostic("object_required", "Task requires a JSON object"))
    return StrictJSONAssessment(valid=True, format_valid=not diagnostics,
                                top_level_type=_top_level_type(value), diagnostics=tuple(diagnostics))


def parse_production_json(raw: str | None, *, object_required: bool = False) -> ParseResult:
    """Empty → direct dict/list → first fence → first object → first array → raw.

    Every JSON recovery branch accepts only dict/list. A rejected fenced scalar
    proceeds to object/array recovery over the full stripped response.
    """
    strict = check_strict_json(raw, object_required=object_required)
    diagnostics = list(strict.diagnostics)

    def result(value: JsonValue, path: str, success: bool) -> ParseResult:
        return ParseResult(
            raw_response=raw, strict_format_valid=strict.format_valid,
            strict_json_valid=strict.valid, production_parse_success=success,
            production_usable=success and isinstance(value, (dict, list)),
            parser_path=path, parsed_value=value, normalized_value=value,
            diagnostics=tuple(diagnostics),
        )

    if raw is None or not raw.strip():
        return result({}, "empty", False)
    content = raw.strip()
    try:
        value = _decode(content)
        if isinstance(value, (dict, list)):
            return result(value, "direct", True)
        diagnostics.append(_diagnostic("direct_scalar_rejected", "Direct production parsing requires dict/list"))
    except ValueError:
        pass
    fence = _first_fence(content)
    if fence is not None:
        try:
            value = _decode(fence.strip())
        except ValueError:
            diagnostics.append(_diagnostic("first_fence_invalid", "First fenced block is not JSON"))
        else:
            if isinstance(value, (dict, list)):
                return result(value, "fence", True)
            diagnostics.append(_diagnostic("first_fence_scalar_rejected", "Fenced production parsing requires dict/list"))
    for opening, closing, path in (("{", "}", "object"), ("[", "]", "array")):
        span = _first_balanced(content, opening, closing)
        if span is None:
            continue
        try:
            value = _decode(content[span[0]:span[1]])
        except ValueError:
            diagnostics.append(_diagnostic(f"first_{path}_invalid", f"First balanced {path} is not JSON"))
        else:
            return result(value, path, True)
    diagnostics.append(_diagnostic("raw_fallback", "No structured production result"))
    return result({"raw": raw}, "raw_fallback", False)


def parse_extraction(raw: str | None) -> ParseResult:
    result = parse_production_json(raw, object_required=True)
    usable = result.production_parse_success and isinstance(result.parsed_value, dict)
    diagnostics = result.diagnostics
    if result.production_parse_success and not isinstance(result.parsed_value, dict):
        diagnostics += (_diagnostic("non_dict_extraction_output", "Extraction requires an object"),)
    return result.model_copy(update={"production_usable": usable, "diagnostics": diagnostics})


def parse_qa(raw: str | None) -> ParseResult:
    result = parse_production_json(raw, object_required=True)
    usable = result.production_parse_success and isinstance(result.parsed_value, dict)
    diagnostics = result.diagnostics
    if not isinstance(result.parsed_value, dict):
        diagnostics += (_diagnostic("non_dict_qa_output", "QA consumes an empty object for non-dict output"),)
    # The raw fallback marker is parser metadata, not a QA result field. Keep it
    # in parsed_value while exposing {} as the consumed QA object, per Stage 2.
    return result.model_copy(update={
        "normalized_value": result.parsed_value if usable else {},
        "production_usable": usable, "diagnostics": diagnostics,
    })


def parse_voicemail(raw: str | None) -> ParseResult:
    _validate_raw(raw)
    text = raw if raw is not None else ""
    upper = text.upper()
    decision = "CONVERSATION" if "CONVERSATION" in upper else (
        "VOICEMAIL" if "VOICEMAIL" in upper else None
    )
    # Python str.strip() defines the surrounding whitespace allowance. Case is
    # not normalized for strict compliance.
    strict = text.strip() in {"CONVERSATION", "VOICEMAIL"}
    diagnostics = []
    if not text.strip():
        diagnostics.append(_diagnostic("blank_output", "No classifier output"))
    if not strict:
        diagnostics.append(_diagnostic("strict_label_invalid", "Expected one exact uppercase label"))
    if decision is None:
        diagnostics.append(_diagnostic("undecided", "Response contains neither classification keyword"))
    return ParseResult(
        raw_response=raw, strict_format_valid=strict, strict_json_valid=None,
        production_parse_success=decision is not None, production_usable=decision is not None,
        parser_path="label" if decision is not None else "none",
        parsed_value=decision, normalized_value=decision, diagnostics=tuple(diagnostics),
    )


def parse_summary(raw: str | None) -> ParseResult:
    _validate_raw(raw)
    nonblank = raw is not None and bool(raw.strip())
    return ParseResult(
        raw_response=raw, strict_format_valid=nonblank, strict_json_valid=None,
        production_parse_success=nonblank, production_usable=nonblank,
        parser_path="text" if nonblank else "empty", parsed_value=raw, normalized_value=raw,
        diagnostics=() if nonblank else (_diagnostic("blank_output", "No summary text"),),
    )
