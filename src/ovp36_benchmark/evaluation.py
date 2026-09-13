"""Pure post-run measurements and private, explicit human review. No model judge."""

import hashlib
import math
import re
from typing import Annotated, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from . import adapters, parsing
from .client import TokenUsage
from .contracts import load_contract
from .dataset import hash_dataset, validate_curated_cases
from .identity import SHA256, ExecutionKey, canonical_json_bytes
from .persistence import AttemptRecorded
from .replay import CapturedReplayCase
from .requests import PreparedRequest
from .schemas import (
    BenchmarkCase, ContractExpected, CriticalCheckResult, EvidenceRef,
    MetricValue, QAEvaluationExpected, SafeIdentifier, SafeMetadata, ScoreResult,
    Source, StrictModel, Task, VoicemailExpected,
)

EVALUATION_SCHEMA_VERSION = "ovp36-evaluation-v1"
EVALUATOR_VERSION = "ovp36-scorer-v1"
CheckStatus = Literal["pass", "fail", "pending", "not_applicable"]
Count = Annotated[int, Field(ge=0)]
QA_TAGS = tuple(re.findall(r"^- ([A-Z_]+) -", load_contract("qa").system_template, re.M))
CATEGORIES = (
    "user_preferences", "decisions", "unresolved_questions", "action_items", "prior_failures",
    "user_constraints", "downstream_relevant", "purpose", "key_behaviors", "tools",
    "transitions", "prohibitions", "optional_behaviors",
)
TASK_METRICS = {
    Task.EXTRACTION: ("field_total", "field_exact_count", "field_accuracy", "required_key_coverage",
        "all_fields_correct", "whole_object_match", "unexpected_key_count", "known_value_accuracy",
        "missing_policy_accuracy", "type_correctness", "latest_value_accuracy", "stale_value_count",
        "missing_output_count", "incorrect_value_count", "known_field_count", "missing_field_count"),
    Task.VOICEMAIL: ("binary_accuracy", "no_decision_rate"),
    Task.QA: ("tag_true_positive", "tag_false_negative", "tag_false_positive", "unannotated_tag_count",
        "invalid_tag_count", "duplicate_tag_count", "invalid_tag_entry_count", "annotation_coverage",
        "annotation_scoped_precision", "annotation_scoped_recall", "annotation_scoped_f1",
        "sentiment_match", "score_coverage", "score_distance", "score_in_range"),
}


class EvaluationError(ValueError):
    """Safe harness category, never a candidate-quality failure."""


def checked(model, value):
    """Detach/revalidate even unchecked model_copy instances; discard private errors."""
    if not isinstance(value, model):
        raise EvaluationError("invalid_evaluation_input")
    try:
        result = model.model_validate_json(canonical_json_bytes(value.model_dump(mode="json", warnings="error")))
    except (ValidationError, ValueError):
        failure = EvaluationError("invalid_evaluation_input")
    else:
        return result
    raise failure


def evaluation_fingerprint(domain: str, value: object) -> str:
    """Separate public evaluation domain; raw evidence is hashed only in memory."""
    return hashlib.sha256((EVALUATION_SCHEMA_VERSION + ":" + domain + "\0").encode()
                          + canonical_json_bytes(value)).hexdigest()


class EvaluationCheck(StrictModel):
    check_id: SafeIdentifier
    kind: SafeIdentifier
    subject: SafeIdentifier
    method: Literal["automatic", "human"]
    status: CheckStatus
    critical: bool = False
    categories: tuple[SafeIdentifier, ...] = ()

    @field_validator("categories")
    @classmethod
    def known_categories(cls, value):
        if any(name not in CATEGORIES for name in value):
            raise ValueError("unknown_check_category")
        return value


class CaseEvaluation(StrictModel):
    schema_version: Literal["ovp36-evaluation-v1"] = EVALUATION_SCHEMA_VERSION
    evaluator_version: SafeIdentifier = EVALUATOR_VERSION
    evaluator_fingerprint: SHA256
    evaluation_id: SHA256
    dataset_hash: SHA256
    case_fingerprint: SHA256
    evidence_fingerprint: SHA256
    content_fingerprint: SHA256 | None = None
    case_id: SafeIdentifier
    task: Task
    contract_id: SafeIdentifier
    source: Source
    exercise_kind: Literal["model", "adapter_control", "response_contract", "transport_contract"]
    purpose: Literal["candidate", "functional_stub", "control"]
    critical: bool
    gold_available: bool
    execution_key: ExecutionKey | None = None
    attempt_index: Count | None = None
    execution_state: Literal["finalized", "not_attempted", "unfinalized", "control"]
    transport_status: Literal["success", "timeout", "connection_error", "http_error", "protocol_error"] | None = None
    transport_error_code: SafeIdentifier | None = None
    http_status: int | None = None
    retryable: bool | None = None
    content_state: Literal["text", "empty", "whitespace", "null", "content_absent", "response_absent", "not_available", "not_applicable"]
    strict_json_valid: bool | None = None
    strict_format_valid: bool | None = None
    production_parse_success: bool | None = None
    production_usable: bool | None = None
    parser_path: Literal["empty", "direct", "fence", "object", "array", "raw_fallback", "label", "text", "none"] | None = None
    structure_valid: bool | None = None
    diagnostic_codes: tuple[SafeIdentifier, ...] = ()
    predicted_label: Literal["CONVERSATION", "VOICEMAIL"] | None = None
    expected_label: Literal["CONVERSATION", "VOICEMAIL"] | None = None
    voicemail_stratum: Literal["labelled", "ambiguous_labelled", "unlabelled"] | None = None
    predicted_tags: tuple[SafeIdentifier, ...] = ()
    unannotated_tags: tuple[SafeIdentifier, ...] = ()
    latency_ms: Annotated[float, Field(ge=0)] | None = None
    usage: TokenUsage | None = None
    response_model: SafeMetadata | None = None
    finish_reason: SafeIdentifier | None = None
    omitted_metadata: tuple[Literal["finish_reason", "response_model"], ...] = ()
    checks: tuple[EvaluationCheck, ...] = ()
    score: ScoreResult

    @field_validator("predicted_tags", "unannotated_tags")
    @classmethod
    def known_tags(cls, value):
        if any(tag not in QA_TAGS for tag in value):
            raise ValueError("unknown_source_tag")
        return value

    @model_validator(mode="after")
    def result_bindings(self):
        if self.score.case_id != self.case_id or self.score.exercise_kind != self.exercise_kind:
            raise ValueError("evaluation_score_binding_mismatch")
        if len({c.check_id for c in self.checks}) != len(self.checks):
            raise ValueError("duplicate_evaluation_check")
        critical = tuple(CriticalCheckResult(check_id=c.check_id, status=c.status) for c in self.checks if c.critical)
        if critical != self.score.critical_checks:
            raise ValueError("critical_check_binding_mismatch")
        control = self.exercise_kind != "model"
        if control != (self.purpose == "control") or control != (self.execution_state == "control"):
            raise ValueError("evaluation_purpose_mismatch")
        if control:
            if self.execution_key is not None or self.attempt_index is not None:
                raise ValueError("control_has_no_execution")
        elif self.execution_key is None or self.execution_key.case_id != self.case_id:
            raise ValueError("evaluation_execution_mismatch")
        identity = {name: getattr(self, name) for name in ("dataset_hash", "case_id", "case_fingerprint",
            "evidence_fingerprint", "evaluator_fingerprint", "purpose", "execution_state", "attempt_index")}
        identity["evaluator_version"] = self.evaluator_version
        identity["execution_key"] = self.execution_key.model_dump(mode="json") if self.execution_key else None
        if self.evaluation_id != evaluation_fingerprint("result", identity):
            raise ValueError("evaluation_identity_mismatch")
        return self

    @property
    def control_status(self):
        if self.exercise_kind == "model":
            return None
        # QA-026's response contradictions are successful negative-control evidence.
        checks = [c for c in self.checks if c.kind == "control_assertion" or c.method == "human"]
        return _rollup(checks)

    @property
    def critical_status(self):
        if not self.critical:
            return "not_applicable"
        return _rollup([c for c in self.checks if c.critical])


class HumanReviewItem(StrictModel):
    evaluation_id: SHA256
    case_id: SafeIdentifier
    contract_id: SafeIdentifier
    check: EvaluationCheck
    candidate_output: str | None = Field(repr=False, exclude=True)
    question: str = Field(repr=False, exclude=True)
    statement: str = Field(repr=False, exclude=True)
    evidence: tuple[EvidenceRef, ...] = Field(repr=False, exclude=True)
    snippets: tuple[str, ...] = Field(repr=False, exclude=True)


class ReviewDecision(StrictModel):
    schema_version: Literal["ovp36-review-v1"] = "ovp36-review-v1"
    evaluation_id: SHA256
    check_id: SafeIdentifier
    status: CheckStatus
    notes: str = Field(default="", repr=False)


def _rollup(checks):
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "pending" for c in checks):
        return "pending"
    if checks and all(c.status == "pass" for c in checks):
        return "pass"
    return "unavailable"


def _metric(value=None, numerator=None, denominator=None):
    return MetricValue(value=None if value is None else float(value),
                       numerator=None if numerator is None else float(numerator),
                       denominator=None if denominator is None else float(denominator))


def _rate(n, d):
    return _metric(n / d if d else None, n, d)


def _check(checks, name, ok=None, *, kind="semantic", subject="case", human=False,
           critical=False, categories=()):
    checks.append(EvaluationCheck(check_id=name, kind=kind, subject=subject,
        method="human" if human else "automatic", critical=critical, categories=categories,
        status=("pending" if human else "not_applicable") if ok is None else ("pass" if ok else "fail")))


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _equal(left, right):
    if _number(left) and _number(right):
        return left == right
    return type(left) is type(right) and left == right


def sentence_count_by_rule(text: str) -> int:
    """V1: punctuation runs before whitespace/end, optional closing quotes/brackets.

    Count nonempty spans plus a trailing fragment. Decimals do not split. This
    deliberately diagnostic rule splits abbreviations; human review adjudicates.
    """
    # ponytail: punctuation diagnostic only; linguistic compliance stays human-reviewed.
    return sum(bool(part.strip()) for part in re.split(r'''[.!?]+["'”’)\]]*(?:\s+|$)''', text))


def _sentence(text, bounds, metrics, checks):
    if bounds is not None:
        count = sentence_count_by_rule(text)
        metrics["sentence_count_by_rule"] = _metric(count)
        metrics["sentence_range_by_rule_ok"] = _rate(int(bounds[0] <= count <= bounds[1]), 1)
        _check(checks, "sentence_compliance", kind="sentence_compliance", human=True)


def _score_extraction(case, parsed, metrics, checks):
    gold = case.expected
    usable = parsed.production_usable
    output = parsed.parsed_value if usable else {}
    totals = dict(exact=0, present=0, known=0, known_ok=0, missing=0, missing_ok=0,
                  typed=0, typed_total=0, latest=0, latest_ok=0, stale=0, incorrect=0)
    for index, variable in enumerate(case.input.variables):
        field = gold.fields[variable.name]
        present = variable.name in output
        value = output.get(variable.name)
        typed = present and ({"string": type(value) is str, "boolean": type(value) is bool,
                              "number": _number(value)}[variable.type])
        if field.state == "known":
            correct = usable and typed and any(_equal(value, v) for v in field.acceptable_values)
            totals["known"] += 1
            totals["known_ok"] += correct
            totals["typed_total"] += present
            totals["typed"] += typed
        else:
            policy = field.missing_policy
            correct = usable and ((not present and policy.allow_absent)
                                   or (present and any(_equal(value, v) for v in policy.values)))
            totals["missing"] += 1
            totals["missing_ok"] += correct
        stale = present and not correct and any(_equal(value, v) for v in field.superseded_values)
        totals["exact"] += correct
        totals["present"] += present
        totals["stale"] += stale
        totals["incorrect"] += present and not correct
        totals["latest"] += bool(field.superseded_values)
        totals["latest_ok"] += bool(field.superseded_values) and correct
        prefix = f"field.{index}"
        for suffix, ok in (("present", present), ("correct", correct), ("not_stale", not stale)):
            _check(checks, prefix + "." + suffix, ok, kind="field_" + suffix, subject=prefix)
        _check(checks, prefix + ".type", typed if present and field.state == "known" else None,
               kind="field_type", subject=prefix)
    n = len(gold.fields)
    extras = len(set(output) - set(gold.fields)) if usable else 0
    for name, value in {"field_total": n, "field_exact_count": totals["exact"],
        "known_field_count": totals["known"], "missing_field_count": totals["missing"],
        "missing_output_count": n-totals["present"], "incorrect_value_count": totals["incorrect"],
        "unexpected_key_count": extras, "stale_value_count": totals["stale"]}.items():
        metrics[name] = _metric(None if name == "unexpected_key_count" and not usable else value)
    for name, a, b in (("field_accuracy", totals["exact"], n),
        ("required_key_coverage", totals["present"], n), ("known_value_accuracy", totals["known_ok"], totals["known"]),
        ("missing_policy_accuracy", totals["missing_ok"], totals["missing"]),
        ("type_correctness", totals["typed"], totals["typed_total"]),
        ("latest_value_accuracy", totals["latest_ok"], totals["latest"]),
        ("all_fields_correct", int(usable and totals["exact"] == n), 1),
        ("whole_object_match", int(usable and totals["exact"] == n and extras == 0), 1)):
        metrics[name] = _rate(a, b)
    for check in gold.critical_checks:
        _check(checks, "critical." + check.id, usable and totals["exact"] == n and extras == 0,
               kind="critical_contract", critical=True)


def _score_voicemail(case, parsed, metrics, checks, data):
    gold = case.expected
    data.update(predicted_label=parsed.normalized_value, expected_label=gold.label,
        voicemail_stratum="unlabelled" if gold.label is None else ("ambiguous_labelled" if gold.ambiguous else "labelled"))
    if gold.label is not None:
        correct = parsed.normalized_value == gold.label
        metrics["binary_accuracy"] = _rate(int(correct), 1)
        metrics["no_decision_rate"] = _rate(int(parsed.normalized_value is None), 1)
        _check(checks, "classification", correct)
        for check in gold.critical_checks:
            _check(checks, "critical." + check.id, correct, kind="critical_contract", critical=True)
    else:
        _check(checks, "unlabelled_inspection", human=True, kind="unlabelled_inspection")


def _fact_checks(gold, checks):
    required = getattr(gold, "required_facts", getattr(gold, "summary_facts", ()))
    critical = set(getattr(gold, "critical_facts", ()))
    for index, fact in enumerate(required):
        categories = tuple(name for name in CATEGORIES if fact.id in getattr(gold, name, ()))
        _check(checks, f"required_fact.{index}", kind="required_fact", subject=f"required_fact.{index}",
               human=True, critical=fact.id in critical, categories=categories)
    for attr, kind in (("superseded_facts", "stale_current_state"), ("forbidden_facts", "forbidden_claim"),
                       ("safe_to_omit", "optional_inclusion"), ("corrections", "correction")):
        for index, _ in enumerate(getattr(gold, attr, ())):
            _check(checks, f"{kind}.{index}", kind=kind, subject=f"{attr}.{index}", human=True)
    _check(checks, "unsupported_claims", kind="unsupported_claims", human=True)
    _check(checks, "contradictions", kind="contradictions", human=True)
    for check in gold.critical_checks:
        _check(checks, "critical." + check.id, kind="critical_contract", human=True, critical=True)


def _qa_structure(parsed, metrics, checks, data):
    output = parsed.normalized_value if parsed.production_usable else {}
    required = ("tags", "overall_sentiment", "call_quality_score", "summary")
    tags = output.get("tags")
    tag_list = type(tags) is list
    valid_entries = []
    invalid = invalid_vocab = 0
    if tag_list:
        for entry in tags:
            if (type(entry) is not dict or type(entry.get("tag")) is not str
                    or type(entry.get("reason")) is not str):
                invalid += 1
                continue
            if entry["tag"] not in QA_TAGS:
                invalid_vocab += 1
            else:
                valid_entries.append(entry["tag"])
    sentiment = output.get("overall_sentiment")
    sentiment_ok = type(sentiment) is str and sentiment in ("positive", "neutral", "negative")
    score = output.get("call_quality_score")
    score_ok = _number(score) and 1 <= score <= 10
    summary_ok = type(output.get("summary")) is str
    structure = parsed.production_usable and tag_list and not invalid and not invalid_vocab and sentiment_ok and score_ok and summary_ok
    data["structure_valid"] = bool(structure)
    metrics["structure_complete"] = _rate(int(structure), 1)
    metrics["unexpected_key_count"] = _metric(len(set(output)-set(required)) if parsed.production_usable else None)
    for name in required:
        _check(checks, "structure." + name, name in output, kind="field_presence", subject=name)
    if any(name not in output for name in required):
        data["diagnostic_codes"] += ("missing_qa_fields",)
    data["predicted_tags"] = tuple(sorted(set(valid_entries)))
    if tag_list:
        metrics.update(invalid_tag_count=_metric(invalid_vocab), invalid_tag_entry_count=_metric(invalid),
                       duplicate_tag_count=_metric(len(valid_entries)-len(set(valid_entries))))
    return output, tag_list and not invalid and not invalid_vocab, sentiment_ok, score_ok


def _score_qa(case, parsed, metrics, checks, data):
    output, tags_ok, sentiment_ok, score_ok = _qa_structure(parsed, metrics, checks, data)
    gold = case.expected
    _check(checks, "qa_structure", data["structure_valid"], kind="structure", critical=case.critical)
    e, f, p = set(gold.expected_tags), set(gold.forbidden_tags), set(data["predicted_tags"])
    data["unannotated_tags"] = tuple(sorted(p-e-f))
    metrics["annotation_coverage"] = _rate(len(e | f), len(QA_TAGS))
    if type(output.get("tags")) is list:
        metrics["unannotated_tag_count"] = _metric(len(p-e-f))
    if tags_ok:
        tp, fn, fp = len(p & e), len(e-p), len(p & f)
        metrics.update(tag_true_positive=_metric(tp), tag_false_negative=_metric(fn),
                       tag_false_positive=_metric(fp), unannotated_tag_count=_metric(len(p-e-f)))
        metrics.update(annotation_scoped_precision=_rate(tp, tp+fp),
                       annotation_scoped_recall=_rate(tp, tp+fn), annotation_scoped_f1=_rate(2*tp, 2*tp+fp+fn))
        for tag in QA_TAGS:
            if tag in e | f:
                for name, count in (("tp", int(tag in p & e)), ("fn", int(tag in e-p)), ("fp", int(tag in p & f))):
                    metrics[f"annotation_scoped.{tag}.{name}"] = _metric(count)
        _check(checks, "expected_tags_present", not (e-p), critical=case.critical)
        _check(checks, "forbidden_tags_absent", not (p & f), critical=case.critical)
    for index, entry in enumerate(output.get("tags", []) if type(output.get("tags")) is list else []):
        if type(entry) is dict and type(entry.get("reason")) is str:
            _check(checks, f"tag_grounding.{index}", kind="tag_grounding", subject=f"tags.{index}", human=True)
    for tag in data["unannotated_tags"]:
        _check(checks, "unannotated." + tag, kind="unannotated_tag", subject=tag, human=True)
    if sentiment_ok:
        match = output["overall_sentiment"] == gold.expected_sentiment
        metrics["sentiment_match"] = _rate(int(match), 1)
        _check(checks, "sentiment_match", match)
    metrics["score_coverage"] = _rate(int(score_ok), 1)
    if score_ok:
        score = output["call_quality_score"]
        lo, hi = gold.quality_score_range or (gold.quality_score_target, gold.quality_score_target)
        distance = max(lo-score, 0, score-hi)
        metrics["score_distance"] = _rate(distance, 1)
        metrics["score_in_range"] = _rate(int(distance == 0), 1)
        _check(checks, "score_in_range", distance == 0)
    _fact_checks(gold, checks)
    if type(output.get("summary")) is str:
        _sentence(output["summary"], (1, 2), metrics, checks)


def _parse(task, raw):
    if task == Task.EXTRACTION:
        return parsing.parse_extraction(raw)
    if task == Task.VOICEMAIL:
        return parsing.parse_voicemail(raw)
    if task == Task.QA:
        return parsing.parse_qa(raw)
    return parsing.parse_summary(raw)


def _parsed_fields(parsed, data, metrics):
    for name in ("strict_json_valid", "strict_format_valid", "production_parse_success", "production_usable", "parser_path"):
        data[name] = getattr(parsed, name)
    data["diagnostic_codes"] = tuple(d.code for d in parsed.diagnostics)
    for name in ("strict_json_valid", "strict_format_valid", "production_parse_success", "production_usable"):
        value = getattr(parsed, name)
        if value is not None:
            metrics[name] = _rate(int(value), 1)


def _case_data(case):
    if isinstance(case, CapturedReplayCase):
        # Captured payloads cannot be round-tripped through ordinary serialization.
        try:
            case = CapturedReplayCase(contract=case.contract, captured=case.captured)
        except ValidationError:
            failure = EvaluationError("invalid_captured_evaluation_case")
        else:
            failure = None
        if failure:
            raise failure
        return dict(case_id=case.case_id, task=case.captured.task, contract_id=case.captured.task.value,
                    source=Source.OVP34_REPLAY, exercise_kind="model", critical=False, gold_available=False,
                    case_fingerprint=evaluation_fingerprint("case", case.identity_projection()))
    case = checked(BenchmarkCase, case)
    if case.source == Source.OVP34_REPLAY:
        raise EvaluationError("captured_replay_case_required")
    return dict(case_id=case.id, task=case.task, contract_id=case.contract_id, source=case.source,
                exercise_kind=case.exercise.kind, critical=case.critical, gold_available=case.expected is not None,
                case_fingerprint=hash_dataset((case,)))


def _finish(data, metrics, checks, *, assessed):
    identity = {name: data.get(name) for name in ("dataset_hash", "case_id", "case_fingerprint",
        "evidence_fingerprint", "evaluator_fingerprint", "purpose", "execution_state", "attempt_index")}
    identity["evaluator_version"] = EVALUATOR_VERSION
    key = data.get("execution_key")
    identity["execution_key"] = key.model_dump(mode="json") if key else None
    score = ScoreResult(case_id=data["case_id"], exercise_kind=data["exercise_kind"],
        status="not_scored" if not assessed else ("pending_review" if any(c.status == "pending" for c in checks) else "complete"),
        metrics=metrics, critical_checks=tuple(CriticalCheckResult(check_id=c.check_id, status=c.status)
                                             for c in checks if c.critical))
    try:
        result = CaseEvaluation(**data, evaluator_version=EVALUATOR_VERSION,
            evaluation_id=evaluation_fingerprint("result", identity), score=score, checks=tuple(checks))
    except ValidationError:
        failure = EvaluationError("invalid_evaluation_identity")
    else:
        return result
    raise failure


def evaluate_model_output(case, *, request: PreparedRequest, attempt: AttemptRecorded | None,
                          dataset_hash: str, evaluator_fingerprint: str, purpose: str,
                          execution_key: ExecutionKey | None = None, execution_state: str = "finalized") -> CaseEvaluation:
    """Effect-free evaluation. Caller proves finalization; evaluate_run does so publicly."""
    data = _case_data(case)
    if data["exercise_kind"] != "model" or purpose not in ("candidate", "functional_stub"):
        raise EvaluationError("model_evaluation_required")
    if not isinstance(request, PreparedRequest):
        raise EvaluationError("prepared_request_required")
    if (request.case_id != data["case_id"] or request.task != data["task"] or request.source != data["source"]):
        raise EvaluationError("case_request_mismatch")
    if isinstance(case, CapturedReplayCase) and (request.captured is None or request.message_fingerprint != case.message_fingerprint):
        raise EvaluationError("captured_request_mismatch")
    if attempt is not None:
        attempt = checked(AttemptRecorded, attempt)
        if execution_key is not None and execution_key != attempt.execution_key:
            raise EvaluationError("attempt_key_mismatch")
        execution_key = attempt.execution_key
    if execution_key is None or (execution_key.case_id != data["case_id"] or execution_key.request_fingerprint != request.request_fingerprint):
        raise EvaluationError("execution_key_mismatch")
    if (execution_state not in ("finalized", "not_attempted", "unfinalized")
            or (execution_state == "not_attempted") != (attempt is None)):
        raise EvaluationError("invalid_execution_state")
    data.update(dataset_hash=dataset_hash, evaluator_fingerprint=evaluator_fingerprint, purpose=purpose,
        execution_key=execution_key, execution_state=execution_state, attempt_index=attempt.attempt_index if attempt else None,
        evidence_fingerprint=evaluation_fingerprint("evidence", attempt.model_dump(mode="json") if attempt else None),
        content_state="not_available", diagnostic_codes=())
    metrics = {name: _metric() for name in TASK_METRICS.get(data["task"], ())}
    checks = []
    if isinstance(case, BenchmarkCase) and isinstance(case.expected, VoicemailExpected):
        data.update(expected_label=case.expected.label, voicemail_stratum="unlabelled" if case.expected.label is None
                    else "ambiguous_labelled" if case.expected.ambiguous else "labelled")
    if attempt is not None:
        result = attempt.result
        data.update(transport_status=result.status, latency_ms=result.latency_ms,
                    transport_error_code=result.error_code, http_status=result.http_status, retryable=result.retryable)
        response = result.response
        if response is not None:
            data.update(usage=response.usage, response_model=response.response_model, finish_reason=response.finish_reason,
                        omitted_metadata=response.omitted_metadata,
                        content_fingerprint=evaluation_fingerprint("content", response.raw_content))
            raw = response.raw_content
            data["content_state"] = ("content_absent" if not response.content_present else "null" if raw is None
                                     else "empty" if raw == "" else "whitespace" if not raw.strip() else "text")
        else:
            raw = None
            data["content_state"] = "response_absent"
        if execution_state != "finalized" or result.status != "success" or response is None:
            return _finish(data, metrics, checks, assessed=False)
        # Content-absent evidence remains distinct even for unusual allowed Stage 3B combinations.
        raw = raw if response.content_present else None
        parsed = _parse(data["task"], raw)
        _parsed_fields(parsed, data, metrics)
        metrics["output_characters"] = _metric(len(raw) if raw is not None else 0)
        if not data["gold_available"]:
            if data["task"] == Task.QA:
                _qa_structure(parsed, metrics, checks, data)
            elif data["task"] == Task.VOICEMAIL:
                data["predicted_label"] = parsed.normalized_value
            return _finish(data, metrics, checks, assessed=False)
        if data["task"] == Task.EXTRACTION:
            data["structure_valid"] = parsed.production_usable
            _score_extraction(case, parsed, metrics, checks)
        elif data["task"] == Task.VOICEMAIL:
            _score_voicemail(case, parsed, metrics, checks, data)
        elif data["task"] == Task.QA:
            _score_qa(case, parsed, metrics, checks, data)
        else:
            _check(checks, "nonblank", parsed.production_usable, kind="availability", critical=case.critical)
            _fact_checks(case.expected, checks)
            _sentence(raw or "", None if case.task == Task.CONTEXT_SUMMARY else case.expected.sentence_range, metrics, checks)
            chars = sum(len(m.get("content", "")) for m in request.messages.to_messages()
                        if m.get("role") == "user" and type(m.get("content")) is str)
            metrics["output_to_user_input_character_ratio"] = _rate(len(raw or ""), chars)
        return _finish(data, metrics, checks, assessed=True)
    return _finish(data, metrics, checks, assessed=False)


def _adapter_observations(case):
    assertions = case.expected.assertions
    if case.task == Task.EXTRACTION:
        rendered = adapters.render_extraction(case.input, load_contract(case.contract_id))
        history = rendered[-1]["content"].split("\n\nConversation history:\n", 1)[1]
        observed = {"formatted_history": history}
        if case.id.endswith("ex-015"):
            retained = [i for i, message in enumerate(case.input.messages)
                        if adapters.render_extraction(case.input.model_copy(update={"messages": (message,)}))[-1]["content"].split("\n\nConversation history:\n", 1)[1]]
            observed.update(retained_message_indices=retained,
                            omitted_message_indices=[i for i in range(len(case.input.messages)) if i not in retained])
        if case.id.endswith("ex-016"):
            tool, following = history.split("\nuser: ", 1)
            body = tool.split("\n", 1)[1]
            suffix = "...(truncated)"
            observed.update(retained_characters=len(body)-len(suffix) if body.endswith(suffix) else len(body),
                            truncation_suffix=suffix if body.endswith(suffix) else "", following_message="user: " + following)
        return observed
    if case.task == Task.CONTEXT_SUMMARY:
        selection = adapters.select_summary_context(case.input)
        start = selection.last_summarized_index + 1 - len(selection.messages)
        observed = dict(last_summarized_index=selection.last_summarized_index,
            request_created=adapters.render_runtime_summary(case.input) is not None,
            selected_indices=list(range(start, selection.last_summarized_index+1)),
            preserved_indices=list(range(selection.last_summarized_index+1, len(case.input.messages))))
        if case.id.endswith("cs-023"):
            current = case.exercise.current_messages
            applied = adapters.apply_context_summary(current, selection.last_summarized_index, assertions["summary_text"])
            observed.update(summary_text=assertions["summary_text"], expected_summary_message=applied[1].content,
                preserved_current_indices=list(range(selection.last_summarized_index+1, len(current))) if applied[2:] == current[selection.last_summarized_index+1:] else [],
                use_current_system=applied[0] == current[0])
        return observed
    raise EvaluationError("unknown_adapter_control")


def _assertion_equal(actual, expected):
    if type(expected) is dict:
        return type(actual) is dict and actual.keys() == expected.keys() and all(
            _assertion_equal(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return type(actual) is list and len(actual) == len(expected) and all(
            _assertion_equal(a, b) for a, b in zip(actual, expected))
    return _equal(actual, expected)


def evaluate_control_case(case, *, dataset_hash: str, evaluator_fingerprint: str) -> CaseEvaluation:
    case, = validate_curated_cases((case,))
    if case.exercise.kind == "model":
        raise EvaluationError("control_evaluation_required")
    data = _case_data(case)
    data.update(dataset_hash=dataset_hash, evaluator_fingerprint=evaluator_fingerprint, purpose="control",
        execution_state="control", content_state="not_applicable", diagnostic_codes=(),
        evidence_fingerprint=evaluation_fingerprint("control", case.model_dump(mode="json")))
    metrics, checks = {}, []
    if case.exercise.kind == "adapter_control":
        observed = _adapter_observations(case)
    elif case.exercise.kind == "transport_contract":
        if not case.id.endswith("cs-025") or case.exercise.response != "timeout":
            raise EvaluationError("unknown_transport_control")
        observed = dict(transport_event="timeout", semantic_quality_applicable=False,
                        source_timeout_seconds=load_contract("context_summary").source_settings.timeout_seconds)
        data["transport_status"] = "timeout"
    else:
        raw = case.exercise.raw_output
        data.update(content_state="null" if raw is None else "empty" if not raw else "whitespace" if not raw.strip() else "text",
                    content_fingerprint=evaluation_fingerprint("content", raw))
        parsed = _parse(case.task, raw)
        _parsed_fields(parsed, data, metrics)
        if isinstance(case.expected, QAEvaluationExpected):
            if not case.id.endswith("qa-026"):
                raise EvaluationError("unknown_semantic_control")
            _score_qa(case, parsed, metrics, checks, data)
            failed = {c.check_id for c in checks if c.status == "fail"}
            # This negative control passes when a reviewer identifies fabrication;
            # it does not ask the reviewer to approve the supplied false response.
            checks = [c.model_copy(update={"critical": False}) for c in checks if c.method == "automatic"]
            _check(checks, "critical.matrix-critical", kind="negative_grounding", human=True, critical=True)
            _check(checks, "control.negative_stimulus", parsed.production_usable and data["structure_valid"]
                and {"forbidden_tags_absent", "sentiment_match", "score_in_range"} <= failed,
                kind="control_assertion")
            return _finish(data, metrics, checks, assessed=True)
        observed = {name: getattr(parsed, name) for name in (
            "strict_json_valid", "strict_format_valid", "parser_path", "production_usable", "parsed_value", "normalized_value")}
        observed["decision"] = parsed.normalized_value
        if case.task == Task.QA:
            output = parsed.normalized_value
            defaults = {"tags": [], "summary": "", "overall_sentiment": None, "call_quality_score": None}
            observed["missing_fields"] = [name for name in defaults if name not in output]
            requested_defaults = case.expected.assertions.get("source_defaults", {})
            if type(requested_defaults) is not dict or requested_defaults.keys() - defaults.keys():
                raise EvaluationError("unknown_source_default")
            observed["source_defaults"] = {name: output.get(name, defaults[name]) for name in requested_defaults}
        elif case.task == Task.CONTEXT_SUMMARY:
            selection = adapters.select_summary_context(case.input)
            observed.update(last_summarized_index=selection.last_summarized_index,
                context_unchanged=adapters.apply_context_summary(case.input.messages, selection.last_summarized_index, raw) == case.input.messages)
    if not isinstance(case.expected, ContractExpected) or set(case.expected.assertions)-observed.keys():
        raise EvaluationError("unknown_control_assertion")
    for name, expected in case.expected.assertions.items():
        # JSON-semantic assertion equality, including bool versus numeric distinction.
        _check(checks, "control." + name, _assertion_equal(observed[name], expected), kind="control_assertion")
    for check in case.expected.critical_checks:
        _check(checks, "critical." + check.id, all(c.status == "pass" for c in checks), kind="control_assertion", critical=True)
    return _finish(data, metrics, checks, assessed=True)


def apply_review_decisions(evaluation, decisions) -> CaseEvaluation:
    evaluation = checked(CaseEvaluation, evaluation)
    decisions = tuple(checked(ReviewDecision, item) for item in decisions)
    checks = {c.check_id: c for c in evaluation.checks}
    seen = set()
    for decision in decisions:
        if (decision.evaluation_id != evaluation.evaluation_id or decision.check_id not in checks
                or decision.check_id in seen or checks[decision.check_id].method != "human"):
            raise EvaluationError("invalid_review_binding")
        seen.add(decision.check_id)
        checks[decision.check_id] = checks[decision.check_id].model_copy(update={"status": decision.status})
    updated = tuple(checks.values())
    score = evaluation.score.model_copy(update={
        "status": "not_scored" if evaluation.score.status == "not_scored" else "pending_review" if any(c.status == "pending" for c in updated) else "complete",
        "critical_checks": tuple(CriticalCheckResult(check_id=c.check_id, status=c.status) for c in updated if c.critical)})
    return evaluation.model_copy(update={"checks": updated, "score": score})


def build_human_review_items(case, evaluation, *, raw_output) -> tuple[HumanReviewItem, ...]:
    evaluation = checked(CaseEvaluation, evaluation)
    if _case_data(case)["case_fingerprint"] != evaluation.case_fingerprint:
        raise EvaluationError("review_case_mismatch")
    if type(raw_output) is not str and raw_output is not None:
        raise EvaluationError("invalid_review_output")
    if evaluation.content_fingerprint is not None and evaluation_fingerprint("content", raw_output) != evaluation.content_fingerprint:
        raise EvaluationError("review_output_mismatch")
    if isinstance(case, CapturedReplayCase):
        return ()  # No historical gold or invented factual checklist.
    gold = case.expected
    source = case.input.model_dump(mode="json")
    items = []
    for check in evaluation.checks:
        if check.method != "human":
            continue
        evidence, statement = (), ""
        if check.kind == "required_fact":
            facts = getattr(gold, "required_facts", getattr(gold, "summary_facts", ()))
            fact = facts[int(check.subject.rsplit(".", 1)[1])]
            statement, evidence = fact.statement, fact.evidence
        elif check.kind in ("stale_current_state", "forbidden_claim", "optional_inclusion"):
            attr, index = check.subject.split(".")
            fact = getattr(gold, attr)[int(index)]
            statement, evidence = fact.statement, fact.evidence
        elif check.kind == "correction":
            correction = gold.corrections[int(check.subject.split(".")[-1])]
            old = next(f for f in gold.superseded_facts if f.id == correction.old_fact_id)
            new = next(f for f in gold.required_facts if f.id == correction.new_fact_id)
            statement, evidence = old.statement + " -> " + new.statement, old.evidence + new.evidence
        elif check.kind in ("critical_contract", "negative_grounding"):
            statement = next(c.description for c in gold.critical_checks if "critical." + c.id == check.check_id)
            if check.kind == "negative_grounding":
                evidence = (EvidenceRef(input_pointer="/transcript"),)
        elif check.kind == "tag_grounding":
            parsed = parsing.parse_qa(raw_output)
            entry = parsed.normalized_value["tags"][int(check.subject.split(".")[-1])]
            statement = entry["reason"]
            evidence = gold.tag_evidence.get(entry.get("tag"), (EvidenceRef(input_pointer="/transcript"),))
        if not evidence:
            if case.task == Task.QA:
                pointers = ("/transcript", "/node_summary", "/previous_conversation_summary", "/metrics")
            elif case.task == Task.QA_CONVERSATION_SUMMARY:
                pointers = ("/prior_transcript",)
            elif case.task == Task.QA_NODE_SUMMARY:
                pointers = ("/agent_prompt", "/custom_tools", "/outgoing_edges")
            elif case.task == Task.CONTEXT_SUMMARY:
                selection = adapters.select_summary_context(case.input)
                start = selection.last_summarized_index + 1 - len(selection.messages)
                pointers = tuple(f"/messages/{i}" for i in range(start, selection.last_summarized_index + 1))
            else:
                pointers = ("/messages",)
            evidence = tuple(EvidenceRef(input_pointer=p) for p in pointers)
        if check.kind == "sentence_compliance":
            bounds = (1, 2) if case.task == Task.QA else case.expected.sentence_range
            statement = f"Source sentence range: {bounds[0]}-{bounds[1]}. Punctuation count is diagnostic only."
        snippets = []
        for ref in evidence:
            value = source
            for part in ref.input_pointer.split("/")[1:]:
                part = part.replace("~1", "/").replace("~0", "~")
                value = value[int(part)] if isinstance(value, list) else value[part]
            if type(value) is not str:
                value = canonical_json_bytes(value).decode()
            snippets.append(value[ref.start_char:ref.end_char] if ref.start_char is not None else value)
        question = {
            "required_fact": "Is the required meaning accurately preserved, including qualifications?",
            "stale_current_state": "Is obsolete information avoided as CURRENT state? Historical mention is allowed.",
            "forbidden_claim": "Is the forbidden assertion absent?",
            "optional_inclusion": "If included, is this fact accurate? Omission passes without penalty.",
            "correction": "Is the old-to-new correction and current state preserved?",
            "tag_grounding": "Is the reason supported by the supplied context without invented evidence?",
            "unannotated_tag": "Is this unannotated source-vocabulary prediction supported by context?",
            "unsupported_claims": "Are all additional claims supported by the supplied context?",
            "contradictions": "Does the output avoid contradicting the supplied context?",
            "sentence_compliance": "Does the text meet the source sentence range, considering linguistic ambiguity?",
            "critical_contract": "Is the authored critical requirement satisfied?",
            "negative_grounding": "Confirm that the supplied hang-up threat claim is unsupported by the transcript.",
            "unlabelled_inspection": "Inspect the decision without assigning a binary gold label.",
        }[check.kind]
        if case.task == Task.QA and check.kind in ("required_fact", "sentence_compliance"):
            question = "Evaluate the summary field only. " + question
        items.append(HumanReviewItem(evaluation_id=evaluation.evaluation_id, case_id=evaluation.case_id,
            contract_id=evaluation.contract_id, check=check, candidate_output=raw_output, question=question,
            statement=statement, evidence=evidence, snippets=tuple(snippets)))
    return tuple(items)
