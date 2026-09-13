"""JSONL datasets and curated inventory validation; no fixtures or execution."""

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from pydantic import BaseModel

from .contracts import load_contract

from .schemas import (
    BenchmarkCase, Difficulty, EvidenceRef, ExtractionExpected, FactAnnotation,
    QAEvaluationExpected, Source, SummaryExpected, Task, VoicemailExpected, validate_safe_identifier,
)


class DatasetError(ValueError):
    """Invalid JSONL, duplicate IDs, or a matrix coverage mismatch."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _validate_unique_ids(cases: Iterable[BenchmarkCase]) -> tuple[BenchmarkCase, ...]:
    items = tuple(cases)
    seen = set()
    for case in items:
        if case.id in seen:
            raise DatasetError("duplicate case ID")
        seen.add(case.id)
    return items


def load_cases(paths: str | Path | Iterable[str | Path]) -> tuple[BenchmarkCase, ...]:
    """Load every line strictly, preserving file/line order; never skip bad rows."""
    sources = (paths,) if isinstance(paths, (str, Path)) else paths
    cases = []
    seen = set()
    for file_number, source in enumerate(sources, 1):
        failure = None
        try:
            with Path(source).open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        json.loads(line, object_pairs_hook=_unique_object,
                                   parse_constant=_reject_constant)
                        case = BenchmarkCase.model_validate_json(line)
                    except ValueError:
                        failure = DatasetError(f"file{file_number}:{line_number}: invalid case")
                    if failure is not None:
                        break
                    if case.id in seen:
                        failure = DatasetError(f"file{file_number}:{line_number}: duplicate case ID")
                        break
                    seen.add(case.id)
                    cases.append(case)
        except UnicodeError:
            failure = DatasetError(f"file{file_number}: dataset must be UTF-8")
        except OSError:
            failure = DatasetError(f"file{file_number}: dataset read failed")
        # Raise outside the handler: `from None` alone retains __context__.
        if failure is not None:
            raise failure
    return tuple(cases)


def select_cases(
    cases: Iterable[BenchmarkCase], *,
    tasks: Iterable[Task] | None = None,
    sources: Iterable[Source] | None = None,
    difficulties: Iterable[Difficulty] | None = None,
    critical: bool | None = None,
    tags: Iterable[str] | None = None,
    exercise_kinds: Iterable[str] | None = None,
) -> tuple[BenchmarkCase, ...]:
    """Intersect filters; tags require all requested tags. None means unrestricted."""
    task_set = None if tasks is None else {Task(task) for task in tasks}
    source_set = None if sources is None else {Source(source) for source in sources}
    difficulty_set = None if difficulties is None else {Difficulty(d) for d in difficulties}
    tag_set = set(tags or ())
    exercise_set = None if exercise_kinds is None else set(exercise_kinds)
    allowed = {"model", "adapter_control", "response_contract", "transport_contract"}
    if exercise_set is not None and not exercise_set <= allowed:
        raise ValueError("unknown exercise kind")
    if critical is not None and type(critical) is not bool:
        raise ValueError("critical filter must be a boolean or None")
    return tuple(
        case for case in cases
        if (task_set is None or case.task in task_set)
        and (source_set is None or case.source in source_set)
        and (difficulty_set is None or case.difficulty in difficulty_set)
        and (critical is None or case.critical is critical)
        and tag_set <= set(case.tags)
        and (exercise_set is None or case.exercise.kind in exercise_set)
    )


@dataclass(frozen=True)
class MatrixCoverage:
    missing_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_ids and not self.unexpected_ids


def validate_matrix_coverage(
    cases: Iterable[BenchmarkCase], expected_ids: Iterable[str], *,
    require_complete: bool = True,
) -> MatrixCoverage:
    """Compare with caller-supplied matrix IDs; never fabricate absent fixtures."""
    actual = {case.id for case in _validate_unique_ids(cases)}
    expected = set(expected_ids)
    coverage = MatrixCoverage(tuple(sorted(expected - actual)), tuple(sorted(actual - expected)))
    if require_complete and not coverage.complete:
        raise DatasetError(
            f"matrix coverage mismatch: missing={coverage.missing_ids}, "
            f"unexpected={coverage.unexpected_ids}"
        )
    return coverage


def hash_dataset(cases: Iterable[BenchmarkCase]) -> str:
    """SHA-256 of normalized cases sorted by ID, independent of JSONL layout.

    Case/message sequence contents, gold, provenance, and contract IDs contribute
    to the hash. File ordering and JSON object key ordering do not.
    """
    ordered = sorted(_validate_unique_ids(cases), key=lambda case: case.id)
    payload = json.dumps(
        [case.model_dump(mode="json") for case in ordered],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Matrix metadata only, NOT authored benchmark cases. e/n/h/a = difficulty;
# the final set of ordinals in each row identifies noncritical scenarios.
_SUITE = (
    (Task.EXTRACTION, "ex", "eeennnhnnhhnnnnhhnnhaaannh", {9, 13, 14, 15, 16, 23, 24, 25}),
    (Task.CONTEXT_SUMMARY, "cs", "ennnnnnhhhnnhhhhhaahnhhaah", {1, 11, 12, 17, 21, 25}),
    (Task.VOICEMAIL, "vm", "eeennhhheennnnnhhhhhaaanhh", {19, 20, 21, 22, 23, 24}),
    (Task.QA, "qa", "eeeeenennnehhhhhhhhhhhhnaannaahh", {24, 27, 28, 29, 30}),
    (Task.QA_CONVERSATION_SUMMARY, "qs", "ennhhhnhaa", {7}),
    (Task.QA_NODE_SUMMARY, "ns", "ennhha", set()),
)
_CONTROLS = {
    "adapter_control": {"EX-013", "EX-014", "EX-015", "EX-016", "CS-001", "CS-021", "CS-023"},
    "response_contract": {"EX-024", "EX-025", "EX-026", "CS-024", "VM-022", "VM-023", "VM-024",
                          "QA-026", "QA-027", "QA-028", "QA-029", "QA-030"},
    "transport_contract": {"CS-025"},
}
QA_METRIC_FIELDS = (
    "call_duration_seconds", "num_turns", "avg_latency_seconds",
    "avg_ttfb_seconds", "max_latency_seconds",
)


def _matrix_entry(case: BenchmarkCase) -> tuple[int, int, str]:
    validate_safe_identifier(case.id)
    match = re.fullmatch(r"(curated|adversarial)-(ex|cs|vm|qa|qs|ns)-(\d{3})", case.id)
    if match is None or match[1] != case.source.value:
        raise ValueError("invalid_curated_id")
    ordinal = int(match[3])
    for position, (task, prefix, difficulties, noncritical) in enumerate(_SUITE):
        if prefix != match[2]:
            continue
        if task != case.task or not 1 <= ordinal <= len(difficulties):
            raise ValueError("invalid_matrix_reference")
        difficulty = dict(e="easy", n="normal", h="hard", a="adversarial")[difficulties[ordinal - 1]]
        source = "adversarial" if difficulty == "adversarial" else "curated"
        matrix_id = f"{prefix.upper()}-{ordinal:03}"
        exercise = next((kind for kind, ids in _CONTROLS.items() if matrix_id in ids), "model")
        if (case.difficulty.value != difficulty or case.source.value != source
                or case.critical != (ordinal not in noncritical) or case.exercise.kind != exercise
                or [tag for tag in case.tags if tag.startswith("matrix:")] != [f"matrix:{matrix_id}"]):
            raise ValueError("matrix_allocation_mismatch")
        return position, ordinal, matrix_id
    raise ValueError("invalid_matrix_reference")


def _validate_evidence(evidence: EvidenceRef, input: object) -> None:
    value = input
    for token in evidence.input_pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("invalid_evidence_pointer")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9]\d*", token) and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError("unresolved_evidence_pointer")
    if evidence.start_char is not None and (
        not isinstance(value, str) or evidence.end_char > len(value)
    ):
        raise ValueError("invalid_evidence_span")


def _walk_evidence(value: object, input: object) -> None:
    if isinstance(value, EvidenceRef):
        _validate_evidence(value, input)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _walk_evidence(getattr(value, name), input)
    elif isinstance(value, dict):
        for item in value.values():
            _walk_evidence(item, input)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _walk_evidence(item, input)


def _grounded(facts: tuple[FactAnnotation, ...]) -> None:
    if not facts or any(not fact.evidence for fact in facts):
        raise ValueError("missing_fact_evidence")


def _validate_gold(case: BenchmarkCase) -> None:
    expected = case.expected
    _walk_evidence(expected, case.input.model_dump(mode="json"))
    if isinstance(expected, ExtractionExpected):
        if any(gold.state == "known" and not gold.evidence for gold in expected.fields.values()):
            raise ValueError("missing_field_evidence")
    if isinstance(expected, SummaryExpected):
        _grounded(expected.required_facts)
        if expected.superseded_facts:
            _grounded(expected.superseded_facts)
        sentence_range = {Task.QA_CONVERSATION_SUMMARY: (3, 5), Task.QA_NODE_SUMMARY: (2, 4)}.get(case.task)
        if sentence_range and expected.sentence_range != sentence_range:
            raise ValueError("summary_sentence_annotation_mismatch")
    if isinstance(expected, QAEvaluationExpected):
        tags = set(re.findall(r"^- ([A-Z_]+) -", load_contract(case.contract_id).system_template, re.MULTILINE))
        if (not set(expected.expected_tags + expected.forbidden_tags) <= tags
                or len(set(expected.expected_tags)) != len(expected.expected_tags)
                or len(set(expected.forbidden_tags)) != len(expected.forbidden_tags)
                or any(not refs for refs in expected.tag_evidence.values())):
            raise ValueError("invalid_qa_tag_annotations")
        _grounded(expected.summary_facts)
    if case.task == Task.QA:
        metrics = case.input.metrics
        if metrics.keys() != set(QA_METRIC_FIELDS):
            raise ValueError("invalid_qa_metric_fields")
        if type(metrics["num_turns"]) is not int or metrics["num_turns"] < 0:
            raise ValueError("invalid_qa_metrics")
        if any(value is not None and (type(value) not in (int, float) or value < 0)
               for key, value in metrics.items() if key != "num_turns"):
            raise ValueError("invalid_qa_metrics")
        if case.input.scope == "node" and metrics["call_duration_seconds"] is not None:
            raise ValueError("node_duration_must_be_null")


def validate_curated_cases(
    cases: Iterable[BenchmarkCase], *, require_complete: bool = False,
) -> tuple[BenchmarkCase, ...]:
    """Validate authored matrix metadata and evidence; preserve canonical order.

    Partial collections support test-local fixtures and explicit selections. This
    does not attest synthetic origin or judge semantic gold; human review is required.
    """
    if type(require_complete) is not bool:
        raise DatasetError("invalid_complete_flag")
    validated, positions, seen = [], [], set()
    for number, case in enumerate(cases, 1):
        if not isinstance(case, BenchmarkCase):
            raise DatasetError(f"case:{number}: invalid curated case")
        failure = None
        try:
            case = BenchmarkCase.model_validate_json(case.model_dump_json(exclude_unset=True, warnings="error"))
            if case.source not in (Source.CURATED, Source.ADVERSARIAL) or "synthetic" not in case.tags:
                raise ValueError("synthetic_source_required")
            if case.provenance is not None:
                if case.provenance.historical_run_id is not None:
                    raise ValueError("historical_provenance_forbidden")
                validate_safe_identifier(case.provenance.source_ref)
                if not case.provenance.source_ref.startswith("synthetic-"):
                    raise ValueError("synthetic_provenance_required")
                for value in (case.provenance.source_revision, case.provenance.artifact_hash):
                    if value is not None:
                        validate_safe_identifier(value)
            contract = load_contract(case.contract_id)
            if contract.task != case.task or contract.completeness != "complete":
                raise ValueError("invalid_case_contract")
            position, ordinal, _ = _matrix_entry(case)
            if case.id in seen:
                raise ValueError("duplicate_curated_id")
            _validate_gold(case)
        except ValueError:
            failure = DatasetError(f"case:{number}: invalid curated case")
        if failure is not None:
            raise failure
        seen.add(case.id)
        positions.append((position, ordinal))
        validated.append(case)
    if positions != sorted(positions):
        raise DatasetError("noncanonical_curated_order")
    if require_complete and positions != [(p, i) for p, row in enumerate(_SUITE) for i in range(1, len(row[2]) + 1)]:
        raise DatasetError("incomplete_curated_inventory")
    return tuple(validated)


def load_curated_cases(root: str | Path) -> tuple[BenchmarkCase, ...]:
    """Read six explicit files under root; never discover or create fixtures."""
    cases = []
    for task, *_ in _SUITE:
        batch = load_cases(Path(root) / f"{task.value}.jsonl")
        if any(case.task != task for case in batch):
            raise DatasetError("curated_file_task_mismatch")
        cases.extend(batch)
    return validate_curated_cases(cases, require_complete=True)


def curated_review_rows(cases: Iterable[BenchmarkCase]) -> tuple[dict[str, object], ...]:
    """Small deterministic inventory projection, not a score/report or prompt dump."""
    rows = []
    for case in validate_curated_cases(cases):
        expected = case.expected
        if isinstance(expected, ExtractionExpected):
            annotation = {name: {"state": gold.state, "acceptable_values": gold.acceptable_values,
                                "missing_policy": gold.missing_policy.model_dump(mode="json") if gold.missing_policy else None,
                                "superseded_values": gold.superseded_values}
                          for name, gold in expected.fields.items()}
        elif isinstance(expected, SummaryExpected):
            annotation = {"required_facts": {fact.id: fact.statement for fact in expected.required_facts},
                          "critical_facts": expected.critical_facts,
                          "forbidden_facts": {fact.id: fact.statement for fact in expected.forbidden_facts},
                          "corrections": [c.model_dump(mode="json") for c in expected.corrections],
                          "optional_behaviors": expected.optional_behaviors,
                          "sentence_range": expected.sentence_range}
        elif isinstance(expected, QAEvaluationExpected):
            annotation = {"tags": expected.expected_tags, "sentiment": expected.expected_sentiment,
                          "score_target": expected.quality_score_target, "score_range": expected.quality_score_range}
        elif isinstance(expected, VoicemailExpected):
            annotation = {"label": expected.label, "ambiguous": expected.ambiguous, "rationale": expected.rationale}
        else:
            # Detach containers so a review consumer cannot mutate fixture annotations.
            annotation = {"assertions": json.loads(json.dumps(expected.assertions))}
        rows.append({"case_id": case.id, "matrix_id": _matrix_entry(case)[2],
                     "contract": case.contract_id, "source": case.source.value,
                     "exercise_kind": case.exercise.kind, "difficulty": case.difficulty.value,
                     "critical": case.critical, "short_scenario": " ".join(case.notes.split())[:160],
                     "expected_annotation_summary": annotation})
    return tuple(rows)
