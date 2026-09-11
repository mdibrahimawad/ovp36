"""Read-only JSONL dataset operations; no real fixtures or replay reconstruction."""

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from .schemas import BenchmarkCase, Difficulty, Source, Task


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
            raise DatasetError(f"duplicate case ID: {case.id}")
        seen.add(case.id)
    return items


def load_cases(paths: str | Path | Iterable[str | Path]) -> tuple[BenchmarkCase, ...]:
    """Load every line strictly, preserving file/line order; never skip bad rows."""
    sources = (paths,) if isinstance(paths, (str, Path)) else paths
    cases = []
    seen = set()
    for source in sources:
        path = Path(source)
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        # Reject ambiguous JSON keys and non-standard NaN/Infinity
                        # before Pydantic's JSON-mode strict schema validation.
                        json.loads(line, object_pairs_hook=_unique_object,
                                   parse_constant=_reject_constant)
                        case = BenchmarkCase.model_validate_json(line)
                    except ValidationError as exc:
                        errors = exc.errors(include_input=False, include_context=False, include_url=False)
                        details = "; ".join(
                            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                            for error in errors[:3]
                        )
                        raise DatasetError(
                            f"{path}:{line_number}: invalid case ({details})"
                        ) from None
                    except ValueError as exc:
                        raise DatasetError(
                            f"{path}:{line_number}: invalid case ({exc})"
                        ) from None
                    if case.id in seen:
                        raise DatasetError(f"{path}:{line_number}: duplicate case ID: {case.id}")
                    seen.add(case.id)
                    cases.append(case)
        except UnicodeError:
            raise DatasetError(f"{path}: dataset must be UTF-8") from None
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
