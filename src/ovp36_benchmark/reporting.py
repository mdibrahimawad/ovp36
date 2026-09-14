"""Read-only execution joins, per-contract reports, and private immutable artifacts."""

from collections import Counter, defaultdict
from collections.abc import Sequence
import json
import os
from pathlib import Path
import stat
import tempfile

from pydantic import ValidationError

from .contracts import hash_contract, load_contract
from .dataset import hash_dataset, validate_curated_cases
from .evaluation import (
    CaseEvaluation, EvaluationError, QA_TAGS, ReviewDecision, apply_review_decisions,
    checked, evaluate_model_output, evaluation_fingerprint,
)
from .identity import ExecutionKey, canonical_json_bytes, fingerprint_execution_plan
from .persistence import AttemptRecorded, ExecutionFinalized, RunManifest, load_manifest, read_events
from .replay import ReplayDataset
from .runner import PlannedExecution


def evaluate_run(dataset, *, plan, results_root, expected_manifest, evaluator_fingerprint,
                 purpose) -> tuple[CaseEvaluation, ...]:
    """Join the full original plan to a strict journal snapshot, without mutations."""
    manifest = checked(RunManifest, expected_manifest)
    if purpose not in ("candidate", "candidate_smoke", "functional_stub"):
        raise EvaluationError("invalid_evaluation_purpose")
    if load_manifest(results_root, manifest.run_id) != manifest:
        raise EvaluationError("evaluation_manifest_mismatch")
    if isinstance(dataset, ReplayDataset):
        # Reconstruction revalidates the captured boundary; never use adapters.
        try:
            dataset = ReplayDataset(cases=dataset.cases, source_artifact_hash=dataset.source_artifact_hash,
                                    selection_artifact_hash=dataset.selection_artifact_hash)
        except ValidationError:
            failure = EvaluationError("invalid_replay_evaluation_dataset")
        else:
            failure = None
        if failure:
            raise failure
        digest = dataset.dataset_hash
        cases = {c.case_id: c for c in dataset.cases}
    else:
        dataset = validate_curated_cases(dataset)
        digest = hash_dataset(dataset)
        cases = {c.id: c for c in dataset}
    if digest != manifest.dataset_hash:
        raise EvaluationError("evaluation_dataset_mismatch")
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        raise EvaluationError("invalid_evaluation_plan")
    plan = tuple(plan)
    keys, projections = [], []
    for item in plan:
        if not isinstance(item, PlannedExecution):
            raise EvaluationError("invalid_evaluation_plan")
        try:
            item = PlannedExecution(request=item.request, repetition_index=item.repetition_index)
            projection = item.identity_projection()
            key = ExecutionKey(run_id=manifest.run_id, **projection)
        except ValidationError:
            failure = EvaluationError("invalid_evaluation_plan")
        else:
            failure = None
        if failure:
            raise failure
        if (key.case_id not in cases or key.repetition_index >= manifest.safe_config.repetitions
                or item.request.model != manifest.safe_config.requested_model):
            raise EvaluationError("evaluation_plan_binding_mismatch")
        case = cases[key.case_id]
        if not isinstance(dataset, ReplayDataset):
            if case.exercise.kind != "model" or item.request.source != case.source or item.request.task != case.task:
                raise EvaluationError("evaluation_case_binding_mismatch")
            if manifest.contract_hashes.get(case.contract_id) != hash_contract(load_contract(case.contract_id)):
                raise EvaluationError("evaluation_contract_mismatch")
        projections.append(projection)
        keys.append(key)
    if len(set(keys)) != len(keys) or fingerprint_execution_plan(projections) != manifest.ordered_execution_plan_hash:
        raise EvaluationError("evaluation_plan_hash_mismatch")
    # Complete validation happens before any evaluation result is returned.
    events = read_events(results_root, manifest.run_id)
    key_set = set(keys)
    attempts, finalized, latest = {}, {}, {}
    for event in events:
        if event.execution_key not in key_set:
            raise EvaluationError("evaluation_foreign_execution")
        if isinstance(event, AttemptRecorded):
            attempts[event.execution_key, event.attempt_index] = event
            latest[event.execution_key] = event
        elif isinstance(event, ExecutionFinalized):
            finalized[event.execution_key] = event.attempt_index
    results = []
    for item, key in zip(plan, keys, strict=True):
        final = key in finalized
        attempt = attempts[key, finalized[key]] if final else latest.get(key)
        results.append(evaluate_model_output(cases[key.case_id], request=item.request, attempt=attempt,
            execution_key=key, execution_state="finalized" if final else "unfinalized" if attempt else "not_attempted",
            dataset_hash=digest, evaluator_fingerprint=evaluator_fingerprint, purpose=purpose))
    return tuple(results)


def _ratio(n, d):
    return dict(value=n/d if d else None, numerator=n, denominator=d)


def _metric_summary(rows):
    names = sorted({name for row in rows for name in row.score.metrics})
    result = {}
    for name in names:
        all_values = [row.score.metrics.get(name) for row in rows]
        values = [m for m in all_values if m is not None and m.value is not None]
        rates = [m for m in all_values if m is not None and m.denominator is not None]
        if rates:
            item = _ratio(sum(m.numerator or 0 for m in rates), sum(m.denominator for m in rates))
        else:
            item = {"value": sum(m.value for m in values) if values else None,
                    "numerator": None, "denominator": None}
        item.update(available_records=len(values), unavailable_records=len(rows)-len(values),
                    pending_review_records=sum(row.score.status == "pending_review" for row in rows))
        result[name] = item
    return result


def _check_summary(rows):
    counts = defaultdict(Counter)
    for row in rows:
        for check in row.checks:
            counts[check.kind][check.status] += 1
            if check.critical:
                counts["critical_check"][check.status] += 1
            for category in check.categories:
                counts["category." + category][check.status] += 1
    output = {}
    for kind, values in sorted(counts.items()):
        output[kind] = {state: values[state] for state in ("pass", "fail", "pending", "not_applicable")}
        output[kind]["resolved_pass_rate"] = _ratio(values["pass"], values["pass"]+values["fail"])
        output[kind]["review_coverage"] = _ratio(values["pass"]+values["fail"], sum(values.values()))
    return output


def _critical_summary(rows):
    rows = [r for r in rows if r.critical]
    result = dict(selected=len(rows), evaluated=sum(r.critical_status != "unavailable" for r in rows),
                  unique_selected=len({r.case_id for r in rows}))
    for status in ("pass", "fail", "pending", "unavailable"):
        selected = [r for r in rows if r.critical_status == status]
        result[status] = len(selected)
        result[status + "_case_ids"] = sorted({r.case_id for r in selected})
    # Failures can coexist with pending checks; do not lose these during rollup.
    for status in ("fail", "pending"):
        result[status + "_checks"] = sorted({(r.case_id, c.check_id) for r in rows for c in r.checks
                                             if c.critical and c.status == status})
        result[status + "_checks"] = [list(pair) for pair in result[status + "_checks"]]
    result["resolved_pass_rate"] = _ratio(result["pass"], result["pass"]+result["fail"])
    return result


def _voicemail_summary(rows):
    output = {}
    for stratum in ("labelled", "ambiguous_labelled", "unlabelled"):
        selected = [r for r in rows if r.voicemail_stratum == stratum]
        eligible = [r for r in selected if r.score.metrics.get("binary_accuracy") is not None
                    and r.score.metrics["binary_accuracy"].value is not None]
        values = dict(selected=len(selected), evaluated=len(eligible), unavailable=len(selected)-len(eligible))
        if stratum == "unlabelled":
            values.update(evaluated=sum(r.production_parse_success is not None for r in selected),
                          binary_accuracy=None, classes={}, no_decision_rate=None)
            values["unavailable"] = len(selected)-values["evaluated"]
        else:
            correct = sum(r.predicted_label == r.expected_label for r in eligible)
            values.update(correct=correct, incorrect=len(eligible)-correct, binary_accuracy=_ratio(correct, len(eligible)),
                          no_decision_rate=_ratio(sum(r.predicted_label is None for r in eligible), len(eligible)))
            values["classes"] = {}
            for label in ("CONVERSATION", "VOICEMAIL"):
                tp = sum(r.expected_label == label and r.predicted_label == label for r in eligible)
                fp = sum(r.expected_label != label and r.predicted_label == label for r in eligible)
                fn = sum(r.expected_label == label and r.predicted_label != label for r in eligible)
                values["classes"][label] = dict(tp=tp, fp=fp, fn=fn, support=tp+fn,
                    precision=_ratio(tp, tp+fp), recall=_ratio(tp, tp+fn), f1=_ratio(2*tp, 2*tp+fp+fn))
            values["human_as_voicemail"] = sum(r.expected_label == "CONVERSATION" and r.predicted_label == "VOICEMAIL" for r in eligible)
            values["voicemail_as_human"] = sum(r.expected_label == "VOICEMAIL" and r.predicted_label == "CONVERSATION" for r in eligible)
            values["human_as_voicemail_rate"] = _ratio(values["human_as_voicemail"], sum(r.expected_label == "CONVERSATION" for r in eligible))
            values["voicemail_as_human_rate"] = _ratio(values["voicemail_as_human"], sum(r.expected_label == "VOICEMAIL" for r in eligible))
        output[stratum] = values
    return output


def _qa_tags(rows):
    per_tag = {}
    for tag in QA_TAGS:
        counts = {}
        for name in ("tp", "fp", "fn"):
            measurements = [r.score.metrics.get(f"annotation_scoped.{tag}.{name}") for r in rows]
            counts[name] = sum(m.value for m in measurements if m is not None and m.value is not None)
        tp, fp, fn = (counts[n] for n in ("tp", "fp", "fn"))
        opportunities = sum(f"annotation_scoped.{tag}.tp" in r.score.metrics for r in rows)
        per_tag[tag] = dict(**counts, annotated_evaluated_opportunities=opportunities,
            precision=_ratio(tp, tp+fp), recall=_ratio(tp, tp+fn), f1=_ratio(2*tp, 2*tp+fp+fn))
    macro = {}
    for metric in ("precision", "recall", "f1"):
        values = [m[metric]["value"] for m in per_tag.values() if m[metric]["value"] is not None]
        macro[metric] = dict(**_ratio(sum(values), len(values)), defined_tags=len(values), unavailable_tags=len(QA_TAGS)-len(values))
    return dict(scope="annotation_scoped", per_tag=per_tag, macro=macro)


def _evaluations(values):
    rows = tuple(checked(CaseEvaluation, r) for r in values)
    if len({r.evaluation_id for r in rows}) != len(rows):
        raise EvaluationError("duplicate_evaluation")
    if len({(r.dataset_hash, r.evaluator_version, r.evaluator_fingerprint) for r in rows}) > 1:
        raise EvaluationError("mixed_evaluation_basis")
    if len({r.purpose for r in rows if r.exercise_kind == "model"}) > 1:
        raise EvaluationError("mixed_evaluation_purpose")
    for row in rows:
        if row.score.case_id != row.case_id or row.score.exercise_kind != row.exercise_kind:
            raise EvaluationError("evaluation_score_binding_mismatch")
    return rows


def aggregate_evaluations(evaluations) -> dict:
    rows = _evaluations(evaluations)
    groups = defaultdict(list)
    models = [r for r in rows if r.exercise_kind == "model"]
    controls = [r for r in rows if r.exercise_kind != "model"]
    for row in models:
        # Never mix unlabelled historical evidence into synthetic gold denominators.
        family = "historical" if row.source.value == "ovp34_replay" else "synthetic"
        groups[row.purpose + ":" + family + ":" + row.contract_id].append(row)
    contracts = {}
    for name, selected in sorted(groups.items()):
        report = dict(selected=len(selected), source_counts=dict(sorted(Counter(r.source.value for r in selected).items())),
            execution_states=dict(sorted(Counter(r.execution_state for r in selected).items())),
            transport_states=dict(sorted(Counter(r.transport_status or "not_available" for r in selected).items())),
            content_states=dict(sorted(Counter(r.content_state for r in selected).items())),
            semantic_gold_available=sum(r.gold_available for r in selected),
            metrics=_metric_summary(selected), checks=_check_summary(selected), critical=_critical_summary(selected))
        if selected[0].task.value == "voicemail":
            report["voicemail"] = _voicemail_summary(selected)
        if selected[0].task.value == "qa" and any(r.gold_available for r in selected):
            report["annotation_scoped_tags"] = _qa_tags(selected)
        contracts[name] = report
    control_groups = {}
    for kind in ("adapter_control", "response_contract", "transport_contract"):
        selected = [r for r in controls if r.exercise_kind == kind]
        control_groups[kind] = dict(selected=len(selected),
            outcomes=dict(sorted(Counter(r.control_status for r in selected).items())),
            case_outcomes={r.case_id: r.control_status for r in sorted(selected, key=lambda r: r.case_id)})
    basis = ({"dataset_hash": rows[0].dataset_hash, "evaluator_version": rows[0].evaluator_version,
              "evaluator_fingerprint": rows[0].evaluator_fingerprint,
              "run_ids": sorted({r.execution_key.run_id for r in rows if r.execution_key})} if rows else None)
    return dict(evaluation_basis=basis, contracts=contracts, controls=control_groups,
                critical_models=_critical_summary(models), critical_controls=_critical_summary(controls))


def _directory(path, *, create=False):
    """Check every component: never follow symlinks; private artifact tree only."""
    path = Path(path)
    for component in (path, *path.parents):
        if component.is_symlink():
            raise EvaluationError("artifact_symlink")
    if create and not path.exists():
        path.mkdir(mode=0o700)
        _sync_directory(path.parent)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_mode & 0o500 != 0o500:
        raise EvaluationError("artifact_directory_permissions")
    return path


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_private(path):
    _directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or not info.st_mode & 0o400:
            raise EvaluationError("artifact_file_permissions")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def _publish(path, content):
    _directory(path.parent)
    if path.exists() or path.is_symlink():
        if _read_private(path) != content:
            raise EvaluationError("artifact_content_conflict")
        return
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix=".evaluation-", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink()
            _sync_directory(path.parent)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError("duplicate_artifact_key")
        result[key] = value
    return result


def _decisions(evaluations, decisions):
    rows = _evaluations(evaluations)
    by_id = {r.evaluation_id: r for r in rows}
    groups = defaultdict(list)
    decisions = tuple(checked(ReviewDecision, d) for d in decisions)
    for decision in decisions:
        if decision.evaluation_id not in by_id:
            raise EvaluationError("unknown_review_evaluation")
        groups[decision.evaluation_id].append(decision)
    updated = tuple(apply_review_decisions(r, groups[r.evaluation_id]) for r in rows)
    ordered = tuple(sorted(decisions, key=lambda d: (d.evaluation_id, d.check_id)))
    return ordered, updated


def load_review_decisions(path, *, evaluations) -> tuple[ReviewDecision, ...]:
    try:
        content = _read_private(Path(path))
        if content and not content.endswith(b"\n"):
            raise EvaluationError("incomplete_review_artifact")
        decisions = []
        for line in content.decode("utf-8").splitlines():
            data = json.loads(line, object_pairs_hook=_unique)
            canonical_json_bytes(data)  # Reject non-finite constants/overflow before validation.
            decisions.append(ReviewDecision.model_validate_json(canonical_json_bytes(data)))
        ordered, _ = _decisions(evaluations, decisions)
        return ordered
    except (OSError, ValueError):
        failure = EvaluationError("invalid_review_artifact")
    raise failure


def write_evaluation_artifacts(results_root, *, evaluations, review_decisions=None) -> Path:
    """Create-only, private POSIX snapshots. Single publisher; no journal repair."""
    rows = _evaluations(evaluations)
    if not rows:
        raise EvaluationError("empty_evaluation_batch")
    is_control = {r.exercise_kind != "model" for r in rows}
    if len(is_control) != 1 or len({r.execution_key.run_id for r in rows if r.execution_key}) > 1:
        raise EvaluationError("mixed_artifact_batch")
    # Automatic records must be original checks; reviews are separate complete revisions.
    if any(c.method == "human" and c.status != "pending" for r in rows for c in r.checks):
        raise EvaluationError("automatic_artifact_contains_review")
    decisions, updated = _decisions(rows, () if review_decisions is None else review_decisions)
    automatic = b"".join(canonical_json_bytes(r.model_dump(mode="json")) + b"\n" for r in rows)
    automatic_summary = canonical_json_bytes(aggregate_evaluations(rows))
    batch_id = evaluation_fingerprint("batch", [r.evaluation_id for r in rows])
    review_bytes = b"".join(canonical_json_bytes(d.model_dump(mode="json")) + b"\n" for d in decisions)
    review_hash = evaluation_fingerprint("review", [d.model_dump(mode="json") for d in decisions])
    try:
        root = _directory(Path(results_root), create=True)
        if True in is_control:
            parent = _directory(root / "control-evaluations", create=True)
        else:
            run_id = rows[0].execution_key.run_id
            run = _directory(root / run_id)
            manifest = load_manifest(root, run_id)
            if manifest.dataset_hash != rows[0].dataset_hash:
                raise EvaluationError("artifact_manifest_mismatch")
            parent = _directory(run / "evaluations", create=True)
        batch = _directory(parent / batch_id, create=True)
        _publish(batch / "automatic.jsonl", automatic)
        _publish(batch / "summary-automatic.json", automatic_summary)
        if review_decisions is not None:
            reviews = _directory(batch / "reviews", create=True)
            _publish(reviews / (review_hash + ".jsonl"), review_bytes)
            _publish(batch / ("summary-" + review_hash + ".json"), canonical_json_bytes(aggregate_evaluations(updated)))
        return batch
    except (OSError, ValueError):
        failure = EvaluationError("evaluation_artifact_publication_failed")
    raise failure
