"""Private, diagnostic QA comparison with a historical baseline, never gold.

The Stage 4A loader owns replay validation. The additional raw-file pass only
joins baseline outputs after validating the complete dataset and its provenance.
"""

import hashlib
import json
import math
from pathlib import Path

from .identity import canonical_json_bytes
from .parsing import parse_qa
from .persistence import AttemptRecorded, ExecutionFinalized, load_manifest, read_events
from .replay import ReplayContract, ReplayDataset, ReplayError, load_ovp34_replay
from .reporting import _directory, _publish


QA_OBSERVATION_IDS = (
    "c276db54ca990a74", "31b24d3491a50db8", "34da3fd07076ab1e", "4162e59759efd5c8",
    "de5409211c79ea8c", "fd352d76033d559b", "c032b38b0cac1cca", "8afbd319f5e0b261",
    "81b63df3a246dd12", "9fbf1f50391d2a43",
)
# Exact file-byte hashes frozen in SOURCE_AUDIT.md section 1.2. No CLI bypass.
CANONICAL_SOURCE_HASH = "dbea5016e0445cad2952c2cba861782d75c1944a1c83ae826ca6c936e31c35e6"
CANONICAL_SELECTION_HASH = "419f21cce3fa5c49645b0d237114aad521251f1ecda8207b9e220d92696b4ee1"


def select_qa_cases(dataset, observation_ids=QA_OBSERVATION_IDS):
    """Subset an already fully validated dataset, retaining canonical CSV order."""
    wanted = set(observation_ids)
    by_id = {case.source_observation_id: case for case in dataset.cases}
    if not wanted or len(wanted) != len(observation_ids) or not wanted <= by_id.keys():
        raise ReplayError("invalid_qa_selection")
    if any(by_id[key].contract != ReplayContract.QA_EVALUATION for key in wanted):
        raise ReplayError("non_qa_selection")
    return ReplayDataset(cases=tuple(case for case in dataset.cases if case.source_observation_id in wanted),
                         source_artifact_hash=dataset.source_artifact_hash,
                         selection_artifact_hash=dataset.selection_artifact_hash)


def load_qa_comparison_inputs(source_path, selection_path):
    """Full loader validation and canonical hashes precede any selection or join."""
    dataset = load_ovp34_replay(source_path, selection_path=selection_path)
    if (dataset.source_artifact_hash != CANONICAL_SOURCE_HASH
            or dataset.selection_artifact_hash != CANONICAL_SELECTION_HASH):
        raise ReplayError("canonical_artifact_mismatch")
    selected = select_qa_cases(dataset)
    if tuple(case.source_observation_id for case in selected.cases) != QA_OBSERVATION_IDS:
        raise ReplayError("canonical_qa_order_mismatch")
    content = Path(source_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != dataset.source_artifact_hash:
        raise ReplayError("baseline_source_changed")
    # This exact byte snapshot has already passed the full strict replay loader.
    # Do not reconstruct requests or interpret historical generation settings.
    baselines = {}
    for line in content.splitlines():
        raw = json.loads(line)["raw"]
        if raw["id"] not in QA_OBSERVATION_IDS:
            continue
        output = raw.get("output")
        if (raw.get("model") != "qwen3.5" or type(output) is not dict
                or type(output.get("content")) is not str or not output["content"].strip()):
            raise ReplayError("invalid_historical_qa_output")
        baselines[raw["id"]] = dict(name=raw["name"], model=raw["model"], raw_output=output["content"])
    if set(baselines) != set(QA_OBSERVATION_IDS):
        raise ReplayError("missing_historical_qa_output")
    return dataset, selected, baselines


def _side(raw_output):
    parsed = parse_qa(raw_output)
    value = parsed.normalized_value
    parsed_value = parsed.parsed_value if parsed.production_parse_success else None
    encoding_status = "available" if parsed.production_parse_success else "not_parsed"
    try:
        canonical_json_bytes(parsed_value)
    except ValueError:
        # Production accepts NaN/Infinity; strict artifact JSON cannot encode them.
        # Preserve the exact raw output and mark this projection unavailable.
        parsed_value, encoding_status = None, "nonfinite_parsed_result_omitted"
    return dict(raw_output=raw_output, parsed_result=parsed_value,
                tags=value.get("tags"), sentiment=value.get("overall_sentiment"),
                call_quality_score=value.get("call_quality_score"), summary=value.get("summary"),
                parse_status=dict(parser_path=parsed.parser_path, strict_json_valid=parsed.strict_json_valid,
                    strict_format_valid=parsed.strict_format_valid,
                    production_parse_success=parsed.production_parse_success,
                    production_usable=parsed.production_usable, parsed_result_encoding=encoding_status,
                    diagnostics=[d.code for d in parsed.diagnostics]))


def _tag_names(tags):
    if type(tags) is not list or any(type(t) is not dict or type(t.get("tag")) is not str for t in tags):
        return None
    return set(t["tag"] for t in tags)


def compare_qa(case, baseline, *, candidate_model, attempt):
    """Diagnostic field differences only; missing/invalid fields remain unavailable."""
    qwen = _side(baseline["raw_output"])
    response = attempt.result.response
    candidate = _side(response.raw_content if response is not None else None)
    qwen.update(role="historical_baseline", model=baseline["model"])
    candidate.update(role="candidate", model=candidate_model)
    transport = attempt.result.model_dump(mode="json", exclude={"response"})
    transport["response_metadata"] = response.model_dump(mode="json", exclude={"raw_content"}) if response else None
    qtags, ctags = _tag_names(qwen["tags"]), _tag_names(candidate["tags"])
    successful = attempt.result.status == "success"
    tag_comparable = successful and qtags is not None and ctags is not None
    sentiments = (qwen["sentiment"], candidate["sentiment"])
    sentiment_comparable = successful and all(type(s) is str and s in ("positive", "neutral", "negative")
                                             for s in sentiments)
    scores = (qwen["call_quality_score"], candidate["call_quality_score"])
    score_comparable = successful and all(type(s) in (int, float) and 1 <= s <= 10 and math.isfinite(s)
                                         for s in scores)
    record = dict(observation_id=case.source_observation_id, qa_observation_name=baseline["name"],
                  candidate_model=candidate_model, captured_request_messages=case.messages.to_messages(),
                  execution_key=attempt.execution_key.model_dump(mode="json"),
                  qwen=qwen, candidate=candidate, transport=transport,
                  differences=dict(tags=dict(overlap=sorted(qtags & ctags) if tag_comparable else None,
                      qwen_only=sorted(qtags - ctags) if tag_comparable else None,
                      candidate_only=sorted(ctags - qtags) if tag_comparable else None),
                      sentiment_agreement=sentiments[0] == sentiments[1] if sentiment_comparable else None,
                      absolute_score_difference=abs(scores[0] - scores[1]) if score_comparable else None))
    # Invalid field projections can themselves contain non-finite values. Keep
    # raw evidence plus an explicit omission list, never write NaN as JSON.
    for side in (qwen, candidate):
        side["omitted_nonfinite_fields"] = []
        for field in ("tags", "sentiment", "call_quality_score", "summary"):
            try:
                canonical_json_bytes(side[field])
            except ValueError:
                side[field] = None
                side["omitted_nonfinite_fields"].append(field)
    return record


def write_qa_comparisons(results_root, *, manifest, selected, baselines):
    """Publish ten private create-only files using the existing POSIX publisher.

    Caller first runs evaluate_run to validate the full dataset/plan/journal join.
    Historical payloads are copied only to this explicitly private report, never
    into the manifest, journal, or ordinary automatic evaluations.
    """
    if load_manifest(results_root, manifest.run_id) != manifest:
        raise ReplayError("comparison_manifest_mismatch")
    events = read_events(results_root, manifest.run_id)
    attempts = {(e.execution_key, e.attempt_index): e for e in events if isinstance(e, AttemptRecorded)}
    final = {e.execution_key.case_id: attempts[e.execution_key, e.attempt_index]
             for e in events if isinstance(e, ExecutionFinalized)}
    if set(final) != {case.case_id for case in selected.cases}:
        raise ReplayError("comparison_finalized_cases_mismatch")
    parent = _directory(Path(results_root) / manifest.run_id / "qa-comparison", create=True)
    for case in selected.cases:
        record = compare_qa(case, baselines[case.source_observation_id],
                            candidate_model=manifest.safe_config.requested_model, attempt=final[case.case_id])
        record.update(schema_version="ovp36-private-qa-comparison-v1",
                      source_artifact_hash=selected.source_artifact_hash,
                      selection_artifact_hash=selected.selection_artifact_hash)
        _publish(parent / (case.source_observation_id + ".json"), canonical_json_bytes(record) + b"\n")
    return parent
