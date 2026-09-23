"""Assemble private evidence and validate AI-review coverage; never grade outputs.

Judgments must be supplied by the reviewing assistant after reading the complete
case. This module does not turn automatic checks into semantic verdicts and does
not write human review decisions.
"""

import hashlib
import json
from pathlib import Path

from compare_candidates import DEFAULT_BATCH, ROOT, read, save
from ovp36_benchmark.persistence import AttemptRecorded, ExecutionFinalized, read_events


def outputs(batch=DEFAULT_BATCH):
    progress = read(batch / "progress.json")
    runs = {(row["phases"][phase]["run_id"], phase): model
            for model, row in progress.items() for phase in ("suite", "replay") if phase in row["phases"]}
    runs[(read(batch / "freeze.json")["baseline_run_id"], "suite")] = "m11"
    records = {}
    for (run_id, phase), model in runs.items():
        events = read_events(ROOT / "results", run_id)
        finalized = {e.execution_key for e in events if isinstance(e, ExecutionFinalized)}
        for event in events:
            if isinstance(event, AttemptRecorded) and event.execution_key in finalized:
                result = event.result.model_dump(mode="json")
                raw = (result.get("response") or {}).get("raw_content")
                key = (run_id, event.execution_key.case_id)
                records[key] = dict(model_key=model, phase=phase, run_id=run_id,
                    case_id=event.execution_key.case_id, result=result,
                    output_sha256=hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None)
    return records


def case_inputs(batch=DEFAULT_BATCH):
    cases = {c["id"]: c for path in (ROOT / "data/curated").glob("*.jsonl")
             for line in path.read_text().splitlines() if (c := json.loads(line))}
    requests = {r["case_id"]: r for r in read(batch / "requests.json")}
    return cases, requests


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def record_review(record, *, verdict, severity, input_quote, output_quote, explanation,
                  ambiguity=None, batch=DEFAULT_BATCH):
    cases, requests = case_inputs(batch)
    if record["output_sha256"] is None:
        raise ValueError("unavailable output cannot receive a semantic judgment")
    if verdict not in ("acceptable", "reject", "uncertain") or severity not in ("none", "minor", "major", "critical"):
        raise ValueError("explicit review verdict/severity required")
    raw = record["result"]["response"]["raw_content"]
    if (not isinstance(input_quote, str) or not isinstance(output_quote, str) or not explanation
            or (output_quote == "" and raw != "")):
        raise ValueError("exact evidence and explanation required")
    input_texts = strings(requests[record["case_id"]]) + strings(cases.get(record["case_id"], {}))
    input_matches = any(input_quote in text if input_quote else text == "" for text in input_texts)
    if not input_matches or output_quote not in raw:
        raise ValueError("review evidence is not an exact input/output quotation")
    review = dict(review_type="AI", reviewer="Codex assistant", run_id=record["run_id"],
        case_id=record["case_id"], phase=record["phase"], output_sha256=record["output_sha256"],
        verdict=verdict, severity=severity, input_evidence=input_quote, output_evidence=output_quote,
        explanation=explanation, ambiguity_requiring_human_review=ambiguity,
        human_approval=False, complete_input_reference="requests.json#" + record["case_id"],
        raw_output_reference=f"results/{record['run_id']}/journal.jsonl")
    save(batch / "ai-reviews" / record["run_id"] / (record["case_id"] + ".json"), review)


def coverage(batch=DEFAULT_BATCH):
    inventory = outputs(batch)
    reviews = {}
    for path in (batch / "ai-reviews").glob("*/*.json"):
        review = read(path)
        key = (review["run_id"], review["case_id"])
        if key in reviews or key not in inventory:
            raise ValueError("duplicate or unknown review identity")
        record = inventory[key]
        if (review["review_type"] != "AI" or review["human_approval"] is not False
                or review["output_sha256"] is None or review["output_sha256"] != record["output_sha256"]):
            raise ValueError("invalid review type or stale output hash")
        reviews[key] = review
    available = {k for k, r in inventory.items() if r["output_sha256"] is not None}
    return dict(required_executions=len(inventory), available_outputs=len(available),
        unavailable_outputs=len(inventory)-len(available), reviewed_outputs=len(reviews),
        unfinished=[dict(run_id=k[0], case_id=k[1]) for k in sorted(available-reviews.keys())],
        human_reviews_approved=0)


if __name__ == "__main__":
    print(json.dumps(coverage(), indent=2))
