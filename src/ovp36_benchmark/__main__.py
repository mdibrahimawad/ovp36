"""Explicit local candidate commands; no serving or model policy."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from .adapters import prepare_curated_plan
from .client import ModelClient
from .config import load_config, resolve_endpoint
from .contracts import hash_contract, load_contract
from .dataset import hash_dataset, load_curated_cases
from .evaluation import evaluate_control_case, evaluation_fingerprint
from .identity import fingerprint_execution_plan
from .persistence import build_manifest
from .reporting import aggregate_evaluations, evaluate_run, write_evaluation_artifacts
from .runner import run_plan
from .qa_replay_comparison import load_qa_comparison_inputs, write_qa_comparisons
from .replay import prepare_replay_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DATASET_HASH = "519b02e5b0ceb1b32882377d7dd58178dead8a22e6305821e3ba2633d008b9ba"
SMOKE_CASE_IDS = (
    "curated-ex-001", "curated-cs-002", "curated-vm-001", "curated-qa-001",
    "curated-qs-001", "adversarial-ns-006",
)


class SmokeError(ValueError):
    """Internal fixed codes only, with no configuration or provider payload."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's original message can contain arbitrary argument values.
        self.print_usage(sys.stderr)
        self.exit(2, "candidate-smoke: invalid_arguments\n")


def _arguments(argv):
    parser = _Parser(prog="python -m ovp36_benchmark",
                     description="Run a local candidate smoke, full curated suite, or QA replay comparison.")
    parser.add_argument("command", choices=("candidate-smoke", "candidate-suite", "qa-replay-compare"))
    parser.add_argument("--config", required=True, help="Private local TOML configuration")
    parser.add_argument("--results-root", default="results", help="Private artifacts directory (default: results)")
    parser.add_argument("--source", help="Private canonical final-100 raw observations JSONL")
    parser.add_argument("--selection", help="Canonical OOB selection CSV")
    args = parser.parse_args(argv)
    if args.command == "qa-replay-compare" and (not args.source or not args.selection):
        parser.error("missing_replay_artifacts")
    if args.command != "qa-replay-compare" and (args.source or args.selection):
        parser.error("unexpected_replay_artifacts")
    return args


def _validate_loopback(base_url):
    valid = False
    try:
        parts = urlsplit(base_url)
        port = parts.port
        # Literal comparison is numeric IPv4 validation without DNS or aliases.
        valid = (parts.scheme == "http" and parts.hostname == "127.0.0.1"
                 and port is not None and 1 <= port <= 65535
                 and parts.netloc == f"127.0.0.1:{port}"
                 and parts.path in ("/v1", "/v1/") and "?" not in base_url and "#" not in base_url)
    except ValueError:
        pass
    if not valid:
        raise SmokeError("numeric_loopback_v1_url_required")


def _implementation_fingerprints():
    # Reuse the documented public implementation fingerprint policy for both
    # harness and evaluator identity. Paths in this mapping are repo-relative.
    sources = sorted((REPOSITORY_ROOT / "src/ovp36_benchmark").glob("*.py"))
    sources += [REPOSITORY_ROOT / "src/ovp36_benchmark/prompts/contracts.json", REPOSITORY_ROOT / "uv.lock"]
    components = {p.relative_to(REPOSITORY_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sources}
    return evaluation_fingerprint("implementation", components), components["uv.lock"]


def _local_config(config_path, results_root):
    """Share the existing local smoke configuration and output boundary."""
    failure = None
    try:
        config = load_config(config_path)
        endpoint = resolve_endpoint(config.endpoint)
    except (OSError, ValueError):
        failure = SmokeError("invalid_configuration")
    if failure is not None:
        raise failure
    _validate_loopback(endpoint.base_url)
    if config.repetitions != 1 or config.concurrency != 1:
        raise SmokeError("one_repetition_and_concurrency_required")
    if config.evaluation != "exploratory":
        raise SmokeError("exploratory_designation_required")
    if not (config.experiment_label == "candidate-smoke" or config.experiment_label.startswith("candidate-smoke-")):
        raise SmokeError("candidate_smoke_experiment_label_required")

    root = Path(os.path.abspath(results_root))
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise SmokeError("results_root_symlink")
    # Inside the checkout, artifacts belong in the existing ignored results tree.
    # External private directories also support operator storage and temp tests.
    if root.is_relative_to(REPOSITORY_ROOT) and not root.is_relative_to(REPOSITORY_ROOT / "results"):
        raise SmokeError("results_root_must_be_ignored_or_external")
    return config, endpoint, root


async def candidate_smoke(config_path, results_root):
    return await _curated_candidate(config_path, results_root, full_suite=False)


async def candidate_suite(config_path, results_root):
    return await _curated_candidate(config_path, results_root, full_suite=True)


async def _curated_candidate(config_path, results_root, *, full_suite):
    """Compose existing APIs; all preflight checks precede client construction."""
    config, endpoint, root = _local_config(config_path, results_root)
    cases = load_curated_cases(REPOSITORY_ROOT / "data/curated")
    digest = hash_dataset(cases)
    if digest != FROZEN_DATASET_HASH:
        raise SmokeError("frozen_dataset_mismatch")
    if full_suite:
        selected = tuple(c for c in cases if c.exercise.kind == "model")
        if len(cases) != 126 or len(selected) != 106:
            raise SmokeError("full_suite_inventory_required")
        expected_ids = tuple(c.id for c in selected)
        purpose = "candidate"
    else:
        selected = tuple(c for c in cases if c.id in SMOKE_CASE_IDS)
        if tuple(c.id for c in selected) != SMOKE_CASE_IDS or any(c.exercise.kind != "model" for c in selected):
            raise SmokeError("six_model_cases_required")
        expected_ids = SMOKE_CASE_IDS
        purpose = "candidate_smoke"
    # Full-suite preparation receives all 126 cases; the adapter API filters controls.
    plan = prepare_curated_plan(cases if full_suite else selected, config=config, resolved_model=endpoint.model)
    if tuple(p.request.case_id for p in plan) != expected_ids or any(p.repetition_index != 0 for p in plan):
        raise SmokeError("full_suite_execution_plan_required" if full_suite else "six_execution_plan_required")
    implementation, lock = _implementation_fingerprints()
    controls = tuple(evaluate_control_case(c, dataset_hash=digest, evaluator_fingerprint=implementation)
                     for c in cases if c.exercise.kind != "model") if full_suite else ()
    manifest = build_manifest(config, resolved_model=endpoint.model, dataset_hash=digest,
        ordered_execution_plan_hash=fingerprint_execution_plan([p.identity_projection() for p in plan]),
        contract_hashes={c.contract_id: hash_contract(load_contract(c.contract_id)) for c in selected},
        harness_code_fingerprint=implementation, dependency_lock_fingerprint=lock)
    async with ModelClient(endpoint, timeout_seconds=config.timeout_seconds) as client:
        summary = await run_plan(plan, client=client, results_root=root, expected_manifest=manifest)
    evaluations = evaluate_run(cases, plan=plan, results_root=root, expected_manifest=manifest,
                               evaluator_fingerprint=implementation, purpose=purpose)
    report = aggregate_evaluations((*evaluations, *controls))
    write_evaluation_artifacts(root, evaluations=evaluations)
    if controls:
        write_evaluation_artifacts(root, evaluations=controls)
    result = dict(run_id=manifest.run_id, purpose=purpose, selected_case_count=len(selected),
                execution=summary.model_dump(), evaluation_count=len(evaluations),
                contracts={name: dict(selected=group["selected"], transport_states=group["transport_states"])
                           for name, group in report["contracts"].items()},
                pending_human_review_checks=sum(c.method == "human" and c.status == "pending"
                                               for row in evaluations for c in row.checks))
    if full_suite:
        result.update(control_evaluation_count=len(controls), controls=report["controls"],
                      pending_control_human_review_checks=sum(c.method == "human" and c.status == "pending"
                                                             for row in controls for c in row.checks))
    return result


async def qa_replay_compare(config_path, results_root, source_path, selection_path):
    config, endpoint, root = _local_config(config_path, results_root)
    dataset, selected, baselines = load_qa_comparison_inputs(source_path, selection_path)
    plan = prepare_replay_plan(selected, config=config, resolved_model=endpoint.model)
    implementation, lock = _implementation_fingerprints()
    manifest = build_manifest(config, resolved_model=endpoint.model, dataset_hash=dataset.dataset_hash,
        ordered_execution_plan_hash=fingerprint_execution_plan([p.identity_projection() for p in plan]),
        contract_hashes={"qa": hash_contract(load_contract("qa"))},
        harness_code_fingerprint=implementation, dependency_lock_fingerprint=lock)
    async with ModelClient(endpoint, timeout_seconds=config.timeout_seconds) as client:
        summary = await run_plan(plan, client=client, results_root=root, expected_manifest=manifest)
    evaluations = evaluate_run(dataset, plan=plan, results_root=root, expected_manifest=manifest,
                               evaluator_fingerprint=implementation, purpose="candidate_smoke")
    write_evaluation_artifacts(root, evaluations=evaluations)
    write_qa_comparisons(root, manifest=manifest, selected=selected, baselines=baselines)
    return dict(run_id=manifest.run_id, purpose="candidate_smoke", comparison="historical_baseline",
                selected_case_count=len(selected.cases), execution=summary.model_dump(),
                evaluation_count=len(evaluations))


def main(argv=None):
    args = _arguments(argv)
    try:
        if args.command == "qa-replay-compare":
            operation = qa_replay_compare(args.config, args.results_root, args.source, args.selection)
        elif args.command == "candidate-suite":
            operation = candidate_suite(args.config, args.results_root)
        else:
            operation = candidate_smoke(args.config, args.results_root)
        result = asyncio.run(operation)
    except SmokeError as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"{args.command}: interrupted", file=sys.stderr)
        return 130
    except Exception:
        # Never expose exception text/tracebacks containing paths or payloads.
        print(f"{args.command}: orchestration_failed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
