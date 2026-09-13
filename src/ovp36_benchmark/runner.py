"""Sequential prepared-request execution. V1 records one attempt, with no retries.

This policy is a benchmark normalization, not a product retry-fidelity claim.
Caller owns the injected client; this function owns the journal it opens.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, InstanceOf, model_validator

from .client import ModelClient
from .identity import (
    ExecutionKey, canonical_json_bytes, fingerprint_execution_plan, fingerprint_request,
)
from .persistence import AttemptRecorded, ResultJournal, RunManifest
from .requests import PreparedRequest
from .schemas import StrictModel


MAX_ATTEMPTS = 1
Count = Annotated[int, Field(ge=0)]


class RunnerValidationError(ValueError):
    """Safe plan/state error; no private inputs or retained validation exceptions."""


class PlannedExecution(StrictModel):
    # Require a prepared instance; never reconstruct requests from payload dictionaries.
    request: InstanceOf[PreparedRequest] = Field(repr=False, exclude=True)
    repetition_index: Count

    def identity_projection(self) -> dict[str, object]:
        return {
            "case_id": self.request.case_id,
            "request_fingerprint": fingerprint_request(self.request),
            "repetition_index": self.repetition_index,
        }


class RunSummary(StrictModel):
    planned: Count
    skipped_completed: Count
    resumed_unfinished: Count
    attempts_sent: Count
    executions_finalized: Count
    successes: Count
    failures: Count

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if (self.planned != self.skipped_completed + self.executions_finalized
                or self.executions_finalized != self.successes + self.failures):
            raise ValueError("inconsistent_run_counts")
        return self


def _preflight(
    plan: Sequence[PlannedExecution], expected_manifest: RunManifest,
) -> tuple[tuple[PlannedExecution, ...], tuple[ExecutionKey, ...], RunManifest]:
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        raise RunnerValidationError("invalid_plan_sequence")
    snapshot = tuple(plan)
    if not isinstance(expected_manifest, RunManifest):
        raise RunnerValidationError("invalid_expected_manifest")
    failure = None
    try:
        # Detach and revalidate safe metadata before any journal/filesystem operation.
        manifest = RunManifest.model_validate_json(canonical_json_bytes(
            expected_manifest.model_dump(mode="json", warnings="error"),
        ))
    except ValueError:
        failure = RunnerValidationError("invalid_expected_manifest")
    if failure is not None:
        raise failure
    if manifest.safe_config.concurrency != 1:
        raise RunnerValidationError("sequential_concurrency_required")

    validated, keys, projections = [], [], []
    seen = set()
    for item in snapshot:
        if not isinstance(item, PlannedExecution):
            raise RunnerValidationError("invalid_planned_execution")
        try:
            # Recheck fields even for instances made through unchecked model_copy().
            item = PlannedExecution(request=item.request, repetition_index=item.repetition_index)
            projection = item.identity_projection()
            key = ExecutionKey(run_id=manifest.run_id, **projection)
        except ValueError:
            failure = RunnerValidationError("invalid_planned_execution")
        if failure is not None:
            raise failure
        if item.repetition_index >= manifest.safe_config.repetitions:
            raise RunnerValidationError("repetition_out_of_range")
        if item.request.model != manifest.safe_config.requested_model:
            raise RunnerValidationError("request_model_mismatch")
        if key in seen:
            raise RunnerValidationError("duplicate_execution")
        seen.add(key)
        validated.append(item)
        keys.append(key)
        projections.append(projection)
    if fingerprint_execution_plan(projections) != manifest.ordered_execution_plan_hash:
        raise RunnerValidationError("execution_plan_hash_mismatch")
    return tuple(validated), tuple(keys), manifest


async def run_plan(
    plan: Sequence[PlannedExecution], *, client: ModelClient,
    results_root: str | Path, expected_manifest: RunManifest,
) -> RunSummary:
    """Execute the full original plan, skipping completed keys without reordering.

    An injected client's endpoint, timeout and physical server are caller-owned
    composition responsibilities, not independently attested here. No artificial
    cancellation checkpoints: cancellation propagates from the awaited client.
    A crash before durable attempt evidence can still cause a resend on restart.
    """
    plan, keys, manifest = _preflight(plan, expected_manifest)
    key_set = set(keys)
    skipped = resumed = sent = finalized = successes = failures = 0
    with ResultJournal.open(results_root, manifest) as journal:
        completed = journal.completed_keys()
        attempts: dict[ExecutionKey, list[AttemptRecorded]] = {}
        if not completed <= key_set:
            raise RunnerValidationError("journal_execution_outside_plan")
        for event in journal.records():
            if event.execution_key not in key_set:
                raise RunnerValidationError("journal_execution_outside_plan")
            if isinstance(event, AttemptRecorded):
                attempts.setdefault(event.execution_key, []).append(event)
        if any(len(records) > MAX_ATTEMPTS for key, records in attempts.items() if key not in completed):
            raise RunnerValidationError("incompatible_unfinished_attempts")

        for item, key in zip(plan, keys, strict=True):
            if key in completed:
                skipped += 1
                continue
            if key in attempts:
                resumed += 1
                previous = attempts[key][0]
                attempt_index, terminal = previous.attempt_index, previous.result
            else:
                attempt_index = 0
                sent += 1
                terminal = await client.send_once(item.request)
                journal.append_attempt(key, attempt_index, terminal)
            # retryable and all candidate text remain evidence, never retry criteria.
            journal.finalize(key, attempt_index)
            finalized += 1
            if terminal.status == "success":
                successes += 1
            else:
                failures += 1
    return RunSummary(
        planned=len(plan), skipped_completed=skipped, resumed_unfinished=resumed,
        attempts_sent=sent, executions_finalized=finalized, successes=successes, failures=failures,
    )
