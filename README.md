# OVP-36 OOB model benchmark

Standalone research benchmark. **Stages 1–4B2:** strict schemas, TOML configuration,
read-only JSONL dataset operations, versioned prompt contracts, and response
parsing, immutable request preparation, pure identity primitives, and a common
one-shot AsyncOpenAI client, immutable run manifests, private result journals,
and a sequential prepared-request runner with frozen OVP-34 replay loading and
plan preparation. Stage 4B1 adds source-aligned curated adapters, inventory
validation, model-only plan preparation, and review projections. Stage 4B2 adds
the fixed 126-case synthetic curated/adversarial inventory and its review projection.
The literal fixtures and draft semantic annotations retain their external human-review
requirements; exploratory candidate runs do not constitute approval. Stage 4C1 adds deterministic
evaluation, separate controls, post-run reporting, and private human-review
revisions. Stage 4C2 adds a deterministic localhost HTTP server and real
AsyncOpenAI E2E smoke. Stage 5B adds the model-agnostic `candidate-smoke` operator
command and distinct `candidate_smoke` evaluation purpose. Local Granite candidate
evaluation has now run, including the 106 model-case curated suite and a separate
ten-case historical QA comparison. Private results remain ignored; these local
research runs do not establish production readiness or GB10 performance.

OVP-36 has four product-level OOB areas: extraction, runtime context summary,
voicemail, and post-call QA. The benchmark has six source-derived task contracts:
`extraction`, `context_summary`, `voicemail`, `qa`, `qa_conversation_summary`, and
`qa_node_summary`. The last two are supporting QA operations. These task values
are benchmark taxonomy, not current routing labels.

The requirements are `docs/BENCHMARK_SPEC.md`, `docs/SOURCE_AUDIT.md`, and
`docs/CASE_MATRIX.md`, together with the approved source corrections from plan
review. The Stage 2 evidence correction updates those documents with the final
OVP-34 telemetry findings and exact source contracts.

## Development

Python **>=3.12,<3.13** is required. With uv installed:

```sh
uv sync --locked
.venv/bin/python -m unittest discover -s tests -v
```

Without uv, install this package using Python 3.12 in an isolated environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

The uv lockfile pins runtime dependencies; the pip alternative honors the exact
OpenAI/HTTPX pins and resolves other ranges from `pyproject.toml`. Tests use
synthetic cases in temporary directories and need no model, credentials, or
external services. Stage 4C2 tests require permission to bind an ephemeral
`127.0.0.1` socket; their guard rejects external connections and DNS.

## Data contracts

`schemas.py` defines all six task IDs and their typed input/gold models. It also
defines explicit `model`, `adapter_control`, `response_contract`, and
`transport_contract` exercises. Contract assertions cannot substitute for gold
in model exercises. Result foundation schemas require an exercise identity;
aggregation and case-to-request execution wiring are deferred.

- Models reject unknown fields and incorrect primitive types. JSON arrays become
  tuples; JSON enum strings become enum values. Direct Python construction uses
  typed enums and tuples under Pydantic strict validation.
- Models are frozen. Opaque JSON dictionaries, such as QA metrics, are not deeply
  frozen and must be treated as read-only. Dataset operations never mutate them.
- Only `ovp34_replay` cases may set `expected` to null. This rule also requires
  explicit expected assertions for curated/adversarial control cases.
- Extraction types are string, number, and boolean. Known gold must match the
  declared type; every missing field needs an explicit missing-value policy.
- Voicemail input stores ordered `messages`, including assistant/tool contexts,
  after the frozen system instruction. The curated adapter prepends that instruction.
- The three summary gold types share one fact-annotation schema. Named categories
  reference required-fact IDs; correction pairs reference old/new facts.
- QA gold uses a score target or an ordered acceptable range. Tags remain strings
  so a later frozen prompt contract can validate its vocabulary.
- Evidence offsets are half-open character spans. Structural validation checks
  pointers/offsets; curated validation resolves them relative to case.input and checks span bounds.

`load_cases(path_or_paths)` reads UTF-8 JSONL into a tuple of cases. It rejects
invalid/blank lines, duplicate object keys, non-standard numeric constants,
invalid schemas, and duplicate IDs, including across files. Errors identify the
file ordinal and line without paths, rejected values, or retained lower-level
exception cause/context. Expected filesystem failures also become DatasetError.

`select_cases(...)` intersects task/source/difficulty/critical/exercise filters;
tag filtering requires all supplied tags. No filter means unrestricted; an empty
task/source/difficulty/exercise selection matches no cases.

`hash_dataset(cases)` hashes all normalized case content in sorted ID order.
Whitespace, JSON object key order, and file ordering do not affect it. Message
order, gold, exercise kind, and provenance do. It is a dataset content hash, not
a future execution-order or prompt-artifact hash.

`validate_matrix_coverage(cases, expected_ids)` checks caller-supplied IDs and
raises on missing/unexpected IDs. `require_complete=False` returns gaps for
partial-suite development. No curated or replay data is created in Stage 1.

The final OVP-34 replay target has 242 captured requests and outputs: 40
extraction, 56 runtime summaries, 22 voicemail, 72 QA, and 52 QA
previous-conversation summaries. Stage 4A reads a private local export and
preserves captured historical trace messages, with observation references to
outputs and other provenance retained in that external source. These messages
are the best available captured trace/model-facing evidence,
not independently verified historical HTTP bytes. For the 56 runtime summaries,
current tracing reconstructs the summary request after generation; historical
wire equality is not independently proven. Keep those observations and their
qualification; never regenerate them. Historical outputs are baseline behavior,
not automatic gold. Curated/adversarial requests use source-aligned adapters;
historical replay bypasses adapters. Private replay data is never included.

## Curated preparation and authored inventory (Stages 4B1–4B2)

Stage 4B1 adapter and inventory infrastructure is complete. Stage 4B2 now supplies
all six `data/curated/*.jsonl` files: 126 independently synthetic scenarios,
including 106 model exercises and 20 controls. These are draft annotations awaiting
external semantic review, not candidate-derived truth or completed human review.
All VM-017–021 annotations and all QA tag/sentiment/score judgments require explicit
human inspection of the literal wording before candidate evaluation. VM-021's
null label marks insufficient evidence, not a third model label; Stage 4C will
freeze its metric treatment. QA score bands are case annotations, not acceptance
thresholds. No scoring, client execution, model call, or server start occurs here.

`dataset.py` provides `validate_curated_cases(cases, require_complete=False)`,
`load_curated_cases(root)`, and `curated_review_rows(cases)`. Reuse BenchmarkCase
and hash_dataset; public case repr and semantic hashing remain unchanged.
Validation rechecks typed cases, loaded contract/task equality, synthetic tags,
lowercase source/family/ordinal IDs, matrix category/difficulty/critical/exercise
allocations, duplicate IDs, evidence pointers/spans, summary annotations, and
source QA tag/metric vocabulary. It rejects noncanonical order rather than sorting.
Synthetic origin and semantic correctness still require human review.

The complete loader reads only these names under the explicitly supplied root:
`extraction.jsonl`, `context_summary.jsonl`, `voicemail.jsonl`, `qa.jsonl`,
`qa_conversation_summary.jsonl`, `qa_node_summary.jsonl`. Within each file, matrix
ordinals ascend. A missing, incomplete, misfiled, or malformed inventory fails;
no files are discovered, generated, or substituted. Partial validation supports
small test fixtures and explicitly selected subsets in the same relative order.

IDs use `curated-ex-001` / `adversarial-ex-021` spelling, with `matrix:EX-001`
tags. Matrix metadata in code describes the frozen allocation, not authored
scenarios. Explicitly adversarial matrix rows use source adversarial; the others
use curated. Optional provenance uses safe `synthetic-...` references without
historical run IDs. This is not a mechanism for sanitizing private source data.

`adapters.py` provides six pure renderers, `select_summary_context`,
`format_summary_transcript`, and the pure `apply_context_summary` control helper.
Prompts come from the unchanged contracts catalog. Runtime selection recognizes
pending async tools and developer completion messages, using original indices;
no-summary selection produces None from render_runtime_summary. A model exercise
with no request is an error, not a silently skipped model case.

```python
from ovp36_benchmark.dataset import load_curated_cases, hash_dataset, curated_review_rows
from ovp36_benchmark.adapters import prepare_curated_plan
from ovp36_benchmark.identity import fingerprint_execution_plan

# root/config/model are explicit caller inputs; preparation never sends requests.
cases = load_curated_cases(root)
rows = curated_review_rows(cases)
plan = prepare_curated_plan(cases, config=config, resolved_model=resolved_model)
dataset_hash = hash_dataset(cases)
plan_hash = fingerprint_execution_plan([item.identity_projection() for item in plan])
```

Preparation uses only prepare_generated_request, config.generation, and the
explicit resolved candidate model. There is no generation_overrides API or
implicit 4000-token override. Product runtime-summary 4000/30 seconds remain
source provenance. Exactly one configured token-limit field is retained.

The inventory contains 126 scenarios: 106 model exercises and 20 controls.
Only model exercises enter the repetition-major plan, giving 106 * repetitions
for the complete authored suite. Controls never enter candidate quality or latency
denominators. Stage 4C must evaluate/report controls separately. Preparation
performs no client construction, runner execution, journal access, or scoring.

Review rows contain IDs, contract/category, exercise, difficulty, critical flag,
a short scenario, and gold summaries; they do not render or print prompts.
The suite has 109 curated and 17 adversarial rows, with 100 critical entries:
96 model exercises and four controls. Tests freeze exact IDs, control/critical
allocations, evidence resolution (including runtime selected-slice visibility),
QA context-pair equality, hash stability, and 106/212 preparation counts for one/two
repetitions. Source-shaped synthetic QA metrics are consistent with the authored
transcript gaps; they are not measured model performance.

All 126 authored cases need human review before any candidate run. Extraction
policies, voicemail ambiguity, summary facts, QA score annotations, optional node
behavior, and control assignments must be reviewed independently of model output.

## Historical replay preparation (Stage 4A)

`replay.py` exposes `ReplayContract`, `CapturedReplayCase`, `ReplayDataset`,
`ReplayError`, `load_ovp34_replay(source_path, *, selection_path)`, and
`prepare_replay_plan(replay, *, config, resolved_model)`. Both paths are explicit;
no machine-specific location, model server, or endpoint is discovered or contacted.

```python
from ovp36_benchmark.replay import load_ovp34_replay, prepare_replay_plan
from ovp36_benchmark.identity import fingerprint_execution_plan

replay = load_ovp34_replay(raw_path, selection_path=selection_path)
plan = prepare_replay_plan(replay, config=config, resolved_model=resolved_model)
dataset_hash = replay.dataset_hash
ordered_execution_plan_hash = fingerprint_execution_plan(
    [execution.identity_projection() for execution in plan]
)
```

The selection input is the verified OVP-34 `oob_jobs.csv`; its rows determine IDs
and canonical case order. Raw JSONL supplies observation payloads, even when its
file order differs. The loader requires exactly 40 extraction, 56 runtime summary,
22 voicemail, 72 QA evaluation, and 52 QA conversation-summary cases. There is
no historical node/script-summary subtype among the six overall task contracts.
Generic wrappers and other observations are excluded by selection ID, never by
message deduplication or reconstructed attribution heuristics.

Both artifacts are read once as bytes. CSV parsing rejects invalid framing,
missing required columns (`observation_id`, `job_type`, `name`, `trace_id`), duplicate
headers/IDs, and inconsistent name/category/reference mappings. Every raw JSONL
row is validated, including unrelated rows: strict UTF-8, unique JSON keys, finite
numbers, object envelopes, consistent IDs/traces, and newline termination. Missing
selected observations or wrong counts fail closed without returning a partial set.
Selected input must contain exactly `messages`, a list of JSON objects. No internal
message schema inserts content, drops unknown fields, or rewrites argument strings.

Case IDs are `ovp34-` plus the full validated 16-character lowercase hexadecimal
observation ID. Frozen cases wrap the existing `CapturedReplayRequest`; private
captures are excluded from repr and ordinary serialization. Historical output and
provider configuration are not stored in replay cases. Later baseline comparison
can retrieve them by observation ID from the external source. Runtime summaries
carry the trace-reconstruction caveat; they are replayed without regeneration.

`fingerprint_replay_dataset()` hashes ordered safe projections containing exactly
`case_id`, `contract`, `source_observation_id`, and `message_fingerprint`, using the
existing versioned domain `ovp34-replay-dataset`. Semantic identity changes with
selected observations, contracts, messages, or case order. It ignores candidate
settings, historical output, unrelated rows, file location, and semantically
irrelevant JSON/CSV formatting. `hash_dataset()` remains for `BenchmarkCase` data.

`source_artifact_hash` and `selection_artifact_hash` identify the exact parsed bytes
as in-memory provenance only. They do not enter dataset, request, or plan identity.
Official private-data acceptance compares these hashes with the audited pair in
`docs/SOURCE_AUDIT.md`; the public loader does not enforce those particular hashes.
Synthetic fixtures exercise the same strict loader without bypass flags.

Preparation uses only `prepare_captured_request()` with the explicit resolved
candidate model and current `config.generation`, including exactly one token-limit
field. It produces repetition-major order: all CSV-ordered cases at repetition
zero, then all cases at repetition one, and so on. No runner execution, manifest
creation, journal opening, prompt rendering, or model call occurs in Stage 4A.
Errors expose categories/line numbers only; translated private failures retain
neither exception cause nor context.

For local private acceptance, invoke the APIs above in a Python session with
caller-supplied paths and fixed synthetic candidate configuration. Check both
audited artifact hashes, subtype counts, 242 unique IDs, 14 voicemail assistant
tool-call messages with absent content, and 14 tool-message references. Compare
all 242 original selected message sequences with prepared messages, then repeat
loading/preparation and compare case IDs, semantic dataset hash, and plan hash.
Change candidate model/settings and verify request fingerprints change while
message/dataset identity stays fixed. Print only counts, safe hashes, and pass/fail
categories. Do not execute the plan or print historical messages/outputs. No
tracked private-data validation script or CLI is required.

## Configuration

This example describes the general configuration schema. Dataset-to-runner APIs
exist; Stage 5B adds only the narrow `candidate-smoke` command documented below,
not a general benchmark CLI. The example's settings are not candidate policy.

```toml
experiment_label = "local-development"
evaluation = "exploratory" # or "canonical"
timeout_seconds = 30.0
concurrency = 1
repetitions = 1

[endpoint]
alias = "local"
base_url_env = "OVP36_BASE_URL"
model_env = "OVP36_MODEL"
# api_key_env = "OVP36_API_KEY" # omit for endpoints without authentication

[generation]
temperature = 0.0
max_tokens = 4000
token_limit_field = "max_tokens" # or "max_completion_tokens"
top_p = 1.0
# seed = 7
```

`load_config(path)` uses `tomllib` without reading environment values.
`resolve_endpoint(config.endpoint)` resolves only named variables. A missing or
blank referenced variable raises an error. Literal `base_url`/`model` may be used
instead of their environment references, but not alongside them.

API keys can only be supplied through environment references. URLs containing
userinfo, query parameters, or fragments are rejected. Resolved endpoint values
are excluded from ordinary serialization and repr. `redact_config(config)` retains
its original unresolved-config semantics, including a literal URL when configured;
it is **not** a persistence projection. Keep local configuration and secrets out
of Git. Use `persistent_config_projection(config, resolved_model=...)` for safe
identity metadata; it constructs an explicit allowlist without dumping config.
It includes alias, requested model, generation, timeout, repetitions, concurrency,
experiment label, evaluation designation, and declared safe server metadata.
Literal endpoint URLs and URL-derived hashes are never persisted or used for
request/run identity. Environment names/values and credentials are excluded.

Endpoint alias is explicit operator-declared safe experiment identity: 1–64
ASCII letters/digits/underscore/hyphen/dot, beginning with a letter or digit.
Traversal-like `..`, dotted IPv4 patterns, and obvious credential prefixes are
rejected. Server metadata, persisted model names, and experiment labels allow
1–200 ASCII letters/digits/underscore/hyphen/dot/plus, optionally one
`namespace/name` separator; each segment starts with a letter or digit. URLs,
absolute/traversal paths, IP patterns, controls/whitespace, credential assignments,
and common token prefixes are rejected. At Stage 3A, `model_artifact_hash` and
`chat_template_hash` are operator-declared safe identity metadata; their validation
does not enforce a digest algorithm or length. If used, their concrete digest
algorithm and format must be frozen before canonical serving metadata is populated.
These checks cannot detect disguised secrets; supply deliberate nonsecret labels.

Canonical configuration requires a checkpoint, revision, artifact hash, chat
template hash, or runtime **and** runtime version. Exploratory configurations may
leave these unknown. This is a minimum declaration, not proof of server equality:
update alias or safe server metadata when material server behavior changes. A
URL-only change with identical declared identity intentionally leaves run ID
unchanged.

The generation limit is explicit for every configuration. Current API
`service_factory.py` gives non-realtime services a default `max_tokens` cap;
`LLM_MAX_TOKENS_CAP_REASONING` defaults to 2048 for reasoning models such as Qwen3
and is environment-overridable. Speaches receives this through
`SpeachesLLMSettings(model=model, max_tokens=max_tokens)`.

Runtime summary passes the explicit one-shot Python argument **max_tokens=4000**
to Pipecat with a **30-second timeout**. Newer supplied Pipecat builds parameters
containing both token-limit keys and sets `max_completion_tokens=4000` because
that key exists. It does not clear the configured service `max_tokens`. With
Qwen source defaults, the constructed dictionary can therefore contain both
`max_tokens=2048` and `max_completion_tokens=4000` before SDK/provider handling.
The 56 historical summary observations have empty `modelParameters` (`{}`), so
their exact historical wire token-limit fields are not independently known.

Stage 3A intentionally normalizes candidate benchmark requests to exactly one
selected token-limit field to avoid ambiguous dual-limit behavior. This is not
a claim of byte-for-byte reproduction of the product's provider parameter
dictionary. Product settings remain provenance for later per-task candidate
configuration; neither the source defaults nor the example above chooses
Granite's canonical field/value. `token_limit_field` defaults to `max_tokens`
only for backward compatibility with Stage 1 configs. Unset seed is omitted and
stream is fixed false. Historical Qwen voicemail used max_tokens=2048; that
separate observation does not assign a Granite limit either.

No Granite checkpoint, quantization, or serving runtime is selected. Their
configuration remains a decision before local serving. GB10/DGX access and
hardware measurements follow local validation in a later stage.

## Request preservation and identity (Stage 3A)

`MessageSnapshot(messages)` accepts an ordered list of JSON objects, stores one
canonical JSON string, and returns fresh graphs through `to_messages()`. It
preserves missing versus null fields, unknown fields, nested structures, exact
strings and tool arguments, tool calls, and tool-result IDs. Object keys may be
sorted; arrays and string whitespace are never normalized. Non-finite numbers,
non-string keys, tuples, and other non-JSON values are rejected. This preserves
JSON semantics, not original serialization bytes. `ContextMessage` is unchanged.

The two preparation functions are separate:

- `CapturedReplayRequest` → `prepare_captured_request()` → `PreparedRequest`:
  verifies the supplied captured fingerprint, preserves the same snapshot, and
  attaches explicit candidate settings. Historical generation/output remain
  provenance only. No contract rendering or task-schema conversion occurs.
- Already-created message objects → `prepare_generated_request()` →
  `PreparedRequest`: accepts curated/adversarial metadata and candidate settings;
  rejects `ovp34_replay`. Future adapters supply these messages.

`PreparedRequest.to_request_body()` builds data only: model, messages, temperature,
one completion limit, top_p, optional seed, and stream=False. There are no
top-level tools/tool_choice fields. Message-level tool fields remain intact.
Private snapshots, historical outputs/settings, and qualifications are hidden
from request repr and ordinary model serialization. Dispatch data is private;
later manifests should reference case/provenance IDs and fingerprints instead
of copying historical inputs. Generic model serialization is not a manifest.

`identity.py` uses SHA-256 with `ovp36-v1:<domain>` plus a NUL separator and
canonical JSON. Domains separate messages, request, safe-config, and run hashes.
`canonical_json_bytes()` sorts keys, retains sequence/string/type distinctions,
and uses deterministic ASCII JSON escapes. This convention is versioned, not an
RFC 8785 implementation. `fingerprint_messages()` hashes messages only;
`fingerprint_request()` hashes dispatch data only; `fingerprint_safe_config()`
hashes the explicit persistent config projection.

`make_run_id()` receives dataset hash, ordered plan hash, safe config fingerprint,
requested model, alias, server metadata, relevant contract hashes, harness-code
fingerprint, and dependency-lock fingerprint explicitly. It never inspects Git,
files, environment, timestamps, URLs, or credentials. Callers must supply the
matching validated config fingerprint and identity components. `ExecutionKey`
contains run ID, case ID, request fingerprint, and zero-based repetition index.
These primitives perform no execution, result writing, or resume behavior.

## One-shot model client (Stage 3B)

`ModelClient(resolved_endpoint, timeout_seconds=T)` owns an HTTPX client and an
`AsyncOpenAI` instance, pinned to **openai==2.54.0 / httpx==0.28.1**. Use it as an
async context manager or call `await client.aclose()`; repeated close is safe.
`await client.send_once(prepared_request)` dispatches
`PreparedRequest.to_request_body()` directly through
`sdk.with_raw_response.chat.completions.create(**body)`. The
prepared request supplies the model and generation settings. Both preparation
paths use this same client; historical replay still bypasses adapters.

Actual outgoing JSON passed MockTransport fidelity tests for current replay
shapes: absent/null content, tool calls and IDs, exact argument strings, unknown
fields, nested structures, whitespace, Unicode, and ordering. This establishes
JSON-message preservation, not original historical HTTP-byte equality. Tests
block real socket/DNS access; no model server is needed or started.

SDK retries and HTTP transport retries are explicitly zero; redirects and HTTP
environment configuration are disabled. No retry loop exists here. Before SDK
construction, the presence of `OPENAI_CUSTOM_HEADERS`, `OPENAI_ORG_ID`,
`OPENAI_PROJECT_ID`, or `OPENAI_LOG` in the environment raises `ClientConfigurationError` naming
only the variable, including when its value is empty or whitespace.
The client never mutates the environment. These settings can add or override
headers or enable SDK logging independently of HTTPX `trust_env=False` in the
pinned SDK. Exact OpenAI 2.54.0 source reads `OPENAI_LOG` during import and can
enable OpenAI/HTTPX logging. The constructor refuses this ambient switch before
dispatch so it cannot silently change benchmark privacy/determinism. The client
itself emits no provider payload logs; external application/root logging remains
outside this library's control. The guard does not undo import-time logging
configuration or change process logger levels.

An explicitly resolved API key wins over ambient `OPENAI_API_KEY`. Without a
configured key, the SDK receives `ovp36-local-no-auth`, a syntactic credential
placeholder, not a secret. It produces an Authorization Bearer header accepted
by the intended local servers. The client does not log or persist credentials,
the placeholder, URLs, payloads, or raw exceptions, and does not configure logging.

The same T sets HTTPX/SDK operation timeouts and an `asyncio.timeout(T)` deadline
on the SDK await. External task cancellation propagates. Timeout does not prove
server-side generation stopped. `latency_ms` measures complete client-observed
request/completion latency, including response extraction; it is not TTFT,
decode time, or GPU latency.

Immutable results capture content (including null/absent distinction), finish
reason, reported model, and nullable usage. Missing counts remain None, reported
zero remains zero, and missing totals are never derived. Malformed completion
envelopes/counts produce non-retryable protocol errors; empty/null content and
poor task text can still be transport successes. Error results contain only safe
classification fields, never exception objects or provider error bodies. HTTP
408/429/500/502/503/504 and connection/timeouts are marked retryable as evidence;
the Stage 3D v1 runner performs no automatic retries, regardless of this flag.

OpenAI 2.54.0 can coerce malformed nested usage values (`true` to `1`, `"2"` to
`2`). Before SDK parsing, the transient raw response's `http_response.json()`
must be a JSON object. The client validates only the five consumed usage counts
and their containing objects against original JSON types. Booleans, strings,
floats (including `2.0`), negative integers, arrays, and objects are rejected as
counts. Missing/null usage is accepted as None; missing/null counts and details
are allowed, and unknown usage fields are ignored. The synchronous `raw.parse()`
then supplies the usual SDK completion and content field-set semantics without
another dispatch. Raw bytes, JSON objects, and headers are never stored in results.
Malformed usage produces non-retryable `invalid_usage_schema`; undecodable JSON
produces `invalid_response_json`, and a non-object top level produces
`invalid_response_schema`. These codes contain no provider data.

Provider-controlled response text is hidden from repr but remains available to
later parsing. Ordinary result serialization is **not** a safe persistence
projection. Stage 3C applies the explicit private storage projection below.
Stage 3D executes prepared requests as described below; Stage 4A supplies replay
loading/preparation. Stage 4B1 provides task adapters; Stage 4B2 supplies their
synthetic inventory.

## Local persistence (Stage 3C)

`persistence.py` provides synchronous `build_manifest(...)`,
`load_manifest(results_root, run_id)`, `read_events(results_root, run_id)`, and
`ResultJournal.open(results_root, expected_manifest)`. The journal supports
`append_attempt(execution_key, attempt_index, result)`,
`finalize(execution_key, attempt_index)`, `records()`, `completed_keys()`,
`close()`, and synchronous context management. The caller explicitly supplies
the results root and every identity component; persistence discovers no Git,
endpoint, dataset, or dependency fingerprints.

```text
results/                         # ignored private evidence
  <validated SHA-256 run_id>/
    manifest.json                # ovp36-manifest-v1
    journal.jsonl                # ovp36-journal-v1
```

The immutable manifest contains only the approved safe configuration projection
and identity components, with the separate Stage 3A identity version. Optional
seed/server fields retain their existing omission semantics. New manifests use
public Stage 3A identity functions. Resume requires a strictly loaded manifest
to match both the directory run ID and the caller's rebuilt expected manifest
exactly; existing bytes are never rewritten. Standalone loading validates the
stored schema/path, not independent recomputation of its safe-config hash.

The journal contains two event kinds: `attempt` stores one returned Stage 3B
result; `finalized` references the latest recorded attempt without duplicating
its output. `attempt_index` starts at zero and is contiguous per `ExecutionKey`;
it is separate from `repetition_index` and is not part of execution identity.
Only finalized keys enter `completed_keys()`, including finalized failures.
Unfinished attempts remain readable. The Stage 3D v1 runner durably records each
returned result before finalizing it; Stage 3C itself makes neither retry nor
finalization decisions and adds no cross-field constraints to Stage 3B results.

Candidate `raw_content` is preserved exactly, including private-looking text,
only in ignored private journal artifacts. It is not redacted or keyword-scanned
and is hidden from repr/errors. Historical messages, baseline outputs, provenance
payloads, endpoints, credentials, headers, and raw HTTP material are not copied.
Provider `finish_reason` and `response_model` are retained only when valid under
`SafeIdentifier` and `SafeMetadata`, respectively. Rejected non-null values become
null with field-name-only `omitted_metadata`, ordered `finish_reason` then
`response_model`. Originally null fields receive no omission marker.

Reading fails closed on malformed JSON/UTF-8, duplicate keys, schema or sequence
violations, non-finite numbers (including overflow), blank lines, and any
unterminated final record, even otherwise valid JSON. A missing/empty journal
with a valid manifest is empty evidence; a journal without a manifest is invalid.
There is no automatic tail repair or valid-prefix success. Errors expose safe
categories/line numbers only. Parsed/scored structures remain deferred; candidate
strings containing `NaN` or `Infinity` are ordinary text in standard JSON.

The supported target is a local POSIX filesystem and **one sequential writer
object per run in one process**. New directories request `0700`, files `0600`;
group/other access is rejected on existing artifacts. More restrictive owner
modes are allowed when sufficient for the operation. Symlinks and wrong artifact
types are rejected; existing permissions are never broadened. These mode checks
do not cover ACLs, administrator access, or network/non-POSIX security policy.

If the results root is missing, its parent must already exist. Creating the root
fsyncs that parent; creating the run directory fsyncs the root. Manifest creation
writes/fsyncs a private temporary file, publishes with a create-only hard link,
fsyncs the run directory, removes that temporary name, and fsyncs the directory
again. Unsupported hard-link publication fails without an overwrite fallback.
Journal appends handle short writes and fsync the file; first creation also
fsyncs the run directory. In-memory state advances only after success. An uncertain
write/fsync failure poisons the writer until close/reopen and strict validation.
This is not HTTP/filesystem transactionality, exactly-once inference, universal
power-loss protection, or concurrent-writer safety: a model call can happen before
durable evidence exists. No database, new dependency, or timestamps are introduced.

## Sequential prepared-request execution (Stage 3D)

`runner.py` exposes frozen strict `PlannedExecution(request, repetition_index)`,
`RunSummary`, safe `RunnerValidationError`, and the asynchronous function:

```python
summary = await run_plan(
    plan, client=client, results_root=results_root, expected_manifest=expected_manifest,
)
```

The caller supplies an ordered sequence of already-created `PlannedExecution`
objects and owns the `ModelClient`; the runner never closes it. The runner owns
the `ResultJournal` it opens and closes it on completion, persistence errors,
unexpected client exceptions, and cancellation. It uses public APIs only.
There is no dataset loading, prompt rendering, parsing, scoring, or server startup.
Captured requests remain on the direct prepared-request path without adapters.

Build the manifest's `ordered_execution_plan_hash` using the public
`identity.fingerprint_execution_plan([item.identity_projection() for item in plan])`.
Each entry has exactly `case_id`, `request_fingerprint`, and `repetition_index`;
the versioned hash domain is `execution-plan`. List order matters; object-key
order does not. The helper validates existing identity field semantics, while
the runner rejects duplicate execution keys. No run ID or private payload is
included in the plan projection.

Before opening any artifacts, the runner snapshots plan order, validates items
and repetition bounds, computes actual request fingerprints/execution keys,
rejects duplicates, checks request model equality with the manifest's requested
model, requires concurrency exactly one, and checks the full ordered plan hash.
An empty plan must match the hash of `[]`. After opening the journal, every
persisted key must belong to the full supplied plan, and every unfinished
execution must have at most one recorded attempt, before any model dispatch.

Per-request generation settings are preserved and need not equal manifest
generation, which is run-level declared/default metadata. Effective settings
are bound through request fingerprint → plan fingerprint → run ID. The runner
does not attest that the injected client's URL, timeout, or physical server
matches declared endpoint alias, timeout, or server metadata; those remain
caller composition/preparation responsibilities.

V1 fixes `MAX_ATTEMPTS = 1`: a fresh execution sends once, durably appends attempt
zero, and finalizes it for every returned status. `retryable` remains unchanged
evidence, never a retry decision. Empty/null output, bad JSON, missing usage, and
finish reasons do not trigger another send. This is a **benchmark normalization**,
not a claim of product retry fidelity. Any later retry policy needs separate
review and different harness/run identity. There is no backoff, jitter, retry
delay, or artificial cancellation sleep.

Resume skips finalized keys, including failures and valid older multi-attempt
histories. Exactly one unfinished attempt is finalized without resending,
regardless of status; more than one unfinished attempt fails preflight. Append
or finalization failure stops execution immediately. Unexpected client errors
and cancellation propagate without fabricated evidence. After uncertain writes,
strict reopening may find finalized, unfinished, or corrupt state; no automatic
tail repair occurs. A crash before durable attempt evidence can cause a resend
on restart, so this does not guarantee exactly-once model execution.

`RunSummary` contains only seven nonnegative strict integer counts: `planned`,
`skipped_completed`, `resumed_unfinished`, `attempts_sent`, `executions_finalized`,
`successes`, and `failures`. Counts describe this invocation: sends made,
unfinished executions resumed, and newly finalized terminal statuses. It enforces
`planned == skipped_completed + executions_finalized` and
`executions_finalized == successes + failures`. These are execution counts,
not quality scores. No summary file or private content is added; candidate text
remains only in the ignored private journal. New translated validation errors
have safe messages and retain neither exception cause nor context.

## Prompt contracts (Stage 2)

```python
from ovp36_benchmark.contracts import load_contract, render_contract, hash_contract

contract = load_contract("context_summary", version=1)
rendered = render_contract(contract, {"formatted_transcript": "USER: A resolved fact."})
assert rendered.source_settings.max_tokens == 4000
assert rendered.source_settings.timeout_seconds == 30.0
fingerprint = hash_contract(contract)
```

Packaged `prompts/contracts.json` contains explicit IDs, versions, source
references, completeness, and exact prompt strings. Hashes include text and
metadata; JSON file layout is irrelevant, but text whitespace is significant.
Rendering requires named string values and rejects missing/unknown variables.
Substituted content is not recursively interpreted as a template.

All seven contracts are complete: `extraction`, `extraction.skyassist`,
`context_summary`, `voicemail`, `qa_conversation_summary`, `qa_node_summary`,
and `qa`. Extraction accepts already-formatted `variable_lines` and
`formatted_history`. Its variable-line template retains `VariableType.string`,
`VariableType.number`, or `VariableType.boolean`, and its user text starts with
exactly two newlines. An optional nonempty `extraction_prompt` is appended after
exactly two newlines. `extraction.skyassist` freezes the reviewed instruction and
origin/destination/travel_date variable definitions.

Completeness describes source knowledge, not adapter implementation. The full
voicemail prompt uses message context; the node-summary contract uses the
source-known node-description algorithm documented in SOURCE_AUDIT. The full
SkyAssist QA `system_template` preserves literal double-brace placeholders.
`load_contract` exposes all these unchanged; `render_contract` refuses tasks
requiring dedicated adapters, including QA substitution. Stage 4B1 supplies those
adapters and history formatting/runtime-summary selection/application. Partial-contract
inspection remains supported for explicitly incomplete resources; no current
resource is marked partial.

## Response parsing (Stage 2)

```python
from ovp36_benchmark.parsing import check_strict_json, parse_extraction, parse_qa

raw = 'Answer: {"name":"Alex"}'
assert not check_strict_json(raw).valid
result = parse_extraction(raw)
assert result.production_parse_success
assert result.parser_path == "object"
assert result.schema_valid is None  # task field validation/scoring is deferred
```

`parse_production_json` applies this order: blank/None → `{}`; direct dict/list;
first fenced block; first balanced object; first balanced array; exact raw
fallback. It does not repair JSON or retry later candidates within a branch.
If an invalid first fence contains no object, a later object's recovery through
the first-object branch is possible. Object recovery precedes array recovery.
Quoted delimiters and escaped quotes do not change balancing. As in production,
backslashes escape the next character even outside strings in malformed input.

Every production JSON branch accepts only dict/list. A scalar in the first fence
is rejected, then object/array recovery examines the full stripped response;
if neither succeeds, the original response is retained in raw fallback.
Production uses exactly the case-sensitive regex below and only its first match;
arbitrary language headers are not removed:

````text
r"```(?:json)?\s*([\s\S]*?)\s*```"
````

Strict assessment handles the entire JSON document and reports its top-level
type, object-required violations, blank output, Markdown, and surrounding prose.
Strict scalar JSON can be syntactically valid while direct production parsing
rejects it. Strict assessment rejects non-standard NaN/Infinity constants.
Production uses Python `json.loads` on stripped content and preserves its
non-finite values, including overflow from syntactically valid numbers such as
`1e999`. `ParseResult` alone allows these values and serializes them as Python
JSON constants, preserving them on round trip rather than converting to null.
Case, configuration, and scoring schemas retain finite-number constraints.
Standards-compliant encoding of parsed/scored structures remains a future
decision. Stage 3C persists candidate text and transport evidence only, never
`ParseResult` or its non-finite Python objects.

`parse_extraction` retains parsed values without type repair. `parse_qa` retains
the original parsed value but exposes `{}` as the consumed value for non-dicts,
empty responses, or parser-generated raw fallbacks. Actual dict fields are not
filled or coerced; schema validation and missing-field scoring are deferred.
The raw fallback marker is identified by parser path, not by the presence of a
legitimate field named `raw`.

`parse_voicemail` uppercases the response and checks CONVERSATION before
VOICEMAIL using substring matching. Its independent strict check accepts only an
exact uppercase label after Python `str.strip()` removes surrounding whitespace.
`parse_summary` preserves returned text without JSON parsing; its format/usability
flags indicate nonblank text only, not factuality or sentence-count compliance.

All helpers preserve `raw_response`, parser path, parsed/normalized values, and
diagnostics. Blank input and raw fallback are not parse successes. Parseable
output does not establish task schema or semantic correctness; `schema_valid`
is nullable to distinguish unassessed output from an actual schema failure.

## Stage 4C1: evaluation and review

`evaluation.py` measures finalized completion evidence without calling a model.
It reuses the production parsers and keeps strict format, recovered usability,
structure, and semantic checks separate. There is no LLM/embedding judge, no
candidate acceptance threshold, and no weighted cross-contract score.

`evaluate_model_output(case, request=..., attempt=..., dataset_hash=...,
evaluator_fingerprint=..., purpose=...)` accepts a prepared request and preserved
`AttemptRecorded`. Model purpose is `candidate`, `candidate_smoke`, or
`functional_stub`. The pure function
assumes its caller has established finalization; use `reporting.evaluate_run()`
for a journal-backed run. That public join verifies the expected manifest,
dataset, original ordered plan, and execution references before evaluating only
finalized attempts. Missing and unfinalized executions remain unavailable. It
never sends, retries, finalizes, or repairs evidence.

`evaluate_control_case(case, dataset_hash=..., evaluator_fingerprint=...)` handles
all 20 non-model exercises without a client. Nineteen controls can pass fully
automatically. QA-026 retains a pending human check confirming that the supplied
hang-up-threat claim is unsupported. Its automatic semantic contradictions are
negative-control evidence, not a candidate result. Control reports are separate
from candidate metrics and from the 96 critical model cases.

Automatic extraction comparisons are typed and exact; each missing-value policy
is respected. Voicemail preserves source decision precedence. VM-021 remains
unlabelled and outside binary metrics. QA precision/recall/F1 are explicitly
**annotation-scoped**, with unannotated predictions and annotation coverage
reported alongside them. Invalid/unavailable structured fields do not acquire
correctness credit from production defaults. Metric records retain numerator and
denominator; aggregate reports disclose unavailable records and pending review.
Zero denominators produce null. QA score bands are case expectations only.

Free-text facts, grounding, corrections, stale-current-state claims, and
optionality produce `pass`, `fail`, `pending`, or `not_applicable` checks.
`ScoreResult.status="complete"` means processing finished, even when checks fail.
Summary categories reference existing facts without duplicating them. NS-006's
optional actions remain required *summary distinctions*; they are not silently
reclassified as safe to omit.

`sentence_count_by_rule` counts nonempty spans separated by runs of `.`, `!`, or
`?` before whitespace/end, with optional closing quotes/brackets, plus a trailing
fragment. It does not split a decimal point without following whitespace. It can
split abbreviations such as `Dr.`. `sentence_range_by_rule_ok` is therefore a
format diagnostic, with a separate human sentence-compliance check: QS 3–5,
NS 2–4, QA summary 1–2, and no runtime-summary sentence constraint.

`build_human_review_items(case, evaluation, raw_output=...)` verifies the case and
output fingerprints and resolves relevant local evidence. These are private
transient views: output, questions, facts, and snippets are excluded from normal
repr/serialization. They are not automatic artifacts or safe log messages.
`ReviewDecision` stores only version, evaluation ID, check ID, status, and private
notes; notes are hidden from repr. `apply_review_decisions()` returns a new view
and rejects stale, unknown, duplicate, or automatic-check decisions.

`aggregate_evaluations()` reports per-contract metrics, review coverage, and
critical failures/pending checks, with controls and historical evidence kept
separate. Historical replay has no semantic gold and receives only applicable
availability, parsing, and structure measurements. Captured requests are never
regenerated through adapters.

Evaluation identity has its own schema/version domain and binds dataset, case,
execution/attempt, privately hashed evidence, evaluator fingerprint, and purpose.
The evaluator fingerprint is caller-supplied. For reproducibility, calculate it
with `evaluation_fingerprint("implementation", components)`, where `components`
is a mapping from repository-relative paths to SHA-256 file-byte digests of all
`src/ovp36_benchmark/*.py`, `src/ovp36_benchmark/prompts/contracts.json`, and
`uv.lock`. Tests use controlled synthetic digests. A changed implementation or
review revision creates a new artifact; raw execution evidence stays unchanged.

`write_evaluation_artifacts(results_root, evaluations=..., review_decisions=...)`
publishes private create-only snapshots:

```text
results/<run_id>/evaluations/<batch_id>/automatic.jsonl
results/<run_id>/evaluations/<batch_id>/summary-automatic.json
results/<run_id>/evaluations/<batch_id>/reviews/<review_hash>.jsonl
results/<run_id>/evaluations/<batch_id>/summary-<review_hash>.json
results/control-evaluations/<batch_id>/...
```

Automatic rows contain only validated IDs, fingerprints, safe labels/statuses,
finite measurements, and check results. Raw candidate text stays in the ignored
journal; raw historical messages/outputs are not duplicated. Parser objects are
not serialized wholesale. Review revisions are complete immutable decision sets,
not another event journal. Omit `review_decisions` for automatic artifacts only;
provide an empty sequence to publish an explicit empty revision.

Publication requires private POSIX directories/files, rejects symlinks and
insecure permissions, fsyncs before create-only publication and after directory
changes, and reuses an existing file only on exact byte equality. Artifacts under
`results/` remain ignored. `load_review_decisions()` reads strict finite JSONL and
validates bindings. Neither evaluation nor publication prints private payloads.
The existing client, runner, persistence, fixtures, and dependencies are unchanged.

## Stage 4C2: deterministic local HTTP smoke

`test_server.py` supplies `StubExchange(expected_request, response_body,
status_code=200)` and the single-use `DeterministicChatServer(exchanges)` context
manager. It snapshots exchanges as finite canonical JSON, binds only to
`127.0.0.1:0` without hostname lookup, and exposes `base_url` ending in `/v1`.
Only `POST /v1/chat/completions` is supported. One background thread compares
each request against the next `PreparedRequest.to_request_body()` snapshot,
including field presence, nulls, ordering, whitespace, and typed values. Token
limits come from the prepared request; the server has no decoding policy.

Request bodies require a valid Content-Length, are limited to 1 MiB, and have a
bounded read deadline. Responses use JSON, byte Content-Length, and connection
close. Access logs and server tracebacks are disabled. Harness failures retain
safe codes; `assert_complete()` and normal context exit reject failures or
unconsumed exchanges. Cleanup closes the listener and joins the server thread,
including on exceptions. The server has no LLM, GPU, external API, streaming,
model discovery, or production authentication.

`tests/test_local_e2e.py` runs the real AsyncOpenAI HTTP transport through the
unchanged runner, manifest/journal, evaluation, and reporting APIs. Exactly
EX-001, CS-002, VM-001, QA-001, QS-001, and NS-006 are dispatched once in the main
smoke as `ovp36-deterministic-stub`, with evaluation purpose `functional_stub`.
Fixed responses live only in test code: they are plumbing fixtures, not benchmark
gold or candidate results. EX-024 and CS-025 are evaluated separately as controls.
Semantic checks initially remain pending; selected test-local review decisions
exercise separate reviewed views without an automatic semantic judge.

The test-local socket guard permits only the allocated loopback destination,
blocks DNS, and records forbidden attempts even if a library catches the error.
SDK environment settings are isolated and restored. All journal/report artifacts
live in temporary directories and are removed. A matched HTTP 500 exchange proves
one attempt and one finalization without a retry, leaving semantic quality
unavailable. Local client latency is functional evidence only: it is not
representative candidate latency, throughput, or hardware performance.

Run just this boundary with:

```sh
.venv/bin/python -m unittest discover -s tests -p test_local_e2e.py -v
```

## Stage 5B: local candidate-smoke command

From this source checkout, after separately selecting and starting an approved
local model server:

```sh
.venv/bin/python -m ovp36_benchmark candidate-smoke \
  --config candidate.local.toml \
  --results-root results
```

The command neither chooses nor starts a model. It reuses `load_config()`,
`resolve_endpoint()`, `prepare_curated_plan()`, `build_manifest()`, `ModelClient`,
`run_plan()`, `evaluate_run()`, aggregation, and artifact publication. No new
dependency or serving-framework integration is required.

Use the existing TOML schema with `evaluation="exploratory"`, `repetitions=1`,
`concurrency=1`, and an `experiment_label` equal to `candidate-smoke` or beginning
with `candidate-smoke-`. In the general example above, replace the experiment
label and choose generation settings for the separately approved model/server;
no token value in that example is imposed by the command. Endpoint/model values
resolve from their configured environment references (for example
`OVP36_BASE_URL` and `OVP36_MODEL`). If authentication is needed, reference the
key through `api_key_env`; never put it in TOML or the URL. Keep the four guarded
SDK settings `OPENAI_CUSTOM_HEADERS`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, and
`OPENAI_LOG` absent. Client defaults already disable environment proxies and
redirects.

Before constructing the client, the command requires
`http://127.0.0.1:<port>/v1` (an optional final slash is accepted). The port must
be explicit and nonzero. Hostnames, other addresses, HTTPS, embedded credentials,
queries, fragments, and the full completion route are rejected without DNS or
network access. The SDK appends `/chat/completions`. There is no production socket
monkeypatch: stronger socket/DNS guards remain test-only.

The full frozen 126-case dataset is validated and hashed, but only these six model
cases enter the plan, in canonical order:

```text
curated-ex-001
curated-cs-002
curated-vm-001
curated-qa-001
curated-qs-001
adversarial-ns-006
```

Controls and historical replay are not dispatched or evaluated by this command.
One configured generation object is preserved for all six requests, including
the selected token-limit field, temperature, top_p, and optional seed. No per-task
overrides or model-optimal defaults are added. Stream remains false.

Actual repository bytes supply identity. The command builds the documented
`evaluation_fingerprint("implementation", components)` over all package Python
files, prompt contracts, and `uv.lock`, and uses it for both harness and evaluator
identity. The lock fingerprint is its SHA-256 file-byte digest; contract, dataset,
and plan hashes use existing public helpers. The full dataset hash and six-entry
plan hash bind the manifest, along with configured safe model identity, endpoint
alias, settings, and declared server metadata. No URL or credential is added to
identity. Model selection and accurate server/checkpoint declarations remain
operator responsibilities; no files should change during a run.

`--results-root` defaults to `results`. Inside the checkout, the command permits
only the existing ignored `results/` tree; external private directories are also
accepted. Symlink paths are rejected. Existing persistence enforces private
permissions and immutable manifests/journals; reporting writes the existing
automatic artifacts below the run. Config files ending in `.local.toml` and the
default results tree remain ignored. Do not commit private artifacts.

Evaluations use `purpose="candidate_smoke"`, distinct in identity and reporting
from `functional_stub`, full `candidate` evaluations, and controls. Semantic facts
and grounding remain pending human checks. The command does not interactively
review, create review decisions, judge with a model, or add quality thresholds.
Its console output contains only run ID, purpose, counts, safe contract/status
labels, and pending human-review counts; it does not print payloads, endpoints,
keys, or paths. Locate artifacts under the supplied results root and printed run ID.

Invalid arguments exit 2; configuration/invariant/harness failures exit nonzero
with safe codes. Successfully published evaluations exit 0 even for poor quality,
pending reviews, or normally journaled HTTP/transport/timeout/protocol failures.
There are no retries. Resume skips finalized executions, including failures, and
finalizes one unfinished attempt without resending. Inspect evidence before a
deliberate rerun; use a distinct smoke experiment label for a new run and retain
old artifacts. Changing only the URL does not create a new run identity.

Local smoke establishes request/response and parser compatibility, basic
functional behavior, automatic evaluation plumbing, human-review workflow, and
obvious catastrophic failures. It does not establish GB10 latency, production p95,
throughput/jobs per second, GPU memory sizing, dialogue/TTS coexistence, contention
reduction, or capacity planning. Local client latency is evidence of functional
execution only, not representative benchmark performance.

## Private ten-case historical QA comparison

`qa-replay-compare` uses the same private TOML, explicit loopback endpoint,
configured model/generation settings, and sequential runner as `candidate-smoke`.
It neither starts a server nor selects a model. With the approved local server
running and the configuration's `OVP36_BASE_URL` and `OVP36_MODEL` variables set:

```sh
.venv/bin/python -m ovp36_benchmark qa-replay-compare \
  --config candidate.local.toml \
  --source "$OVP34_RAW_OBSERVATIONS" \
  --selection "$OVP34_OOB_SELECTION" \
  --results-root results
```

Set the two artifact variables to the private canonical final-100 observations
JSONL and `oob_jobs.csv`. No historical payload or machine-specific path belongs
in tracked `data/`. This command first calls the existing `load_ovp34_replay()`
on the **full 242-case dataset**, then checks both exact file-byte hashes frozen
in `SOURCE_AUDIT.md` §1.2. Only after that does it select the ten observation IDs
in `qa_replay_comparison.QA_OBSERVATION_IDS`, retaining CSV order. There is no
checksum bypass or alternate loader. A second, hash-verified raw-file pass joins
each historical `qwen3.5` output from `raw.output.content` by observation ID.

`prepare_replay_plan()` sends captured messages directly, including their
already-rendered prompts; curated adapters are never involved. The full dataset
hash binds the manifest; the ten selected executions bind the plan hash. Candidate
settings replace historical generation settings, without rewriting messages.
Keep the smoke config's exploratory designation, one repetition, concurrency one,
and `candidate-smoke` experiment-label convention. The different dataset/plan
already separates this run from the six curated smoke cases. Use a new experiment
label for a deliberate rerun; existing finalized successes and failures resume
without another model call.

Existing manifest, journal, and automatic evaluations remain under
`results/<run_id>/`. Additional private, immutable files are published at:

```text
results/<run_id>/qa-comparison/<observation_id>.json
```

These explicitly private reports include captured messages, exact baseline and
candidate raw text, parsed results, tags (including reasons), sentiments, scores,
summaries, tag intersections/differences, sentiment agreement, absolute score
differences, parser/strict-format status, and journaled transport/model metadata.
They include execution identity and both source artifact hashes. Qwen is labeled
`historical_baseline`; agreement is diagnostic, not correctness, gold, an LLM
judgment, or a global semantic score. Existing evaluation utilities report
candidate structure without inventing historical gold.

Direct and fenced JSON use the existing production-compatible QA parser, with
strict-format assessment kept separate. Unavailable/invalid comparison fields
produce `null` differences rather than false agreement or zero error. Tags are
compared by exact tag string as sets; order, duplicate count, and reasons remain
in each side's raw/parsed evidence. Scores must be finite numbers in 1–10, excluding
booleans. Production non-finite JSON extensions remain intact in raw text; parsed
results containing them are explicitly marked omitted from the strict JSON
projection, with affected extracted fields listed in `omitted_nonfinite_fields`.

The existing private POSIX publisher enforces directories with no group/other
access, private files, symlink rejection, fsync, and create-only publication.
Keep all these reports local and ignored. Historical payloads are not copied into
manifests, journals, or ordinary automatic evaluations. Console output contains
only run identity, labels, and counts. The command does not tune token limits or
timeouts: inspect finish reasons and errors before interpreting comparisons.
Local execution does not establish GB10/DGX performance.

## Full curated/adversarial candidate suite

After separately starting the approved local server, explicitly invoke:

```sh
OVP36_BASE_URL='http://127.0.0.1:8080/v1' \
OVP36_MODEL='mlx-community/granite-4.0-h-tiny-8bit' \
.venv/bin/python -m ovp36_benchmark candidate-suite \
  --config candidate.local.toml \
  --results-root results
```

`candidate-suite` shares curated orchestration with `candidate-smoke`. It loads
and validates all 126 frozen cases, verifies the existing dataset hash, and passes
the full dataset to `prepare_curated_plan()`. Before client construction it requires
exactly the 106 model-case IDs, in canonical order, with repetition index zero.
The plan contains 19 extraction, 21 context-summary, 23 voicemail, 27 QA evaluation,
10 QA conversation-summary, and 6 QA node-summary requests. The six-case smoke
selection and its behavior remain unchanged.

The existing local configuration rules apply: explicit numeric loopback endpoint,
one repetition, concurrency one, exploratory configuration, and the existing
`candidate-smoke` experiment-label convention. The command preserves every
configured generation/timeout setting, including the chosen token-limit field.
It does not start a server or adjust the model. Evaluation uses `purpose="candidate"`;
the configuration's exploratory designation remains unchanged. The full 106-entry
plan distinguishes its run identity from the six-case smoke. Existing manifest,
client, sequential runner, journal, and resume behavior are reused. A fresh run
sends 106 requests; resume skips finalized executions, including failures.

The 20 controls (7 adapter, 12 response-contract, 1 transport-contract) are evaluated
locally with `evaluate_control_case()`. They never enter candidate inference or
candidate quality/latency denominators. Model and control evaluations are published
as separate batches using the existing private artifact writer:

```text
results/<run_id>/manifest.json
results/<run_id>/journal.jsonl
results/<run_id>/evaluations/<batch_id>/automatic.jsonl
results/<run_id>/evaluations/<batch_id>/summary-automatic.json
results/control-evaluations/<control_batch_id>/automatic.jsonl
results/control-evaluations/<control_batch_id>/summary-automatic.json
```

Automatic metrics remain separated by task contract. Human checks remain pending,
including the existing negative-grounding control. Console output reports model
execution counts, per-contract transport counts, separate control outcomes, and
separate pending human-review counts. No LLM judge, global cross-task quality score,
or routing recommendation is added. Help/argument inspection does not execute
models. Tests exercise the command using synthetic responses and temporary results.

## Private sequential model comparison

The local operator script composes the existing smoke, suite, and ten-case replay
commands without changing their datasets, prompts, parsers, scoring, or journals:

```bash
.venv/bin/python scripts/compare_candidates.py run \
  --batch private/model-comparison-20260916
```

The ignored batch directory holds the pinned `models.json`, `freeze.json`,
per-model artifact checksums/native templates, local configurations, startup
probes, sampled server RSS, raw HTTP sidecars, and `progress.json`. It requires
the existing private canonical replay inputs and a separate installed MLX runtime.
Only one owned loopback server is loaded at a time. Completed phases are reused;
the existing immutable journal handles interrupted request execution. Blocked
models remain explicit and do not prevent subsequent independent candidates.
No model requests are made on script import. Download/probe/startup activity is
outside the benchmark's request latency measurements.

For Gemma with MLX-LM 0.31.3, the wrapper removes only the checkpoint's unused
shared-KV projection/norm tensors, matching Transformers' Gemma loader policy.
All remaining weights still undergo strict native loading. This compatibility
step and discarded tensor names are recorded privately; source weights are never
rewritten. Native text-only loading excludes vision/audio modules.

AI reviews are separate private evidence sidecars, not human approval records.
`scripts/review_candidates.py` only validates identities, output hashes, exact
quoted evidence, and coverage; it does not produce semantic judgments or call a
judge. Run it with the benchmark Python to list outstanding full-suite/replay
reviews. Human-review requirements and all automatic results remain intact.
