# OVP-36 OOB model benchmark

Standalone research benchmark. **Stages 1–4B2:** strict schemas, TOML configuration,
read-only JSONL dataset operations, versioned prompt contracts, and response
parsing, immutable request preparation, pure identity primitives, and a common
one-shot AsyncOpenAI client, immutable run manifests, private result journals,
and a sequential prepared-request runner with frozen OVP-34 replay loading and
plan preparation. Stage 4B1 adds source-aligned curated adapters, inventory
validation, model-only plan preparation, and review projections. Stage 4B2 adds
the fixed 126-case synthetic curated/adversarial inventory and its review projection. The literal fixtures and draft semantic annotations still
require external human review before candidate runs. Scoring, reports, and serving
remain unimplemented; no candidate evaluation has been run.

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
synthetic cases in temporary directories and need no network, model, credentials,
or external services.

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

This example describes the schema; there is no benchmark CLI or dataset-to-runner wiring yet:

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
