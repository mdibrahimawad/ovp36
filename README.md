# OVP-36 OOB model benchmark

Standalone research benchmark. **Stages 1–3C:** strict schemas, TOML configuration,
read-only JSONL dataset operations, versioned prompt contracts, and response
parsing, immutable request preparation, pure identity primitives, and a common
one-shot AsyncOpenAI client, immutable run manifests, and private result journals.
Runner, task adapters, scoring, reports, and serving remain unimplemented.

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
aggregation and execution are deferred.

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
  after the frozen system instruction. Prepending that instruction belongs to
  the future adapter.
- The three summary gold types share one fact-annotation schema. Named categories
  reference required-fact IDs; correction pairs reference old/new facts.
- QA gold uses a score target or an ordered acceptable range. Tags remain strings
  so a later frozen prompt contract can validate its vocabulary.
- Evidence offsets are half-open character spans. Structural validation checks
  pointers/offsets; resolving them against inputs belongs to fixture validation
  when actual curated fixtures are added.

`load_cases(path_or_paths)` reads UTF-8 JSONL into a tuple of cases. It rejects
invalid/blank lines, duplicate object keys, non-standard numeric constants,
invalid schemas, and duplicate IDs, including across files. Errors identify the
source file and line without echoing the raw fixture.

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
previous-conversation summaries. Future replay reads a private local export and
preserves captured historical trace messages, outputs, and task/span/provenance
IDs. These messages are the best available captured trace/model-facing evidence,
not independently verified historical HTTP bytes. For the 56 runtime summaries,
current tracing reconstructs the summary request after generation; historical
wire equality is not independently proven. Keep those observations and their
qualification; never regenerate them. Historical outputs are baseline behavior,
not automatic gold. Curated/adversarial requests will use source-aligned adapters;
historical replay bypasses adapters. No replay loader or private replay data is
included.

## Configuration

This example describes the schema; there is no executable benchmark runner yet:

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
408/429/500/502/503/504 and connection/timeouts are marked retryable for a future
runner; redirects and other HTTP statuses are not retried.

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
Stage 3D execution/retry orchestration, adapters, and replay loading remain
unimplemented.

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
Unfinished attempts remain readable. The later runner must durably record each
returned result before deciding whether to retry or finalize; Stage 3C makes
neither decision and adds no cross-field constraints to Stage 3B result semantics.

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
requiring their future adapters, including QA substitution. History formatting
and runtime-summary selection/application also remain deferred. Partial-contract
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
