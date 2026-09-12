# OVP-36 OOB model benchmark

Standalone research benchmark. **Stages 1–3A:** strict schemas, TOML configuration,
read-only JSONL dataset operations, versioned prompt contracts, and response
parsing, plus immutable request preparation and pure identity primitives. Model
client, runner, result persistence, task adapters, scoring, reports, and serving
remain unimplemented.

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

The uv lockfile pins runtime dependencies; the pip alternative resolves the
version range from `pyproject.toml`. Tests use synthetic cases in temporary
directories and need no network, model, credentials, or external services.

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
Standards-compliant result-file encoding is a future reporting decision; no
result writer is implemented here.

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
