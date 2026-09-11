# OVP-36 OOB model benchmark

Standalone research benchmark. **Stages 1–2:** strict schemas, TOML configuration,
read-only JSONL dataset operations, versioned prompt contracts, and response
parsing. Model execution, task adapters, scoring, reports, and serving remain
unimplemented.

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
preserves exact captured model-facing messages, outputs, and task/span/provenance
IDs. Historical outputs are baseline behavior, not automatic gold. Curated and
adversarial requests will use source-aligned adapters. No replay loader or raw
private replay data is included in Stage 2.

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
top_p = 1.0
# seed = 7
```

`load_config(path)` uses `tomllib` without reading environment values.
`resolve_endpoint(config.endpoint)` resolves only named variables. A missing or
blank referenced variable raises an error. Literal `base_url`/`model` may be used
instead of their environment references, but not alongside them.

API keys can only be supplied through environment references. URLs containing
userinfo, query parameters, or fragments are rejected. Resolved endpoint values
are excluded from ordinary serialization and repr; `redact_config(config)` saves
the unresolved configuration. Keep local configuration and secrets out of Git.

The generation limit is explicit for every configuration. The approved runtime
summary contract uses **max_tokens=4000** and a **30-second timeout**; future
task-aware configuration/adapters will enforce these canonical settings. This
Stage 1 schema is task-neutral and does not infer a task from those numbers.
Historical Qwen voicemail used max_tokens=2048; the product service factory's
default cap depends on model identity. Neither fact assigns a Granite limit.

No Granite checkpoint, quantization, or serving runtime is selected. Their
configuration remains a decision before local serving. GB10/DGX access and
hardware measurements follow local validation in a later stage.

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
