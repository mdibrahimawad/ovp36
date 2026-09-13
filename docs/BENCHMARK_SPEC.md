# OVP-36 OOB Model Benchmark Specification

Status: Draft v0.1
Owner: Ibrahim Awad
Purpose: Research-only model evaluation for OVP-36 before any Olegos product integration

## 1. Objective

Build a standalone, reproducible benchmark and test server that evaluates whether IBM Granite can safely handle one or more Olegos out-of-band (OOB) LLM workloads with quality comparable to the current dialogue-model baseline and, later, with better serving characteristics on approved GB10/DGX research hardware.

The benchmark MUST answer two separate questions:

1. Quality gate: can the candidate model perform each OOB job correctly and in a production-compatible output format?
2. Hardware gate: on approved GB10/DGX research hardware, is the candidate materially attractive in latency, throughput, concurrency, and memory use?

Passing one gate does not imply passing the other.

The benchmark MUST make a per-job decision. It MUST NOT assume that one replacement model has to serve every OOB workload.

## 2. Scope

OVP-36 has four product-level OOB areas: variable extraction, runtime context
summarization, voicemail, and post-call QA. The benchmark distinguishes six
source-derived task contracts: `extraction`, `context_summary`, `voicemail`,
`qa`, `qa_conversation_summary`, and `qa_node_summary`. The latter two support QA;
these task values are benchmark taxonomy, not current product routing labels.

In scope:

- variable extraction
- runtime context summarization / context compaction
- voicemail detection / classification
- post-call QA
- QA-support conversation summaries
- QA node/script summaries where relevant to the same QA model path
- production-like prompt construction for each job
- production-like parsing and validation
- baseline-vs-candidate comparison
- local functional and quality testing
- later GB10/DGX performance testing
- reproducible result capture and reporting

Out of scope for this research repo:

- modifying olegos-api
- modifying olegos-mps
- modifying olegos-pipecat
- changing production tier tables
- changing user-facing configuration
- deploying into production or shared Olegos infrastructure
- starting model servers on shared/company GPU infrastructure without operator approval
- proving end-to-end OVP-36 routing/isolation inside Olegos
- claiming capacity improvement from token-share percentages alone

## 3. Source basis and fidelity rule

This specification is based on the current source snapshots supplied for:

- olegos-api develop
- olegos-mps develop
- olegos-pipecat develop
- olegos-docs develop
- frozen OVP-34 evidence and benchmark artifacts

Before product integration, source assumptions MUST be revalidated against current origin/develop because the product repositories move quickly.

Core fidelity rule:

> Reproduce the model-facing OOB contract, not the entire Olegos runtime.

The benchmark MUST emulate what the OOB LLM sees and what Olegos expects back. It SHOULD NOT reproduce workflow execution, telephony, STT, TTS, databases, Redis, Langfuse, MPS, or the live dialogue pipeline unless one of those components is strictly necessary to preserve the model-facing contract.

## 4. Why a standalone OOB laboratory

The research architecture should isolate model quality and serving behavior from unrelated product infrastructure.

Conceptual curated/adversarial flow (historical replay bypasses adapters):

benchmark case
-> task adapter
-> OpenAI-compatible model client
-> standalone model server
-> raw response
-> production-like parser
-> task-specific scorer
-> result record
-> aggregate report

The same benchmark runner MUST be able to target multiple endpoints without code changes, for example:

- local Granite server
- Granite on approved GB10/DGX research hardware
- current Qwen baseline endpoint if approved and available
- future candidate models

Endpoint/model selection should be configuration, not hard-coded task logic. For OVP-34 replay, the request path preserves captured historical trace messages unchanged; source-aligned task adapters construct requests for curated/adversarial cases.

## 5. Observed workload priorities from OVP-34

Frozen OVP-34 evidence observed these OOB jobs:

- variable extraction: 40 jobs
- runtime context summarization: 56 jobs
- voicemail detection: 22 jobs
- post-call QA: 72 jobs
- QA previous-conversation summaries: 52 jobs (the observed QA-support subtype)

Runtime context summarization was the largest measured live-window OOB decode component and therefore gets the deepest live-window quality coverage.

The benchmark MUST preserve the distinction between:

- runtime context summarization: llm-context-summarization
- QA rolling summaries: conversation-summary-before-<node>

They are separate OOB operations with different purposes and prompts.

## 6. Dataset strategy

Use three complementary data layers.

### 6.1 Layer A: frozen OVP-34 replay set

The final OVP-34 100-run telemetry has exactly **242 selected OOB observations**, each with captured historical trace input messages and an output: 40 extraction, 56 runtime summary, 22 voicemail, 72 post-call QA, and 52 QA previous-conversation summaries. These messages are the best available captured trace/model-facing evidence, not a blanket claim of verified historical HTTP wire equality.

For runtime context summaries, current tracing reconstructs the summary request
after generation; historical wire equality is not independently proven. Retain
all 56 observations with this qualification. Do not regenerate their messages or
automatically apply the same caveat to unrelated tasks. All 242 selected input
objects contain only `messages`; message-level tool fields must be preserved.

Use a private/local export and a separate manifest selecting those canonical observations. Replay the exact captured request message sequence, roles, and content, including already-rendered system messages; do not reconstruct or regenerate prompts from conversational text. Preserve historical outputs and task/span/provenance IDs. Keep raw/private replay material ignored and uncommitted. Missing local exports are availability problems, not a reason to invent replacement requests.

The runner/client design must preserve two distinct request paths:

- OVP-34 historical replay: captured trace messages preserved exactly -> send directly; DO NOT regenerate with task adapters.
- Curated/adversarial cases: Olegos-shaped task input -> source-aligned adapter -> generated model-facing messages.

Stage 3A implements separate captured/generated preparation paths and immutable
transport snapshots. Stages 3B–3D add the one-shot client, private persistence,
and sequential prepared-request runner. Stage 4A adds the historical replay loader
and plan preparation; task adapters remain unimplemented and strict curated
task-input schemas remain separate.

Stage 4A uses the verified `oob_jobs.csv` as the explicit selected-ID and canonical
case-order authority; raw JSONL supplies the captured messages. It validates all
raw rows and exact subtype counts without porting wrapper attribution or
deduplicating messages. Cases use `ovp34-` plus the complete 16-character lowercase
hexadecimal observation ID. Captured messages pass only through `MessageSnapshot`,
`CapturedReplayRequest`, and `prepare_captured_request()`. Historical outputs and
other original provenance remain external, retrievable by source observation ID.

Semantic replay dataset identity hashes the ordered projections containing exactly
`case_id`, `contract`, `source_observation_id`, and `message_fingerprint` through
the public `fingerprint_replay_dataset()` helper and versioned domain
`ovp34-replay-dataset`. Selected observations, contract mapping, message semantics,
and canonical order affect it. Historical output, unrelated telemetry, source
paths, candidate settings, and semantically irrelevant artifact formatting do not.
Raw/selection artifact SHA-256 values describe the exact bytes parsed, retained
as in-memory provenance; they are not semantic dataset identity inputs. Official
real-data acceptance checks the audited artifact hashes separately. The public
loader requires semantic/structural validity and counts, not official checksums.

`prepare_replay_plan(replay, *, config, resolved_model)` uses new candidate settings
from `config.generation` and the explicit resolved candidate model. It preserves
the existing single token-limit-field normalization and expands repetition-major:
all CSV-ordered cases at repetition zero, then all cases at repetition one, etc.
The caller computes the ordered execution-plan hash using the existing public
helper. Stage 4A performs no model calls, journal operations, automatic manifest
creation, scoring, or serving. It creates no curated/adversarial cases.

Historical baseline output is reference behavior, NOT automatically ground truth.

For each replay case, distinguish:

- baseline_output: output produced by the current/baseline model
- gold/expected: independently defined expected behavior where annotation is possible
- candidate_output: output produced by Granite

### 6.2 Layer B: curated correctness set

Create deterministic, human-reviewable scenarios with explicit expected outputs or scoring annotations.

This set is the primary source for objective accuracy claims.

### 6.3 Layer C: adversarial/edge set

Create difficult cases specifically intended to expose production-relevant failures, including:

- missing information
- user corrections
- contradictions
- irrelevant details
- long contexts
- ambiguous wording
- tool responses
- repeated facts
- misleading keywords
- malformed-looking content
- output-format pressure
- facts appearing only once or far back in context

### 6.4 Stage 4B implementation boundary

Stage 4B1 implements source-aligned renderers and curated-dataset infrastructure,
using small synthetic test-local fixtures. Stage 4B2 authors, validates, and
human-reviews the final six JSONL files only after 4B1 external review.

The frozen inventory is 126 scenarios: 106 model exercises and 20 controls.
Per-contract total/model/control counts are 26/19/7 extraction, 26/21/5 runtime
summary, 26/23/3 voicemail, 32/27/5 QA, 10/10/0 prior summary, and 6/6/0 node summary.
Controls are inventory evidence, never model requests, candidate latency samples,
or candidate quality denominator entries. Stage 4C reports them separately.

Curated validation reuses BenchmarkCase and hash_dataset. It enforces lowercase
IDs (for example curated-ex-001), reviewed matrix metadata and canonical order,
contract/input/gold consistency, and evidence resolution relative to case.input.
No general repr change or new curated hash domain is introduced. The only message
role extension is developer for the source async-tool completion boundary.

The complete loader reads six explicit task-named JSONL files in extraction,
context_summary, voicemail, qa, qa_conversation_summary, qa_node_summary order,
with ascending matrix ordinals in each. No final files are created during 4B1.
prepare_curated_plan filters model exercises in this relative order and expands
repetition-major. Every request uses the explicit resolved model and exactly
config.generation; no generation_overrides interface or silent source-cap override
exists. Prepared settings and order remain bound by existing request/plan identity.

All six generated paths call prepare_generated_request. Historical replay stays
on its unchanged captured path. No model execution, scoring, journal creation,
server operation, or numerical acceptance policy is added by 4B1. Synthetic
origin and the correctness of authored gold require human review before Granite.

## 7. Common case schema

Each immutable benchmark case SHOULD contain at least:

- id
- task
- source: ovp34_replay | curated | adversarial
- difficulty: easy | normal | hard | adversarial
- critical: boolean
- input payload
- expected/gold annotations
- tags describing the scenario
- notes/rationale
- source provenance where applicable

Execution results MUST be stored separately from cases and SHOULD contain at least:

- run_id
- case_id
- timestamp
- model identifier
- endpoint identifier/alias
- model/server configuration
- raw output
- parsed output
- prompt/input token count when reported
- completion/output token count when reported
- total token count when reported
- request latency
- time to first token when measurable
- decode duration / tokens-per-second when measurable
- parser status
- task-specific scores
- error/timeout status
- retry count

Input cases MUST NOT be overwritten by results.

Stage 3C implements the raw transport-evidence subset of this eventual result
view, without timestamps, parser output, scores, or reports. An explicitly
supplied results root contains `<run_id>/manifest.json` and
`<run_id>/journal.jsonl`. The immutable manifest uses only safe identity/config
evidence; resume requires exact equality with an expected manifest rebuilt from
authoritative inputs through public Stage 3A functions. It also requires the
stored run ID to match the directory. Existing manifests are never rewritten.

The private ignored journal preserves new candidate output exactly for later
parsing/scoring, even when the candidate repeats private material. Historical
request messages, baseline outputs, provenance payloads, endpoints, credentials,
headers, and raw HTTP material are not duplicated into persistence. Provider
finish reason/model metadata is retained only under the existing safe-label
validators; rejected non-null values become null with field-name omission markers
in fixed order: `finish_reason`, then `response_model`.

## 8. Variable extraction contract

### 8.1 Current production-shaped behavior

Current API source builds a normalized conversation history containing:

- assistant messages
- user messages
- selected tool responses

Tool-response handling currently:

- excludes transition responses matching {"status": "done"}
- prefers a JSON object's `data` field when present
- otherwise strips wrapper keys such as status/status_code
- truncates a single tool response above 2000 characters
- labels retained tool responses with the tool name

The extraction request includes:

- a system instruction stating that the task is structured extraction
- a requirement to return only a valid JSON object with requested variables as top-level keys
- optional node-specific extraction instructions
- variable name, source-rendered enum type, and per-variable hint
- conversation history

The user message begins with exactly two newlines before `Variables to extract:`. Source variable lines use `VariableType.string` (observed in OVP-34), with `VariableType.number` and `VariableType.boolean` implied by current enum formatting. Optional extraction instructions are appended after exactly two newlines to the base system prompt. The frozen SkyAssist instruction and variable hints are recorded verbatim in `SOURCE_AUDIT.md`.

Supported extraction variable types in the current API DTO are:

- string
- number
- boolean

Production parsing is tolerant: the API parser first attempts strict JSON, then handles markdown code blocks, then attempts to extract a JSON object/array from surrounding text. If parsing fails it returns a raw fallback.

### 8.2 Extraction benchmark families

At minimum cover:

- all fields present
- one field missing
- multiple fields missing
- all fields missing
- user corrects origin
- user corrects destination
- user corrects date/value
- multiple corrections
- old and new values both present
- irrelevant cities/names/numbers present
- airport code vs city name
- relative dates preserved as stated where instructed
- string field
- number field
- boolean field
- mixed-type schema
- tool response supplies relevant fact
- tool response supplies irrelevant fact
- tool response contains wrapper metadata
- long tool response / truncation-shaped input
- long conversation
- heavy small talk
- requested field never appears
- JSON-sensitive characters
- model attempts to infer a missing value
- duplicate key/extra key behavior
- malformed or non-JSON model output

Use the SkyAssist-style origin/destination/travel_date scenario as one production-shaped subset, but do not restrict the benchmark to only flight-search fields.

### 8.3 Extraction metrics

Report at least:

- strict JSON validity rate
- production-parser validity rate
- required-key coverage
- unexpected-extra-key rate
- field-level accuracy
- whole-object exact match
- type correctness
- missing/unknown correctness
- hallucinated-value rate
- correction/latest-value accuracy
- critical-case pass rate

Strict JSON and production-parser validity MUST be separate metrics.

## 9. Runtime context summarization contract

### 9.1 Current production-shaped behavior

Current API source runs background context summarization on eligible node transitions when context compaction is enabled.

Current manager configuration:

- target_context_tokens = 4000
- min_messages_after_summary = 2
- summarization_timeout = 30 seconds
- skips summarization when context message count <= 6

Runtime summary supplies the one-shot Python argument `max_tokens=4000`.
Current API service-factory settings also carry a default `max_tokens` cap;
`LLM_MAX_TOKENS_CAP_REASONING` defaults to 2048 for reasoning models such as Qwen3
and is environment-overridable. Speaches receives it through
`SpeachesLLMSettings(model=model, max_tokens=max_tokens)`.

Newer supplied Pipecat builds both token-limit keys, sets
`max_completion_tokens=4000` because that key exists, and does not clear an
already configured service `max_tokens`. With Qwen source defaults, the
constructed parameter dictionary can contain both `max_tokens=2048` and
`max_completion_tokens=4000` before SDK/provider handling. All 56 historical
summary observations have empty `modelParameters` (`{}`); they do not establish
the exact historical wire token-limit fields.

The benchmark intentionally normalizes candidate requests to exactly one selected
limit field to avoid ambiguous dual-limit behavior. This does not claim
byte-for-byte reproduction of the product's provider parameter dictionary.
Product settings remain provenance for later per-task candidate configuration;
Granite's canonical field/value is not chosen here.

The current Pipecat default summarization prompt asks the model to preserve:

- key facts, decisions, and agreements
- context needed to continue the conversation
- user preferences and requirements
- unresolved questions and action items

It asks the model to omit greetings, small talk, redundant information, and resolved tangents.

The runtime path keeps recent messages outside the generated summary and reinserts the summary into the LLM context while preserving the current system message.

The benchmark MUST distinguish quality of the generated summary from correctness of the surrounding product context-rewrite algorithm. The research server is evaluating the model, not reimplementing the full engine.

### 9.2 Runtime summary benchmark families

At minimum cover:

- short eligible context
- medium context
- long context
- very long context near practical model limits
- multiple topics
- facts scattered throughout the conversation
- one critical fact appearing only once
- critical fact very early in context
- user preference retention
- decision/agreement retention
- unresolved question retention
- action-item retention
- irrelevant small talk
- repeated information
- contradictory statements
- user correction/superseded value
- multiple corrections
- tool call/result information represented in context
- resolved tangent that should be omitted
- unresolved tangent that should remain
- immediate recent context that would normally remain unsummarized

Also include boundary/behavior cases representing the product semantics:

- <=6 messages: no-summary control case for simulator logic
- >6 messages: eligible case
- timeout/error handling at the benchmark/server layer

### 9.3 Runtime summary annotations

Each curated summary case SHOULD define:

- required_facts
- critical_facts
- superseded_facts
- forbidden_facts
- user_preferences
- decisions
- unresolved_questions
- action_items
- information intentionally safe to omit

### 9.4 Runtime summary metrics

Report at least:

- critical fact recall
- overall required fact recall
- forbidden/superseded fact inclusion rate
- hallucination rate
- contradiction rate
- latest-value/correction accuracy
- preference retention
- unresolved-question retention
- action-item retention
- compression ratio
- output length/tokens
- critical-case pass rate

Lexical metrics such as ROUGE MAY be included as secondary diagnostics, but MUST NOT be used as the primary correctness gate.

## 10. Voicemail detection contract

### 10.1 Current production-shaped behavior

Current API creates a separate LLM instance for the voicemail sub-pipeline when voicemail detection is enabled. If configured to use the workflow LLM, that service is created from the workflow configuration.

Current Pipecat classifier expects the model response to contain one of two labels:

- CONVERSATION
- VOICEMAIL

The full default outbound-classifier system prompt is frozen in `SOURCE_AUDIT.md`. Its input is a message context: the historical requests include user, assistant + tool, assistant + tool + user, and user + user sequences after the system instruction. Do not collapse these into a single transcript string.

The detector can also be configured with a long-speech timeout; current API default lookup is 8 seconds.

### 10.2 Voicemail benchmark families

At minimum cover:

Human conversation:

- Hello?
- Hi
- name speaking
- Who is this?
- Can I help you?
- angry human response
- confused human response
- very short human response
- human says the word voicemail
- human asks whether the caller left a voicemail
- human references a previous voicemail

Voicemail/automated:

- classic personal voicemail greeting
- professional voicemail greeting
- leave-name-and-number instruction
- mailbox full
- mailbox not set up
- number not in service
- carrier error/availability message
- office closed/business hours message
- long voicemail greeting
- short automated greeting

Ambiguous/adversarial:

- truncated transcript
- partial first-turn transcript
- automation that sounds conversational
- human speech that sounds scripted
- voicemail greeting containing a question
- noisy/garbled text representation

### 10.3 Voicemail metrics

Report at least:

- accuracy
- VOICEMAIL precision/recall/F1
- CONVERSATION precision/recall/F1
- false-human-as-voicemail rate
- false-voicemail-as-human rate
- exact-format rate
- production-decision parse rate
- critical-case pass rate

False-human-as-voicemail MUST be reported separately because it may terminate a real human call.

## 11. Post-call QA contract

### 11.1 Current production-shaped behavior

Current API can run QA per node and fall back to whole-call QA when node IDs are unavailable.

For per-node QA it constructs:

- node-specific transcript with timestamps
- precomputed metrics
- a summary of the node's intended purpose/script
- a rolling summary of prior-node conversation
- the configured QA system prompt

The QA LLM receives a user message containing the current transcript and a rendered system instruction containing the QA context. The full frozen SkyAssist system template is documented in `SOURCE_AUDIT.md`, including its literal double-brace placeholders. Historical replay retains the already-rendered system messages. The Node Purpose section was empty in all 72 historical OVP-34 QA observations; this is not a general production constraint.

The current SkyAssist QA shape expects JSON containing:

- tags
- overall_sentiment
- call_quality_score
- summary

Production parsing is tolerant through the same JSON parser and non-dict top-level outputs are effectively treated as unusable/empty QA results.

### 11.2 QA benchmark families

Build gold cases for:

- clean call / no tags
- unclear conversation
- assistant loop/repetition
- improper assistant reply
- frustrated user
- user not understanding
- hearing issues
- dead air represented through transcript/metrics
- user requesting unsupported feature
- assistant lacks empathy
- user detects AI
- multiple simultaneous QA tags
- negative wording without actual frustration
- one isolated failure inside otherwise good call
- evidence only in previous-node context
- evidence only in current-node transcript
- conflicting evidence
- insufficient evidence
- long transcript with sparse failure
- tool-call context relevant to QA

### 11.3 QA gold annotations

Each QA case SHOULD define:

- expected tag set
- forbidden tag set where useful
- expected sentiment
- acceptable quality-score range or target
- evidence spans/facts supporting each expected tag
- expected summary facts

### 11.4 QA metrics

Report at least:

- valid/production-parseable JSON rate
- tag micro precision/recall/F1
- tag macro precision/recall/F1
- per-tag precision/recall/F1
- false-positive tag rate
- false-negative tag rate
- sentiment accuracy
- quality-score MAE or distance-to-acceptable-range
- evidence-grounding accuracy
- QA summary factuality/required-fact recall
- critical-case pass rate

Do not score Granite only by agreement with historical Qwen output. Compare both candidate and baseline to gold where gold exists.

## 12. QA-support summary contracts

### 12.1 Rolling previous-conversation summary

Current QA code sends the prior transcript as a user message prefixed by `## Conversation` and uses a concise QA-oriented conversation-summary system prompt.

Benchmark:

- state carried across nodes
- previous failures retained
- user constraints retained
- prior decisions retained
- irrelevant detail compressed
- contradictions/corrections handled
- no hallucinated events

Metrics:

- required fact recall
- critical fact recall
- hallucination rate
- correction/latest-value accuracy
- compression ratio
- usefulness for downstream QA

### 12.2 Node/script summary

Current QA code can summarize agent/start nodes from:

- node name
- agent prompt
- custom tool descriptions
- outgoing edge labels/conditions

The node-summary system prompt asks for a concise 2-4 sentence description of purpose, expected behaviors, and important nuances for later QA.

Benchmark cases should vary:

- simple node
- complex node
- node with tools
- node with many outgoing transitions
- node with safety/negative constraints
- node where a subtle requirement is essential for QA

Metrics:

- purpose retention
- key behavior retention
- tool/transition retention where relevant
- prohibited-behavior retention
- hallucination rate
- concise length compliance

Node summaries may be cached in product, so their operational frequency is different from rolling conversation summaries; keep reporting separate.

## 13. Task adapter architecture

Implement one adapter per logical OOB contract:

- ExtractionAdapter
- RuntimeContextSummaryAdapter
- VoicemailAdapter
- QAEvaluationAdapter
- QAConversationSummaryAdapter
- QANodeSummaryAdapter

Each adapter is responsible for:

1. validating the benchmark case input
2. constructing production-shaped system/user messages
3. supplying task-specific generation parameters when required
4. handing a prepared request to the runner, which invokes the common model client
5. applying production-like parsing
6. returning a normalized task result for scoring

Adapters MUST NOT contain candidate-specific hacks.

Adapters SHOULD expose prompt/version metadata so a result can be traced back to the exact benchmark contract.

## 14. Common model client

Use one configurable OpenAI-compatible chat-completions client where feasible.

Required configuration SHOULD include:

- base URL
- model name
- API key/token if needed
- timeout
- temperature
- max tokens
- optional seed if supported
- concurrency

Keep historical/source generation evidence separate from candidate configuration:
runtime summary supplies the Python argument `max_tokens=4000`, while current
source can retain both provider limit fields as described in section 9.1.
Historical summary wire limits are unverified; historical Qwen voicemail exposes
`max_tokens=2048`. Service-factory defaults depend on model identity and environment.
Do not infer Granite settings from these observations. Freeze candidate settings
before canonical evaluation.

The runner MUST be endpoint-driven so local and GB10/DGX runs use the same benchmark logic.

No company endpoint, credential, or secret may be committed to the repository.

## 15. Standalone test server

The research server should expose the minimum serving functionality needed by the benchmark. Prefer an OpenAI-compatible `/v1/chat/completions` surface if supported by the selected Granite serving stack.

The research server is NOT an Olegos API clone.

It should support:

- model load/start
- chat completion requests
- deterministic/low-temperature quality runs
- token usage when available
- request/error logging
- health/readiness check
- configuration through CLI/env/config file

Later GB10/DGX deployment should reuse the same serving interface.

## 16. Runner behavior

The benchmark runner MUST support:

- selecting one task or all tasks
- selecting one dataset layer or all layers
- filtering by tags/difficulty/criticality
- deterministic run IDs
- configurable endpoint/model
- sequential quality mode
- controlled concurrency performance mode
- any future bounded transport-retry policy requires separate review and changed harness/run identity; Stage 3D v1 has no automatic retries
- timeout classification
- resume without overwriting prior results
- machine-readable JSONL result output
- aggregate report generation

A failed request is a benchmark result, not something silently dropped.

Stage 3C supplies persistence; Stage 3D v1 implements sequential prepared-request
execution with `MAX_ATTEMPTS = 1`, an explicit benchmark normalization rather
than a claim of product retry fidelity. Each fresh execution calls `send_once()`
once, durably appends attempt zero, and finalizes it for every returned status.
`retryable` remains unchanged evidence only. There are no retries, delays,
backoff, jitter, or artificial cancellation sleeps. Per-execution
`attempt_index` is contiguous from zero and separate from `repetition_index` and
`ExecutionKey`. A `finalized` event references the latest attempt. Only finalized
keys are completed, including finalized failures; unfinished evidence is retained.
Persistence does not add new Stage 3B result cross-field invariants.

Before opening journal artifacts, Stage 3D snapshots plan order, validates strict
prepared execution items and repetition bounds, computes actual request hashes
and execution keys, rejects duplicates, requires request model equality with
the manifest requested model and concurrency exactly one, and checks the full
ordered execution-plan hash. The public `fingerprint_execution_plan()` uses the
existing versioned `execution-plan` hash domain over exactly `case_id`,
`request_fingerprint`, and `repetition_index`, preserving list order. Empty plans
must match the hash of `[]`. No separate message hash validation is required.

After opening, all persisted keys must belong to the full supplied plan before
any dispatch. Finalized executions are skipped, including failures and valid
multi-attempt histories. Exactly one unfinished attempt is finalized without
resending regardless of status; more than one unfinished attempt is incompatible
and raises safe `RunnerValidationError` before dispatch. Append/finalization
failures stop immediately. Reopening after uncertain writes trusts strict journal
state, whether finalized, unfinished, or corrupt. Client bugs and cancellation
propagate without synthetic evidence. A crash before durable evidence can still
cause resend on restart.

The caller owns the injected client; the runner owns and always closes its journal.
Client URL/alias correspondence, timeout, and physical-server identity remain
caller composition responsibilities; the runner does not inspect private client
or journal fields. `RunSummary` has only strict nonnegative counts `planned`,
`skipped_completed`, `resumed_unfinished`, `attempts_sent`, `executions_finalized`,
`successes`, and `failures`. Counts are per invocation, with
`planned == skipped_completed + executions_finalized` and
`executions_finalized == successes + failures`; they are not quality metrics.
No summary file is persisted. New translated validation errors have safe text
and neither exception cause nor context. Stage 3D adds no dataset loading,
adapters, parsing, scoring, reports, quality gates, or serving.

Journal reading fails closed on corrupt framing, JSON/UTF-8/schema violations,
duplicate keys, non-finite numbers, or invalid event sequences. Even valid JSON
without a final newline is corruption. There is no automatic tail repair. All
persistent outer JSON is standard JSON; parsed/scored structures remain deferred.

The local POSIX implementation supports one sequential writer per run. It rejects
group/other access and insufficient owner permissions, without requiring exact
owner modes or changing existing permissions. Directory creation fsyncs its
parent. Manifest publication uses a file-fsynced private temporary file and a
create-only hard link, with run-directory fsync after publication and temporary
name cleanup. Journal appends fsync the file and, on first creation, the run
directory. Uncertain write/fsync failures poison the writer until close/reopen
and strict reading. This does not guarantee exactly-once inference, HTTP/file
transactionality, universal power-loss durability, or concurrent-writer safety.

## 17. Repetition and determinism

Stage 3A provides pure versioned SHA-256 identity primitives and an explicit safe
persistent configuration projection. `EndpointConfig.alias` is operator-declared
experiment identity, validated as a bounded identifier. Server metadata, persisted
model names, and experiment labels are bounded nonsecret metadata (policy in
README). Literal endpoint URLs and URL-derived hashes are never persisted or
included in identity. API keys, environment references/values, and timestamps are
excluded. Legacy `redact_config()` is not suitable for persistence.

Canonical configuration requires at least one useful declared server identity:
checkpoint, revision, artifact hash, chat-template hash, or runtime/version pair.
Exploratory configuration permits unknown metadata. Operators must update safe
declared identity when materially changing servers; changing only the runtime URL
does not change run ID. This guard cannot prove physical-server equivalence.

Generation configuration retains the explicit numeric `max_tokens` value and
adds `token_limit_field` (`max_tokens` or `max_completion_tokens`). The former is
the backward-compatible harness default, not a Granite quality decision. Prepared
requests contain exactly one completion-limit key, optional seed, and stream=False.
This is explicit candidate request normalization to avoid ambiguous dual-limit
behavior, not byte-for-byte reproduction of the product's provider parameter
dictionary. Product runtime-summary 4000 remains source provenance, not a
universal limit or a selected Granite canonical field/value.

Stage 3D does not require per-request generation to equal manifest generation.
Manifest generation is run-level declared/default metadata. Actual prepared
settings are preserved and identity-bound through request fingerprint → ordered
execution-plan hash → run ID. The run still requires one declared requested model.

Quality runs SHOULD default to deterministic or near-deterministic generation where the server/model supports it.

For critical tasks, support repeated-run stability checks. Record disagreement rate across repeats.

Do not mix stochastic stability experiments with the canonical quality score without clearly labeling them.

## 18. Baseline comparison

Where a current Qwen baseline endpoint or frozen outputs are available, report:

- candidate vs gold
- baseline vs gold
- candidate regression/improvement relative to baseline
- candidate vs historical baseline output as a diagnostic only

Historical baseline output MUST NOT be treated as unquestionable truth.

## 19. Quality gates

Exact thresholds should be frozen before the canonical Granite evaluation to avoid moving the goalposts.

Initial gate structure:

Extraction:
- zero critical hallucinations
- very high production-parser validity
- no material regression in field accuracy vs baseline

Runtime summary:
- no critical-fact regression
- zero/near-zero critical hallucinations
- correction/latest-value behavior at least baseline-comparable
- useful compression

Voicemail:
- no unacceptable human-as-voicemail errors
- classification F1 at least baseline-comparable
- production-compatible label output

QA:
- tag F1 and critical-tag behavior baseline-comparable
- no unacceptable false-pass/false-fail pattern
- valid structured output

QA-support summaries:
- critical information retention baseline-comparable
- no material hallucination regression

Final numeric thresholds MUST be documented before official results are generated.

## 20. Local phase

Local testing is for:

- server correctness
- model loading feasibility
- prompt/adapter correctness
- parser correctness
- benchmark-runner correctness
- qualitative/quantitative task quality
- token/output-length behavior
- rough functional latency only

Local laptop results MUST NOT be presented as evidence of GB10/DGX production serving suitability.

## 21. GB10/DGX research phase

After the standalone system is runnable and local quality evidence exists, request approved research compute access from Timur.

Run the same benchmark artifacts against Granite on the approved research namespace/hardware.

Measure at least where technically available:

- end-to-end request latency p50/p95
- TTFT p50/p95 for streamed measurements
- decode tokens/sec
- aggregate tokens/sec
- requests/sec
- concurrency scaling
- memory/unified-memory usage
- model load/startup time
- timeout rate
- server error rate
- OOM/failure behavior

Suggested initial controlled concurrency levels:

- 1
- 2
- 4
- 8

Adjust only to the safe limits of the approved environment.

Do not start extra CUDA/vLLM processes on shared Olegos serving hardware without explicit operator approval.

## 22. Performance methodology

Separate quality and performance modes.

Quality mode:

- low/zero temperature
- sequential or low concurrency
- complete raw outputs retained
- emphasis on correctness

Performance mode:

- fixed representative request sets
- warmup separated from measured iterations
- concurrency explicitly controlled
- server/model configuration frozen and recorded
- p50/p95 reported, not averages alone
- failures and timeouts included in the report

Do not infer GPU capacity improvement from the OVP-34 16.14% live-window OOB decode-token share. Isolation/token relocation and measured hardware capacity are different claims.

## 23. Reporting

Generate both machine-readable and human-readable outputs.

Machine-readable:

- JSONL per request/result
- aggregate JSON/CSV summary

Human-readable report should include:

- environment and model configuration
- dataset version/hash
- prompt/adapter version
- case counts by task/source/difficulty
- task-specific quality metrics
- critical failures
- baseline comparison
- latency/performance metrics where applicable
- errors/timeouts/OOMs
- limitations
- per-job recommendation

## 24. Per-job decision matrix

The final decision MUST be per workload, for example:

| Job | Quality Gate | GB10 Gate | Decision |
|---|---|---|---|
| Extraction | PASS/FAIL | PASS/FAIL | candidate / keep baseline |
| Runtime summary | PASS/FAIL | PASS/FAIL | candidate / keep baseline |
| Voicemail | PASS/FAIL | PASS/FAIL | candidate / keep baseline |
| QA | PASS/FAIL | PASS/FAIL | candidate / keep baseline |
| QA support summaries | PASS/FAIL | PASS/FAIL | candidate / keep baseline |

A failure on one job MUST NOT automatically reject Granite for every other job.

## 25. Proposed repository structure

```text
ovp36-oob-model-benchmark/
├── README.md
├── pyproject.toml
├── docs/
│   ├── BENCHMARK_SPEC.md
│   ├── SOURCE_AUDIT.md
│   ├── LOCAL_RUNBOOK.md
│   └── GB10_RUNBOOK.md
├── configs/
│   └── models/
├── cases/
│   ├── extraction/
│   ├── context_summary/
│   ├── voicemail/
│   ├── qa/
│   ├── qa_conversation_summary/
│   └── qa_node_summary/
├── src/
│   └── ovp36_benchmark/
│       ├── client.py
│       ├── runner.py
│       ├── schemas.py
│       ├── adapters/
│       ├── metrics/
│       └── reporting/
├── tests/
└── results/              # gitignored
```

The exact tree can change during implementation review, but separation of cases, adapters, client, scoring, tests, and generated results should be preserved.

## 26. Engineering requirements

- Python 3.12 unless research-server constraints require a documented exception.
- Typed code for public/internal benchmark interfaces.
- Pydantic/dataclass schemas for case/result configuration where useful.
- No secrets in source control.
- No hard-coded company infrastructure addresses.
- Clear error taxonomy.
- Deterministic fixture tests for parsers/scorers.
- Unit tests for each adapter.
- Golden tests for prompt construction where practical.
- Results directory ignored by Git.
- Raw model output always retained for auditability.
- Every aggregate score must be reproducible from saved per-case results.
- Benchmark should fail loudly on invalid case schema instead of silently skipping.
- Version/hash the benchmark dataset and prompt contracts.

## 27. Required tests before first model evaluation

At minimum:

- case schema validation
- extraction prompt golden test
- extraction production-like parser tests
- context-summary prompt golden test
- <=6-message no-summary simulator behavior test
- voicemail prompt/label parser tests
- QA prompt rendering test
- QA non-dict/malformed output test
- QA-support summary prompt tests
- model client timeout/error tests
- runner resume/no-overwrite test
- scorer deterministic tests
- report aggregation test

## 28. Safety and interpretation traps

Do not:

- equate OOB token share with GPU capacity savings
- count QA rolling summaries as runtime context summarization
- treat historical Qwen output as ground truth
- optimize prompts specifically for Granite in a way the baseline does not receive
- use generic academic benchmarks as the primary OVP-36 quality evidence
- hide malformed output behind the tolerant parser; report strict and tolerant success separately
- silently discard failed requests
- compare local Mac performance directly to GB10/DGX performance
- start GPU serving processes on shared Olegos infrastructure without approval
- integrate Olegos before the research quality/hardware gates justify it

## 29. Implementation order

Phase 0 - freeze documentation

1. SOURCE_AUDIT.md: exact model-facing contracts from current Olegos source.
2. BENCHMARK_SPEC.md: this document, reviewed and accepted.
3. CASE_MATRIX.md or equivalent: enumerated test families and target counts.
4. IMPLEMENTATION_PLAN.md: exact modules, interfaces, tests, and milestones.

Phase 1 - benchmark skeleton

5. case/result schemas
6. common model client
7. adapter interfaces
8. result writer and basic runner
9. unit tests

Phase 2 - task implementations

10. extraction adapter + cases + scorer
11. runtime summary adapter + cases + scorer
12. voicemail adapter + cases + scorer
13. QA adapter + cases + scorer
14. QA-support summary adapters + cases + scorers

Phase 3 - standalone Granite serving

15. minimal local server/run configuration
16. smoke test
17. canonical local quality run
18. baseline comparison where available

Phase 4 - GB10/DGX research

19. request approved compute access once the standalone system is runnable
20. deploy/run only in the research namespace/environment approved by Timur
21. canonical hardware benchmark
22. per-job decision report

Phase 5 - integration decision

23. only jobs that pass both quality and hardware gates become OVP-36 integration candidates
24. product routing/integration is designed separately against current product source

## 30. Definition of done for the research benchmark

The research benchmark is ready for a GB10/DGX request when:

- the repo can be installed from documented instructions
- the standalone model endpoint can be started locally or an endpoint can be configured
- all task adapters are implemented or the explicitly approved subset is implemented
- curated cases have gold annotations
- parser/scorer tests pass
- a local benchmark run produces immutable raw results and aggregate reports
- prompt/config/model metadata are captured
- no company secrets/infrastructure addresses are committed
- local limitations are documented

The research phase is complete when:

- canonical quality results are available per job
- canonical approved-hardware results are available per job
- failure cases are documented
- a per-job candidate/keep-baseline recommendation is produced
- claims are limited to what was actually measured
