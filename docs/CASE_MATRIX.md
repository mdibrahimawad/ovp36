# OVP-36 OOB Model Benchmark — Case Matrix

Status: Draft v0.1 — freeze before implementation  
Owner: Ibrahim Awad  
Purpose: Enumerate the benchmark scenarios that must exist before Codex implements the runner/server.

## 1. Rules for this matrix

This matrix is derived from `BENCHMARK_SPEC.md` and `SOURCE_AUDIT.md`.

The implementation must not silently add, remove, or redefine benchmark semantics. If a case needs to change after implementation starts, update this document first.

Each case will eventually become an immutable data fixture. Model outputs and scores must be stored separately.

The matrix has two complementary parts:

1. **Curated/adversarial cases** — explicit human-defined expected behavior.
2. **OVP-34 replay cases** — captured historical trace messages and outputs from the final OVP-34 telemetry, preserved exactly. All 242 selected observations have both. Historical Qwen output is baseline behavior, not automatically ground truth; curated/adversarial requests instead come from source-aligned adapters.

OVP-36 has four product-level OOB areas and six source-derived benchmark task
contracts: `extraction`, `context_summary`, `voicemail`, `qa`,
`qa_conversation_summary`, `qa_node_summary`. These are benchmark taxonomy, not
current routing labels; the two supporting summaries belong to QA.

## 2. Common case fields

Every concrete case fixture should include:

- `id`
- `task`
- `source`: `curated`, `adversarial`, or `ovp34_replay`
- `difficulty`: `easy`, `normal`, `hard`, or `adversarial`
- `critical`: boolean
- `input`
- `expected`
- `tags`
- `notes`
- `provenance` when applicable

Every result record should include:

- `run_id`
- `case_id`
- `model`
- `endpoint_alias`
- `safe_server_metadata` (allowlisted, operator-declared server identity)
- `raw_output`
- `parsed_output`
- `parser_status`
- `latency_ms`
- `ttft_ms` when available
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- task-specific scores
- timeout/error fields

Result records and manifests MUST NOT persist literal `base_url`, URL-derived
hashes, API keys, Authorization/header values, or arbitrary raw server
configuration. Use the explicit safe configuration projection; neither full
`ResolvedEndpoint` nor raw `RunConfig` is a persisted result contract.

Stage 3C stores the raw transport-evidence subset of the eventual result view in
ignored `<results_root>/<run_id>/manifest.json` and `journal.jsonl`. The caller
supplies the root. The manifest is immutable; resume strictly checks its path/run
ID and exact equality with the caller's expected manifest rebuilt through public
Stage 3A identity functions. No timestamps or parsed/scored structures are stored.

Each `attempt` records candidate transport evidence; `finalized` references the
latest attempt without copying output again. Per-execution `attempt_index` starts
at zero and is contiguous, separate from `repetition_index` and execution identity.
Only finalized keys are completed, including finalized failures. Unfinished
attempts remain available to later orchestration; Stage 3C has no retry policy
and imposes no new Stage 3B result cross-field semantics.

Candidate raw text is exact private ignored evidence, including any private
material the candidate repeats. Historical messages, baseline outputs, and
provenance payloads remain in the immutable private source dataset and are not
duplicated into the journal. Unsafe provider finish reason/model values become
null; `omitted_metadata` contains only their names in fixed order (`finish_reason`,
`response_model`). Safe diagnostics never echo candidate or rejected provider text.

The journal reader fails closed on corrupt JSON/UTF-8, duplicate keys, non-finite
numbers, schema/sequence violations, and any unterminated final record. There is
no automatic tail repair. One sequential writer per run is supported on local
POSIX filesystems. Private modes reject group/other access and require sufficient
owner capabilities. File and directory fsync accompany create-only manifest
publication, directory creation, and initial journal creation; later appends
fsync the journal. Uncertain writes poison the writer pending close/reopen and
strict validation. Filesystem guarantees vary; there is no exactly-once inference
or HTTP/filesystem transaction, and concurrent independent writers are unsupported.

---

# 3. Variable extraction matrix

Production fidelity requirements:

- use the Olegos-shaped extraction prompt;
- preserve variable name/type/hint, the two leading user-message newlines, and source enum spellings (`VariableType.string`, with number/boolean counterparts);
- support only current production types: string, number, boolean;
- reproduce current conversation/tool normalization where applicable;
- score strict JSON separately from production-parser compatibility;
- distinguish `"unknown"`/missing behavior from hallucinated values.

| ID | Difficulty | Critical | Scenario | Expected behavior |
|---|---|---:|---|---|
| EX-001 | easy | yes | Origin, destination, date all explicitly stated | Return all three exactly/semantically correctly |
| EX-002 | easy | yes | Destination and date stated, origin absent | Origin must be `unknown`; do not infer |
| EX-003 | easy | yes | Only origin stated | All unstated requested values remain `unknown` |
| EX-004 | normal | yes | Origin corrected later | Final/corrected origin wins |
| EX-005 | normal | yes | Destination corrected later | Final/corrected destination wins |
| EX-006 | normal | yes | Date corrected later | Final/corrected date wins |
| EX-007 | hard | yes | Several old and new values conflict | Use latest clearly accepted values; no merging of obsolete values |
| EX-008 | normal | yes | Relative date such as "next Friday" | Preserve date as stated when contract asks for preservation |
| EX-009 | normal | no | Airport code instead of city | Extract stated airport code, not invented city normalization |
| EX-010 | hard | yes | Unrelated cities mentioned in small talk | Do not extract irrelevant city as origin/destination |
| EX-011 | hard | yes | User discusses another person's trip | Extract only values belonging to requested caller context |
| EX-012 | normal | yes | Relevant value appears in tool response | Retain value if production-normalized tool response is part of context |
| EX-013 | normal | no | Tool response wrapped in `{"data": ...}` | Adapter exposes only `data` content as production does |
| EX-014 | normal | no | Tool response has `status` and `status_code` wrappers | Wrapper fields removed before model-facing transcript |
| EX-015 | normal | no | Transition tool response equals `{"status":"done"}` | Response omitted from extraction transcript |
| EX-016 | hard | no | Tool response above 2000 chars | Adapter truncates one tool response according to source behavior |
| EX-017 | hard | yes | Long transcript with target facts near beginning | Correct fields retained despite distractors |
| EX-018 | normal | yes | Number variable | Valid numeric value and correct type/semantic value |
| EX-019 | normal | yes | Boolean variable | Correct boolean value; no string hallucination where parser/schema expects boolean |
| EX-020 | hard | yes | Mixed string/number/boolean variables | All field types correct |
| EX-021 | adversarial | yes | Requested field never appears but many plausible hints exist | Must not guess; missing/unknown is correct |
| EX-022 | adversarial | yes | Prompt-injection-looking text inside conversation | Treat transcript as data; still perform extraction contract |
| EX-023 | adversarial | no | Quotes/newlines/JSON-sensitive characters in value | Produce consumable JSON preserving intended value |
| EX-024 | normal | no | Candidate wraps JSON in Markdown code fence | Strict-format fail; production-parser compatibility may pass |
| EX-025 | normal | no | Candidate adds prose before JSON | Strict-format fail; production-parser compatibility scored independently |
| EX-026 | hard | yes | Candidate returns malformed JSON | Parser behavior recorded; semantic case fails if no usable structured result |

Primary extraction metrics:

- whole-object exact/semantic match;
- field accuracy;
- type accuracy;
- missing/unknown accuracy;
- hallucinated-value rate;
- correction accuracy;
- strict JSON validity;
- production-parser compatibility;
- extra-key rate.

---

# 4. Runtime context summarization matrix

Production fidelity requirements:

- this means `llm-context-summarization`, not QA `conversation-summary-before-*`;
- use the current Pipecat/Olegos summary prompt;
- adapter tests preserve the source trigger behavior and message-selection rules;
- quality tests score retained facts, corrections, open questions, action items, contradictions, and hallucinations;
- recent messages preserved outside the summary are adapter behavior, not text-generation targets.

| ID | Difficulty | Critical | Scenario | Expected behavior |
|---|---|---:|---|---|
| CS-001 | easy | no | Context has <= 6 messages | Adapter should not request summarization |
| CS-002 | normal | yes | Basic eligible multi-turn context | Concise faithful summary |
| CS-003 | normal | yes | Important fact appears very early | Critical fact retained |
| CS-004 | normal | yes | User preference stated once | Preference retained |
| CS-005 | normal | yes | Explicit decision/agreement | Decision retained |
| CS-006 | normal | yes | Unresolved user question | Open question retained |
| CS-007 | normal | yes | Pending action item | Action item retained |
| CS-008 | hard | yes | Destination corrected later | Latest value represented; obsolete destination not treated as current |
| CS-009 | hard | yes | Multiple corrections across fields | Final state correct for every changed field |
| CS-010 | hard | yes | Contradictory statements without clear resolution | Summary preserves uncertainty instead of inventing resolution |
| CS-011 | normal | no | Lots of greetings/small talk | Nonessential chatter omitted |
| CS-012 | normal | no | Repeated same fact many times | Redundancy compressed without losing fact |
| CS-013 | hard | yes | Multi-topic conversation | Relevant state from all active topics retained |
| CS-014 | hard | yes | One critical fact buried in long transcript | Critical fact retained |
| CS-015 | hard | yes | Old fact is superseded by a new fact | Old value not presented as current |
| CS-016 | hard | yes | Tool result changes conversation state | Summary reflects relevant tool-derived state where included by production formatting |
| CS-017 | hard | no | Resolved tangent plus unresolved main issue | Tangent compressed/omitted; unresolved issue preserved |
| CS-018 | adversarial | yes | Transcript contains unsupported claim bait | No hallucinated facts |
| CS-019 | adversarial | yes | Similar names/numbers/dates across turns | No entity/value conflation |
| CS-020 | hard | yes | Long context approaching candidate limits | Completes successfully or records explicit timeout/error; never silently truncates without metadata |
| CS-021 | normal | no | Recent two messages are outside summarization slice | Adapter selects/preserves messages according to production rule |
| CS-022 | hard | yes | Critical fact sits exactly near summary/preserved-message boundary | No loss or duplication that changes meaning |
| CS-023 | hard | yes | New messages conceptually arrive after summary input snapshot | Adapter/unit test ensures snapshot/preserved-context semantics are represented correctly |
| CS-024 | adversarial | yes | Model returns empty summary | Failure recorded; production-like fallback semantics tested |
| CS-025 | adversarial | no | Request exceeds timeout | Timeout recorded separately from semantic quality |
| CS-026 | hard | yes | Long transcript with multiple unresolved items and corrections | High-recall state summary without stale facts |

Primary context-summary metrics:

- critical-fact recall;
- overall required-fact recall;
- stale/superseded-fact error rate;
- contradiction rate;
- hallucination rate;
- user-preference retention;
- action-item retention;
- unresolved-question retention;
- compression ratio;
- output length;
- timeout/error rate.

---

# 5. Voicemail detection matrix

Production fidelity requirements:

- expected classes are `CONVERSATION` and `VOICEMAIL`;
- use the full frozen classifier prompt and preserve source-shaped message contexts, including assistant/tool messages;
- score exact-label compliance separately from production decision behavior;
- production checks for `CONVERSATION` first, then `VOICEMAIL`;
- a response containing both words is therefore production-decided as `CONVERSATION`;
- false human -> voicemail is a critical operational error.

| ID | Difficulty | Critical | Scenario | Gold |
|---|---|---:|---|---|
| VM-001 | easy | yes | "Hello?" | CONVERSATION |
| VM-002 | easy | yes | "John speaking." | CONVERSATION |
| VM-003 | easy | yes | Human asks "Who's calling?" | CONVERSATION |
| VM-004 | normal | yes | Very short human "Yeah?" | CONVERSATION |
| VM-005 | normal | yes | Hesitant human greeting | CONVERSATION |
| VM-006 | hard | yes | Human says "I saw your voicemail" | CONVERSATION |
| VM-007 | hard | yes | Human asks "Are you leaving me a voicemail?" | CONVERSATION |
| VM-008 | hard | yes | Human mentions automated voicemail system | CONVERSATION |
| VM-009 | easy | yes | Standard personal voicemail greeting | VOICEMAIL |
| VM-010 | easy | yes | "Please leave your name and number after the beep" | VOICEMAIL |
| VM-011 | normal | yes | Professional/business voicemail | VOICEMAIL |
| VM-012 | normal | yes | Mailbox full message | VOICEMAIL |
| VM-013 | normal | yes | Voicemail not set up | VOICEMAIL |
| VM-014 | normal | yes | Number unavailable/not in service carrier recording | VOICEMAIL |
| VM-015 | normal | yes | Business closed recording | VOICEMAIL |
| VM-016 | hard | yes | Long automated greeting without "voicemail" keyword | VOICEMAIL |
| VM-017 | hard | yes | IVR-like automated first turn | Expected class explicitly annotated after source-aligned review |
| VM-018 | hard | yes | Ambiguous automated/human-like greeting | Expected class explicitly annotated; mark ambiguous |
| VM-019 | hard | no | Truncated voicemail greeting | Correct if enough evidence; otherwise benchmark records ambiguity policy |
| VM-020 | hard | no | Truncated human greeting | Must avoid unsafe human->voicemail overclassification |
| VM-021 | adversarial | no | Empty/noisy text representation | No unsupported hard decision; parser/decision behavior recorded |
| VM-022 | adversarial | no | Model output contains both `CONVERSATION` and `VOICEMAIL` | Strict fail; production decision must resolve to CONVERSATION |
| VM-023 | adversarial | no | Model output contains neither keyword | Production decision = no classification |
| VM-024 | normal | no | Extra prose plus one correct label | Strict-format fail; production decision may still pass |
| VM-025 | hard | yes | Human greeting is long and formal like a recording | CONVERSATION |
| VM-026 | hard | yes | Recording says "press 1" / menu options | VOICEMAIL/automation according to production classifier contract |

Primary voicemail metrics:

- accuracy;
- per-class precision/recall/F1;
- human->voicemail false-positive rate;
- voicemail->human false-negative rate;
- exact-label rate;
- production-decision accuracy;
- undecided rate.

---

# 6. Post-call QA matrix

Production fidelity requirements:

- use frozen SkyAssist QA prompt/labels for the primary suite;
- input includes node summary, previous conversation summary, metrics, and current node transcript;
- response requires JSON with `tags`, `overall_sentiment`, `call_quality_score`, and `summary`;
- score semantic correctness and evidence grounding, not just JSON validity.

| ID | Difficulty | Critical | Scenario | Expected QA behavior |
|---|---|---:|---|---|
| QA-001 | easy | yes | Clean coherent node | No tags; sensible sentiment/score/summary |
| QA-002 | easy | yes | Conversation is incoherent | UNCLEAR_CONVERSATION |
| QA-003 | easy | yes | Assistant repeats same question several times | ASSISTANT_IN_LOOP |
| QA-004 | easy | yes | Assistant answers unrelated question | ASSISTANT_REPLY_IMPROPER |
| QA-005 | easy | yes | User clearly frustrated | USER_FRUSTRATED |
| QA-006 | normal | yes | User repeatedly says they do not understand | USER_NOT_UNDERSTANDING |
| QA-007 | easy | yes | "Can you hear me?" / hearing failure | HEARING_ISSUES |
| QA-008 | normal | yes | Metrics/transcript support long silence | DEAD_AIR |
| QA-009 | normal | yes | User asks for unsupported feature | USER_REQUESTING_FEATURE |
| QA-010 | normal | yes | User describes distress; assistant ignores it | ASSISTANT_LACKS_EMPATHY |
| QA-011 | easy | yes | User explicitly says "Are you a bot?" | USER_DETECTS_AI |
| QA-012 | hard | yes | Two true tags simultaneously | Both true tags, no invented extras |
| QA-013 | hard | yes | Three+ true tags simultaneously | High tag recall without over-tagging |
| QA-014 | hard | yes | Negative wording but not frustration | Avoid USER_FRUSTRATED false positive |
| QA-015 | hard | yes | Assistant repeats a phrase once appropriately | Avoid ASSISTANT_IN_LOOP false positive |
| QA-016 | hard | yes | User says "I don't understand" once, then resolves issue | Apply source-aligned annotation; avoid automatic keyword-only tagging |
| QA-017 | hard | yes | "I can't hear the price" meaning "didn't catch information", not audio fault | Avoid HEARING_ISSUES if context does not support it |
| QA-018 | hard | yes | User mentions "bot" while discussing another service | Avoid USER_DETECTS_AI false positive |
| QA-019 | hard | yes | Evidence needed from previous conversation summary | Correct judgment uses prior context |
| QA-020 | hard | yes | Conflicting evidence across current transcript | Explain/tag only what evidence supports |
| QA-021 | hard | yes | Critical evidence occurs far apart in long node transcript | Correct tags despite distance |
| QA-022 | hard | yes | Node purpose makes response improper even though wording sounds polite | Use node summary to judge behavior |
| QA-023 | hard | yes | Metrics contradict textual impression | Use both inputs; no unsupported claim |
| QA-024 | normal | no | Whole-call fallback with no node IDs | Correctly evaluates whole transcript |
| QA-025 | adversarial | yes | Keyword bait for several tags but no actual behavior | Avoid false positives |
| QA-026 | adversarial | yes | Model invents evidence in tag reason | Evidence-grounding failure |
| QA-027 | normal | no | Markdown-wrapped JSON | Strict fail; production parser compatibility scored |
| QA-028 | normal | no | Top-level array instead of dict | Production-like parser/coercion behavior recorded |
| QA-029 | adversarial | no | Missing `tags` | Production default `[]`; schema/semantic score reflects missing content |
| QA-030 | adversarial | no | Missing summary/sentiment/score fields | Parser behavior recorded; completeness fails |
| QA-031 | hard | yes | Same transcript, different node purpose | QA result may legitimately differ; proves node-summary conditioning works |
| QA-032 | hard | yes | Same transcript, different prior context | QA result may legitimately differ; proves previous-summary conditioning works |

Primary QA metrics:

- tag micro/macro precision, recall, F1;
- per-tag precision/recall;
- false-positive/false-negative counts;
- sentiment accuracy;
- call-quality score MAE against annotated band/score;
- summary factuality;
- evidence-grounding accuracy;
- strict JSON/schema validity;
- production-parser compatibility.

---

# 7. QA-support summary matrix

This is separate from runtime context summarization.

## 7.1 Prior-conversation summaries

Production purpose: summarize all previous-node conversation in 3-5 sentences for downstream QA.

| ID | Difficulty | Critical | Scenario | Expected behavior |
|---|---|---:|---|---|
| QS-001 | easy | yes | Short prior conversation | Concise factual 3-5 sentence summary |
| QS-002 | normal | yes | Multiple prior topics | Relevant state from each topic retained |
| QS-003 | normal | yes | User frustration happened earlier | Frustration nuance retained for downstream QA |
| QS-004 | hard | yes | Earlier assistant mistake matters later | Failure retained accurately |
| QS-005 | hard | yes | User corrected an important fact | Final state retained; stale value not treated as current |
| QS-006 | hard | yes | Conflicting prior evidence | Preserve uncertainty/conflict |
| QS-007 | normal | no | Lots of irrelevant chatter | Compress noise |
| QS-008 | hard | yes | Long previous conversation | Retain QA-relevant facts without hallucination |
| QS-009 | adversarial | yes | One critical fact appears only once | Critical fact retained |
| QS-010 | adversarial | yes | Unsupported implication bait | No invented events or sentiment |

## 7.2 Node/script summaries

Production purpose: produce a concise 2-4 sentence description of what a node/script is supposed to accomplish and behaviors relevant to QA.

| ID | Difficulty | Critical | Scenario | Expected behavior |
|---|---|---:|---|---|
| NS-001 | easy | yes | Simple node prompt | Accurate purpose summary |
| NS-002 | normal | yes | Node with multiple required behaviors | All critical behaviors represented |
| NS-003 | normal | yes | Node with tool-use requirement | Tool requirement captured |
| NS-004 | hard | yes | Node contains prohibitions ("never invent", etc.) | Prohibition preserved |
| NS-005 | hard | yes | Long complex node prompt | Important behavior retained, trivia compressed |
| NS-006 | adversarial | yes | Similar required vs optional behaviors | Do not promote optional behavior to mandatory |

Primary QA-support metrics:

- critical fact/behavior recall;
- factuality;
- stale-state error rate;
- hallucination rate;
- downstream-QA-relevant information retention;
- length/compression compliance.

---

# 8. OVP-34 replay matrix

Use a separate manifest selecting the 242 canonical OOB observations from a private/local export of the final OVP-34 100-run telemetry. Every selected observation has captured model-facing input messages and an output.

These messages are the best available captured trace/model-facing evidence.
For the 56 runtime summaries, current tracing reconstructs the summary request
after generation; historical wire equality is not independently proven. Retain
all 56 with that caveat and bypass adapters for all historical replay. Exact
preservation refers to captured JSON message semantics, not verified original
HTTP bytes. Stage 3A provides request preparation only, not replay loading.

| Replay family | Captured observations | Exact-message replay target |
|---|---:|---:|
| Variable extraction | 40 | 40 |
| Runtime context summarization | 56 | 56 |
| Voicemail detection | 22 | 22 |
| Post-call QA | 72 | 72 |
| QA previous-conversation summaries | 52 | 52 |
| **Total** | **242** | **242** |

The 52 summary observations are previous-conversation summaries, not node/script summaries. The Node Purpose section was empty in the 72 historical QA observations; current production still supports node summaries.

Replay requirements:

- preserve exact captured message order, roles, content, and already-rendered system messages; do not reconstruct prompts from conversation text;
- preserve task/span/provenance/run IDs in private/local metadata;
- preserve exact historical output separately as baseline behavior, never automatic gold;
- annotate gold independently when practical;
- keep raw/private exports ignored and uncommitted;
- report unavailable/incomplete local exports explicitly rather than fabricating requests;
- never mix replay cases with curated cases in aggregate reporting without showing both breakdowns.

Replay loading is a later stage. Curated/adversarial cases use source-aligned adapters; replay uses the captured historical requests.

---

# 9. Critical-case policy

A candidate model must not be accepted for a job solely because its average score is high.

Critical cases represent production-dangerous failures.

Examples:

- extraction hallucinates a missing value;
- summary drops or reverses a critical corrected fact;
- human is classified as voicemail;
- QA invents evidence;
- QA misses a clearly defined severe failure;
- QA-support summary fabricates prior events.

Reports must show:

- overall score;
- critical-case pass rate;
- count/list of critical failures;
- baseline-vs-candidate deltas.

A later go/no-go threshold will be frozen before looking at candidate results.

---

# 10. Minimum curated suite size

The first implementation should target:

- Extraction: 26 cases
- Runtime context summary: 26 cases
- Voicemail: 26 cases
- Post-call QA: 32 cases
- QA-support prior-conversation summary: 10 cases
- QA node/script summary: 6 cases

Total initial curated/adversarial suite:

**126 cases**

This is intentionally separate from the 242 captured OVP-34 replay observations.

The suite can grow after the first implementation, but these IDs and their intended semantics should not be silently repurposed.

---

# 11. Implementation order

Implement adapters/scorers in this order:

1. variable extraction — easiest objective structured-output path;
2. voicemail classification — simple classification path and good parser-contract test;
3. runtime context summarization — highest live-window OOB priority;
4. QA-support summaries;
5. post-call QA — richest input contract and scoring.

This is implementation order only. It does not change production importance.

---

# 12. Freeze gate before coding

Before Codex writes implementation code, we should have:

- `docs/BENCHMARK_SPEC.md`
- `docs/SOURCE_AUDIT.md`
- `docs/CASE_MATRIX.md`

Then Codex's first task should be **read-only planning only**:

- inspect these three documents;
- inspect the empty/current research repo;
- propose exact package structure, schemas, adapters, scorer interfaces, server/client boundaries, tests, and commands;
- identify ambiguities;
- make no edits.

Only after that plan is reviewed should implementation begin.
