# OVP-36 OOB Model Benchmark — Source Audit

Status: **design input / read-only audit**  
Purpose: define the production-facing contracts the standalone OVP-36 benchmark must reproduce without copying the full Olegos runtime.

## 1. Source baseline used for this audit

This audit is based on the current source snapshots supplied for the research task:

- `olegos-api-develop`
- `olegos-mps-develop`
- `olegos-pipecat-develop`
- `olegos-docs-develop`

The Olegos API repository instructions say `develop` is the integration branch and moves quickly. Therefore these snapshots are sufficient for the standalone benchmark design, but **before any later product integration/MR work, the real repositories must be fetched/rebased and the audit rechecked against latest `origin/develop`.**

This document does not authorize or perform product changes, deployments, MPS edits, or model-server starts.

### 1.1 Final OVP-34 evidence

The higher-fidelity review of the final OVP-34 100-run raw Langfuse telemetry and frozen SkyAssist configuration established exactly **242 selected OOB observations**, each with captured model-facing input messages and an output: 40 extraction, 56 runtime context summary, 22 voicemail, 72 post-call QA, and 52 QA previous-conversation summaries.

There are two fidelity paths:

- **OVP-34 replay:** read a private/local export, select those 242 canonical observations, and preserve the exact captured message sequence, content, roles, historical output, and task/span/provenance IDs. Do not reconstruct prompts from conversational text or regenerate already-rendered system messages.
- **Curated/adversarial cases:** generate requests through source-aligned task adapters.

Historical outputs remain baseline behavior, not automatic gold. Keep raw/private exports ignored and uncommitted. If a local export is absent or incomplete, report that availability problem; do not fabricate inputs. No replay loader is implemented in Stage 2.

---

## 2. What the standalone benchmark should replicate

The benchmark should **not** recreate the full Olegos application.

It should recreate only the model-facing contract for each OOB job:

1. the information Olegos selects for the OOB request;
2. the prompt/system instruction Olegos constructs;
3. the message shape sent to the LLM;
4. generation/output constraints that materially affect the task;
5. production-style parsing/decision behavior;
6. the expected semantic result used for quality scoring.

The benchmark should not require:

- telephony;
- STT/TTS;
- the workflow engine;
- Redis/Postgres/MinIO;
- MPS routing;
- Langfuse;
- Kubernetes;
- Olegos UI;
- production credentials;
- a live Olegos workflow.

The source-aligned curated/adversarial path is:

```text
benchmark case
    -> task adapter reproducing Olegos model-facing request
    -> common OpenAI-compatible model endpoint
    -> candidate model
    -> production-like parser/decision adapter
    -> task-specific scorer
    -> immutable result record/report
```

---

## 3. Current LLM-routing baseline relevant to OVP-36

### 3.1 API tier resolution

Source: `api/services/pipecat/tier_resolver.py`

The API treats the names:

- `default`
- `fast`
- `accurate`
- `lite`
- `zen`

as logical Speaches LLM tier names. `resolve_llm_tier()` fetches the MPS table from:

```text
GET /api/v1/llm/tiers
```

and, when the configured model is a tier, replaces the LLM config's:

- provider;
- model;
- base URL.

The table is cached per API process for five minutes. If the MPS fetch fails or the tier is absent, resolution fails safe by leaving the configuration unchanged.

The design intent documented in source is that pricing/cost calculation sees the **resolved concrete model**, not the logical tier name.

### 3.2 Service factory

Source: `api/services/pipecat/service_factory.py`

`create_llm_service(user_config)`:

1. calls `resolve_llm_tier(user_config)`;
2. reads the resolved provider/model/API key;
3. forwards provider-specific options, including Speaches `base_url`;
4. calls `create_llm_service_from_provider(...)`;
5. applies the configured context-window guard where supported.

The factory applies a default per-call max-token cap. Models whose name is treated as reasoning-capable (`gpt-5*`, `o1*`, `qwen3*`, `deepseek-r1*`, `deepseek-v3*`) receive the roomier reasoning cap; other models receive the normal cap. Runtime context summary explicitly overrides `max_tokens=4000`; the historical Qwen voicemail path exposes `max_tokens=2048`. These are distinct current-source/historical observations, not a universal candidate policy. Because factory defaults depend on model identity, candidate generation limits must be decided and recorded before canonical evaluation. Do not infer Granite limits of 2048 or 512 from the historical Qwen runs.

### 3.3 Current MPS table

Source: `olegos-mps/defaults.yaml`

At this snapshot the tier table is:

```text
default   -> speaches / qwen3.5 / <configured MPS LLM endpoint>
fast      -> speaches / qwen3.5 / <configured MPS LLM endpoint>
accurate  -> speaches / qwen3.5 / <configured MPS LLM endpoint>
lite      -> speaches / qwen3.5 / <configured MPS LLM endpoint>
zen       -> speaches / qwen3.5 / <configured MPS LLM endpoint>
```

In this audited snapshot, `default`, `fast`, `accurate`, `lite`, and `zen` all resolve to the same `qwen3.5` serving endpoint. The private address is intentionally omitted.

There are **no OVP-36 job-specific tiers** such as extraction/summary/classify/QA in this snapshot.

The MPS endpoint simply serves the YAML table via `GET /api/v1/llm/tiers`.

### 3.4 OVP-36 implication

For the research benchmark we do **not** need to modify or emulate MPS. The benchmark needs a model endpoint abstraction so the same cases can later target:

- a local candidate server;
- a GB10/DGX research candidate server;
- a baseline Qwen endpoint where approved;
- another candidate model.

Actual MPS job-tier routing is a later integration step after evidence exists.

---

# 4. OOB task 1 — Variable extraction

## 4.1 Product source path

Primary source:

```text
api/services/workflow/pipecat_engine_variable_extractor.py
```

Trigger/engine integration:

```text
api/services/workflow/pipecat_engine.py
```

DTO/schema:

```text
api/services/workflow/dto.py
```

Parser:

```text
api/services/gen_ai/json_parser.py
```

## 4.2 When it runs

`PipecatEngine._perform_variable_extraction_if_needed(...)` runs only when the node has:

- `extraction_enabled` true; and
- non-empty `extraction_variables`.

Extraction can run as a background task during the workflow. On normal call termination, pending extractions are awaited and the current node receives a final synchronous extraction. The voicemail/pipeline-error immediate-abort path does not perform the normal final extraction.

## 4.3 Which LLM it uses today

`VariableExtractionManager` calls:

```text
self._engine.inference_llm.run_inference_with_usage(...)
```

`PipecatEngine` sets:

```text
self.inference_llm = inference_llm or llm
```

Therefore:

- in a normal non-realtime text LLM pipeline, extraction falls back to the same LLM service instance used by dialogue;
- in realtime speech-to-speech mode, the pipeline constructs a separate text `inference_llm` because the realtime service does not support the OOB text inference method.

This shared/fallback behavior is one reason OVP-36 exists.

## 4.4 Supported variable types

Source: `api/services/workflow/dto.py`

Production extraction variables support:

```text
string
number
boolean
```

The benchmark should test these types and should not invent arrays/objects as production extraction variable types unless current source later adds them.

## 4.5 Conversation representation

The extraction manager builds a normalized text transcript from the active LLM context.

It includes:

- textual `assistant` messages;
- textual `user` messages;
- supported text segments from list-form content;
- Gemini-style textual content where applicable;
- selected tool responses.

Each ordinary message becomes approximately:

```text
assistant: <content>
user: <content>
```

Tool responses are labelled:

```text
[Tool Response: <tool_name>]
<cleaned response>
```

Production cleanup rules:

- transition response exactly equal to `{"status": "done"}` is omitted;
- if a tool response is JSON and has a `data` key, only `data` is used;
- otherwise wrapper keys `status` and `status_code` are removed;
- one tool response is capped at 2,000 characters and then suffixed with `...(truncated)`;
- unparseable JSON tool output is retained as raw text.

The benchmark adapter should reproduce these transformations for cases that include tool messages.

## 4.6 Prompt contract

Production base system instruction:

```text
You are an assistant tasked with extracting structured data from the conversation. Return ONLY a valid JSON object with the requested variables as top-level keys. Do not wrap the JSON in markdown.
```

If the workflow node has a custom `extraction_prompt`, it is appended after exactly two newline characters.

Current source constructs the user message as:

```python
user_prompt = (
    "\n\nVariables to extract:\n"
    f"{vars_description}"
    "\n\nConversation history:\n"
    f"{conversation_history}"
)
```

The initial two newlines are part of the model-facing contract. Each variable line uses `f"- {v.name} ({v.type}): {v.prompt}"`. Because `v.type` is a `VariableType` enum, the observed string representation is `VariableType.string`; current source implies `VariableType.number` and `VariableType.boolean` for those types. The DTO types remain string/number/boolean.

The frozen SkyAssist extraction-specific system instruction is:

```text
Extract the flight search details stated by the caller. Do not infer missing information. If a requested value was not provided, return "unknown".
```

Frozen SkyAssist variable descriptions are:

```text
- origin (VariableType.string): Departure city or airport requested by the caller. Return "unknown" if not stated.
- destination (VariableType.string): Destination city or airport requested by the caller. Return "unknown" if not stated.
- travel_date (VariableType.string): Travel date requested by the caller. Preserve it as stated by the caller. Return "unknown" if not stated.
```

The benchmark must preserve this text, including the leading user-message newlines and two-newline system append. Stage 2 freezes the contracts; extraction history formatting belongs to the later adapter. Historical replay uses the captured messages unchanged.

## 4.7 Production-like output parsing

Production calls `parse_llm_json(raw_response)`.

For None/empty/blank input, return `{}`. Otherwise use `content = raw_content.strip()` and try, in order:

1. `json.loads(content)`, accepting direct success only for a dict/list;
2. the FIRST `re.search` match of `r"```(?:json)?\s*([\s\S]*?)\s*```"`, stripping and parsing the captured contents, accepting only a dict/list;
3. a balanced object starting at the FIRST `{`;
4. a balanced array starting at the FIRST `[`;
5. fallback `{"raw": original_raw_response}`, preserving the original unstripped response.

Every production JSON parsing branch accepts only a dict/list. An invalid or scalar first fence proceeds to object scanning and then array scanning over the full stripped content; it does not trigger another fence search. If neither succeeds, return raw fallback. The regex only treats lowercase `json` as the optional label, rather than stripping arbitrary language names.

Object/array scanning follows production ordering literally: consume a pending escape; otherwise a backslash sets the escape flag and skips that character; otherwise a double quote toggles string state; skip characters inside strings; then count opening/closing delimiters. Backslashes escape the next character even outside a string in malformed input. A malformed first balanced candidate does not cause a search for a second candidate of the same kind.

Production uses Python `json.loads`, which can accept non-finite values such as NaN or Infinity inside dict/list results. Preserve that production result and parser path; do not replace it with raw fallback because a benchmark schema prefers finite JSON. Strict RFC-format assessment remains independent and may reject those constants. Parser-result serialization must preserve these values without silently converting them to null; future standards-compliant result export needs an explicit lossless encoding policy.

The extraction manager returns an empty dict for a `None` response.

Important benchmark consequence: score **both**:

- strict instruction compliance / strict JSON;
- production-parser compatibility.

A response may violate the instruction while still being consumable by production.

## 4.8 Benchmark cases implied by source

Must include at least:

- all fields present;
- one/multiple fields missing;
- string/number/boolean types;
- user correction of an earlier value;
- conflicting old/new values;
- irrelevant entities that should not be extracted;
- relevant tool response;
- irrelevant tool response;
- `data`-wrapped tool response;
- `status`/`status_code` wrappers;
- transition-tool response that must be omitted;
- oversized tool response/truncation;
- long conversation;
- malformed model JSON;
- Markdown-wrapped JSON;
- extra prose around JSON;
- hallucination trap.

The frozen SkyAssist example additionally requires origin, destination, and travel date semantics with missing values represented as `unknown` and the date preserved as stated by the caller.

---

# 5. OOB task 2 — Runtime context summarization

## 5.1 Product source path

Olegos integration:

```text
api/services/workflow/pipecat_engine_context_summarizer.py
api/services/workflow/pipecat_engine.py
```

Underlying summarization utility:

```text
olegos-pipecat/src/pipecat/utils/context/llm_context_summarization.py
```

## 5.2 When it is enabled

The workflow configuration controls `context_compaction_enabled`.

For non-realtime workflows, when enabled, `PipecatEngine.initialize()` constructs `ContextSummarizationManager`.

Realtime speech-to-speech workflows explicitly disable this Olegos context-compaction path because the realtime service manages conversation state separately.

## 5.3 Trigger condition

After a node transition, if there was a previous node and the summarization manager exists, the engine starts background context summarization.

The manager cancels an already-running summarization if a newer transition starts another one.

Before calling the model, it returns without doing anything when the context has six or fewer messages.

## 5.4 Current Olegos summary configuration

Olegos overrides Pipecat defaults with:

```text
target_context_tokens = 4000
min_messages_after_summary = 2
summarization_timeout = 30 seconds
```

Pipecat's `LLMService._generate_summary()` passes `frame.target_context_tokens` directly as the request's `max_tokens` value:

```python
run_inference(
    summary_context,
    max_tokens=frame.target_context_tokens,
    system_instruction=frame.summarization_prompt,
)
```

Therefore the canonical current production-shaped runtime-summary request uses **`max_tokens=4000`**, with a **30-second timeout**. `target_context_tokens` retains its semantic name in Pipecat, but the current implementation uses its value as the actual per-request completion-token limit. Record these effective settings in the benchmark's production-fidelity metadata.

## 5.5 Prompt contract

Olegos uses Pipecat's default summary system prompt in this snapshot. The exact frozen current-source prompt is:

```text
You are summarizing a conversation between a user and an AI assistant.

Your task:
1. Create a concise summary that preserves:
   - Key facts, decisions, and agreements
   - Important context needed to continue the conversation
   - User preferences and requirements mentioned
   - Any unresolved questions or action items

2. Format:
   - Use clear, factual statements
   - Group related information
   - Prioritize information likely to be referenced later
   - Keep the summary concise to fit within the specified token budget

3. Omit:
   - Greetings and small talk
   - Redundant information
   - Tangential discussions that were resolved

The conversation transcript follows. Generate only the summary, no other text.
```

The user-side content is formatted as:

```text
Conversation history:
<formatted transcript>
```

The benchmark should preserve the exact production prompt text in a fixture or source-derived constant rather than paraphrasing it ad hoc.

## 5.6 Which messages are summarized

After Olegos's six-or-fewer-message skip check, the underlying Pipecat utility selects the summarization range as follows:

1. Preserve the first message separately only when `messages[0]` is a system message; begin summarization after it when present.
2. Preserve the last `min_messages_to_keep` messages outside the proposed summarization range. Olegos uses two.
3. If an unresolved function/tool-call sequence occurs inside that range, stop before the earliest unresolved function call.

The selected messages are formatted into a transcript, skipping `LLMSpecificMessage` objects. For each normal message:

- use `msg.get("role", "unknown")`;
- retain string content;
- for list content, keep items whose `type == "text"` and join their text with ONE space;
- otherwise convert content using `str(...)`;
- when text is non-empty, append `{ROLE_UPPERCASE}: {text}`;
- independently append each tool call as `TOOL_CALL: {name}({arguments})`;
- if the role is `tool`, ALSO append `TOOL_RESULT[{tool_call_id}]: {text}`.

Entries are joined with TWO newlines. A tool-role message with content can therefore produce both entries, as observed in historical OVP-34 runtime-summary requests:

```text
TOOL: {"status": "done"}

TOOL_RESULT[call_id]: {"status": "done"}
```

Do not deduplicate these entries or apply extraction's transition-tool omission to runtime summaries. This is the source contract for a future adapter; Stage 2 does not implement this formatter.

The formatted transcript is sent using the `Conversation history:` wrapper shown above.

When the generated summary is applied, Olegos:

1. snapshots the then-current context so messages that arrived while generation was running are preserved;
2. retains the current system message if present;
3. inserts a user message generated from the template:

```text
Conversation summary: {summary}
```

4. appends all messages after `last_summarized_index` from that current context, including the preserved tail and messages that arrived while generation was running.

The benchmark's **quality task** does not need to recreate the entire asynchronous context mutation engine, but it should test the same summary prompt and the same transcript selection/formatting rules. Separate unit tests for the adapter should verify which messages are selected/preserved.

## 5.7 Failure behavior

If summary text is empty or no valid last index is returned, Olegos keeps the full context.

Timeout is 30 seconds in Olegos's manager. A newer transition can cancel a running summary.

The standalone model-quality benchmark should record timeout/error separately from semantic quality. Later GB10 runs should use the same timeout policy unless the experiment explicitly compares another setting.

## 5.8 Telemetry behavior to preserve conceptually

Olegos traces this operation as:

```text
llm-context-summarization
```

The underlying Pipecat `_generate_summary()` reports OOB usage under a dedicated usage label. API tracing captures only the labelled OOB report so a normal live dialogue completion that finishes during background summarization is not accidentally attributed to the summary.

The standalone benchmark does not need Langfuse, but result records must keep each request's task identity unambiguous.

## 5.9 Benchmark cases implied by source

Must include:

- exactly/under six messages as an adapter trigger-behavior test;
- long multi-node transcript;
- required facts early in history;
- correction/superseded facts;
- user preferences;
- decisions/agreements;
- unresolved question;
- action item;
- irrelevant small talk;
- redundant repeated facts;
- multiple topics;
- tool-call/tool-result context where relevant;
- long-context tail cases;
- critical fact appearing once;
- recent messages that should remain outside the summary;
- hallucination/contradiction traps;
- timeout/error handling.

Context-summary quality should be weighted heavily because OVP-34 found it to be the largest directly observed live-window OOB completion-token source.

---

# 6. OOB task 3 — Voicemail detection/classification

## 6.1 Product source path

Olegos wiring:

```text
api/services/pipecat/run_pipeline.py
```

Classifier implementation:

```text
olegos-pipecat/src/pipecat/extensions/voicemail/voicemail_detector.py
```

## 6.2 When it runs

Workflow config section:

```text
voicemail_detection
```

Voicemail detection runs only when enabled and the workflow is **not** realtime. Realtime workflows explicitly disable this path.

## 6.3 Which LLM it uses today

Olegos deliberately creates a **separate LLM service instance** for the voicemail sub-pipeline so frame linking does not interfere with the main pipeline.

If:

```text
use_workflow_llm = true
```

then Olegos calls `create_llm_service(user_config)`, so the separate instance still resolves to the workflow's same configured logical tier/model/endpoint.

If false, provider/model/API key come from the voicemail-specific configuration.

The default model fallback in that custom path is `gpt-4.1`.

## 6.4 Timing input

Olegos passes `long_speech_timeout`, defaulting to:

```text
8.0 seconds
```

Pipecat describes this as an optional early-classification trigger for a continuously speaking first turn, useful for long voicemail greetings when interim transcripts are available.

The standalone text benchmark cannot reproduce real audio timing, but its case metadata should mark long/truncated-first-turn cases. Hardware/integration tests later can validate the actual streaming timing behavior if necessary.

## 6.5 Prompt contract

The complete current default classifier system prompt is frozen exactly:

```text
You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number and you need to determine if a human answered or if the call went to voicemail based on the provided text.

HUMAN ANSWERED - LIVE CONVERSATION (respond "CONVERSATION"):
- Personal greetings: "Hello?", "Hi", "Yeah?", "John speaking"
- Interactive responses: "Who is this?", "What do you want?", "Can I help you?"
- Conversational tone expecting back-and-forth dialogue
- Questions directed at the caller: "Hello? Anyone there?"
- Informal responses: "Yep", "What's up?", "Speaking"
- Natural, spontaneous speech patterns
- Immediate acknowledgment of the call

VOICEMAIL SYSTEM (respond "VOICEMAIL"):
- Automated voicemail greetings: "Hi, you've reached [name], please leave a message"
- Phone carrier messages: "The number you have dialed is not in service", "Please leave a message", "All circuits are busy"
- Professional voicemail: "This is [name], I'm not available right now"
- Instructions about leaving messages: "leave a message", "leave your name and number"
- References to callback or messaging: "call me back", "I'll get back to you"
- Carrier system messages: "mailbox is full", "has not been set up"
- Business hours messages: "our office is currently closed"

Respond with ONLY "CONVERSATION" if a person answered, or "VOICEMAIL" if it's voicemail/recording.
```

The 22 historical OVP-34 requests show these role sequences after the system instruction:

- user;
- assistant + tool;
- assistant + tool + user;
- user + user.

Classifier input is therefore a message context, not only a transcript string. `VoicemailInput.messages` preserves these roles and content. A future curated adapter prepends the frozen system instruction. Historical replay sends the full captured request directly, including its system message, without passing through a task adapter or stripping/reconstructing messages. Do not recreate Pipecat's context aggregator.

Custom prompts are allowed; the implementation warns if they do not contain both required classification keywords.

## 6.6 Production decision parser

The classifier does **not** require the response to equal the label exactly.

It uppercases the completed response and checks in order:

1. if it contains `CONVERSATION`, classify as human;
2. else if it contains `VOICEMAIL`, classify as voicemail;
3. else make no classification decision.

That order creates an important benchmark requirement: score both strict label compliance and actual production decision behavior.

A response that contains both words is production-classified as `CONVERSATION` because that branch is checked first.

## 6.7 Action after detection

A `VOICEMAIL` decision triggers Olegos's event handler, which ends the workflow run with reason `voicemail_detected` and aborts immediately.

Therefore a false positive — real human classified as voicemail — is operationally more severe than an ordinary text-label error. Report false-human-as-voicemail separately.

## 6.8 Benchmark cases implied by source

Must include:

- obvious personal greeting;
- `John speaking`/equivalent;
- interactive human question;
- very short human response;
- hesitant/informal human response;
- human saying the word `voicemail` in conversation;
- human discussing a voicemail received earlier;
- standard personal voicemail greeting;
- professional voicemail;
- leave-name-and-number instruction;
- mailbox full/not set up;
- number not in service/carrier recording;
- business closed message;
- long automated greeting;
- IVR-like/ambiguous automation;
- truncated first-turn transcript;
- empty/noisy text representation;
- output containing both labels;
- output containing neither label;
- extra prose around the correct label.

---

# 7. OOB task 4 — Post-call QA

## 7.1 Product source paths

```text
api/services/workflow/qa/analysis.py
api/services/workflow/qa/llm_config.py
api/services/workflow/qa/conversation.py
api/services/workflow/qa/metrics.py
api/services/workflow/qa/node_summary.py
api/services/workflow/qa/tracing.py
```

## 7.2 Overall flow

The primary QA path is per-node analysis after a completed workflow run.

It:

1. reads `realtime_feedback_events` from run logs;
2. splits events by node;
3. falls back to whole-call QA if node IDs are not available;
4. resolves the QA LLM configuration;
5. ensures node/script summaries exist;
6. computes per-node metrics;
7. generates a summary of prior-node conversation for later nodes;
8. renders the configured QA system prompt with placeholders;
9. sends the current node transcript as the user message;
10. parses the returned JSON;
11. records node-level results.

## 7.3 Which LLM it uses today

If `qa_use_workflow_llm` is false, QA can use explicitly configured provider/model/key.

Otherwise it resolves the workflow owner's/org's LLM config and importantly calls the same tier resolver used by the main pipeline. That means a logical `default` tier becomes the concrete MPS provider/model/base URL before the QA service is built.

A QA-specific `qa_model` value other than `default` can override the resolved workflow model.

QA runs outside the live pipeline and constructs its service using `create_llm_service_from_provider(...)` plus shared provider-specific kwargs.

## 7.4 Per-node prompt shape

For each node, production builds:

- `node_summary` — what that script/node was supposed to do;
- `previous_conversation_summary` — condensed prior-node context;
- `transcript` — current node transcript;
- `metrics` — JSON-formatted precomputed call metrics.

These values render into the QA node's configured system prompt.

The model then receives a user message:

```text
## Transcript
<current node transcript>
```

The standalone QA adapter should reproduce these four system-prompt inputs and the user-message shape.

## 7.5 Frozen SkyAssist QA template and schema

The frozen OVP-34 SkyAssist system template is complete and source-known. Preserve these literal double-brace placeholders; actual Olegos template substitution belongs to the later QA adapter:

```text
You are a QA analyst evaluating a specific segment of a voice AI conversation.

## Node Purpose
{{node_summary}}

## Previous Conversation Context (For start of conversation, previous conversation summary can be empty.)
{{previous_conversation_summary}}

## Tags to evaluate

Examine the conversation carefully and identify which of the following tags apply:

- UNCLEAR_CONVERSATION - The conversation is not coherent or clear, messages don't connect logically
- ASSISTANT_IN_LOOP - The assistant asks the same question multiple times or gets stuck repeating itself
- ASSISTANT_REPLY_IMPROPER - The assistant did not reply properly to the user's question/query or seems confused by what the user said
- USER_FRUSTRATED - The user seems angry, frustrated, or is complaining about something in the call
- USER_NOT_UNDERSTANDING - The user explicitly says they don't understand or repeatedly asks for clarification
- HEARING_ISSUES - Either party can't hear the other ("hello?", "are you there?", "can you hear me?")
- DEAD_AIR - Unusually long silences in the conversation (use the timestamps to judge)
- USER_REQUESTING_FEATURE - The user asks for something the assistant can't fulfill
- ASSISTANT_LACKS_EMPATHY - The assistant ignores the user's personal situation or emotional state and continues pitching or pushing the agenda.
- USER_DETECTS_AI - The user suspects or identifies that they are talking to an AI/robot/bot rather than a real human.

## Call metrics (pre-computed)

Use these alongside the transcript for your analysis:
{{metrics}}

## Output format

Return ONLY a valid JSON object (no markdown):
{
    "tags": [
        {
            "tag": "TAG_NAME",
            "reason": "Short reason with evidence from the transcript"
        }
    ],
    "overall_sentiment": "positive|neutral|negative",
    "call_quality_score": <1-10>,
    "summary": "1-2 sentence summary of this segment"
}

If no tags apply, return an empty tags list. Always provide sentiment, score, and summary.
```

Historical replay uses the already-rendered system message captured by telemetry, not this template rendered again. In all **72 historical OVP-34 QA observations**, the rendered `Node Purpose` section was empty. This is an observed property of that dataset, not a universal production rule: current source supports node summaries.

The frozen QA configuration asks the evaluator to consider tags including:

- `UNCLEAR_CONVERSATION`
- `ASSISTANT_IN_LOOP`
- `ASSISTANT_REPLY_IMPROPER`
- `USER_FRUSTRATED`
- `USER_NOT_UNDERSTANDING`
- `HEARING_ISSUES`
- `DEAD_AIR`
- `USER_REQUESTING_FEATURE`
- `ASSISTANT_LACKS_EMPATHY`
- `USER_DETECTS_AI`

and return a JSON object containing:

```text
tags: [{tag, reason}, ...]
overall_sentiment: positive|neutral|negative
call_quality_score: 1-10
summary: short segment summary
```

The benchmark should use this frozen contract as a production-shaped QA suite, while keeping the architecture configurable enough to load another QA prompt/label set later.

## 7.6 Production-like parser

Production calls `parse_llm_json(...)`, then:

- if the parsed result is not a dict, it is coerced to `{}`;
- `tags` defaults to `[]`;
- `summary` defaults to `""`;
- score becomes `call_quality_score` when present;
- sentiment becomes `overall_sentiment` when present.

The benchmark should therefore score:

- strict JSON/schema validity;
- production-parser compatibility;
- semantic QA correctness.

## 7.7 Whole-call fallback

When node IDs are absent, Olegos formats the entire transcript, computes whole-call metrics, renders the same QA system prompt with empty node/previous-summary values, and sends:

```text
## Transcript
<whole transcript>
```

The benchmark should include at least a small fallback set because this is real product behavior.

## 7.8 Benchmark cases implied by source

Must include:

- clean node/no tags;
- each supported tag in isolation;
- multiple simultaneous tags;
- borderline/near-miss examples;
- negative sentiment without user frustration;
- user frustration without explicit profanity;
- repeated assistant question/loop;
- hearing issue;
- dead-air metrics/evidence;
- unsupported feature request;
- user identifies bot/AI;
- assistant misses emotional context;
- misleading keyword trap;
- conflicting evidence;
- evidence separated across a long transcript;
- prior-conversation context required to judge current node;
- whole-call fallback;
- malformed JSON;
- top-level array/non-dict result;
- hallucinated evidence in the `reason` field.

---

# 8. OOB task 5 — QA-support summaries

There are two related summary jobs in the current QA pipeline and they should not be conflated with runtime context compaction.

## 8.1 Prior-conversation summary

Source:

```text
api/services/workflow/qa/analysis.py::_generate_conversation_summary
```

For each later node, QA may summarize all prior-node conversation.

System prompt:

```text
You are summarizing a portion of a voice AI conversation. Produce a concise summary (3-5 sentences) covering key topics, information exchanged, and current state. We would be using this summary in doing a QA of the conversation that the voice AI agent did with someone so try to capture the nuances of the conversation as much as possible.
```

User message:

```text
## Conversation
<prior transcript>
```

Tracing name in product:

```text
conversation-summary-before-<node_name>
```

This is distinct from runtime:

```text
llm-context-summarization
```

The benchmark must treat them as different tasks/adapters even though both are summarization.

## 8.2 Node/script summary

Source:

```text
api/services/workflow/qa/node_summary.py
```

Exact `NODE_SUMMARY_SYSTEM_PROMPT`:

```text
You are analyzing a voice AI agent script. This is only a part of a larger script. Produce a concise summary (2-4 sentences) describing this script purpose, what the agent should accomplish, and key behaviors. We will be using this summary to do a QA on the conversation that the agent would do with someone so try to capture the nuances of the script as much as possible.
```

The exact node-description user formatter is source-known:

1. Start with `Node name: {node_name}`.
2. If an agent prompt exists, append a section `Agent prompt:\n{agent_prompt}`.
3. Collect custom tools as `- {tool_name}`, or `- {tool_name}: {description}` when a description exists.
4. Collect outgoing edges as `- {label}`, or `- {label}: {condition}` when a condition exists.
5. If at least one tool/edge exists, append `Available tools:\n` followed by the items, one per line.
6. Join the major sections with ONE newline.

Fixtures provide already-resolved inputs; database/tool fetching is not part of the benchmark. Formatter implementation is deferred to the adapter stage, not missing source knowledge.

These summaries are generated once per workflow definition when absent and then cached in `node_summaries`. They are therefore real QA LLM work but have different execution frequency from per-run prior-conversation summaries.

## 8.3 Benchmark implications

The benchmark should have separate subtypes:

```text
qa_support_previous_conversation
qa_support_node_script
```

Scoring should emphasize factual coverage required for downstream QA and hallucination avoidance rather than lexical overlap alone.

Must include:

- short/long prior transcript;
- facts required for later QA;
- irrelevant prior conversation;
- conflicting/corrected prior facts;
- node with no tools;
- node with several custom tools;
- node with several outgoing transition edges;
- tool descriptions/edge conditions that materially affect expected agent behavior;
- script containing multiple nuanced requirements;
- hallucination traps.

---

# 9. Cross-cutting production-fidelity requirements

## 9.1 Use adapters, not copied runtime

Each task adapter should own:

```text
case data -> production-shaped model request
raw model response -> production-like parsed/decision result
```

Do not import the full Olegos API runtime into the research package. If small pure prompt/formatting behavior is reproduced, document its source path and add tests that lock its intended behavior.

## 9.2 Keep raw and parsed outputs

Every execution should preserve:

- raw output exactly as received;
- parsed/decision output;
- model identity;
- endpoint/run configuration;
- prompt/completion token counts when reported;
- latency;
- error/timeout state;
- task-specific scores.

Never overwrite a benchmark input case with execution results.

## 9.3 Strict vs production-compatible scoring

At minimum distinguish:

```text
instruction_compliance
production_parse_success
semantic_correctness
```

This matters for both JSON jobs and voicemail because production parsers are more tolerant than the prompts request.

## 9.4 Determinism/repeatability

For quality comparison, default toward deterministic or low-variance generation where the server/model supports it. Record all decoding parameters in run metadata. If repeatability is a concern, run selected cases multiple times rather than silently treating one stochastic sample as definitive.

## 9.5 Token usage

Do not assume missing cache fields mean zero. Current Olegos telemetry explicitly preserves the difference between:

```text
provider reported 0
provider did not report the field
```

The standalone result schema should do the same where the endpoint exposes cache/reasoning details.

## 9.6 Baseline vs gold

Historical Qwen output is a **baseline output**, not automatically truth.

Every curated case should distinguish:

```text
baseline_output
expected/gold annotation
candidate_output
```

Quality decisions should be primarily candidate-vs-gold, with baseline-vs-gold and candidate-vs-baseline used to describe regression/improvement.

---

# 10. OVP-34 workload shape the benchmark should reflect

The final OVP-34 dataset contains exactly 242 selected OOB observations, all with captured model-facing messages and output:

```text
variable extraction          40 jobs
runtime context summary      56 jobs
voicemail detector           22 jobs
post-call QA                 72 jobs
QA previous-conversation     52 jobs
```

Context summarization is the highest-priority live-window workload for deep coverage. Post-call QA plus QA-support summaries form a large post-call pool and must still be isolated from the live dialogue engine in the eventual OVP-36 solution.

The benchmark should therefore avoid equal-weighting only by number of handcrafted examples. Reports should show per-task results separately and, where an aggregate is provided, clearly state the weighting scheme.

---

# 11. Standalone model-server contract

The research benchmark should depend on a narrow OpenAI-compatible chat-completions interface rather than on Olegos/MPS.

Target conceptual endpoint:

```text
POST /v1/chat/completions
```

The common client should make endpoint/model configuration external so the exact same runner can target:

```text
local Granite server
approved GB10/DGX Granite server
approved baseline Qwen server
future candidate server
```

The benchmark should not hard-code a private company endpoint.

The server/client contract should support, at minimum:

- system + user messages;
- non-streaming quality runs;
- model identifier;
- max completion tokens;
- temperature/decoding settings;
- usage fields when the server supplies them;
- clear HTTP/error/timeout handling.

A later hardware runner can additionally exercise concurrency without changing the semantic task adapters.

---

# 12. Local phase vs GB10/DGX phase

## 12.1 Local phase

Purpose:

- prove server/client wiring;
- prove adapters reproduce the intended request shape;
- validate parsers/scorers;
- run functional Granite quality experiments if local hardware/runtime allows;
- generate a reproducible result format.

Local Mac performance must **not** be presented as representative GB10 serving performance.

## 12.2 GB10/DGX phase

After the local benchmark is runnable and reviewed, request the research compute access Timur offered.

Use the same cases/adapters/client and change only the approved endpoint/server environment.

Measure at least:

- request latency p50/p95;
- prompt/completion tokens;
- decode/output throughput where measurable;
- aggregate requests/sec or jobs/sec;
- concurrency behavior;
- memory/GPU usage;
- failures/timeouts/OOM;
- model load/startup characteristics when relevant.

Concurrency levels should be explicitly configured rather than hidden in code. Initial candidate levels can include 1/2/4/8, adjusted to the approved environment.

Do not independently start another model process on a shared production-serving GPU. GB10/DGX tests must use the approved research environment/resources.

---

# 13. Quality decision architecture

There is **no single mandatory all-jobs winner**.

Decision output should be per job, for example:

```text
extraction       PASS/FAIL quality | PASS/FAIL hardware | decision
context summary  PASS/FAIL quality | PASS/FAIL hardware | decision
voicemail        PASS/FAIL quality | PASS/FAIL hardware | decision
QA               PASS/FAIL quality | PASS/FAIL hardware | decision
QA support       PASS/FAIL quality | PASS/FAIL hardware | decision
```

This allows a candidate to serve classification/extraction while a stronger model remains on QA, which is compatible with the intended job-tier direction.

The benchmark must define task-specific quality gates **before final candidate results are used for the decision**.

---

# 14. Required adapter modules implied by this audit

The implementation plan should provide equivalents of:

```text
ExtractionAdapter
RuntimeContextSummaryAdapter
VoicemailAdapter
PostCallQAAdapter
QASupportPreviousConversationAdapter
QASupportNodeScriptAdapter
```

These can share common abstractions, but the two summarization families must remain semantically distinct.

Each adapter should expose a small contract conceptually equivalent to:

```text
build_request(case) -> model request
parse_response(raw_response) -> production-like parsed result
```

Task scorers remain separate from request adapters.

---

# 15. Source-to-test traceability requirement

Every production-fidelity behavior copied into the research benchmark should be traceable in `SOURCE_AUDIT.md` or code comments/tests to a source path.

Examples:

```text
Extraction tool response 2,000-char cap
  -> api/services/workflow/pipecat_engine_variable_extractor.py

Context summary 4000 target / keep 2 / 30s
  -> api/services/workflow/pipecat_engine_context_summarizer.py

Voicemail CONVERSATION-first substring decision
  -> olegos-pipecat/src/pipecat/extensions/voicemail/voicemail_detector.py

QA per-node prompt composition
  -> api/services/workflow/qa/analysis.py

QA conversation/node summary prompts
  -> api/services/workflow/qa/node_summary.py

Tier resolution
  -> api/services/pipecat/tier_resolver.py

MPS current tier table
  -> olegos-mps/defaults.yaml
```

This traceability is what lets us claim the benchmark is Olegos-shaped without importing Olegos itself.

---

# 16. Open items to freeze in CASE_MATRIX / implementation plan

This audit establishes the product-facing contracts. The next design artifacts must still freeze:

1. exact curated case count per task and difficulty bucket;
2. access to and validation of the private/local export of the 242 captured OVP-34 requests and outputs;
3. gold-label annotation process;
4. task-specific scoring formulas and thresholds;
5. baseline endpoint availability/approval;
6. exact Granite model identifier and serving runtime;
7. local runtime feasibility on Ibrahim's Mac;
8. final model generation parameters per task;
9. GB10/DGX benchmark concurrency schedule;
10. reporting format and go/no-go gates.

These decisions belong in `CASE_MATRIX.md` and the implementation plan. They should be reviewed before Codex writes the benchmark implementation.

---

# 17. Audit conclusion

The current source supports the planned **mini Olegos OOB lab** architecture.

The five OOB workload families are already identifiable at their call sites, and the application knows which job it is executing. The benchmark therefore does not need an intent classifier or a replica of the Olegos workflow engine.

The correct research boundary is:

```text
realistic Olegos-shaped case
    -> task-specific production request adapter
    -> common candidate model endpoint
    -> production-like parser/decision adapter
    -> semantic scorer
```

This is sufficient to answer the pre-integration questions:

- Can Granite perform each OOB job correctly?
- Which jobs can safely move to it?
- What regressions appear relative to gold/baseline?
- Once moved to approved GB10/DGX research hardware, is the model actually advantageous under Olegos-shaped load?

Only after those gates pass should OVP-36 product routing/MPS integration be designed against the latest live repositories.
