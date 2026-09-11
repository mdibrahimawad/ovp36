"""Strict data contracts; no prompt execution, parsing, or scoring logic.

Models are frozen and sequences are tuples. Opaque JSON payloads (for example
precomputed QA metrics) remain JSON containers; consumers must treat them as
read-only. Dataset functions never modify their inputs.
"""

from enum import StrEnum
import math
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, JsonValue as PydanticJsonValue, StringConstraints,
    field_validator, model_validator,
)

NonEmpty = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveNumber = Annotated[float, Field(gt=0)]
Scalar = str | int | float | bool | None


def _finite_json(value: PydanticJsonValue) -> PydanticJsonValue:
    # Pydantic's recursive JsonValue does not inherit allow_inf_nan=False.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    return value


JsonValue = Annotated[PydanticJsonValue, AfterValidator(_finite_json)]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class Source(StrEnum):
    CURATED = "curated"
    ADVERSARIAL = "adversarial"
    OVP34_REPLAY = "ovp34_replay"


class Difficulty(StrEnum):
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class Task(StrEnum):
    EXTRACTION = "extraction"
    CONTEXT_SUMMARY = "context_summary"
    VOICEMAIL = "voicemail"
    QA = "qa"
    QA_CONVERSATION_SUMMARY = "qa_conversation_summary"
    QA_NODE_SUMMARY = "qa_node_summary"


class Provenance(StrictModel):
    source_ref: NonEmpty
    source_revision: NonEmpty | None = None
    artifact_hash: NonEmpty | None = None
    historical_run_id: NonEmpty | None = None
    notes: str = ""


class EvidenceRef(StrictModel):
    input_pointer: NonEmpty
    start_char: Annotated[int, Field(ge=0)] | None = None
    end_char: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def valid_span(self) -> Self:
        if not self.input_pointer.startswith("/"):
            raise ValueError("input_pointer must be a JSON pointer starting with /")
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("character offsets must be provided together")
        if self.start_char is not None and self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char (half-open span)")
        return self


class FactAnnotation(StrictModel):
    id: NonEmpty
    statement: NonEmpty
    evidence: tuple[EvidenceRef, ...] = ()
    accepted_phrases: tuple[NonEmpty, ...] = ()


class CriticalCheck(StrictModel):
    id: NonEmpty
    description: NonEmpty


class CriticalCheckResult(StrictModel):
    check_id: NonEmpty
    status: Literal["pass", "fail", "pending", "not_applicable"]
    explanation: str = ""


class TextItem(StrictModel):
    type: Literal["text"]
    text: str


class FunctionCall(StrictModel):
    name: NonEmpty
    arguments: str


class ToolCall(StrictModel):
    id: NonEmpty
    type: Literal["function"]
    function: FunctionCall


class ContextMessage(StrictModel):
    role: Literal["system", "assistant", "user", "tool", "function"]
    content: str | tuple[TextItem, ...] | None
    tool_calls: tuple[ToolCall, ...] = ()
    function_call: FunctionCall | None = None
    tool_call_id: NonEmpty | None = None
    name: NonEmpty | None = None


class LLMSpecificMessage(StrictModel):
    """Fixture representation only; does not import Pipecat."""

    kind: Literal["llm_specific"]
    payload: JsonValue


Message = ContextMessage | LLMSpecificMessage


class ModelExercise(StrictModel):
    kind: Literal["model"]


class AdapterControlExercise(StrictModel):
    kind: Literal["adapter_control"]
    current_messages: tuple[Message, ...] | None = None


class ResponseContractExercise(StrictModel):
    kind: Literal["response_contract"]
    raw_output: str | None


class TransportContractExercise(StrictModel):
    kind: Literal["transport_contract"]
    response: Literal["timeout", "http_error", "disconnect"]
    http_status: Annotated[int, Field(ge=400, le=599)] | None = None

    @model_validator(mode="after")
    def status_matches_response(self) -> Self:
        if (self.response == "http_error") != (self.http_status is not None):
            raise ValueError("http_status is required only for http_error")
        return self


ExerciseSpec = Annotated[
    ModelExercise | AdapterControlExercise | ResponseContractExercise
    | TransportContractExercise,
    Field(discriminator="kind"),
]


class ExtractionVariable(StrictModel):
    name: NonEmpty
    type: Literal["string", "number", "boolean"]
    hint: str


class ExtractionInput(StrictModel):
    messages: tuple[Message, ...]
    variables: Annotated[tuple[ExtractionVariable, ...], Field(min_length=1)]
    extraction_prompt: str | None = None

    @model_validator(mode="after")
    def unique_variables(self) -> Self:
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("extraction variable names must be unique")
        return self


class RuntimeContextSummaryInput(StrictModel):
    messages: tuple[Message, ...]
    context_compaction_enabled: bool
    has_previous_node: bool
    is_realtime: bool


class VoicemailInput(StrictModel):
    # Context after the classifier system instruction; preserve message roles.
    messages: tuple[Message, ...]
    is_partial: bool = False
    is_truncated: bool = False
    long_speech_timeout_seconds: PositiveNumber = 8.0


class QAEvaluationInput(StrictModel):
    scope: Literal["node", "whole_call"]
    transcript: str
    metrics: dict[str, JsonValue]
    node_summary: str
    previous_conversation_summary: str

    @model_validator(mode="after")
    def whole_call_context(self) -> Self:
        if self.scope == "whole_call" and (
            self.node_summary or self.previous_conversation_summary
        ):
            raise ValueError("whole_call requires empty node and previous summaries")
        return self


class QAConversationSummaryInput(StrictModel):
    prior_transcript: str


class CustomTool(StrictModel):
    name: NonEmpty
    description: str


class OutgoingEdge(StrictModel):
    label: NonEmpty
    condition: str


class QANodeSummaryInput(StrictModel):
    node_type: Literal["agent", "start"]
    node_name: NonEmpty
    agent_prompt: str | None
    custom_tools: tuple[CustomTool, ...]
    outgoing_edges: tuple[OutgoingEdge, ...]


class MissingValuePolicy(StrictModel):
    """Explicit per-case policy, including null or string sentinels if specified."""

    allow_absent: bool
    values: tuple[Scalar, ...]

    @model_validator(mode="after")
    def has_representation(self) -> Self:
        if not self.allow_absent and not self.values:
            raise ValueError("missing policy must allow at least one representation")
        return self


class ExtractionFieldExpected(StrictModel):
    state: Literal["known", "missing"]
    acceptable_values: tuple[Scalar, ...] = ()
    missing_policy: MissingValuePolicy | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    superseded_values: tuple[Scalar, ...] = ()

    @model_validator(mode="after")
    def state_has_gold(self) -> Self:
        if self.state == "known" and not self.acceptable_values:
            raise ValueError("known fields require acceptable_values")
        if self.state == "missing" and (
            self.missing_policy is None or self.acceptable_values
        ):
            raise ValueError("missing fields require a policy and no known values")
        return self


class ExtractionExpected(StrictModel):
    fields: dict[NonEmpty, ExtractionFieldExpected]
    critical_checks: tuple[CriticalCheck, ...] = ()


class Correction(StrictModel):
    old_fact_id: NonEmpty
    new_fact_id: NonEmpty


class SummaryExpected(StrictModel):
    """Common fact rubric for all three summary tasks; no semantic scoring here."""

    required_facts: tuple[FactAnnotation, ...]
    critical_facts: tuple[NonEmpty, ...] = ()
    superseded_facts: tuple[FactAnnotation, ...] = ()
    forbidden_facts: tuple[FactAnnotation, ...] = ()
    user_preferences: tuple[NonEmpty, ...] = ()
    decisions: tuple[NonEmpty, ...] = ()
    unresolved_questions: tuple[NonEmpty, ...] = ()
    action_items: tuple[NonEmpty, ...] = ()
    prior_failures: tuple[NonEmpty, ...] = ()
    user_constraints: tuple[NonEmpty, ...] = ()
    downstream_relevant: tuple[NonEmpty, ...] = ()
    purpose: tuple[NonEmpty, ...] = ()
    key_behaviors: tuple[NonEmpty, ...] = ()
    tools: tuple[NonEmpty, ...] = ()
    transitions: tuple[NonEmpty, ...] = ()
    prohibitions: tuple[NonEmpty, ...] = ()
    optional_behaviors: tuple[NonEmpty, ...] = ()
    safe_to_omit: tuple[FactAnnotation, ...] = ()
    corrections: tuple[Correction, ...] = ()
    sentence_range: tuple[PositiveInt, PositiveInt] | None = None
    critical_checks: tuple[CriticalCheck, ...] = ()

    @model_validator(mode="after")
    def fact_references(self) -> Self:
        groups = (self.required_facts, self.superseded_facts,
                  self.forbidden_facts, self.safe_to_omit)
        ids = [fact.id for group in groups for fact in group]
        if len(ids) != len(set(ids)):
            raise ValueError("fact IDs must be unique across annotation groups")
        required = {fact.id for fact in self.required_facts}
        for name in (
            "critical_facts", "user_preferences", "decisions", "unresolved_questions",
            "action_items", "prior_failures", "user_constraints", "downstream_relevant",
            "purpose", "key_behaviors", "tools", "transitions", "prohibitions",
            "optional_behaviors",
        ):
            if not set(getattr(self, name)) <= required:
                raise ValueError(f"{name} must reference required_facts IDs")
        superseded = {fact.id for fact in self.superseded_facts}
        if any(c.old_fact_id not in superseded or c.new_fact_id not in required
               for c in self.corrections):
            raise ValueError("corrections must reference superseded and required facts")
        if self.sentence_range and self.sentence_range[0] > self.sentence_range[1]:
            raise ValueError("sentence_range must be ordered")
        return self


RuntimeContextSummaryExpected = SummaryExpected
QAConversationSummaryExpected = SummaryExpected
QANodeSummaryExpected = SummaryExpected


class VoicemailExpected(StrictModel):
    label: Literal["CONVERSATION", "VOICEMAIL"] | None
    ambiguous: bool
    rationale: NonEmpty
    critical_checks: tuple[CriticalCheck, ...] = ()


class QAEvaluationExpected(StrictModel):
    expected_tags: tuple[NonEmpty, ...]
    forbidden_tags: tuple[NonEmpty, ...] = ()
    expected_sentiment: Literal["positive", "neutral", "negative"]
    quality_score_target: Annotated[float, Field(ge=1, le=10)] | None = None
    quality_score_range: tuple[
        Annotated[float, Field(ge=1, le=10)], Annotated[float, Field(ge=1, le=10)]
    ] | None = None
    tag_evidence: dict[NonEmpty, tuple[EvidenceRef, ...]]
    summary_facts: tuple[FactAnnotation, ...]
    critical_checks: tuple[CriticalCheck, ...] = ()

    @model_validator(mode="after")
    def consistent_gold(self) -> Self:
        if (self.quality_score_target is None) == (self.quality_score_range is None):
            raise ValueError("provide exactly one score target or range")
        if self.quality_score_range and self.quality_score_range[0] > self.quality_score_range[1]:
            raise ValueError("quality_score_range must be ordered")
        if set(self.expected_tags) & set(self.forbidden_tags):
            raise ValueError("expected and forbidden tags must not overlap")
        if set(self.tag_evidence) != set(self.expected_tags):
            raise ValueError("tag_evidence must cover exactly the expected tags")
        return self


class ContractExpected(StrictModel):
    """Named observable assertions for non-model exercises; evaluated later."""

    assertions: Annotated[dict[NonEmpty, JsonValue], Field(min_length=1)]
    critical_checks: tuple[CriticalCheck, ...] = ()


TaskInput = (ExtractionInput | RuntimeContextSummaryInput | VoicemailInput
             | QAEvaluationInput | QAConversationSummaryInput | QANodeSummaryInput)
TaskExpected = ExtractionExpected | SummaryExpected | VoicemailExpected | QAEvaluationExpected

TASK_MODELS = {
    Task.EXTRACTION: (ExtractionInput, ExtractionExpected),
    Task.CONTEXT_SUMMARY: (RuntimeContextSummaryInput, SummaryExpected),
    Task.VOICEMAIL: (VoicemailInput, VoicemailExpected),
    Task.QA: (QAEvaluationInput, QAEvaluationExpected),
    Task.QA_CONVERSATION_SUMMARY: (QAConversationSummaryInput, SummaryExpected),
    Task.QA_NODE_SUMMARY: (QANodeSummaryInput, SummaryExpected),
}


class BenchmarkCase(StrictModel):
    schema_version: Literal["1"] = "1"
    id: NonEmpty
    task: Task
    source: Source
    difficulty: Difficulty
    critical: bool
    contract_id: NonEmpty
    exercise: ExerciseSpec
    input: TaskInput
    expected: TaskExpected | ContractExpected | None
    tags: tuple[NonEmpty, ...] = ()
    notes: str = ""
    provenance: Provenance | None = None

    @property
    def is_model_quality_case(self) -> bool:
        return self.exercise.kind == "model"

    @model_validator(mode="after")
    def task_contract(self) -> Self:
        input_type, expected_type = TASK_MODELS[self.task]
        if not isinstance(self.input, input_type):
            raise ValueError("input does not match task")
        if self.expected is None:
            if self.source != Source.OVP34_REPLAY:
                raise ValueError("only unannotated ovp34_replay cases may omit expected")
            return self
        if isinstance(self.expected, ContractExpected):
            if self.is_model_quality_case:
                raise ValueError("model exercises require task gold, not contract assertions")
        elif not isinstance(self.expected, expected_type):
            raise ValueError("expected does not match task")
        if isinstance(self.input, ExtractionInput) and isinstance(self.expected, ExtractionExpected):
            variables = {v.name: v.type for v in self.input.variables}
            if set(self.expected.fields) != set(variables):
                raise ValueError("extraction gold must cover exactly the requested fields")
            types = {"string": (str,), "number": (int, float), "boolean": (bool,)}
            for name, gold in self.expected.fields.items():
                if gold.state == "known" and any(
                    type(value) not in types[variables[name]] for value in gold.acceptable_values
                ):
                    raise ValueError("known extraction gold must match the variable type")
        return self


def validate_base_url(value: str) -> str:
    """Reject URL-embedded credentials; tokens belong in environment variables."""
    try:
        parts = urlsplit(value)
        port = parts.port
        valid = (parts.scheme in {"http", "https"} and parts.hostname
                 and parts.username is None and parts.password is None
                 and not parts.query and not parts.fragment
                 and not any(c.isspace() for c in value))
        if not valid or (port is not None and port == 0):
            raise ValueError
    except ValueError:
        raise ValueError("base URL must be HTTP(S), without credentials, query, or fragment") from None
    return value


class EndpointConfig(StrictModel):
    alias: NonEmpty
    base_url: NonEmpty | None = None
    base_url_env: EnvName | None = None
    model: NonEmpty | None = None
    model_env: EnvName | None = None
    api_key_env: EnvName | None = None

    @field_validator("base_url")
    @classmethod
    def safe_url(cls, value: str | None) -> str | None:
        return validate_base_url(value) if value is not None else None

    @model_validator(mode="after")
    def exclusive_references(self) -> Self:
        if (self.base_url is None) == (self.base_url_env is None):
            raise ValueError("provide exactly one base_url or base_url_env")
        if (self.model is None) == (self.model_env is None):
            raise ValueError("provide exactly one model or model_env")
        return self


class GenerationConfig(StrictModel):
    temperature: Annotated[float, Field(ge=0)] = 0.0
    max_tokens: PositiveInt
    seed: int | None = None
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0


class ServerMetadata(StrictModel):
    runtime: NonEmpty | None = None
    runtime_version: NonEmpty | None = None
    model_checkpoint: NonEmpty | None = None
    model_revision: NonEmpty | None = None
    model_artifact_hash: NonEmpty | None = None
    quantization: NonEmpty | None = None
    chat_template_hash: NonEmpty | None = None
    context_limit: PositiveInt | None = None


class RunConfig(StrictModel):
    endpoint: EndpointConfig
    generation: GenerationConfig
    timeout_seconds: PositiveNumber = 30.0
    concurrency: PositiveInt = 1
    repetitions: PositiveInt = 1
    experiment_label: NonEmpty
    evaluation: Literal["canonical", "exploratory"]
    server: ServerMetadata = Field(default_factory=ServerMetadata)


class BenchmarkError(StrictModel):
    stage: Literal["input", "transport", "parser", "schema", "execution"]
    code: NonEmpty
    message: str
    retryable: bool = False


class ParseResult(StrictModel):
    # Production json.loads accepts NaN/Infinity. Preserve them in parser output
    # without weakening the finite-only case/config/score schemas. Do not let
    # Pydantic's default JSON serialization silently replace them with null.
    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="constants")
    raw_response: str | None = None
    strict_format_valid: bool
    strict_json_valid: bool | None
    # None means task schema validation has not been performed.
    schema_valid: bool | None = None
    production_parse_success: bool
    production_usable: bool
    parser_path: Literal[
        "empty", "direct", "fence", "object", "array", "raw_fallback", "label", "text", "none"
    ]
    parsed_value: PydanticJsonValue
    normalized_value: PydanticJsonValue
    diagnostics: tuple[BenchmarkError, ...] = ()


class MetricValue(StrictModel):
    value: float | None
    numerator: float | None = None
    denominator: Annotated[float, Field(ge=0)] | None = None


class ScoreResult(StrictModel):
    """Exercise identity is mandatory so contract checks cannot look like model scores."""

    case_id: NonEmpty
    exercise_kind: Literal["model", "adapter_control", "response_contract", "transport_contract"]
    status: Literal["complete", "pending_review", "not_scored"]
    metrics: dict[NonEmpty, MetricValue]
    critical_checks: tuple[CriticalCheckResult, ...] = ()
