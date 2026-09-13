"""Source-shaped synthetic requests only; no execution, replay, or scoring.

Prompt text is owned by contracts.json. Source rendering follows SOURCE_AUDIT
sections 4–8; provider decoding is deliberately supplied by the caller.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
import json
import re
from typing import TypeVar

from .config import persistent_config_projection
from .contracts import PromptContract, load_contract, render_contract
from .dataset import QA_METRIC_FIELDS, validate_curated_cases
from .requests import PreparedRequest, prepare_generated_request
from .runner import PlannedExecution
from .schemas import (
    BenchmarkCase, ContextMessage, ExtractionInput, GenerationConfig,
    LLMSpecificMessage, Message, QAConversationSummaryInput, QAEvaluationInput,
    QANodeSummaryInput, RunConfig, RuntimeContextSummaryInput, StrictModel,
    Task, VoicemailInput,
)


class AdapterError(ValueError):
    """Safe construction category, without retained lower-level exceptions."""


InputModel = TypeVar("InputModel", bound=StrictModel)


def _checked_input(value: InputModel, expected: type[InputModel]) -> InputModel:
    if not isinstance(value, expected):
        raise AdapterError("invalid_adapter_input")
    failure = None
    try:
        # Revalidate unchecked model_copy changes, retaining omitted optional fields.
        result = expected.model_validate_json(value.model_dump_json(exclude_unset=True, warnings="error"))
    except ValueError:
        failure = AdapterError("invalid_adapter_input")
    if failure is not None:
        raise failure
    return result


def _contract(task: Task, contract: PromptContract | None) -> PromptContract:
    if contract is not None and not isinstance(contract, PromptContract):
        raise AdapterError("invalid_adapter_contract")
    failure = None
    try:
        frozen = load_contract(task.value if contract is None else contract.id)
    except ValueError:
        failure = AdapterError("invalid_adapter_contract")
    if failure is not None:
        raise failure
    if frozen.task != task or (contract is not None and contract != frozen):
        raise AdapterError("invalid_adapter_contract")
    return frozen


def _pair(system: str, user: str) -> list[dict[str, object]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _render_pair(contract: PromptContract, values: dict[str, str]) -> list[dict[str, object]]:
    rendered = render_contract(contract, values)
    return _pair(rendered.system_text, rendered.user_text)


def _message_dict(message: ContextMessage) -> dict:
    return message.model_dump(mode="json", exclude_unset=True)


def _extraction_tool(raw: str, name: str) -> str | None:
    if raw.strip() == '{"status": "done"}':
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if "data" in parsed:
                parsed = parsed["data"]
            else:
                for key in ("status", "status_code"):
                    parsed.pop(key, None)
            formatted = json.dumps(parsed, ensure_ascii=False)
        else:
            formatted = raw
    except (json.JSONDecodeError, TypeError):
        formatted = raw
    if len(formatted) > 2000:
        formatted = formatted[:2000] + "...(truncated)"
    return f"[Tool Response: {name}]\n{formatted}"


def render_extraction(
    input: ExtractionInput, contract: PromptContract | None = None,
) -> list[dict[str, object]]:
    input = _checked_input(input, ExtractionInput)
    contract = _contract(Task.EXTRACTION, contract)
    if contract.extraction_variables and (
        input.variables != contract.extraction_variables or input.extraction_prompt
    ):
        raise AdapterError("frozen_extraction_variables_mismatch")
    names = {call.id: call.function.name for message in input.messages
             if isinstance(message, ContextMessage) and message.role == "assistant"
             for call in message.tool_calls}
    lines = []
    for message in input.messages:
        if isinstance(message, LLMSpecificMessage):
            continue
        content = message.content
        if isinstance(content, tuple):
            content = " ".join(item.text for item in content)
        if message.role in ("assistant", "user") and content:
            lines.append(f"{message.role}: {content}")
        elif message.role == "tool":
            # Production's formatter expects raw tool content to be a string.
            if not isinstance(message.content, str):
                raise AdapterError("extraction_tool_content_must_be_text")
            formatted = _extraction_tool(message.content, names.get(message.tool_call_id, "unknown"))
            if formatted:
                lines.append(formatted)
    values = {
        "variable_lines": "\n".join(contract.variable_line_template.format(
            name=v.name, type=v.type, prompt=v.hint) for v in input.variables),
        "formatted_history": "\n".join(lines),
    }
    if contract.optional_system_append:
        values[contract.optional_system_append] = input.extraction_prompt or ""
    return _render_pair(contract, values)


@dataclass(frozen=True)
class SummarySelection:
    messages: tuple[Message, ...] = field(repr=False)
    last_summarized_index: int


def _pending(content: str) -> bool:
    if content == "IN_PROGRESS":
        return True
    try:
        parsed = json.loads(content)
        return isinstance(parsed, dict) and parsed.get("type") == "async_tool" and parsed.get("status") == "started"
    except (json.JSONDecodeError, ValueError):
        return False


def select_summary_context(input: RuntimeContextSummaryInput) -> SummarySelection:
    input = _checked_input(input, RuntimeContextSummaryInput)
    messages = input.messages
    if (not input.context_compaction_enabled or not input.has_previous_node
            or input.is_realtime or len(messages) <= 6):
        return SummarySelection((), -1)
    start = int(isinstance(messages[0], ContextMessage) and messages[0].role == "system")
    end = len(messages) - 2
    pending = {}
    # Scan only the proposed range, retaining original indices including opaque messages.
    for index in range(start, end):
        message = messages[index]
        if isinstance(message, LLMSpecificMessage):
            continue
        if message.role == "assistant":
            for call in message.tool_calls:
                pending[call.id] = index
        if message.role == "tool" and message.tool_call_id in pending:
            content = message.content if isinstance(message.content, str) else ""
            if not _pending(content):
                pending.pop(message.tool_call_id)
        if message.role == "developer" and isinstance(message.content, str):
            try:
                parsed = json.loads(message.content)
                if (isinstance(parsed, dict) and parsed.get("type") == "async_tool"
                        and parsed.get("status") == "finished"):
                    call_id = parsed.get("tool_call_id")
                    if call_id and call_id in pending:
                        pending.pop(call_id)
            except (json.JSONDecodeError, ValueError):
                pass
    if pending:
        end = min(pending.values())
    if start >= end:
        return SummarySelection((), -1)
    return SummarySelection(messages[start:end], end - 1)


def format_summary_transcript(messages: Iterable[Message]) -> str:
    parts = []
    for message in messages:
        if isinstance(message, LLMSpecificMessage):
            continue
        message = _checked_input(message, ContextMessage)
        content = message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, tuple):
            text = " ".join(item.text for item in content)
        else:
            text = str(content)
        if text:
            parts.append(f"{message.role.upper()}: {text}")
        for call in message.tool_calls:
            parts.append(f"TOOL_CALL: {call.function.name}({call.function.arguments})")
        if message.role == "tool":
            call_id = _message_dict(message).get("tool_call_id", "unknown")
            parts.append(f"TOOL_RESULT[{call_id}]: {text}")
    return "\n\n".join(parts)


def render_runtime_summary(
    input: RuntimeContextSummaryInput, contract: PromptContract | None = None,
) -> list[dict[str, object]] | None:
    contract = _contract(Task.CONTEXT_SUMMARY, contract)
    selection = select_summary_context(input)
    if not selection.messages:
        return None
    return _render_pair(contract, {"formatted_transcript": format_summary_transcript(selection.messages)})


def apply_context_summary(
    current_messages: tuple[Message, ...], last_index: int, summary: str | None,
) -> tuple[Message, ...]:
    """Pure apply-time control, without scheduling or modifying the supplied context."""
    if type(last_index) is not int or (summary is not None and not isinstance(summary, str)):
        raise AdapterError("invalid_summary_application")
    current = tuple(current_messages)
    if not summary or last_index < 0:
        return current
    system = next((m for m in current if isinstance(m, ContextMessage) and m.role == "system"), None)
    replacement = ContextMessage(role="user", content=f"Conversation summary: {summary}")
    return ((system,) if system is not None else ()) + (replacement,) + current[last_index + 1:]


def render_voicemail(
    input: VoicemailInput, contract: PromptContract | None = None,
) -> list[dict[str, object]]:
    input = _checked_input(input, VoicemailInput)
    contract = _contract(Task.VOICEMAIL, contract)
    if any(isinstance(message, LLMSpecificMessage) for message in input.messages):
        raise AdapterError("unsupported_voicemail_message")
    return [{"role": "system", "content": contract.system_text},
            *(_message_dict(message) for message in input.messages)]


def render_qa_evaluation(
    input: QAEvaluationInput, contract: PromptContract | None = None,
) -> list[dict[str, object]]:
    input = _checked_input(input, QAEvaluationInput)
    contract = _contract(Task.QA, contract)
    if input.metrics.keys() != set(QA_METRIC_FIELDS):
        raise AdapterError("invalid_qa_metric_fields")
    values = {
        "node_summary": input.node_summary,
        "previous_conversation_summary": input.previous_conversation_summary,
        "transcript": input.transcript,
        "metrics": json.dumps({key: input.metrics[key] for key in QA_METRIC_FIELDS}, indent=2),
    }
    # The frozen template uses only these simple placeholders. Source substitutes
    # once and then unescapes literal backslash-n in the whole system string.
    system = re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda match: values[match[1]], contract.system_template)
    system = system.replace("\\n", "\n")
    return _pair(system, contract.user_template.format(transcript=input.transcript))


def render_qa_conversation_summary(
    input: QAConversationSummaryInput, contract: PromptContract | None = None,
) -> list[dict[str, object]]:
    input = _checked_input(input, QAConversationSummaryInput)
    contract = _contract(Task.QA_CONVERSATION_SUMMARY, contract)
    return _render_pair(contract, {"transcript": input.prior_transcript})


def render_qa_node_summary(
    input: QANodeSummaryInput, contract: PromptContract | None = None,
) -> list[dict[str, object]]:
    input = _checked_input(input, QANodeSummaryInput)
    contract = _contract(Task.QA_NODE_SUMMARY, contract)
    parts = [f"Node name: {input.node_name}"]
    if input.agent_prompt:
        parts.append(f"Agent prompt:\n{input.agent_prompt}")
    tools = [f"- {tool.name}" + (f": {tool.description}" if tool.description else "")
             for tool in input.custom_tools]
    tools.extend(f"- {edge.label}" + (f": {edge.condition}" if edge.condition else "")
                 for edge in input.outgoing_edges)
    if tools:
        parts.append("Available tools:\n" + "\n".join(tools))
    return _pair(contract.system_text, "\n".join(parts))


_RENDERERS = {
    Task.EXTRACTION: render_extraction, Task.CONTEXT_SUMMARY: render_runtime_summary,
    Task.VOICEMAIL: render_voicemail, Task.QA: render_qa_evaluation,
    Task.QA_CONVERSATION_SUMMARY: render_qa_conversation_summary,
    Task.QA_NODE_SUMMARY: render_qa_node_summary,
}


def prepare_curated_request(
    case: BenchmarkCase, *, model: str, generation: GenerationConfig,
) -> PreparedRequest:
    case, = validate_curated_cases((case,))
    if case.exercise.kind != "model":
        raise AdapterError("control_has_no_model_request")
    failure = None
    try:
        generation = _checked_input(generation, GenerationConfig)
        messages = _RENDERERS[case.task](case.input, load_contract(case.contract_id))
        if messages is None:
            raise AdapterError("model_case_has_no_request")
        request = prepare_generated_request(messages, case_id=case.id, task=case.task,
                                           source=case.source, model=model, generation=generation)
    except ValueError:
        failure = AdapterError("invalid_curated_request")
    if failure is not None:
        raise failure
    return request


def prepare_curated_plan(
    cases: Iterable[BenchmarkCase], *, config: RunConfig, resolved_model: str,
) -> tuple[PlannedExecution, ...]:
    cases = validate_curated_cases(cases)
    config = _checked_input(config, RunConfig)
    failure = None
    try:
        persistent_config_projection(config, resolved_model=resolved_model)
    except ValueError:
        failure = AdapterError("invalid_candidate_configuration")
    if failure is not None:
        raise failure
    requests = tuple(prepare_curated_request(case, model=resolved_model, generation=config.generation)
                     for case in cases if case.exercise.kind == "model")
    return tuple(PlannedExecution(request=request, repetition_index=index)
                 for index in range(config.repetitions) for request in requests)
