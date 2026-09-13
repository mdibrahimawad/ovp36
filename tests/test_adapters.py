"""Synthetic construction tests only; no candidate execution or authored dataset."""

from contextlib import ExitStack
import json
import unittest
from unittest.mock import patch

from test_contracts import EXTRACTION, RUNTIME, CONVERSATION, NODE, GOLDENS
from test_curated_dataset import case, fixture
from ovp36_benchmark.adapters import (
    AdapterError, apply_context_summary, format_summary_transcript,
    prepare_curated_plan, prepare_curated_request, render_extraction,
    render_qa_conversation_summary, render_qa_evaluation, render_qa_node_summary,
    render_runtime_summary, render_voicemail, select_summary_context,
)
from ovp36_benchmark.contracts import load_contract
from ovp36_benchmark.dataset import hash_dataset
from ovp36_benchmark.identity import fingerprint_execution_plan
from ovp36_benchmark.requests import prepare_generated_request
from ovp36_benchmark.schemas import (
    ContextMessage, ExtractionInput, GenerationConfig, QAConversationSummaryInput,
    QAEvaluationInput, QANodeSummaryInput, RunConfig, RuntimeContextSummaryInput, VoicemailInput,
)


def message(role="user", content="Synthetic text", **extra):
    return {"role": role, "content": content, **extra}


def call(call_id="test-call"):
    return message("assistant", None, tool_calls=[{"id": call_id, "type": "function",
                   "function": {"name": "lookup", "arguments": '{ "key": "value" }'}}])


def runtime(messages, **flags):
    return RuntimeContextSummaryInput.model_validate_json(json.dumps({
        "messages": messages, "context_compaction_enabled": True,
        "has_previous_node": True, "is_realtime": False, **flags}))


def extraction(messages, **extra):
    return ExtractionInput.model_validate_json(json.dumps({
        "messages": messages, "variables": [{"name": "x", "type": "string", "hint": "As stated."}], **extra}))


def config(repetitions=2, **generation):
    return RunConfig.model_validate_json(json.dumps({
        "endpoint": {"alias": "synthetic", "base_url_env": "UNUSED_TEST_ENDPOINT", "model": "synthetic-model"},
        "generation": {"temperature": 0.25, "max_tokens": 71, "top_p": 0.8, "seed": 9, **generation},
        "repetitions": repetitions, "experiment_label": "synthetic-construction", "evaluation": "exploratory"}))


class ExtractionAdapterTests(unittest.TestCase):
    def test_exact_prompt_variables_append_and_multipart_history(self):
        input = extraction([message("system", "excluded"), message(content=[{"type": "text", "text": "Blue"},
                           {"type": "text", "text": "please"}]), message("assistant", "Noted.")],
                           variables=[{"name": "x", "type": "string", "hint": "As stated."},
                                      {"name": "n", "type": "number", "hint": "Count."},
                                      {"name": "b", "type": "boolean", "hint": "Confirmed."}],
                           extraction_prompt="Keep literal {values}.")
        self.assertEqual(render_extraction(input), [
            {"role": "system", "content": EXTRACTION + "\n\nKeep literal {values}."},
            {"role": "user", "content": "\n\nVariables to extract:\n- x (VariableType.string): As stated.\n"
             "- n (VariableType.number): Count.\n- b (VariableType.boolean): Confirmed.\n\n"
             "Conversation history:\nuser: Blue please\nassistant: Noted."}])
        self.assertEqual(render_extraction(extraction([]))[0]["content"], EXTRACTION)

    def test_tool_cleanup_names_and_exact_transition_comparison(self):
        input = extraction([call(), message("tool", '{"data":{"city":"Paris"},"status":200}', tool_call_id="test-call"),
                            message("tool", '{"status":200,"status_code":200,"x":1}'),
                            message("tool", ' \n{"status": "done"}\t'),
                            message("tool", '{"status":"done"}')])
        history = render_extraction(input)[1]["content"].split("Conversation history:\n")[1]
        self.assertEqual(history, '[Tool Response: lookup]\n{"city": "Paris"}\n'
                         '[Tool Response: unknown]\n{"x": 1}\n[Tool Response: unknown]\n{}')

    def test_tool_json_scalars_data_and_invalid_raw_text(self):
        for raw, expected in [('42', '42'), ('[1, 2]', '[1, 2]'), ('{bad', '{bad'),
                              ('{"data":null,"status":2}', 'null'), ('{"data":"é"}', '"é"')]:
            with self.subTest(raw=raw):
                self.assertTrue(render_extraction(extraction([message("tool", raw)]))[1]["content"].endswith(expected))
        with self.assertRaises(AdapterError):
            render_extraction(extraction([message("tool", None)]))

    def test_individual_2000_boundary_no_conversation_truncation(self):
        for size in (1999, 2000, 2001):
            raw = "z" * size
            expected = raw if size <= 2000 else "z" * 2000 + "...(truncated)"
            output = render_extraction(extraction([message("tool", raw)]))[1]["content"]
            self.assertTrue(output.endswith(expected))
        long_text = "Synthetic long conversation. " * 500
        input = extraction([message(content=long_text), message("tool", "x" * 2100), message("tool", "y" * 2100)])
        output = render_extraction(input)[1]["content"]
        self.assertIn(long_text, output)
        self.assertEqual(output.count("...(truncated)"), 2)

    def test_corrections_missing_ambiguity_are_supplied_data_not_prompt_hacks(self):
        for transcript in ("Blue, actually green.", "I have not chosen a color.", "Blue or green; undecided."):
            input = extraction([message(content=transcript)], extraction_prompt="Use null for unresolved values.")
            output = render_extraction(input)
            self.assertIn(transcript, output[1]["content"])
            self.assertEqual(output[0]["content"], EXTRACTION + "\n\nUse null for unresolved values.")
        self.assertNotIn("unknown", render_extraction(extraction([]))[0]["content"])

    def test_frozen_variant_cannot_silently_change_variables(self):
        contract = load_contract("extraction.skyassist")
        input = ExtractionInput(messages=(), variables=contract.extraction_variables)
        self.assertEqual(render_extraction(input, contract)[0]["content"], EXTRACTION + "\n\n" + GOLDENS["extraction"])
        with self.assertRaises(AdapterError):
            render_extraction(extraction([]), contract)


class RuntimeAdapterTests(unittest.TestCase):
    def test_eligibility_and_six_message_boundary(self):
        for count in (0, 2, 6):
            self.assertIsNone(render_runtime_summary(runtime([message()] * count)))
        for flags in ({"context_compaction_enabled": False}, {"has_previous_node": False}, {"is_realtime": True}):
            self.assertIsNone(render_runtime_summary(runtime([message()] * 7, **flags)))
        selected = select_summary_context(runtime([message()] * 7))
        self.assertEqual(len(selected.messages), 5)
        self.assertEqual(selected.last_summarized_index, 4)
        self.assertNotIn("Synthetic text", repr(selected))

    def test_initial_system_and_later_system_tail(self):
        messages = [message("system", "initial"), message(content="old fact"), message("system", "injection"),
                    message(content="correction"), message(content="open issue"), message(content="tail1"), message(content="tail2")]
        input = runtime(messages)
        output = render_runtime_summary(input)
        self.assertEqual(output, [{"role": "system", "content": RUNTIME}, {"role": "user", "content":
                         "Conversation history:\nUSER: old fact\n\nSYSTEM: injection\n\nUSER: correction\n\nUSER: open issue"}])
        self.assertEqual(select_summary_context(input).last_summarized_index, 4)

    def test_unresolved_call_and_result_only_in_tail(self):
        base = [message(content="before"), call(), message(content="middle"), message(), message(),
                message("tool", "complete", tool_call_id="test-call"), message()]
        selection = select_summary_context(runtime(base))
        self.assertEqual(selection.last_summarized_index, 0)
        self.assertEqual(selection.messages[0].content, "before")
        base[0] = call()
        base[1] = message()
        self.assertIsNone(render_runtime_summary(runtime(base)))

    def test_pending_sentinels_and_developer_completion(self):
        for content in ("IN_PROGRESS", '{"type":"async_tool","status":"started"}'):
            messages = [message(), call(), message("tool", content, tool_call_id="test-call"), message(), message(), message(), message()]
            self.assertEqual(select_summary_context(runtime(messages)).last_summarized_index, 0)
            for developer in ('{"type":"async_tool","status":"finished","tool_call_id":"test-call"}',
                              '{"type":"async_tool","status":"finished","tool_call_id":"other"}', '{bad'):
                messages[3] = message("developer", developer)
                expected = 4 if '"test-call"' in developer else 0
                self.assertEqual(select_summary_context(runtime(messages)).last_summarized_index, expected)

    def test_resolved_result_multiple_pending_and_opaque_indexing(self):
        messages = [message("system", "initial"), {"kind": "llm_specific", "payload": {"synthetic": True}},
                    call("a"), message("tool", "complete", tool_call_id="a"), call("b"), message(), message(), message()]
        selected = select_summary_context(runtime(messages))
        self.assertEqual(selected.last_summarized_index, 3)
        self.assertEqual(len(selected.messages), 3)
        rendered = format_summary_transcript(selected.messages)
        self.assertNotIn("synthetic", rendered)
        self.assertIn("TOOL_RESULT[a]: complete", rendered)
        self.assertNotIn("b]", rendered)

    def test_literal_none_multitext_tool_dual_rendering(self):
        messages = [call(), message("tool", '{"status": "done"}', tool_call_id="test-call"),
                    message(content=[{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])]
        typed = runtime(messages).messages
        self.assertEqual(format_summary_transcript(typed), 'ASSISTANT: None\n\nTOOL_CALL: lookup({ "key": "value" })\n\n'
                         'TOOL: {"status": "done"}\n\nTOOL_RESULT[test-call]: {"status": "done"}\n\nUSER: one two')
        self.assertEqual(format_summary_transcript(runtime([message("tool", "")]).messages), "TOOL_RESULT[unknown]: ")

    def test_fixed_long_context_retains_entire_selected_slice(self):
        messages = [message(content="Old value: blue.")]
        messages += [message(content=f"Synthetic irrelevant chatter {i}: " + "detail " * 40) for i in range(90)]
        messages += [message(content="Correction: green. Confirmation remains unresolved."),
                     message(content="preserved tail one"), message(content="preserved tail two")]
        input = runtime(messages)
        output = render_runtime_summary(input)[1]["content"]
        self.assertEqual(output, "Conversation history:\n" + "\n\n".join("USER: " + m["content"] for m in messages[:-2]))
        self.assertEqual(select_summary_context(input).last_summarized_index, len(messages) - 3)

    def test_apply_uses_current_system_and_retains_arrivals(self):
        messages = runtime([message("system", "new node"), message(content="old"), message(content="tail"),
                            message(content="new arrival")]).messages
        output = apply_context_summary(messages, 1, "supplied summary")
        self.assertEqual([m.content for m in output], ["new node", "Conversation summary: supplied summary", "tail", "new arrival"])
        self.assertEqual(messages[1].content, "old")
        self.assertEqual(apply_context_summary(messages, -1, "summary"), messages)
        self.assertEqual(apply_context_summary(messages, 1, ""), messages)
        self.assertEqual(apply_context_summary(messages[1:], 0, "summary")[0].role, "user")


class OtherAdapterTests(unittest.TestCase):
    def test_voicemail_exact_ordered_contexts_and_no_extra_question(self):
        for text in ("Hello?", "Please leave a message.", "Press one for support.",
                     "This is the office speaking.", "I saw your voicemail.", ""):
            messages = [call(), message("tool", "synthetic result", tool_call_id="test-call"), message(content=text)]
            input = VoicemailInput.model_validate_json(json.dumps({"messages": messages, "is_partial": True}))
            self.assertEqual(render_voicemail(input), [{"role": "system", "content": GOLDENS["voicemail"]}, *messages])
        opaque = VoicemailInput.model_validate_json('{"messages":[{"kind":"llm_specific","payload":{}}]}')
        with self.assertRaises(AdapterError):
            render_voicemail(opaque)

    def test_qa_template_metrics_one_pass_and_newlines(self):
        data = fixture("qa")["input"]
        data["metrics"] = dict(reversed(list(data["metrics"].items())))
        data["node_summary"] = "First\\nSecond {{metrics}}"
        data["previous_conversation_summary"] = "Prior {{node_summary}}"
        input = QAEvaluationInput.model_validate_json(json.dumps(data))
        output = render_qa_evaluation(input)
        metrics = '{\n  "call_duration_seconds": null,\n  "num_turns": 1,\n  "avg_latency_seconds": null,\n  "avg_ttfb_seconds": null,\n  "max_latency_seconds": null\n}'
        # Replace original placeholders once, using split/join to avoid recursive substitution.
        import re
        values = {"node_summary": data["node_summary"], "previous_conversation_summary": data["previous_conversation_summary"], "metrics": metrics}
        expected = re.sub(r"\{\{(\w+)\}\}", lambda m: values[m[1]], GOLDENS["qa"]).replace("\\n", "\n")
        self.assertEqual(output[0]["content"], expected)
        self.assertIn("First\nSecond {{metrics}}", output[0]["content"])
        self.assertEqual(output[1]["content"], "## Transcript\n" + input.transcript)

    def test_qa_whole_call_and_raw_transcript_preservation(self):
        data = fixture("qa")["input"]
        data.update(scope="whole_call", node_summary="", previous_conversation_summary="",
                    transcript='[0.0s] user: literal\\ntext\n[1.0s] [tool_call]: lookup')
        output = render_qa_evaluation(QAEvaluationInput.model_validate_json(json.dumps(data)))
        self.assertIn("## Node Purpose\n\n", output[0]["content"])
        self.assertEqual(output[1]["content"], "## Transcript\n" + data["transcript"])
        data["metrics"] = {}
        with self.assertRaises(AdapterError):
            render_qa_evaluation(QAEvaluationInput.model_validate_json(json.dumps(data)))

    def test_qa_context_conditioning_changes_only_supplied_sections(self):
        original = case(task="qa").input
        for updates in ({"node_summary": "Handle an unresolved request."},
                        {"previous_conversation_summary": "Earlier failure remains unresolved."}):
            output = render_qa_evaluation(original.model_copy(update=updates))
            self.assertNotEqual(output[0], render_qa_evaluation(original)[0])
            self.assertEqual(output[1], render_qa_evaluation(original)[1])

    def test_conversation_summary_no_tail_or_truncation(self):
        transcript = "[0.0s] user: Earlier fact.\n" * 300 + "[9.0s] user: Corrected fact; unresolved request."
        self.assertEqual(render_qa_conversation_summary(QAConversationSummaryInput(prior_transcript=transcript)),
                         [{"role": "system", "content": CONVERSATION}, {"role": "user", "content": "## Conversation\n" + transcript}])

    def test_node_description_optional_sections_and_order(self):
        data = fixture("qa_node_summary")["input"]
        data.update(custom_tools=[{"name": "lookup", "description": "Find options."}, {"name": "check", "description": ""}],
                    outgoing_edges=[{"label": "continue", "condition": "After confirmation."}, {"label": "finish", "condition": ""}])
        output = render_qa_node_summary(QANodeSummaryInput.model_validate_json(json.dumps(data)))
        self.assertEqual(output, [{"role": "system", "content": NODE}, {"role": "user", "content":
                         "Node name: Synthetic greeting\nAgent prompt:\nGreet the caller.\nAvailable tools:\n"
                         "- lookup: Find options.\n- check\n- continue: After confirmation.\n- finish"}])
        for prompt in (None, ""):
            data.update(node_type="start", agent_prompt=prompt, custom_tools=[], outgoing_edges=[])
            self.assertEqual(render_qa_node_summary(QANodeSummaryInput.model_validate_json(json.dumps(data)))[1]["content"], "Node name: Synthetic greeting")
        data["agent_prompt"] = "Do not invent. " * 1000
        self.assertIn(data["agent_prompt"], render_qa_node_summary(QANodeSummaryInput.model_validate_json(json.dumps(data)))[1]["content"])


class PreparationTests(unittest.TestCase):
    def test_generated_path_all_six_tasks_without_execution(self):
        cases = [case(task=task) for task in ("extraction", "context_summary", "voicemail", "qa", "qa_conversation_summary", "qa_node_summary")]
        with ExitStack() as stack:
            for target in ("ovp36_benchmark.requests.prepare_captured_request", "ovp36_benchmark.requests.CapturedReplayRequest",
                           "ovp36_benchmark.client.ModelClient", "ovp36_benchmark.runner.run_plan",
                           "ovp36_benchmark.persistence.ResultJournal.open", "socket.socket.connect"):
                stack.enter_context(patch(target, side_effect=AssertionError("forbidden path")))
            generated = stack.enter_context(patch("ovp36_benchmark.adapters.prepare_generated_request", wraps=prepare_generated_request))
            plan = prepare_curated_plan(cases, config=config(), resolved_model="synthetic-model")
            self.assertEqual(generated.call_count, 6)
        self.assertEqual(len(plan), 12)
        for item in plan:
            self.assertIsNone(item.request.captured)
            self.assertEqual(item.request.generation, config().generation)
            self.assertEqual(item.request.model, "synthetic-model")

    def test_controls_excluded_relative_order_and_repetition_major_hash(self):
        a = case()
        control = case(ordinal=13, kind="adapter_control", difficulty="normal", critical=False)
        response = case(ordinal=24, kind="response_contract", difficulty="normal", critical=False)
        transport = case(task="context_summary", ordinal=25, kind="transport_contract", difficulty="adversarial", critical=False)
        b = case(task="voicemail")
        cases = [a, control, response, transport, b]
        plan = prepare_curated_plan(cases, config=config(3), resolved_model="synthetic-model")
        self.assertEqual([(p.request.case_id, p.repetition_index) for p in plan],
                         [(c.id, i) for i in range(3) for c in (a, b)])
        again = prepare_curated_plan(cases, config=config(3), resolved_model="synthetic-model")
        self.assertEqual(fingerprint_execution_plan([p.identity_projection() for p in plan]),
                         fingerprint_execution_plan([p.identity_projection() for p in again]))
        for c in (control, response, transport):
            with self.assertRaisesRegex(AdapterError, "control"):
                prepare_curated_request(c, model="synthetic-model", generation=config().generation)

    def test_candidate_settings_affect_request_and_plan_not_dataset(self):
        cases = [case(task="context_summary")]
        dataset_hash = hash_dataset(cases)
        plans = [prepare_curated_plan(cases, config=cfg, resolved_model="synthetic-model")
                 for cfg in (config(), config(max_tokens=89), config(token_limit_field="max_completion_tokens"))]
        self.assertEqual(len({p[0].request.request_fingerprint for p in plans}), 3)
        self.assertEqual(len({fingerprint_execution_plan([e.identity_projection() for e in p]) for p in plans}), 3)
        self.assertEqual(hash_dataset(cases), dataset_hash)
        for plan in plans:
            body = plan[0].request.to_request_body()
            self.assertEqual(len({"max_tokens", "max_completion_tokens"} & body.keys()), 1)
            self.assertFalse(body["stream"])
        self.assertEqual(plans[0][0].request.generation.max_tokens, 71)
        self.assertEqual(load_contract("context_summary").source_settings.max_tokens, 4000)

    def test_invalid_config_ineligible_model_and_replay_rejected_safely(self):
        with self.assertRaises(AdapterError) as caught:
            prepare_curated_plan([case()], config=config(), resolved_model="different-model")
        self.assertIsNone(caught.exception.__context__)
        data = fixture("context_summary")
        data["input"]["messages"] = [message()]
        with self.assertRaises(AdapterError):
            prepare_curated_request(case(data), model="synthetic-model", generation=config().generation)
        data = fixture()
        data["source"] = "ovp34_replay"
        with self.assertRaises(ValueError):
            prepare_curated_plan([case(data)], config=config(), resolved_model="synthetic-model")

    def test_empty_plan_and_payload_detachment(self):
        self.assertEqual(prepare_curated_plan([], config=config(), resolved_model="synthetic-model"), ())
        original = case()
        before = original.model_dump_json()
        request = prepare_curated_request(original, model="synthetic-model", generation=config().generation)
        digest = request.request_fingerprint
        request.to_request_body()["messages"][1]["content"] = "changed"
        self.assertEqual(request.request_fingerprint, digest)
        self.assertEqual(original.model_dump_json(), before)

    def test_public_surface_has_no_generation_override(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(prepare_curated_plan).parameters), ("cases", "config", "resolved_model"))
