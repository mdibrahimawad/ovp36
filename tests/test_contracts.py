import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from pydantic import ValidationError

from ovp36_benchmark.contracts import (
    PromptContract, hash_contract, load_contract, render_contract,
)


EXTRACTION = "You are an assistant tasked with extracting structured data from the conversation. Return ONLY a valid JSON object with the requested variables as top-level keys. Do not wrap the JSON in markdown."
RUNTIME = """You are summarizing a conversation between a user and an AI assistant.

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

The conversation transcript follows. Generate only the summary, no other text."""
GOLDENS = json.loads((Path(__file__).parent / "fixtures/source_contract_goldens.json").read_text())
CONVERSATION = "You are summarizing a portion of a voice AI conversation. Produce a concise summary (3-5 sentences) covering key topics, information exchanged, and current state. We would be using this summary in doing a QA of the conversation that the voice AI agent did with someone so try to capture the nuances of the conversation as much as possible."
NODE = "You are analyzing a voice AI agent script. This is only a part of a larger script. Produce a concise summary (2-4 sentences) describing this script purpose, what the agent should accomplish, and key behaviors. We will be using this summary to do a QA on the conversation that the agent would do with someone so try to capture the nuances of the script as much as possible."
IDS = ("extraction", "extraction.skyassist", "context_summary", "voicemail", "qa_conversation_summary", "qa_node_summary", "qa")


class ContractTests(unittest.TestCase):
    def test_exact_frozen_system_text(self):
        for name, expected in (("extraction", EXTRACTION), ("context_summary", RUNTIME),
                               ("qa_conversation_summary", CONVERSATION), ("qa_node_summary", NODE)):
            with self.subTest(contract=name):
                self.assertEqual(load_contract(name).system_text.encode("utf-8"), expected.encode("utf-8"))
        self.assertEqual(load_contract("voicemail").system_text, GOLDENS["voicemail"])
        self.assertIsNone(load_contract("voicemail").system_fragment)

    def test_runtime_matches_authority_and_settings(self):
        audit = (Path(__file__).resolve().parents[1] / "docs/SOURCE_AUDIT.md").read_text()
        self.assertIn("```text\n" + RUNTIME + "\n```", audit)
        result = render_contract(load_contract("context_summary"), {"formatted_transcript": "USER: Fact\n\nASSISTANT: Agreed"})
        self.assertEqual(result.user_text, "Conversation history:\nUSER: Fact\n\nASSISTANT: Agreed")
        self.assertEqual(result.source_settings.max_tokens, 4000)
        self.assertEqual(result.source_settings.timeout_seconds, 30.0)

    def test_extraction_rendering_and_optional_append(self):
        contract = load_contract("extraction")
        self.assertEqual(contract.variable_line_template, "- {name} (VariableType.{type}): {prompt}")
        values = {"variable_lines": "- name (VariableType.string): stated name\n- age (VariableType.number): stated age",
                  "formatted_history": 'user: {"name": "Alex"}\nuser: age 4'}
        rendered = render_contract(contract, values)
        self.assertEqual(rendered.system_text, EXTRACTION)
        self.assertEqual(rendered.user_text, '\n\nVariables to extract:\n- name (VariableType.string): stated name\n- age (VariableType.number): stated age\n\nConversation history:\nuser: {"name": "Alex"}\nuser: age 4')
        self.assertEqual(render_contract(contract, values | {"extraction_prompt": ""}).system_text, EXTRACTION)
        self.assertEqual(render_contract(contract, values | {"extraction_prompt": "Keep relative dates."}).system_text,
                         EXTRACTION + "\n\nKeep relative dates.")
        self.assertEqual(len(values), 2)

    def test_summary_and_qa_wrappers(self):
        text = "[1.0s] user: Hello?"
        self.assertEqual(render_contract(load_contract("qa_conversation_summary"), {"transcript": text}).user_text,
                         "## Conversation\n" + text)
        self.assertEqual(load_contract("qa").user_template, "## Transcript\n{transcript}")

    def test_skyassist_extraction_golden(self):
        contract = load_contract("extraction.skyassist")
        self.assertEqual(contract.system_text, EXTRACTION + "\n\n" + GOLDENS["extraction"])
        self.assertEqual([v.model_dump(mode="json") for v in contract.extraction_variables], GOLDENS["variables"])
        lines = "\n".join(contract.variable_line_template.format(name=v.name, type=v.type, prompt=v.hint)
                          for v in contract.extraction_variables)
        rendered = render_contract(contract, {"variable_lines": lines, "formatted_history": "user: Hello"})
        self.assertEqual(rendered.user_text, "\n\nVariables to extract:\n" + lines + "\n\nConversation history:\nuser: Hello")
        for kind in ("string", "number", "boolean"):
            self.assertEqual(contract.variable_line_template.format(name="x", type=kind, prompt="hint"),
                             f"- x (VariableType.{kind}): hint")

    def test_qa_literal_template_golden(self):
        contract = load_contract("qa")
        self.assertEqual(contract.system_template, GOLDENS["qa"])
        self.assertIsNone(contract.system_text)
        for name in ("node_summary", "previous_conversation_summary", "metrics"):
            self.assertEqual(contract.system_template.count("{{" + name + "}}"), 1)
        restored = PromptContract.model_validate_json(contract.model_dump_json())
        self.assertEqual(restored.system_template, GOLDENS["qa"])
        changed = PromptContract.model_validate_json(json.dumps(contract.model_dump(mode="json") |
                                                               {"system_template": GOLDENS["qa"] + "\n"}))
        self.assertNotEqual(hash_contract(contract), hash_contract(changed))

    def test_corrected_prompt_goldens_in_source_audit(self):
        audit = (Path(__file__).resolve().parents[1] / "docs/SOURCE_AUDIT.md").read_text()
        for name in ("voicemail", "qa", "extraction"):
            self.assertIn(GOLDENS[name], audit)
        node_section = audit.split("## 8.2 Node/script summary\n", 1)[1].split("## 8.3", 1)[0]
        self.assertIn("```text\n" + NODE + "\n```", node_section)

    def test_complete_contracts_can_require_future_adapters(self):
        for name in IDS:
            self.assertEqual(load_contract(name).completeness, "complete")
            self.assertFalse(load_contract(name).missing_components)
        self.assertEqual(load_contract("voicemail").user_input_kind, "messages")
        self.assertEqual(load_contract("qa_node_summary").user_input_kind, "node_description")
        self.assertIsNone(load_contract("voicemail").source_settings.max_tokens)
        for name in ("voicemail", "qa", "qa_node_summary"):
            for partial in (False, True):
                with self.subTest(name=name, partial=partial), self.assertRaisesRegex(ValueError, "adapter"):
                    render_contract(load_contract(name), allow_partial=partial)

    def test_missing_extra_and_wrong_type_variables_fail(self):
        contract = load_contract("context_summary")
        for values in ({}, {"wrong_name": "text"}, {"formatted_transcript": "x", "extra": "y"}):
            with self.assertRaises(ValueError):
                render_contract(contract, values)
        with self.assertRaises(TypeError):
            render_contract(contract, {"formatted_transcript": 123})

    def test_partial_contracts_cannot_render_as_complete(self):
        data = load_contract("context_summary").model_dump(mode="json") | {
            "completeness": "partial", "system_text": None,
            "system_fragment": "Synthetic fragment", "missing_components": ["system_text"],
        }
        contract = PromptContract.model_validate_json(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "incomplete contract"):
            render_contract(contract)
        rendered = render_contract(contract, {"formatted_transcript": "text"}, allow_partial=True)
        self.assertEqual(rendered.completeness, "partial")
        self.assertEqual(rendered.missing_components, ("system_text",))
        self.assertIsNone(rendered.system_text)
        self.assertEqual(rendered.system_fragment, "Synthetic fragment")
        with self.assertRaises(ValidationError):
            PromptContract.model_validate_json(json.dumps(data | {"completeness": "complete"}))

    def test_hashes_are_stable_and_include_text_metadata(self):
        for name in IDS:
            contract = load_contract(name)
            self.assertEqual(hash_contract(contract), hash_contract(load_contract(name)))
            self.assertRegex(hash_contract(contract), r"^[0-9a-f]{64}$")
        contract = load_contract("context_summary")
        for changes in ({"system_text": RUNTIME + "\n"}, {"version": 2},
                        {"source_settings": {"max_tokens": 3999, "timeout_seconds": 30.0}}):
            changed = PromptContract.model_validate_json(json.dumps(contract.model_dump(mode="json") | changes))
            self.assertNotEqual(hash_contract(contract), hash_contract(changed))
        normalized = json.dumps(contract.model_dump(mode="json"), sort_keys=True,
                                ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(hash_contract(contract), hashlib.sha256(normalized.encode()).hexdigest())

    def test_resource_formatting_does_not_change_hash(self):
        contract = load_contract("context_summary")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contracts.json"
            catalog = {"schema_version": "1", "contracts": [contract.model_dump(mode="json")]}
            path.write_text(json.dumps(catalog, indent=4, sort_keys=True), encoding="utf-8")
            self.assertEqual(hash_contract(contract), hash_contract(load_contract("context_summary", resource_path=path)))
            catalog["contracts"] *= 2
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_contract("context_summary", resource_path=path)

    def test_explicit_ids_and_versions(self):
        for name, version in (("unknown", 1), ("extraction", 2), ("extraction", True),
                              ("extraction", "1"), ("extraction", 0)):
            with self.assertRaises(ValueError):
                load_contract(name, version=version)
        with self.assertRaises(TypeError):
            render_contract(load_contract("voicemail"), allow_partial="false")

    def test_only_simple_template_fields_allowed(self):
        contract = load_contract("extraction").model_dump(mode="json")
        for template in ("{value.attribute}", "{value[0]}", "{value!r}", "{value:>10}", "{"):
            with self.assertRaises(ValidationError):
                PromptContract.model_validate_json(json.dumps(contract | {"user_template": template}))

    def test_no_private_endpoint_data_in_contracts(self):
        for name in IDS:
            text = load_contract(name).model_dump_json()
            self.assertIsNone(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|https?://", text))


if __name__ == "__main__":
    unittest.main()
