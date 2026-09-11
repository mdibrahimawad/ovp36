"""Small synthetic cases, not the curated CASE_MATRIX fixtures."""

import json

from ovp36_benchmark.schemas import BenchmarkCase


def case_data(task: str = "extraction", case_id: str = "synthetic-1") -> dict:
    inputs = {
        "extraction": {
            "messages": [{"role": "user", "content": "My name is Alex."}],
            "variables": [{"name": "name", "type": "string", "hint": "stated name"}],
        },
        "context_summary": {
            "messages": [{"role": "user", "content": "Keep this fact."}],
            "context_compaction_enabled": True, "has_previous_node": True,
            "is_realtime": False,
        },
        "voicemail": {"transcript": "Hello?"},
        "qa": {"scope": "node", "transcript": "[0.0s] user: Hello?",
               "metrics": {}, "node_summary": "Greet the caller.",
               "previous_conversation_summary": ""},
        "qa_conversation_summary": {"prior_transcript": "[0.0s] user: Hello?"},
        "qa_node_summary": {"node_type": "agent", "node_name": "Welcome",
                            "agent_prompt": "Greet the caller.",
                            "custom_tools": [], "outgoing_edges": []},
    }
    expected = {
        "extraction": {"fields": {"name": {"state": "known", "acceptable_values": ["Alex"]}}},
        "voicemail": {"label": "CONVERSATION", "ambiguous": False, "rationale": "Human greeting"},
        "qa": {"expected_tags": [], "expected_sentiment": "neutral",
               "quality_score_range": [8, 10], "tag_evidence": {}, "summary_facts": []},
    }
    return {
        "id": case_id, "task": task, "source": "curated", "difficulty": "normal",
        "critical": True, "contract_id": f"synthetic/{task}/1",
        "exercise": {"kind": "model"}, "input": inputs[task],
        "expected": expected.get(task, {"required_facts": []}), "tags": ["synthetic"],
    }


def parse_case(data: dict) -> BenchmarkCase:
    return BenchmarkCase.model_validate_json(json.dumps(data))
