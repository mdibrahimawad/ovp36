"""Batch orchestration checks use mocked HTTP only; no serving or weights."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("batch_compare", Path(__file__).parents[1] / "scripts/compare_candidates.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_private_sidecars_are_immutable(self):
        tmp_path = self.tmp_path
        target = tmp_path / "sidecar.json"
        batch.save(target, {"review_type": "AI"})
        batch.save(target, {"review_type": "AI"})
        assert target.stat().st_mode & 0o077 == 0
        with self.assertRaisesRegex(ValueError, "immutable"):
            batch.save(target, {"review_type": "human"})
        assert batch.read(target)["review_type"] == "AI"


    def test_frozen_file_change_stops_before_execution(self):
        tmp_path = self.tmp_path
        self.enterContext(patch.object(batch, "ROOT", tmp_path))
        (tmp_path / "source.py").write_text("changed")
        batch.save(tmp_path / "freeze.json", {"files": {"source.py": hashlib.sha256(b"original").hexdigest()}})
        with self.assertRaisesRegex(ValueError, "frozen benchmark changed"):
            batch.verify_freeze(tmp_path)


    def test_phase_uses_existing_suite_and_preserves_raw_reasoning(self):
        tmp_path = self.tmp_path
        import ovp36_benchmark.__main__ as cli
        from ovp36_benchmark.config import ResolvedEndpoint
        from ovp36_benchmark.requests import prepare_generated_request
        from ovp36_benchmark.schemas import GenerationConfig, Source, Task
        request = prepare_generated_request(case_id="test-case", task=Task.EXTRACTION,
            source=Source.CURATED, model="test/model", generation=GenerationConfig(max_tokens=512),
            messages=[{"role": "user", "content": "Unchanged input"}])
        reply = dict(id="test", object="chat.completion", created=1, model="test/model",
            choices=[dict(index=0, message=dict(role="assistant", content='{"x":1}', reasoning="leaked reasoning"),
                          finish_reason="stop")], usage=dict(prompt_tokens=3, completion_tokens=8, total_tokens=11))
        def handler(incoming):
            assert json.loads(incoming.content)["messages"] == request.messages.to_messages()
            return httpx.Response(200, json=reply)
        self.enterContext(patch.object(httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler)))
        self.enterContext(patch.object(batch, "write_config", lambda *_: tmp_path / "unused.toml"))
        for name in ("OPENAI_LOG", "OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
            self.enterContext(patch.dict(os.environ))
            os.environ.pop(name, None)
        calls = []
        original = cli.ModelClient
        async def suite(config, results_root):
            calls.append("suite")
            endpoint = ResolvedEndpoint(alias="test", base_url="http://127.0.0.1:8081/v1", model="test/model")
            async with cli.ModelClient(endpoint, timeout_seconds=30) as client:
                result = await client.send_once(request)
            assert result.status == "success" and result.response.raw_content == '{"x":1}'
            return {"run_id": "test"}
        self.enterContext(patch.object(cli, "candidate_suite", suite))
        result = asyncio.run(batch.phase(tmp_path, {"key": "m01"}, "suite"))
        assert result == {"run_id": "test"} and calls == ["suite"]
        assert cli.ModelClient is original
        captured = batch.read(next((tmp_path / "m01/wire/suite").glob("*.json")))
        assert captured["message_fingerprint"] == request.message_fingerprint
        assert json.loads(captured["exchanges"][0]["response_body"])["choices"][0]["message"]["reasoning"] == "leaked reasoning"


    def test_import_has_no_model_execution(self):
        assert callable(batch.main)
        assert batch.URL == "http://127.0.0.1:8081/v1"
        assert "mlx_lm" not in batch.__dict__

    def test_conversion_uses_checksum_verified_pinned_local_source(self):
        source = self.tmp_path / "pinned-source"
        source.mkdir()
        (source / "model.safetensors").write_bytes(b"public weights")
        calls = []
        def download(repo, **kwargs):
            assert repo == "official/model" and kwargs["revision"] == "pinned-revision"
            return str(source)
        def convert(path, **kwargs):
            assert path == str(source) and "revision" not in kwargs
            assert kwargs["quantize"] and kwargs["q_bits"] == 8 and kwargs["q_group_size"] == 64
            calls.append(path)
            target = Path(kwargs["mlx_path"])
            target.mkdir(parents=True)
            (target / "config.json").write_text(json.dumps({"quantization": {"bits": 8}, "model_type": "test"}))
        modules = {
            "huggingface_hub": SimpleNamespace(snapshot_download=download),
            "mlx_lm.utils": SimpleNamespace(load_tokenizer=lambda _: SimpleNamespace(chat_template="native")),
            "mlx_lm.convert": SimpleNamespace(convert=convert),
        }
        row = dict(key="m10", canonical_id="official/model", thinking=False,
                   canonical=dict(license="apache-2.0", revision="pinned-revision",
                       files=[dict(name="model.safetensors", lfs=dict(sha256=hashlib.sha256(b"public weights").hexdigest()))]),
                   artifact=dict(license="apache-2.0", local_conversion=True))
        with patch.dict(sys.modules, modules), patch.object(batch, "ROOT", self.tmp_path), patch.object(batch.importlib.metadata, "version", return_value="test"):
            batch.artifact(self.tmp_path / "batch", row)
            assert calls == [str(source)]
            assert batch.read(self.tmp_path / "batch/m10/conversion-source.json")["revision"] == "pinned-revision"
            (source / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                batch.artifact(self.tmp_path / "another-batch", row)
            assert len(calls) == 1

    def test_review_coverage_rejects_stale_hashes_and_human_approval(self):
        spec = importlib.util.spec_from_file_location("batch_review", Path(__file__).parents[1] / "scripts/review_candidates.py")
        review = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"compare_candidates": batch}):
            spec.loader.exec_module(review)
        inventory = {("run", "case"): {"output_sha256": "abc"},
                     ("run", "failed"): {"output_sha256": None}}
        self.enterContext(patch.object(review, "outputs", return_value=inventory))
        path = self.tmp_path / "ai-reviews/run/case.json"
        entry = dict(run_id="run", case_id="case", review_type="AI", human_approval=False, output_sha256="abc")
        batch.save(path, entry)
        counts = review.coverage(self.tmp_path)
        assert counts["reviewed_outputs"] == 1 and counts["unavailable_outputs"] == 1
        assert counts["unfinished"] == [] and counts["human_reviews_approved"] == 0
        for changed in ({"output_sha256": "stale"}, {"human_approval": True}, {"review_type": "human"}):
            batch.save(path, entry | changed, replace=True)
            with self.assertRaisesRegex(ValueError, "invalid review type or stale output hash"):
                review.coverage(self.tmp_path)

    def test_semantic_review_requires_real_exact_evidence(self):
        spec = importlib.util.spec_from_file_location("batch_review", Path(__file__).parents[1] / "scripts/review_candidates.py")
        review = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"compare_candidates": batch}):
            spec.loader.exec_module(review)
        self.enterContext(patch.object(review, "case_inputs", return_value=({}, {"case": {"messages": [{"content": "actual input"}]}})))
        record = dict(run_id="run", case_id="case", phase="suite", output_sha256="abc",
                      result={"response": {"raw_content": "actual output"}})
        with self.assertRaisesRegex(ValueError, "not an exact"):
            review.record_review(record, verdict="acceptable", severity="none", input_quote="invented evidence",
                                 output_quote="actual output", explanation="Reviewed", batch=self.tmp_path)
