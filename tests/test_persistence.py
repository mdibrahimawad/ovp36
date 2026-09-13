"""Synthetic local evidence only; no model client construction or network calls."""

import copy
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from ovp36_benchmark.client import AttemptResult, ModelResponse, TokenUsage
from ovp36_benchmark.config import persistent_config_projection
from ovp36_benchmark.identity import (
    IDENTITY_VERSION, ExecutionKey, canonical_json_bytes, fingerprint_safe_config, make_run_id,
)
from ovp36_benchmark.persistence import (
    MANIFEST_SCHEMA_VERSION, JOURNAL_SCHEMA_VERSION, AttemptRecorded,
    DuplicateExecutionError, JournalCorruptionError, ManifestMismatchError,
    PersistenceError, ResultJournal, RunManifest, build_manifest, load_manifest, read_events,
)
from ovp36_benchmark.schemas import EndpointConfig, GenerationConfig, RunConfig, ServerMetadata


def config(**updates):
    return RunConfig(**({
        "endpoint": EndpointConfig(alias="synthetic", base_url="http://synthetic.invalid/v1",
                                   model="synthetic-model", api_key_env="SYNTHETIC_KEY"),
        "generation": GenerationConfig(max_tokens=137), "experiment_label": "synthetic",
        "evaluation": "canonical", "server": ServerMetadata(model_revision="revision-a"),
    } | updates))


def manifest(cfg=None, **updates):
    return build_manifest(cfg or config(), **({
        "resolved_model": "synthetic-model", "dataset_hash": "a" * 64,
        "ordered_execution_plan_hash": "b" * 64, "contract_hashes": {"qa": "c" * 64},
        "harness_code_fingerprint": "d" * 64, "dependency_lock_fingerprint": "e" * 64,
    } | updates))


def result(content="SYNTHETIC_CANDIDATE", **updates):
    return AttemptResult(**({
        "status": "success", "latency_ms": 1.25,
        "response": ModelResponse(raw_content=content, content_present=True,
                                  finish_reason="stop", response_model="synthetic/model",
                                  usage=TokenUsage(prompt_tokens=0, completion_tokens=2)),
    } | updates))


@unittest.skipUnless(os.name == "posix", "local POSIX persistence target")
class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "results"
        self.manifest = manifest()
        self.run = self.root / self.manifest.run_id
        self.key = ExecutionKey(run_id=self.manifest.run_id, case_id="synthetic-case",
                                request_fingerprint="f" * 64, repetition_index=0)

    def open(self):
        journal = ResultJournal.open(self.root, self.manifest)
        self.addCleanup(journal.close)
        return journal

    def write_private(self, path, data):
        path.write_bytes(data)
        path.chmod(0o600)

    def event_data(self):
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result())
            return journal.records()[0].model_dump(mode="json")

    def assert_safe(self, error, *markers):
        for marker in markers:
            self.assertNotIn(marker, str(error))
            self.assertNotIn(marker, repr(error))
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)

    def test_builder_matches_public_identity_and_projection(self):
        cfg = config()
        value = self.manifest
        self.assertIsInstance(value, RunManifest)
        self.assertEqual(value.schema_version, MANIFEST_SCHEMA_VERSION)
        self.assertEqual(value.identity_version, IDENTITY_VERSION)
        self.assertEqual(value.safe_config.model_dump(mode="json"),
                         persistent_config_projection(cfg, resolved_model="synthetic-model"))
        self.assertEqual(value.safe_config_fingerprint,
                         fingerprint_safe_config(cfg, resolved_model="synthetic-model"))
        self.assertEqual(value.run_id, make_run_id(
            dataset_hash=value.dataset_hash, ordered_execution_plan_hash=value.ordered_execution_plan_hash,
            safe_config_fingerprint=value.safe_config_fingerprint, requested_model="synthetic-model",
            endpoint_alias=cfg.endpoint.alias, server=cfg.server, contract_hashes=value.contract_hashes,
            harness_code_fingerprint=value.harness_code_fingerprint,
            dependency_lock_fingerprint=value.dependency_lock_fingerprint,
        ))
        self.assertEqual(value, manifest())
        self.assertFalse(self.root.exists())

    def test_optional_projection_omissions_and_seed_zero(self):
        for seed in (None, 0, 41):
            value = manifest(config(generation=GenerationConfig(max_tokens=137, seed=seed)))
            data = value.model_dump(mode="json")["safe_config"]
            self.assertEqual("seed" in data["generation"], seed is not None)
            self.assertEqual(data["server"], {"model_revision": "revision-a"})
            with ResultJournal.open(self.root, value):
                self.assertEqual(load_manifest(self.root, value.run_id), value)
        value = manifest(config(evaluation="exploratory", server=ServerMetadata()))
        self.assertEqual(value.safe_config.model_dump(mode="json")["server"], {})

    def test_builder_rejects_invalid_digests_labels_and_canonical_config(self):
        for change in ({"dataset_hash": "bad"}, {"resolved_model": "private/bad/path"},
                       {"contract_hashes": {"../private": "a" * 64}},
                       {"contract_hashes": {"qa": "bad"}}):
            with self.subTest(change=change), self.assertRaises(PersistenceError):
                manifest(**change)
        invalid = config().model_copy(update={"server": ServerMetadata()})
        with self.assertRaises(PersistenceError):
            manifest(invalid)

    def test_create_manifest_canonical_and_no_empty_journal(self):
        with self.open():
            data = (self.run / "manifest.json").read_bytes()
            self.assertEqual(data, canonical_json_bytes(self.manifest.model_dump(mode="json")))
            self.assertFalse((self.run / "journal.jsonl").exists())
            self.assertEqual(list(self.run.iterdir()), [self.run / "manifest.json"])
        with self.assertRaises(ValidationError):
            self.manifest.run_id = "a" * 64

    def test_identical_resume_does_not_rewrite_manifest(self):
        self.open().close()
        path = self.run / "manifest.json"
        # Whitespace is not identity. A valid noncanonical existing file stays untouched.
        data = json.dumps(self.manifest.model_dump(mode="json"), indent=2).encode()
        self.write_private(path, data)
        before = path.stat()
        with patch("ovp36_benchmark.persistence.os.link", side_effect=AssertionError("rewrite")):
            self.open().close()
        after = path.stat()
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))

    def test_expected_mismatch_never_overwrites(self):
        self.open().close()
        path = self.run / "manifest.json"
        before = path.read_bytes()
        changed = self.manifest.model_copy(update={"dataset_hash": "f" * 64})
        with self.assertRaises(ManifestMismatchError):
            ResultJournal.open(self.root, changed)
        self.assertEqual(path.read_bytes(), before)

    def test_manifest_path_id_mismatch(self):
        self.open().close()
        data = self.manifest.model_dump(mode="json")
        data["run_id"] = "a" * 64
        self.write_private(self.run / "manifest.json", canonical_json_bytes(data))
        with self.assertRaises(ManifestMismatchError):
            load_manifest(self.root, self.manifest.run_id)

    def test_orphaned_journal_fails_without_creating_manifest(self):
        self.run.mkdir(parents=True, mode=0o700)
        self.root.chmod(0o700)
        self.write_private(self.run / "journal.jsonl", b"")
        with self.assertRaisesRegex(PersistenceError, "orphaned_journal"):
            self.open()
        self.assertFalse((self.run / "manifest.json").exists())

    def test_missing_parent_is_not_created_recursively(self):
        missing = self.root / "missing" / "results"
        with self.assertRaises(PersistenceError):
            ResultJournal.open(missing, self.manifest)
        self.assertFalse(self.root.exists())

    def test_invalid_run_ids_rejected_before_path_use(self):
        for value in ("../private", "f" * 63, "F" * 64, "f" * 64 + "\n", True):
            for reader in (load_manifest, read_events):
                with self.subTest(value=value, reader=reader), self.assertRaises(PersistenceError):
                    reader(self.root, value)
        self.assertFalse(self.root.exists())

    def test_manifest_schema_corruption_and_safe_errors(self):
        self.open().close()
        base = self.manifest.model_dump(mode="json")
        variants = []
        for change in ({"schema_version": "unknown"}, {"identity_version": "unknown"},
                       {"dataset_hash": "PRIVATE_CORRUPT"}, {"extra": "PRIVATE_CORRUPT"}):
            variants.append(json.dumps(base | change).encode())
        missing = copy.deepcopy(base)
        del missing["safe_config_fingerprint"]
        variants.append(json.dumps(missing).encode())
        for field in ("temperature", "top_p", "max_tokens", "token_limit_field"):
            missing = copy.deepcopy(base)
            del missing["safe_config"]["generation"][field]
            variants.append(json.dumps(missing).encode())
        for field in ("seed",):
            null = copy.deepcopy(base)
            null["safe_config"]["generation"][field] = None
            variants.append(json.dumps(null).encode())
        null = copy.deepcopy(base)
        null["safe_config"]["server"]["runtime"] = None
        variants.append(json.dumps(null).encode())
        variants += [b'{"x":1,"x":2}', b'NaN', b'{"x":Infinity}', b'{"x":-Infinity}',
                     b'{"x":[1e999]}', b'[]', b'"PRIVATE_CORRUPT"', b'\xff', b'{',
                     json.dumps(base).encode("utf-16")]
        for data in variants:
            with self.subTest(data=data[:35]):
                self.write_private(self.run / "manifest.json", data)
                with self.assertRaises(PersistenceError) as caught:
                    load_manifest(self.root, self.manifest.run_id)
                self.assert_safe(caught.exception, "PRIVATE_CORRUPT", self.temporary.name)

    def test_symlinks_and_nonregular_files_rejected(self):
        self.open().close()
        target = Path(self.temporary.name) / "target"
        self.write_private(target, b"PRIVATE_TARGET")
        for filename in ("manifest.json", "journal.jsonl"):
            path = self.run / filename
            original = path.read_bytes() if path.exists() else None
            if path.exists():
                path.unlink()
            for kind in ("symlink", "directory", "fifo"):
                with self.subTest(filename=filename, kind=kind):
                    if kind == "symlink":
                        path.symlink_to(target)
                    elif kind == "directory":
                        path.mkdir(mode=0o700)
                    else:
                        os.mkfifo(path, 0o600)
                    with self.assertRaises(PersistenceError):
                        self.open()
                    if kind == "directory":
                        path.rmdir()
                    else:
                        path.unlink()
            if original is not None:
                self.write_private(path, original)
        self.assertEqual(target.read_bytes(), b"PRIVATE_TARGET")

    def test_root_and_run_directory_symlinks_rejected(self):
        target = Path(self.temporary.name) / "actual"
        target.mkdir(mode=0o700)
        self.root.symlink_to(target, target_is_directory=True)
        with self.assertRaises(PersistenceError):
            self.open()
        self.root.unlink()
        self.root.mkdir(mode=0o700)
        self.run.symlink_to(target, target_is_directory=True)
        with self.assertRaises(PersistenceError):
            self.open()
        self.assertEqual(list(target.iterdir()), [])

    def test_stale_temporary_file_is_not_a_manifest(self):
        self.open().close()
        stale = self.run / ".manifest-stale"
        self.write_private(stale, b"PRIVATE_STALE")
        self.open().close()
        self.assertEqual(stale.read_bytes(), b"PRIVATE_STALE")

    def test_unsupported_hardlink_fails_without_fallback(self):
        for failure in (OSError("PRIVATE_OS"), NotImplementedError("PRIVATE_OS")):
            with patch("ovp36_benchmark.persistence.os.link", side_effect=failure):
                with self.assertRaises(PersistenceError) as caught:
                    self.open()
            self.assert_safe(caught.exception, "PRIVATE_OS")
            self.assertFalse((self.run / "manifest.json").exists())
            self.assertEqual(list(self.run.iterdir()), [])

    def test_create_only_publication_refuses_existing_destination(self):
        real_link = os.link
        def collision(source, destination, **kwargs):
            self.write_private(Path(destination), b"EXISTING_MANIFEST")
            return real_link(source, destination, **kwargs)
        with patch("ovp36_benchmark.persistence.os.link", side_effect=collision):
            with self.assertRaises(ManifestMismatchError) as caught:
                self.open()
            self.assert_safe(caught.exception, "EXISTING_MANIFEST")
        self.assertEqual((self.run / "manifest.json").read_bytes(), b"EXISTING_MANIFEST")

    def test_creation_modes_are_private(self):
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result())
        for path in (self.root, self.run, self.run / "manifest.json", self.run / "journal.jsonl"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0)

    def test_restrictive_owner_modes_accepted_when_sufficient(self):
        self.open().close()
        path = self.run / "manifest.json"
        path.chmod(0o400)
        self.run.chmod(0o500)
        self.root.chmod(0o500)
        try:
            self.assertEqual(load_manifest(self.root, self.manifest.run_id), self.manifest)
            with self.open() as journal:
                self.assertEqual(journal.records(), ())
        finally:
            self.root.chmod(0o700)
            self.run.chmod(0o700)
            path.chmod(0o600)

    def test_all_group_other_permission_bits_rejected(self):
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result())
        for path in (self.root, self.run, self.run / "manifest.json", self.run / "journal.jsonl"):
            original = stat.S_IMODE(path.stat().st_mode)
            for bit in (0o040, 0o020, 0o010, 0o004, 0o002, 0o001):
                try:
                    path.chmod(original | bit)
                    with self.subTest(path=path.name, bit=bit), self.assertRaises(PersistenceError):
                        self.open()
                finally:
                    path.chmod(original)

    def test_insufficient_owner_read_and_traverse_permissions(self):
        self.open().close()
        for path, mode in ((self.run / "manifest.json", 0o200), (self.run, 0o400), (self.root, 0o400)):
            original = stat.S_IMODE(path.stat().st_mode)
            try:
                path.chmod(mode)
                with self.assertRaises(PersistenceError):
                    self.open()
            finally:
                path.chmod(original)

    def test_insufficient_owner_create_and_append_permissions(self):
        with self.open() as journal:
            self.run.chmod(0o500)
            try:
                with self.assertRaises(PersistenceError):
                    journal.append_attempt(self.key, 0, result())
            finally:
                self.run.chmod(0o700)
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result())
            path = self.run / "journal.jsonl"
            path.chmod(0o400)
            try:
                self.assertEqual(len(read_events(self.root, self.manifest.run_id)), 1)
                with self.assertRaises(PersistenceError):
                    journal.finalize(self.key, 0)
            finally:
                path.chmod(0o600)

    def test_manifest_directory_and_file_durability_order(self):
        actions = []
        real_sync, real_link, real_unlink = os.fsync, os.link, Path.unlink
        def synced(fd):
            info = os.fstat(fd)
            for label, path in (("parent", Path(self.temporary.name)), ("root", self.root), ("run", self.run)):
                if path.exists() and path.stat().st_ino == info.st_ino:
                    actions.append("sync_" + label)
                    break
            else:
                self.assertTrue(stat.S_ISREG(info.st_mode))
                self.assertEqual(stat.S_IMODE(info.st_mode) & 0o077, 0)
                actions.append("sync_file")
            real_sync(fd)
        def linked(*args, **kwargs):
            actions.append("publish")
            return real_link(*args, **kwargs)
        def unlinked(path, *args, **kwargs):
            actions.append("cleanup")
            return real_unlink(path, *args, **kwargs)
        with patch("ovp36_benchmark.persistence.os.fsync", side_effect=synced), \
                patch("ovp36_benchmark.persistence.os.link", side_effect=linked), \
                patch.object(Path, "unlink", unlinked):
            self.open().close()
        self.assertEqual(actions, ["sync_parent", "sync_root", "sync_file", "publish",
                                   "sync_run", "cleanup", "sync_run"])

    def test_first_journal_and_later_append_sync_order(self):
        actions = []
        real_sync = os.fsync
        with self.open() as journal:
            def synced(fd):
                actions.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
                self.assertEqual(journal.records(), ())
                real_sync(fd)
            with patch("ovp36_benchmark.persistence.os.fsync", side_effect=synced):
                journal.append_attempt(self.key, 0, result())
            self.assertEqual(actions, ["file", "directory"])
            actions.clear()
            def later_synced(fd):
                actions.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
                self.assertEqual(journal.completed_keys(), frozenset())
                real_sync(fd)
            with patch("ovp36_benchmark.persistence.os.fsync", side_effect=later_synced):
                journal.finalize(self.key, 0)
            self.assertEqual(actions, ["file"])

    def test_candidate_content_exact_roundtrip_and_private_repr(self):
        texts = (None, "", "line1\nline2\n", "\ttab\t", '"quotes"', "\\back\\slash\\",
                 " café ☎ العربية ", "null", "NaN", "Infinity", '{"x":42}',
                 "http://synthetic.invalid/private", "SYNTHETIC_CANDIDATE_SECRET")
        with self.open() as journal:
            for index, text in enumerate(texts):
                key = self.key.model_copy(update={"repetition_index": index})
                journal.append_attempt(key, 0, result(text))
                event = journal.records()[-1]
                self.assertEqual(event.result.response.raw_content, text)
                if text:
                    for wrapper in (event, event.result, event.result.response, journal):
                        self.assertNotIn(text, repr(wrapper))
        records = read_events(self.root, self.manifest.run_id)
        self.assertEqual(tuple(item.result.response.raw_content for item in records), texts)
        self.assertEqual(len((self.run / "journal.jsonl").read_bytes().splitlines()), len(texts))

    def test_null_absent_usage_and_zero_are_not_inferred(self):
        with self.open() as journal:
            for index, response in enumerate((
                ModelResponse(raw_content=None, content_present=False, finish_reason=None, response_model=None, usage=None),
                ModelResponse(raw_content=None, content_present=True, finish_reason=None, response_model=None, usage=TokenUsage()),
                result().response,
            )):
                key = self.key.model_copy(update={"repetition_index": index})
                journal.append_attempt(key, 0, result(response=response))
        records = read_events(self.root, self.manifest.run_id)
        self.assertFalse(records[0].result.response.content_present)
        self.assertIsNone(records[0].result.response.usage)
        self.assertTrue(records[1].result.response.content_present)
        self.assertIsNone(records[1].result.response.usage.prompt_tokens)
        self.assertEqual(records[2].result.response.usage.prompt_tokens, 0)
        self.assertIsNone(records[2].result.response.usage.total_tokens)

    def test_provider_metadata_filtering_is_deterministic(self):
        unsafe = "http://synthetic.invalid/REJECTED_PROVIDER"
        variants = (("stop", "synthetic/model", ()), (None, None, ()),
                    (unsafe, "synthetic/model", ("finish_reason",)),
                    ("stop", unsafe, ("response_model",)),
                    (unsafe, unsafe, ("finish_reason", "response_model")))
        with self.open() as journal:
            for index, (finish, model, omitted) in enumerate(variants):
                response = ModelResponse(raw_content="candidate", content_present=True, finish_reason=finish,
                                         response_model=model, usage=None)
                journal.append_attempt(self.key.model_copy(update={"repetition_index": index}),
                                       0, result(response=response))
                stored = journal.records()[-1].result.response
                self.assertEqual(stored.omitted_metadata, omitted)
                self.assertEqual(stored.finish_reason, None if "finish_reason" in omitted else finish)
                self.assertEqual(stored.response_model, None if "response_model" in omitted else model)
                self.assertNotIn(unsafe, repr(stored))
        self.assertNotIn(unsafe.encode(), (self.run / "journal.jsonl").read_bytes())

    def test_omission_marker_corruption_rejected(self):
        base = self.event_data()
        for markers in (["unknown"], ["finish_reason", "finish_reason"],
                        ["response_model", "finish_reason"], ["finish_reason"]):
            data = copy.deepcopy(base)
            data["result"]["response"]["omitted_metadata"] = markers
            # The last variant deliberately retains non-null finish_reason.
            if markers != ["finish_reason"]:
                data["result"]["response"]["finish_reason"] = None
                data["result"]["response"]["response_model"] = None
            self.write_private(self.run / "journal.jsonl", canonical_json_bytes(data) + b"\n")
            with self.assertRaises(JournalCorruptionError):
                read_events(self.root, self.manifest.run_id)

    def test_attempt_retry_success_finalize_and_resume(self):
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result(status="timeout", response=None, retryable=True))
            self.assertEqual(journal.completed_keys(), frozenset())
            journal.append_attempt(self.key, 1, result())
            journal.finalize(self.key, 1)
            before = journal.records()
            self.assertEqual(journal.completed_keys(), frozenset({self.key}))
            self.assertEqual([item.kind for item in before], ["attempt", "attempt", "finalized"])
            self.assertEqual(before[-1].model_dump().keys(),
                             {"schema_version", "kind", "execution_key", "attempt_index"})
        with self.open() as resumed:
            self.assertEqual(resumed.records(), before)
            self.assertEqual(resumed.completed_keys(), frozenset({self.key}))

    def test_all_error_statuses_recordable_and_finalized_failures_completed(self):
        with self.open() as journal:
            for index, status in enumerate(("timeout", "connection_error", "http_error", "protocol_error")):
                key = self.key.model_copy(update={"repetition_index": index})
                journal.append_attempt(key, 0, result(status=status, response=None, error_code="synthetic_error"))
                journal.finalize(key, 0)
                self.assertIn(key, journal.completed_keys())
        self.assertEqual(len(read_events(self.root, self.manifest.run_id)), 8)

    def test_unfinished_success_recovered_without_model_call(self):
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result())
        with self.open() as resumed:
            self.assertEqual(resumed.completed_keys(), frozenset())
            self.assertIsInstance(resumed.records()[0], AttemptRecorded)
            self.assertEqual(resumed.records()[0].result.response.raw_content, "SYNTHETIC_CANDIDATE")
            resumed.finalize(self.key, 0)
            self.assertEqual(len(resumed.records()), 2)

    def test_execution_repetition_and_request_fingerprint_remain_distinct(self):
        keys = (self.key, self.key.model_copy(update={"repetition_index": 1}),
                self.key.model_copy(update={"request_fingerprint": "e" * 64}))
        with self.open() as journal:
            for key in keys:
                journal.append_attempt(key, 0, result())
                journal.finalize(key, 0)
            self.assertEqual(journal.completed_keys(), frozenset(keys))

    def test_invalid_transitions_rejected_before_disk_change(self):
        with self.open() as journal:
            for operation in (lambda: journal.append_attempt(self.key, 1, result()),
                              lambda: journal.finalize(self.key, 0)):
                with self.assertRaises(DuplicateExecutionError):
                    operation()
            self.assertFalse((self.run / "journal.jsonl").exists())
            journal.append_attempt(self.key, 0, result())
            for index in (0, 2):
                before = (self.run / "journal.jsonl").read_bytes()
                with self.assertRaises(DuplicateExecutionError):
                    journal.append_attempt(self.key, index, result())
                self.assertEqual((self.run / "journal.jsonl").read_bytes(), before)
            journal.append_attempt(self.key, 1, result())
            with self.assertRaises(DuplicateExecutionError):
                journal.finalize(self.key, 0)
            journal.finalize(self.key, 1)
            for operation in (lambda: journal.finalize(self.key, 1),
                              lambda: journal.append_attempt(self.key, 2, result())):
                with self.assertRaises(DuplicateExecutionError):
                    operation()

    def test_wrong_execution_run_id_rejected_before_append(self):
        with self.open() as journal:
            key = self.key.model_copy(update={"run_id": "a" * 64})
            with self.assertRaises(DuplicateExecutionError):
                journal.append_attempt(key, 0, result())
            self.assertFalse((self.run / "journal.jsonl").exists())

    def test_stage3b_unusual_model_valid_results_are_preserved(self):
        variants = (result(response=None), result(status="protocol_error"),
                    result(error_code="synthetic_error", http_status=500), result(retryable=True))
        with self.open() as journal:
            for index, value in enumerate(variants):
                journal.append_attempt(self.key.model_copy(update={"repetition_index": index}), 0, value)
                stored = journal.records()[-1].result
                for field in ("status", "latency_ms", "error_code", "http_status", "retryable"):
                    self.assertEqual(getattr(stored, field), getattr(value, field))
                self.assertEqual(stored.response is None, value.response is None)

    def test_field_validation_not_weakened_by_storage(self):
        with self.open() as journal:
            for index in (-1, True, "0"):
                with self.assertRaises(PersistenceError):
                    journal.append_attempt(self.key, index, result())
            for change in ({"latency_ms": math.inf}, {"latency_ms": -1.0}, {"retryable": 1},
                           {"http_status": "500"}, {"error_code": "http://synthetic.invalid/error"}):
                with self.assertRaises(PersistenceError):
                    journal.append_attempt(self.key, 0, result().model_copy(update=change))
            self.assertFalse((self.run / "journal.jsonl").exists())

    def test_missing_empty_journals_and_missing_manifest(self):
        with self.assertRaises(PersistenceError):
            read_events(self.root, self.manifest.run_id)
        self.open().close()
        self.assertEqual(read_events(self.root, self.manifest.run_id), ())
        self.write_private(self.run / "journal.jsonl", b"")
        self.assertEqual(read_events(self.root, self.manifest.run_id), ())

    def test_strict_journal_json_corruption_no_prefix_returned(self):
        base = self.event_data()
        valid = canonical_json_bytes(base) + b"\n"
        variants = [b"\n", b" \n", b"\xff\n", b"{\n", b'{"x":1,"x":2}\n', b"42\n", b"[]\n",
                    b'{"x":NaN}\n', b'{"x":Infinity}\n', b'{"x":-Infinity}\n', b'{"x":[1e999]}\n',
                    b'{"PRIVATE_CORRUPT":', canonical_json_bytes(base)]
        for data in variants:
            for prefix in (b"", valid):
                with self.subTest(data=data[:30], prefix=bool(prefix)):
                    damaged = prefix + data
                    self.write_private(self.run / "journal.jsonl", damaged)
                    with self.assertRaises(JournalCorruptionError) as caught:
                        read_events(self.root, self.manifest.run_id)
                    self.assert_safe(caught.exception, "PRIVATE_CORRUPT", "SYNTHETIC_CANDIDATE")
                    self.assertEqual((self.run / "journal.jsonl").read_bytes(), damaged)
                    with self.assertRaises(JournalCorruptionError):
                        self.open()

    def test_journal_schema_corruption(self):
        base = self.event_data()
        variants = [base | change for change in (
            {"schema_version": "unknown"}, {"kind": "unknown"}, {"extra": "PRIVATE_CORRUPT"},
            {"attempt_index": True}, {"attempt_index": "0"},
        )]
        for field in base:
            missing = copy.deepcopy(base)
            del missing[field]
            variants.append(missing)
        for field in base["result"]:
            missing = copy.deepcopy(base)
            del missing["result"][field]
            variants.append(missing)
        for field in base["result"]["response"]:
            missing = copy.deepcopy(base)
            del missing["result"]["response"][field]
            variants.append(missing)
        missing = copy.deepcopy(base)
        del missing["result"]["response"]["usage"]["prompt_tokens"]
        variants.append(missing)
        for data in variants:
            self.write_private(self.run / "journal.jsonl", canonical_json_bytes(data) + b"\n")
            with self.assertRaises(JournalCorruptionError) as caught:
                read_events(self.root, self.manifest.run_id)
            self.assert_safe(caught.exception, "PRIVATE_CORRUPT", "SYNTHETIC_CANDIDATE")

    def test_malformed_middle_record_reports_line_without_repair(self):
        base = self.event_data()
        first = canonical_json_bytes(base) + b"\n"
        last = canonical_json_bytes(base | {"attempt_index": 1}) + b"\n"
        damaged = first + b'{"PRIVATE_MIDDLE": broken}\n' + last
        self.write_private(self.run / "journal.jsonl", damaged)
        with self.assertRaisesRegex(JournalCorruptionError, "journal.jsonl:2: invalid_record") as caught:
            read_events(self.root, self.manifest.run_id)
        self.assert_safe(caught.exception, "PRIVATE_MIDDLE", "SYNTHETIC_CANDIDATE")
        self.assertEqual((self.run / "journal.jsonl").read_bytes(), damaged)

    def test_journal_nested_usage_and_result_fields_remain_strict(self):
        base = self.event_data()
        variants = []
        for value in (True, "2", 2.0, -1):
            data = copy.deepcopy(base)
            data["result"]["response"]["usage"]["prompt_tokens"] = value
            variants.append(canonical_json_bytes(data) + b"\n")
        line = canonical_json_bytes(base)
        for constant in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            variants.append(line.replace(b'"latency_ms":1.25', b'"latency_ms":' + constant) + b"\n")
        for data in variants:
            self.write_private(self.run / "journal.jsonl", data)
            with self.assertRaises(JournalCorruptionError):
                read_events(self.root, self.manifest.run_id)

    def test_reader_enforces_every_event_transition(self):
        attempt = self.event_data()
        second = attempt | {"attempt_index": 1}
        finalized = {"schema_version": JOURNAL_SCHEMA_VERSION, "kind": "finalized",
                     "execution_key": self.key.model_dump(), "attempt_index": 0}
        wrong = copy.deepcopy(attempt)
        wrong["execution_key"]["run_id"] = "a" * 64
        variants = ([second], [attempt, attempt], [attempt, attempt | {"attempt_index": 2}],
                    [finalized], [attempt, second, finalized], [attempt, finalized, finalized],
                    [attempt, finalized, second], [wrong])
        for records in variants:
            self.write_private(self.run / "journal.jsonl",
                               b"".join(canonical_json_bytes(item) + b"\n" for item in records))
            with self.assertRaises(JournalCorruptionError):
                read_events(self.root, self.manifest.run_id)

    def test_short_writes_finish_manifest_and_journal(self):
        real_write = os.write
        def short_write(fd, data):
            return real_write(fd, data[:17])
        with patch("ovp36_benchmark.persistence.os.write", side_effect=short_write):
            with self.open() as journal:
                journal.append_attempt(self.key, 0, result())
                journal.finalize(self.key, 0)
        self.assertEqual(load_manifest(self.root, self.manifest.run_id), self.manifest)
        self.assertEqual(len(read_events(self.root, self.manifest.run_id)), 2)

    def test_partial_write_failure_poisons_without_truncation(self):
        real_write = os.write
        with self.open() as journal:
            calls = 0
            def failed_write(fd, data):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(fd, data[:9])
                raise OSError("PRIVATE_WRITE_FAILURE")
            with patch("ovp36_benchmark.persistence.os.write", side_effect=failed_write):
                with self.assertRaises(PersistenceError) as caught:
                    journal.append_attempt(self.key, 0, result())
            self.assert_safe(caught.exception, "PRIVATE_WRITE_FAILURE")
            self.assertEqual(journal.records(), ())
            self.assertEqual((self.run / "journal.jsonl").stat().st_size, 9)
            with self.assertRaisesRegex(PersistenceError, "writer_unusable"):
                journal.append_attempt(self.key, 0, result())
        with self.assertRaises(JournalCorruptionError):
            self.open()
        self.assertEqual((self.run / "journal.jsonl").stat().st_size, 9)

    def test_zero_write_poisons_writer(self):
        with self.open() as journal:
            with patch("ovp36_benchmark.persistence.os.write", return_value=0):
                with self.assertRaises(PersistenceError):
                    journal.append_attempt(self.key, 0, result())
            with self.assertRaisesRegex(PersistenceError, "writer_unusable"):
                journal.finalize(self.key, 0)

    def test_file_and_directory_fsync_failure_poison_writer(self):
        real_sync = os.fsync
        for fail_directory in (False, True):
            with self.subTest(fail_directory=fail_directory):
                path = self.run / "journal.jsonl"
                if path.exists():
                    path.unlink()
                with self.open() as journal:
                    def failed_sync(fd):
                        if stat.S_ISDIR(os.fstat(fd).st_mode) == fail_directory:
                            raise OSError("PRIVATE_SYNC_FAILURE")
                        real_sync(fd)
                    with patch("ovp36_benchmark.persistence.os.fsync", side_effect=failed_sync):
                        with self.assertRaises(PersistenceError) as caught:
                            journal.append_attempt(self.key, 0, result())
                    self.assert_safe(caught.exception, "PRIVATE_SYNC_FAILURE")
                    self.assertEqual(journal.records(), ())
                    self.assertEqual(journal.completed_keys(), frozenset())
                    with self.assertRaisesRegex(PersistenceError, "writer_unusable"):
                        journal.finalize(self.key, 0)
                # The complete record may exist despite failed synchronization.
                with self.open() as resumed:
                    self.assertEqual(len(resumed.records()), 1)
                    self.assertEqual(resumed.completed_keys(), frozenset())

    def test_context_manager_close_idempotent_normal_and_exceptional(self):
        journal = self.open()
        with journal:
            pass
        journal.close()
        with self.assertRaises(PersistenceError):
            journal.append_attempt(self.key, 0, result())
        failed = self.open()
        with self.assertRaises(RuntimeError):
            with failed:
                raise RuntimeError("synthetic")
        with self.assertRaises(PersistenceError):
            failed.finalize(self.key, 0)

    def test_failed_finalization_sync_does_not_advance_memory(self):
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result())
            with patch("ovp36_benchmark.persistence.os.fsync", side_effect=OSError("synthetic")):
                with self.assertRaises(PersistenceError):
                    journal.finalize(self.key, 0)
            self.assertEqual(len(journal.records()), 1)
            self.assertEqual(journal.completed_keys(), frozenset())
        with self.open() as resumed:
            # A complete event may be present after uncertain synchronization.
            self.assertEqual(resumed.completed_keys(), frozenset({self.key}))

    def test_programming_errors_are_not_hidden_by_blanket_catch(self):
        with patch("ovp36_benchmark.persistence.make_run_id", side_effect=RuntimeError("synthetic bug")):
            with self.assertRaisesRegex(RuntimeError, "synthetic bug"):
                manifest()

    def test_private_journal_validation_exception_is_not_retained(self):
        data = self.event_data()
        marker = "SYNTHETIC_PRIVATE_VALIDATION_VALUE"
        data["result"]["latency_ms"] = marker
        self.write_private(self.run / "journal.jsonl", canonical_json_bytes(data) + b"\n")
        with self.assertRaises(JournalCorruptionError) as caught:
            read_events(self.root, self.manifest.run_id)
        self.assert_safe(caught.exception, marker, "SYNTHETIC_CANDIDATE")

    def test_private_json_decode_exception_is_not_retained(self):
        self.open().close()
        marker = "SYNTHETIC_PRIVATE_JSON_VALUE"
        self.write_private(self.run / "journal.jsonl", ('{"' + marker + '": broken}\n').encode())
        with self.assertRaises(JournalCorruptionError) as caught:
            read_events(self.root, self.manifest.run_id)
        self.assert_safe(caught.exception, marker)

    def test_private_filesystem_exception_is_not_retained(self):
        marker = "SYNTHETIC_PRIVATE_FILESYSTEM_PATH"
        with patch("ovp36_benchmark.persistence.os.open", side_effect=OSError(marker)):
            with self.assertRaises(PersistenceError) as caught:
                self.open()
        self.assert_safe(caught.exception, marker)

    def test_private_filesystem_read_exceptions_are_not_retained(self):
        self.event_data()
        marker = "SYNTHETIC_PRIVATE_READ_PATH"
        real_open = os.open
        for reader, filename in ((load_manifest, "manifest.json"), (read_events, "journal.jsonl")):
            def failed_open(path, *args, **kwargs):
                if Path(path).name == filename:
                    raise OSError(marker)
                return real_open(path, *args, **kwargs)
            with self.subTest(reader=reader), patch("ovp36_benchmark.persistence.os.open", side_effect=failed_open):
                with self.assertRaises(PersistenceError) as caught:
                    reader(self.root, self.manifest.run_id)
                self.assert_safe(caught.exception, marker)

    def test_private_append_wrapper_exception_is_not_retained(self):
        marker = "SYNTHETIC_PRIVATE_WRAPPER"
        with self.open() as journal:
            with patch("ovp36_benchmark.persistence._check_directory", side_effect=PersistenceError(marker)):
                with self.assertRaises(PersistenceError) as caught:
                    journal.append_attempt(self.key, 0, result())
            self.assert_safe(caught.exception, marker)
            with self.assertRaisesRegex(PersistenceError, "writer_unusable"):
                journal.append_attempt(self.key, 0, result())

    def test_private_invalid_model_inputs_are_not_retained(self):
        marker = "SYNTHETIC_PRIVATE_INVALID_INPUT"
        with self.assertRaises(PersistenceError) as caught:
            manifest(dataset_hash=marker)
        self.assert_safe(caught.exception, marker)
        with self.assertRaises(PersistenceError) as caught:
            load_manifest(self.root, marker)
        self.assert_safe(caught.exception, marker)
        invalid = self.manifest.model_copy(update={"dataset_hash": marker})
        with self.assertRaises(PersistenceError) as caught:
            ResultJournal.open(self.root, invalid)
        self.assert_safe(caught.exception, marker)
        with self.open() as journal:
            for operation in (
                lambda: journal.append_attempt(self.key, marker, result()),
                lambda: journal.finalize(self.key, marker),
                # Nested model instances are revalidated separately before append.
                lambda: journal.append_attempt(self.key.model_copy(update={"case_id": "/" + marker}),
                                               0, result()),
            ):
                with self.assertRaises(PersistenceError) as caught:
                    operation()
                self.assert_safe(caught.exception, marker, "SYNTHETIC_CANDIDATE")
            self.assertFalse((self.run / "journal.jsonl").exists())

    def test_manifest_and_journal_explicit_privacy_boundary(self):
        with self.open() as journal:
            journal.append_attempt(self.key, 0, result("SYNTHETIC_CANDIDATE_SECRET"))
        manifest_bytes = (self.run / "manifest.json").read_bytes()
        journal_bytes = (self.run / "journal.jsonl").read_bytes()
        for forbidden in (b"SYNTHETIC_REQUEST_MESSAGE", b"SYNTHETIC_HISTORICAL_OUTPUT",
                          b"synthetic.invalid", b"SYNTHETIC_KEY", b"Authorization",
                          b"base_url", b"headers", b"api_key"):
            self.assertNotIn(forbidden, manifest_bytes)
            self.assertNotIn(forbidden, journal_bytes)
        self.assertNotIn(b"SYNTHETIC_CANDIDATE_SECRET", manifest_bytes)
        self.assertIn(b"SYNTHETIC_CANDIDATE_SECRET", journal_bytes)
        self.assertEqual(json.loads(journal_bytes)["result"].keys(),
                         {"status", "latency_ms", "response", "error_code", "http_status", "retryable"})


if __name__ == "__main__":
    unittest.main()
