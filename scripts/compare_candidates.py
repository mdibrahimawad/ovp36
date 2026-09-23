"""Private, sequential local batch; compose the existing benchmark unchanged.

Run with the benchmark Python. The `artifact` and `serve` subcommands run in
the separately installed MLX environment. Nothing runs merely on import.
"""

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "private/model-comparison-20260916"
RUNTIME = Path.home() / ".local/share/uv/tools/mlx-lm/bin/python"
SOURCE = Path.home() / "Downloads/OVP34_CODEX_HANDOFF/OVP34_DELIVERY/data/analysis/ovp34/raw/final_100_direct_id/langfuse_observations.jsonl"
SELECTION = Path.home() / "Downloads/OVP34_CODEX_HANDOFF/analysis/ovp34/outputs/oob_jobs.csv"
URL = "http://127.0.0.1:8081/v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, *, replace=False):
    """Private artifacts are immutable except the explicit progress snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists() and not replace:
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact differs: {path.name}")
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        os.chmod(temporary, 0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_freeze(batch):
    frozen = read(batch / "freeze.json")
    for name, expected in frozen["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen benchmark changed: {name}")
    from ovp36_benchmark.__main__ import _implementation_fingerprints
    if _implementation_fingerprints() != (frozen["harness_fingerprint"], frozen["lock_fingerprint"]):
        raise ValueError("implementation fingerprint changed")
    return frozen


def export_requests(batch):
    from ovp36_benchmark.__main__ import FROZEN_DATASET_HASH
    from ovp36_benchmark.adapters import prepare_curated_plan
    from ovp36_benchmark.config import load_config
    from ovp36_benchmark.dataset import hash_dataset, load_curated_cases
    from ovp36_benchmark.qa_replay_comparison import load_qa_comparison_inputs
    from ovp36_benchmark.replay import prepare_replay_plan
    cases = load_curated_cases(ROOT / "data/curated")
    if len(cases) != 126 or hash_dataset(cases) != FROZEN_DATASET_HASH:
        raise ValueError("frozen dataset mismatch")
    config = load_config(ROOT / "candidate.local.toml")
    _, selected, _ = load_qa_comparison_inputs(SOURCE, SELECTION)
    curated = prepare_curated_plan(cases, config=config, resolved_model="batch/preflight")
    replay = prepare_replay_plan(selected, config=config, resolved_model="batch/preflight")
    if len(curated) != 106 or len(replay) != 10:
        raise ValueError("inventory mismatch")
    save(batch / "requests.json", [dict(case_id=p.request.case_id,
        messages=p.request.messages.to_messages(), message_hash=p.request.message_fingerprint,
        source=p.request.source.value) for p in (*curated, *replay)])


def artifact(batch, row):
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer
    directory = batch / row["key"]
    if (directory / "artifact.json").exists():
        previous = read(directory / "artifact.json")
        for name, expected in previous["file_hashes"].items():
            if digest(Path(previous["snapshot"]) / name) != expected:
                raise ValueError("cached artifact changed")
        return
    if row["artifact"]["license"] != "apache-2.0" or row["canonical"]["license"] != "apache-2.0":
        raise ValueError("unapproved license")
    if row["artifact"].get("local_conversion"):
        from mlx_lm.convert import convert
        source = Path(snapshot_download(row["canonical_id"], revision=row["canonical"]["revision"],
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model", "*.txt", "README.md", "LICENSE*"],
            max_workers=4))
        source_hashes = {p.name: digest(p) for p in sorted(source.iterdir()) if p.is_file()}
        for entry in row["canonical"]["files"]:
            if entry.get("lfs") and entry["name"] in source_hashes:
                if entry["lfs"]["sha256"] != source_hashes[entry["name"]]:
                    raise ValueError("canonical conversion source checksum mismatch")
        save(directory / "conversion-source.json", dict(snapshot=str(source),
            revision=row["canonical"]["revision"], file_hashes=source_hashes))
        snapshot = ROOT / "models" / (row["key"] + "-8bit")
        # MLX's save helper resolves a repo ID again without its revision. A
        # pinned local snapshot keeps both loading and saving on the same bytes.
        convert(str(source), mlx_path=str(snapshot), quantize=True,
                q_group_size=64, q_bits=8)
    else:
        snapshot = Path(snapshot_download(row["artifact_id"], revision=row["artifact"]["revision"],
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model", "*.txt", "README.md", "LICENSE*"],
            max_workers=4))
    hashes = {p.name: digest(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    for entry in row["artifact"].get("files", []):
        if entry.get("lfs") and entry["name"] in hashes:
            if entry["lfs"]["sha256"] != hashes[entry["name"]]:
                raise ValueError("artifact LFS checksum mismatch")
    config = read(snapshot / "config.json")
    quantization = config.get("quantization", config.get("quantization_config", {}))
    if quantization.get("bits") != 8:
        raise ValueError("8-bit artifact required")
    tokenizer = load_tokenizer(snapshot)
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError("native template required")
    if row["thinking"] and "enable_thinking" not in template:
        raise ValueError("supported thinking switch missing")
    save(directory / "artifact.json", dict(snapshot=str(snapshot), file_hashes=hashes,
        artifact_hash=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        template_hash=hashlib.sha256(template.encode()).hexdigest(), native_template=template,
        quantization=quantization, model_type=config["model_type"],
        runtime={name: importlib.metadata.version(name) for name in ("mlx-lm", "mlx", "transformers")}))


def check_context(tokenizer, requests, thinking, allocation):
    from mlx_lm.server import process_message_content
    counts = {}
    for request in requests:
        messages = copy.deepcopy(request["messages"])
        process_message_content(messages)
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               tools=None, tokenize=True, **thinking)
        if len(tokens) + 512 > allocation:
            raise ValueError(f"complete input plus output exceeds context: {request['case_id']}")
        counts[request["case_id"]] = len(tokens)
    return counts


def serve(batch, row):
    """Restrict the native MLX server to one pinned snapshot, with no fallback."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import server
    info = read(batch / row["key"] / "artifact.json")
    ignored_shared_kv = []
    if info["model_type"] == "gemma4":
        from mlx_lm.models import gemma4
        native_sanitize = gemma4.Model.sanitize
        def sanitize_shared_kv(model, weights):
            # Transformers Gemma4TextModel explicitly ignores these unused
            # checkpoint tensors in KV-sharing layers. MLX 0.31.3 omits that
            # load-time filter; keep strict loading for every remaining tensor.
            weights = native_sanitize(model, weights)
            prefixes = tuple(f"language_model.model.layers.{i}.self_attn.{name}."
                for i, layer in enumerate(model.layers) if not layer.self_attn.has_kv
                for name in ("k_proj", "v_proj", "k_norm", "v_norm"))
            ignored_shared_kv.extend(k for k in weights if k.startswith(prefixes))
            return {k: v for k, v in weights.items() if not k.startswith(prefixes)}
        gemma4.Model.sanitize = sanitize_shared_kv
    native_provider = server.ModelProvider
    class PinnedProvider(native_provider):
        def __init__(self, args):
            super().__init__(args)
            self._model_map.update({row["artifact_id"]: info["snapshot"], "default_model": info["snapshot"]})

        def load(self, model_path, adapter_path=None, draft_model_path=None):
            if model_path not in (row["artifact_id"], "default_model") or adapter_path is not None:
                raise ValueError("only pinned model is permitted")
            return super().load(model_path, adapter_path, draft_model_path)

        def _load(self, *args):
            start = time.monotonic()
            super()._load(*args)
            if hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest() != info["template_hash"]:
                raise ValueError("loaded template mismatch")
            config = read(Path(info["snapshot"]) / "config.json")
            native_limit = config.get("text_config", config).get("max_position_embeddings")
            allocation = row["context_allocation"]
            if native_limit is not None and native_limit < allocation:
                raise ValueError("native context smaller than allocation")
            counts = check_context(self.tokenizer, read(batch / "requests.json"), row["thinking"], allocation)
            parameters = tree_flatten(self.model.parameters())
            nontext = [k for k, _ in parameters if any(x in k for x in ("vision", "visual", "audio", "multi_modal"))]
            if nontext:
                raise ValueError("unexpected loaded multimodal weights")
            save(batch / row["key"] / ("loaded-" + uuid.uuid4().hex + ".json"), dict(
                load_seconds=time.monotonic()-start, context_allocation=allocation,
                native_context_limit=native_limit, prompt_tokens=counts, no_input_truncation=True,
                thinking=row["thinking"], loaded_nontext_weights=nontext,
                ignored_unused_shared_kv_tensors=ignored_shared_kv,
                loaded_parameter_prefixes=sorted({k.split('.')[0] for k, _ in parameters}),
                resident_array_bytes=sum(v.nbytes for _, v in parameters),
                mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory(),
                text_only=True, model_class=type(self.model).__module__))
    class PinnedHandler(server.APIHandler):
        def handle_models_request(self):
            self._set_completion_headers(200)
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [{"id": row["artifact_id"], "object": "model"}]}).encode())
    native_http = server._run_http_server
    server.ModelProvider = PinnedProvider
    server._run_http_server = lambda host, port, generator: native_http(host, port, generator, handler_class=PinnedHandler)
    sys.argv = ["mlx_lm.server", "--model", row["artifact_id"], "--host", "127.0.0.1", "--port", "8081",
                "--chat-template-args", json.dumps(row["thinking"]), "--log-level", "WARNING"]
    server.main()


def write_config(batch, row):
    info = read(batch / row["key"] / "artifact.json")
    server = dict(runtime="mlx-lm", runtime_version=info["runtime"]["mlx-lm"],
        model_checkpoint=row["canonical_id"], model_revision=row["artifact"]["revision"],
        model_artifact_hash=info["artifact_hash"], quantization="8bit-affine-group64",
        chat_template_hash=info["template_hash"], context_limit=row["context_allocation"])
    base = (ROOT / "candidate.local.toml").read_text()
    base = base.replace('"candidate-smoke-granite-4-h-tiny-8bit"', f'"candidate-smoke-batch-{row["key"]}"')
    base = base.replace('"local-granite-4-h-tiny-8bit"', f'"local-batch-{row["key"]}"')
    text = base + "\n[server]\n" + "\n".join(f"{k} = {json.dumps(v)}" for k, v in server.items()) + "\n"
    path = batch / row["key"] / "candidate.local.toml"
    if path.exists() and path.read_text() != text:
        raise ValueError("immutable local configuration changed")
    if not path.exists():
        with path.open("x") as stream:
            stream.write(text)
    return path


async def phase(batch, row, name):
    import httpx
    import ovp36_benchmark.__main__ as cli
    native_client = cli.ModelClient
    class Capture(httpx.AsyncBaseTransport):
        def __init__(self):
            self.delegate = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
            self.exchanges = []

        async def handle_async_request(self, request):
            response = await self.delegate.handle_async_request(request)
            await response.aread()
            self.exchanges.append(dict(request_body=request.content.decode(), status=response.status_code,
                                       response_body=response.content.decode()))
            return response

        async def aclose(self):
            await self.delegate.aclose()
    class CapturingClient(native_client):
        def __init__(self, endpoint, **kwargs):
            self.capture = Capture()
            super().__init__(endpoint, transport=self.capture, **kwargs)

        async def send_once(self, request):
            result = await super().send_once(request)
            # Persist after the existing client stops its latency clock.
            save(batch / row["key"] / "wire" / name / (request.case_id + "-" + uuid.uuid4().hex + ".json"),
                 dict(case_id=request.case_id, request_fingerprint=request.request_fingerprint,
                      message_fingerprint=request.message_fingerprint, exchanges=self.capture.exchanges,
                      status=result.status))
            self.capture.exchanges = []
            return result
    cli.ModelClient = CapturingClient
    try:
        config = write_config(batch, row)
        if name == "replay":
            return await cli.qa_replay_compare(config, ROOT / "results", SOURCE, SELECTION)
        operation = cli.candidate_smoke if name == "smoke" else cli.candidate_suite
        return await operation(config, ROOT / "results")
    finally:
        cli.ModelClient = native_client


def batch_run(batch):
    import httpx
    frozen = verify_freeze(batch)
    export_requests(batch)
    rows = read(batch / "models.json")
    progress_path = batch / "progress.json"
    progress = read(progress_path) if progress_path.exists() else {}
    env = dict(os.environ, OVP36_BASE_URL=URL, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1")
    for name in ("OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_LOG"):
        env.pop(name, None)
    for row in rows:
        key = row["key"]
        state = progress.setdefault(key, {"status": "pending", "phases": {}})
        if state["status"] in ("complete", "blocked"):
            continue
        directory = batch / key
        directory.mkdir(mode=0o700, exist_ok=True)
        env["OVP36_MODEL"] = row["artifact_id"]
        state["status"] = "running"
        save(progress_path, progress, replace=True)
        print(key, row["artifact_id"], "starting", flush=True)
        process = None
        try:
            with (directory / "artifact.log").open("a") as log:
                subprocess.run([str(RUNTIME), __file__, "artifact", "--batch", str(batch), "--model", key],
                               env=env, stdout=log, stderr=log, check=True)
            write_config(batch, row)
            with socket.socket() as probe_socket:
                if probe_socket.connect_ex(("127.0.0.1", 8081)) == 0:
                    raise RuntimeError("loopback port already occupied; no existing server replaced")
            with (directory / "server.log").open("a") as log:
                process = subprocess.Popen([str(RUNTIME), __file__, "serve", "--batch", str(batch), "--model", key],
                                           env=env, stdout=log, stderr=log)
            state["server_pid"] = process.pid
            save(progress_path, progress, replace=True)
            start = time.monotonic()
            with httpx.Client(timeout=5, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("pinned server exited; inspect private server.log")
                    try:
                        models = client.get(URL + "/models").json()["data"]
                        if [m["id"] for m in models] != [row["artifact_id"]]:
                            raise RuntimeError("wrong model endpoint")
                        break
                    except (httpx.ConnectError, httpx.ReadTimeout):
                        if time.monotonic()-start > 300:
                            raise RuntimeError("server startup timeout")
                        time.sleep(1)
                probe = client.post(URL + "/chat/completions", timeout=180, json=dict(
                    model=row["artifact_id"], messages=[{"role": "user", "content": "What is 2 plus 2? Answer briefly."}],
                    temperature=0, top_p=1, max_tokens=64, stream=False))
                probe.raise_for_status()
                payload = probe.json()
                save(directory / ("warmup-" + uuid.uuid4().hex + ".json"), dict(
                    elapsed_startup_and_warmup_seconds=time.monotonic()-start, response=payload))
                if payload.get("model") != row["artifact_id"]:
                    raise RuntimeError("probe model mismatch")
                message = payload["choices"][0]["message"]
                if row["thinking"] and (message.get("reasoning") or message.get("reasoning_content")
                                         or "<think>" in (message.get("content") or "")):
                    raise RuntimeError("thinking-off probe emitted reasoning")
            for name in ("smoke", "suite", "replay"):
                if name in state["phases"]:
                    continue
                if (directory / (name + ".json")).exists():
                    state["phases"][name] = read(directory / (name + ".json"))
                    save(progress_path, progress, replace=True)
                    continue
                if key == "m11" and name == "suite":
                    if not all(frozen["baseline_compatibility"].values()):
                        raise ValueError("baseline compatibility not established")
                    state["phases"][name] = dict(run_id=frozen["baseline_run_id"], reused=True)
                else:
                    with (directory / (name + ".log")).open("a") as log:
                        operation = subprocess.Popen([sys.executable, __file__, "phase", "--batch", str(batch),
                            "--model", key, "--phase", name], env=env, stdout=log, stderr=log)
                        peak_rss = 0
                        while operation.poll() is None:
                            sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True)
                            if sample.returncode == 0 and sample.stdout.strip():
                                peak_rss = max(peak_rss, int(sample.stdout.strip())*1024)
                            time.sleep(0.25)
                        if operation.returncode:
                            raise RuntimeError(f"{name} orchestration failed; inspect private log")
                    state["phases"][name] = read(directory / (name + ".json"))
                    state["phases"][name]["sampled_server_rss_peak_bytes"] = peak_rss
                save(progress_path, progress, replace=True)
                print(key, name, state["phases"][name]["run_id"], flush=True)
            state["status"] = "complete"
        except Exception as error:
            state.update(status="blocked", blocker=str(error), error_type=type(error).__name__)
            print(key, "blocked:", str(error), flush=True)
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=30)
            state.pop("server_pid", None)
            save(progress_path, progress, replace=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "artifact", "serve", "phase"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--model")
    parser.add_argument("--phase", choices=("smoke", "suite", "replay"))
    args = parser.parse_args()
    save(args.batch / "invocations" / (uuid.uuid4().hex + ".json"), dict(
        command=args.command, model=args.model, phase=args.phase,
        script_sha256=digest(__file__), started_unix=time.time()))
    if args.command == "run":
        batch_run(args.batch.resolve())
        return
    row = next(r for r in read(args.batch / "models.json") if r["key"] == args.model)
    if args.command == "artifact":
        artifact(args.batch, row)
    elif args.command == "serve":
        serve(args.batch, row)
    else:
        verify_freeze(args.batch)
        result = asyncio.run(phase(args.batch, row, args.phase))
        save(args.batch / args.model / (args.phase + ".json"), result)


if __name__ == "__main__":
    main()
