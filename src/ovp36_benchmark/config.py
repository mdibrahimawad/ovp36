"""TOML loading and explicit environment resolution, without HTTP execution."""

import json
import os
from collections.abc import Mapping
from pathlib import Path
import tomllib

from pydantic import Field, SecretStr, TypeAdapter

from .schemas import (
    EndpointConfig, RunConfig, SafeIdentifier, SafeMetadata, ServerMetadata,
    StrictModel, validate_base_url,
)


class ResolvedEndpoint(StrictModel):
    """Runtime-only values are excluded even from ordinary model serialization."""

    alias: SafeIdentifier
    base_url: str = Field(exclude=True, repr=False)
    model: str = Field(exclude=True, repr=False)
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)


def load_config(path: str | Path) -> RunConfig:
    """Load configuration without resolving or embedding environment values."""
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    # JSON-mode strict validation accepts JSON arrays/enums without coercing
    # strings to numbers/bools. TOML dates are deliberately not part of this schema.
    return RunConfig.model_validate_json(json.dumps(data, allow_nan=False))


def resolve_endpoint(
    config: EndpointConfig, environ: Mapping[str, str] | None = None,
) -> ResolvedEndpoint:
    """Resolve only named variables; never expand arbitrary configuration text."""
    environment = os.environ if environ is None else environ

    def read(name: str) -> str:
        value = environment.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"required environment variable is missing or empty: {name}")
        return value

    base_url = read(config.base_url_env) if config.base_url_env else config.base_url
    model = read(config.model_env) if config.model_env else config.model
    key = read(config.api_key_env) if config.api_key_env else None
    validate_base_url(base_url)
    return ResolvedEndpoint(
        alias=config.alias, base_url=base_url, model=model,
        api_key=SecretStr(key) if key is not None else None,
    )


def redact_config(config: RunConfig) -> dict[str, object]:
    """Return a serializable configuration containing references, never API keys.

    Direct API keys are forbidden by EndpointConfig. URL userinfo, query, and
    fragment credentials are rejected rather than copied into saved metadata.
    """
    return config.model_dump(mode="json")


def server_metadata_projection(server: ServerMetadata) -> dict[str, str | int]:
    """Explicit allowlist, shared by safe config and pure run identity."""
    fields = {
        "runtime": server.runtime, "runtime_version": server.runtime_version,
        "model_checkpoint": server.model_checkpoint, "model_revision": server.model_revision,
        "model_artifact_hash": server.model_artifact_hash, "quantization": server.quantization,
        "chat_template_hash": server.chat_template_hash, "context_limit": server.context_limit,
    }
    return {name: value for name, value in fields.items() if value is not None}


def persistent_config_projection(config: RunConfig, *, resolved_model: str) -> dict[str, object]:
    """Allowlisted experiment identity; no URL, URL hash, or environment names.

    Supply the model resolved for dispatch, using a nonsecret model identifier.
    Runtime model paths must instead be assigned a safe serving model name.
    This function performs no environment or filesystem access.
    """
    model = TypeAdapter(SafeMetadata).validate_python(resolved_model, strict=True)
    label = TypeAdapter(SafeMetadata).validate_python(config.experiment_label, strict=True)
    if config.endpoint.model is not None and config.endpoint.model != model:
        raise ValueError("resolved model differs from configured literal model")
    generation = config.generation
    settings = {
        "temperature": generation.temperature, "max_tokens": generation.max_tokens,
        "token_limit_field": generation.token_limit_field, "top_p": generation.top_p,
    }
    if generation.seed is not None:
        settings["seed"] = generation.seed
    return {
        "endpoint_alias": config.endpoint.alias, "requested_model": model,
        "generation": settings, "timeout_seconds": config.timeout_seconds,
        "repetitions": config.repetitions, "concurrency": config.concurrency,
        "experiment_label": label, "evaluation": config.evaluation,
        "server": server_metadata_projection(config.server),
    }
