"""Versioned source prompt text and simple rendering; no task input formatting."""

from collections.abc import Mapping
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from string import Formatter
from typing import Literal, Self

from pydantic import Field, model_validator

from .schemas import ExtractionVariable, NonEmpty, PositiveInt, PositiveNumber, StrictModel, Task


def _template_fields(template: str) -> set[str]:
    names = set()
    for _, name, spec, conversion in Formatter().parse(template):
        if name is not None:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or spec or conversion:
                raise ValueError("templates support named placeholders only")
            names.add(name)
    return names


class SourceSettings(StrictModel):
    max_tokens: PositiveInt | None = None
    timeout_seconds: PositiveNumber | None = None


class PromptContract(StrictModel):
    id: NonEmpty
    version: PositiveInt
    task: Task
    completeness: Literal["complete", "partial"]
    system_text: NonEmpty | None = None
    # Olegos double-brace substitution belongs to the QA adapter, not format_map.
    system_template: NonEmpty | None = None
    user_input_kind: Literal["template", "messages", "node_description"] = "template"
    user_template: NonEmpty | None = None
    system_fragment: NonEmpty | None = None
    variable_line_template: NonEmpty | None = None
    optional_system_append: NonEmpty | None = None
    extraction_variables: tuple[ExtractionVariable, ...] = ()
    missing_components: tuple[NonEmpty, ...] = ()
    source_refs: tuple[NonEmpty, ...]
    source_settings: SourceSettings = Field(default_factory=SourceSettings)

    @model_validator(mode="after")
    def honest_completeness(self) -> Self:
        if self.system_text is not None and self.system_template is not None:
            raise ValueError("provide system_text or system_template, not both")
        if self.user_input_kind != "template" and self.user_template is not None:
            raise ValueError("message contexts and node descriptions are not user text templates")
        if self.completeness == "complete":
            system_known = self.system_text is not None or self.system_template is not None
            user_known = self.user_input_kind != "template" or self.user_template is not None
            if not system_known or not user_known or self.missing_components:
                raise ValueError("complete contracts require known system/user text and no missing components")
            if self.system_fragment is not None:
                raise ValueError("a source fragment cannot stand in for a complete system prompt")
        elif not self.missing_components:
            raise ValueError("partial contracts must identify missing components")
        for template in (self.user_template, self.variable_line_template):
            if template is not None:
                _template_fields(template)
        if self.optional_system_append and (
            self.system_text is None
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.optional_system_append)
        ):
            raise ValueError("system append requires a known system prompt and a named variable")
        return self


class ContractCatalog(StrictModel):
    schema_version: Literal["1"]
    contracts: tuple[PromptContract, ...]

    @model_validator(mode="after")
    def unique_versions(self) -> Self:
        keys = [(contract.id, contract.version) for contract in self.contracts]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate contract ID/version")
        return self


class RenderedContract(StrictModel):
    contract_id: NonEmpty
    version: PositiveInt
    contract_hash: NonEmpty
    completeness: Literal["complete", "partial"]
    missing_components: tuple[NonEmpty, ...]
    system_text: str | None
    user_text: str | None
    system_fragment: str | None
    source_settings: SourceSettings


def load_contract(
    contract_id: str, *, version: int = 1, resource_path: str | Path | None = None,
) -> PromptContract:
    """Load an explicit version; never silently select the latest version."""
    if type(version) is not int or version <= 0:
        raise ValueError("contract version must be a positive integer")
    resource = Path(resource_path) if resource_path is not None else files(
        "ovp36_benchmark"
    ).joinpath("prompts/contracts.json")
    catalog = ContractCatalog.model_validate_json(resource.read_text(encoding="utf-8"))
    for contract in catalog.contracts:
        if (contract.id, contract.version) == (contract_id, version):
            return contract
    raise ValueError(f"unknown contract/version: {contract_id}/{version}")


def hash_contract(contract: PromptContract) -> str:
    """Hash normalized metadata and exact strings; text whitespace is significant."""
    payload = json.dumps(contract.model_dump(mode="json"), sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_contract(
    contract: PromptContract, values: Mapping[str, str] | None = None, *,
    allow_partial: bool = False,
) -> RenderedContract:
    """Render supplied strings once. History/node formatters belong to adapters.

    Partial rendering is opt-in inspection: unknown components remain None and
    the result retains its partial status. A fragment is never promoted to system_text.
    """
    if type(allow_partial) is not bool:
        raise TypeError("allow_partial must be a boolean")
    if contract.completeness != "complete" and not allow_partial:
        raise ValueError(f"incomplete contract: {contract.id}; missing {contract.missing_components}")
    if contract.system_template is not None:
        raise ValueError("system template substitution requires the QA adapter; load_contract preserves the literal template")
    if contract.user_input_kind != "template":
        raise ValueError(f"{contract.user_input_kind} input requires its task adapter; source contract is known")
    supplied = dict(values or {})
    required = _template_fields(contract.user_template) if contract.user_template else set()
    optional = {contract.optional_system_append} if contract.optional_system_append else set()
    if missing := required - supplied.keys():
        raise ValueError(f"missing template variables: {sorted(missing)}")
    if unexpected := supplied.keys() - required - optional:
        raise ValueError(f"unexpected template variables: {sorted(unexpected)}")
    if any(not isinstance(value, str) for value in supplied.values()):
        raise TypeError("template values must be strings")
    system = contract.system_text
    if contract.optional_system_append and supplied.get(contract.optional_system_append):
        system += "\n\n" + supplied[contract.optional_system_append]
    user = contract.user_template.format_map(supplied) if contract.user_template else None
    return RenderedContract(
        contract_id=contract.id, version=contract.version, contract_hash=hash_contract(contract),
        completeness=contract.completeness, missing_components=contract.missing_components,
        system_text=system, user_text=user, system_fragment=contract.system_fragment,
        source_settings=contract.source_settings,
    )
