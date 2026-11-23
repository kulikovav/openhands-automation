"""Registry of configured execution targets (``AUTOMATION_TARGETS``).

A target names one governed Agent Server an automation can run on: the run is
dispatched as a single Agent Server conversation there instead of the bash
entrypoint. The registry parses and validates the JSON array and resolves
targets for validation, dispatch and capability reporting. It never logs or
returns ``api_key``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final
from urllib.parse import urlparse

from fastapi import HTTPException, status


# Exactly these keys are allowed on an entry; anything else is a startup error.
TARGET_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "name",
        "url",
        "api_key",
        "origin",
        "capability",
        "policy_id",
        "agent_profile_id",
    }
)
_REQUIRED_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {"id", "name", "url", "api_key", "agent_profile_id"}
)

TARGET_FIELD: Final[str] = "target_id"
CODE_TARGET_UNAVAILABLE: Final[str] = "target_unavailable"
CODE_TARGET_REQUIRES_PROMPT_PRESET: Final[str] = "target_requires_prompt_preset"


class TargetConfigError(ValueError):
    """``AUTOMATION_TARGETS`` is malformed; the service must not start on it."""


@dataclass(frozen=True)
class AutomationTarget:
    """One configured execution target. ``api_key`` must never be exposed."""

    id: str
    name: str
    url: str
    api_key: str
    origin: str
    capability: str | None = None
    policy_id: str | None = None
    agent_profile_id: str | None = None

    def public(self) -> dict[str, str]:
        """The capability-report view: no secret and no URL beyond the origin."""
        return {"id": self.id, "name": self.name, "origin": self.origin}


def _origin_of(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def parse_targets(raw: str) -> dict[str, AutomationTarget]:
    """Parse the ``AUTOMATION_TARGETS`` JSON array into targets by id.

    Unset or empty means no targets and legacy behavior is unchanged. Raises
    ``TargetConfigError`` on any malformed entry, naming the entry and key.
    """
    if not raw or not raw.strip():
        return {}
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TargetConfigError(f"AUTOMATION_TARGETS is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise TargetConfigError("AUTOMATION_TARGETS must be a JSON array")

    targets: dict[str, AutomationTarget] = {}
    for index, entry in enumerate(entries):
        where = f"AUTOMATION_TARGETS entry {index}"
        if not isinstance(entry, dict):
            raise TargetConfigError(f"{where} must be an object")
        unknown = set(entry) - TARGET_ENTRY_KEYS
        if unknown:
            raise TargetConfigError(
                f"{where}: unknown key(s) {sorted(unknown)}; "
                f"allowed keys: {sorted(TARGET_ENTRY_KEYS)}"
            )
        missing = _REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            raise TargetConfigError(f"{where}: missing key(s) {sorted(missing)}")
        for key in _REQUIRED_ENTRY_KEYS:
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise TargetConfigError(f"{where}: {key} must be a non-empty string")
        try:
            agent_profile_id = (
                str(uuid.UUID(str(entry["agent_profile_id"])))
                if entry.get("agent_profile_id")
                else None
            )
        except ValueError as exc:
            raise TargetConfigError(
                f"{where}: agent_profile_id must be a UUID"
            ) from exc
        url = entry["url"].strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise TargetConfigError(f"{where}: url must be an http(s) URL")
        target_id = entry["id"].strip()
        if target_id in targets:
            raise TargetConfigError(f"{where}: duplicate target id {target_id!r}")
        capability = entry.get("capability")
        if capability is not None and not isinstance(capability, str):
            raise TargetConfigError(f"{where}: capability must be a string")
        targets[target_id] = AutomationTarget(
            id=target_id,
            name=entry["name"].strip(),
            url=url.rstrip("/"),
            api_key=entry["api_key"],
            origin=(entry.get("origin") or _origin_of(url)).strip(),
            capability=capability,
            policy_id=entry.get("policy_id"),
            agent_profile_id=agent_profile_id,
        )
    return targets


@lru_cache(maxsize=4)
def _parse_cached(raw: str) -> dict[str, AutomationTarget]:
    return parse_targets(raw)


def get_targets() -> dict[str, AutomationTarget]:
    """The configured targets, parsed from the current settings."""
    from openhands.automation.config import get_config

    return _parse_cached(get_config().service.targets)


def resolve_target(target_id: str | None) -> AutomationTarget | None:
    """The configured target named by ``target_id``, or None when unset/removed."""
    if not target_id:
        return None
    return get_targets().get(target_id)


def target_name_of(target_id: str | None) -> str | None:
    """The configured display name of a target id (never a secret)."""
    target = resolve_target(target_id)
    return target.name if target else None


def is_prompt_preset_automation(
    preset_metadata: dict[str, Any] | None,
    plugin_spec: dict[str, Any] | None,
) -> bool:
    """Whether the automation is the prompt preset, the only supported kind."""
    return plugin_spec is None and (preset_metadata or {}).get("preset_type") == (
        "prompt"
    )


def target_field_error(field: str, code: str, message: str) -> HTTPException:
    """The 422 field error shape every target refusal uses."""
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=[{"loc": ["body", field], "msg": message, "type": code}],
    )


def validate_target_selection(
    target_id: str | None,
    *,
    preset_metadata: dict[str, Any] | None,
    plugin_spec: dict[str, Any] | None,
    field: str = TARGET_FIELD,
) -> AutomationTarget | None:
    """Validate one target selection at create, update, enable and dispatch.

    A null target selects the deployment default and is always valid. A
    configured target must exist and the automation must be the prompt preset:
    script, plugin and custom-tarball automations never run on a target, and
    there is no fallback to the default backend.
    """
    if not target_id:
        return None
    target = resolve_target(target_id)
    if target is None:
        raise target_field_error(
            field,
            CODE_TARGET_UNAVAILABLE,
            f"No execution target named {target_id!r} is configured.",
        )
    if not is_prompt_preset_automation(preset_metadata, plugin_spec):
        raise target_field_error(
            field,
            CODE_TARGET_REQUIRES_PROMPT_PRESET,
            (
                f"Target {target.name!r} runs only prompt-preset automations; "
                "script, plugin and custom-tarball automations are refused."
            ),
        )
    return target
