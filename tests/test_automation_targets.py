"""Execution targets: registry parsing, validation, capabilities, run snapshot."""

import json

import pytest
from httpx import AsyncClient

from openhands.automation.config import clear_config_cache
from openhands.automation.targets import (
    TargetConfigError,
    classify_automation,
    parse_targets,
)


pytestmark = pytest.mark.usefixtures("local_file_store")

BASE = "/api/automation/v1"
TARGET_ID = "governed-1"
TARGET_NAME = "Governed Sandbox"
TARGET_SECRET = "sk-target-secret-do-not-leak"
TARGET_ENTRY = {
    "id": TARGET_ID,
    "name": TARGET_NAME,
    "url": "https://governed.example:8443",
    "api_key": TARGET_SECRET,
    "agent_profile_id": "0a1b2c3d-0000-4000-8000-000000000042",
}


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Configuration is cached, so each test reads it fresh and leaves it clean."""
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def targets_env(monkeypatch):
    """One configured target, and nothing else, for the test."""
    monkeypatch.setenv("AUTOMATION_TARGETS", json.dumps([{**TARGET_ENTRY}]))
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def ready_deployment(monkeypatch):
    """A deployment that can accept work, so capabilities report targets."""
    monkeypatch.setenv("AUTOMATION_SERVICE_KEY", "service-key")
    clear_config_cache()


def _raw_body(name: str, **extra) -> dict:
    return {
        "name": name,
        "entrypoint": "python main.py",
        "tarball_path": "https://example.com/automation.tar.gz",
        "trigger": {"type": "cron", "schedule": "0 9 * * 1", "timezone": "UTC"},
        **extra,
    }


def _prompt_body(name: str, **extra) -> dict:
    return {
        "name": name,
        "prompt": "Review open pull requests.",
        "trigger": {"type": "cron", "schedule": "0 9 * * 1", "timezone": "UTC"},
        **extra,
    }


def _error_codes(response) -> list[tuple[str | None, str]]:
    detail = response.json()["detail"]
    return [(item["loc"][-1], item["type"]) for item in detail]


def _configure_kinds(monkeypatch, kinds: list[str] | None) -> None:
    """Point AUTOMATION_TARGETS at one target with these kinds."""
    entry = {**TARGET_ENTRY}
    if kinds is not None:
        entry["kinds"] = kinds
    monkeypatch.setenv("AUTOMATION_TARGETS", json.dumps([entry]))
    clear_config_cache()


# --- Kind classification ---


def test_classify_prompt_preset():
    assert classify_automation({"preset_type": "prompt"}, None, None, None) == "prompt"


def test_classify_plugin_preset_and_plugin_spec():
    assert classify_automation({"preset_type": "plugin"}, None, None, None) == "plugin"
    assert (
        classify_automation(
            {"preset_type": "plugin_spec"},
            {"plugin": {}},
            "plugin-spec://runtime",
            None,
        )
        == "plugin"
    )


def test_classify_script_bundle_from_template_provenance():
    preset = {"template": {"id": "github-pr-reviewer", "version": "1.0.0"}}
    assert (
        classify_automation(preset, None, "oh-internal://uploads/x", "python3 main.py")
        == "script"
    )


def test_classify_external_url_is_a_custom_tarball():
    assert (
        classify_automation(
            None, None, "https://example.com/bundle.tar.gz", "bash run.sh"
        )
        == "custom_tarball"
    )


# --- Registry parsing ---


def test_unset_or_empty_targets_means_no_targets():
    assert parse_targets("") == {}
    assert parse_targets("   ") == {}
    assert parse_targets("[]") == {}


def test_origin_defaults_to_the_url_origin():
    [target] = parse_targets(json.dumps([TARGET_ENTRY])).values()
    assert target.origin == "https://governed.example:8443"


def test_explicit_origin_is_kept():
    entry = {**TARGET_ENTRY, "origin": "https://public.example", "capability": "c1"}
    [target] = parse_targets(json.dumps([entry])).values()
    assert target.origin == "https://public.example"
    assert target.capability == "c1"


def test_absent_kinds_defaults_to_prompt():
    [target] = parse_targets(json.dumps([TARGET_ENTRY])).values()
    assert target.kinds is None
    assert target.allowed_kinds == ("prompt",)


def test_explicit_kinds_are_kept():
    entry = {**TARGET_ENTRY, "kinds": ["plugin", "prompt", "plugin"]}
    [target] = parse_targets(json.dumps([entry])).values()
    assert target.allowed_kinds == ("plugin", "prompt")


def test_empty_kinds_runs_no_automation():
    [target] = parse_targets(json.dumps([{**TARGET_ENTRY, "kinds": []}])).values()
    assert target.allowed_kinds == ()


def test_unknown_kind_is_rejected_by_name():
    with pytest.raises(TargetConfigError, match="kinds must be a list"):
        parse_targets(json.dumps([{**TARGET_ENTRY, "kinds": ["prompt", "host_bash"]}]))


def test_kinds_must_be_a_list():
    with pytest.raises(TargetConfigError, match="kinds must be a list"):
        parse_targets(json.dumps([{**TARGET_ENTRY, "kinds": "prompt"}]))


def test_unknown_entry_key_is_rejected_by_name():
    with pytest.raises(TargetConfigError, match="unknown key.*'bogus'"):
        parse_targets(json.dumps([{**TARGET_ENTRY, "bogus": 1}]))


def test_unknown_entry_key_fails_startup(monkeypatch):
    monkeypatch.setenv("AUTOMATION_TARGETS", json.dumps([{**TARGET_ENTRY, "bogus": 1}]))
    clear_config_cache()
    from openhands.automation.config import get_config

    with pytest.raises(ValueError, match="unknown key"):
        get_config().service


def test_missing_required_key_is_rejected():
    with pytest.raises(TargetConfigError, match="missing key"):
        parse_targets(json.dumps([{"id": "x", "name": "x"}]))


# --- Capabilities ---


class TestCapabilities:
    async def test_targets_list_and_feature_without_secrets(
        self, async_client, ready_deployment, targets_env
    ):
        response = await async_client.get(f"{BASE}/capabilities")
        assert response.status_code == 200
        body = response.json()
        assert body["targets"] == [
            {
                "id": TARGET_ID,
                "name": TARGET_NAME,
                "origin": "https://governed.example:8443",
                "kinds": ["prompt"],
            }
        ]
        assert "governedTargets" in body["features"]
        assert TARGET_SECRET not in response.text

    async def test_capabilities_reports_the_configured_kinds(
        self, async_client, ready_deployment, targets_env, monkeypatch
    ):
        _configure_kinds(monkeypatch, ["prompt", "script", "custom_tarball"])
        response = await async_client.get(f"{BASE}/capabilities")
        [target] = response.json()["targets"]
        assert target["kinds"] == ["prompt", "script", "custom_tarball"]

    async def test_no_targets_means_no_feature(
        self, async_client, ready_deployment, monkeypatch
    ):
        monkeypatch.delenv("AUTOMATION_TARGETS", raising=False)
        clear_config_cache()
        response = await async_client.get(f"{BASE}/capabilities")
        body = response.json()
        assert body["targets"] == []
        assert "governedTargets" not in body["features"]


# --- Create and update validation ---


class TestCreateValidation:
    async def test_null_target_create_keeps_default_behavior(
        self, async_client, targets_env
    ):
        response = await async_client.post(f"{BASE}", json=_raw_body("Legacy"))
        assert response.status_code == 201
        body = response.json()
        assert body["target_id"] is None
        assert body["target_name"] is None

    async def test_custom_tarball_automation_on_target_refused_at_create(
        self, async_client, targets_env
    ):
        response = await async_client.post(
            f"{BASE}", json=_raw_body("Upload", target_id=TARGET_ID)
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_kind_unsupported")]
        message = response.json()["detail"][0]["msg"]
        assert "'custom_tarball'" in message
        assert "prompt" in message

    async def test_script_automation_on_target_refused_at_create(
        self, async_client, targets_env
    ):
        response = await async_client.post(
            f"{BASE}",
            json=_raw_body(
                "Script",
                target_id=TARGET_ID,
                template={"id": "bundle", "version": "1.0.0"},
            ),
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_kind_unsupported")]
        assert "'script'" in response.json()["detail"][0]["msg"]

    async def test_unknown_target_refused_at_create(self, async_client, targets_env):
        response = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id="nope")
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_unavailable")]

    async def test_plugin_automation_on_target_refused_at_create(
        self, async_client, targets_env
    ):
        response = await async_client.post(
            f"{BASE}/preset/plugin",
            json={
                **_prompt_body("Plugin", target_id=TARGET_ID),
                "plugins": [{"source": "github:owner/repo"}],
            },
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_kind_unsupported")]
        assert "'plugin'" in response.json()["detail"][0]["msg"]

    async def test_a_target_runs_each_kind_it_lists(
        self, async_client, targets_env, monkeypatch
    ):
        _configure_kinds(monkeypatch, ["script", "custom_tarball", "plugin"])
        script = await async_client.post(
            f"{BASE}",
            json=_raw_body(
                "Script",
                target_id=TARGET_ID,
                template={"id": "bundle", "version": "1.0.0"},
            ),
        )
        assert script.status_code == 201, script.text
        upload = await async_client.post(
            f"{BASE}", json=_raw_body("Upload", target_id=TARGET_ID)
        )
        assert upload.status_code == 201, upload.text
        plugin = await async_client.post(
            f"{BASE}/preset/plugin",
            json={
                **_prompt_body("Plugin", target_id=TARGET_ID),
                "plugins": [{"source": "github:owner/repo"}],
            },
        )
        assert plugin.status_code == 201, plugin.text

    async def test_prompt_refused_when_the_target_lists_only_plugin(
        self, async_client, targets_env, monkeypatch
    ):
        _configure_kinds(monkeypatch, ["plugin"])
        response = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_kind_unsupported")]

    async def test_empty_kinds_refuses_every_kind_at_create(
        self, async_client, targets_env, monkeypatch
    ):
        _configure_kinds(monkeypatch, [])
        response = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_kind_unsupported")]
        assert response.json()["detail"][0]["msg"] == (
            f"Target {TARGET_NAME!r} does not run 'prompt' automations; it runs none."
        )

    async def test_prompt_preset_on_target_accepted_at_create(
        self, async_client, targets_env
    ):
        response = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["target_id"] == TARGET_ID
        assert body["target_name"] == TARGET_NAME
        assert TARGET_SECRET not in response.text


class TestUpdateValidation:
    async def _create_raw(self, client: AsyncClient, name: str = "Script") -> dict:
        response = await client.post(f"{BASE}", json=_raw_body(name))
        assert response.status_code == 201, response.text
        return response.json()

    async def test_target_on_custom_tarball_automation_refused_at_update(
        self, async_client, targets_env
    ):
        auto = await self._create_raw(async_client)
        response = await async_client.patch(
            f"{BASE}/{auto['id']}", json={"target_id": TARGET_ID}
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_kind_unsupported")]

    async def test_custom_tarball_accepted_when_the_target_lists_it(
        self, async_client, targets_env, monkeypatch
    ):
        auto = await self._create_raw(async_client)
        _configure_kinds(monkeypatch, ["custom_tarball"])
        response = await async_client.patch(
            f"{BASE}/{auto['id']}", json={"target_id": TARGET_ID}
        )
        assert response.status_code == 200, response.text
        assert response.json()["target_id"] == TARGET_ID

    async def test_unknown_target_refused_at_update(self, async_client, targets_env):
        created = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        auto = created.json()
        response = await async_client.patch(
            f"{BASE}/{auto['id']}", json={"target_id": "gone"}
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_unavailable")]

    async def test_live_run_target_change_refused(
        self, async_client, targets_env, monkeypatch
    ):
        created = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        auto = created.json()
        dispatched = await async_client.post(f"{BASE}/{auto['id']}/dispatch")
        assert dispatched.status_code == 201, dispatched.text

        monkeypatch.setenv(
            "AUTOMATION_TARGETS",
            json.dumps([{**TARGET_ENTRY, "id": "governed-2", "name": "Second"}]),
        )
        clear_config_cache()
        refused = await async_client.patch(
            f"{BASE}/{auto['id']}", json={"target_id": "governed-2"}
        )
        assert refused.status_code == 409

        run_id = dispatched.json()["id"]
        cancelled = await async_client.post(f"{BASE}/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        changed = await async_client.patch(
            f"{BASE}/{auto['id']}", json={"target_id": "governed-2"}
        )
        assert changed.status_code == 200, changed.text

    async def test_run_keeps_its_target_snapshot(
        self, async_client, targets_env, monkeypatch
    ):
        created = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        auto = created.json()
        dispatched = await async_client.post(f"{BASE}/{auto['id']}/dispatch")
        run = dispatched.json()
        assert run["target_id"] == TARGET_ID
        assert run["target_name"] == TARGET_NAME

        # End the run, then move the automation to another target: the run
        # keeps the snapshot it was created with.
        await async_client.post(f"{BASE}/runs/{run['id']}/cancel")
        monkeypatch.setenv(
            "AUTOMATION_TARGETS",
            json.dumps([{**TARGET_ENTRY, "id": "governed-2", "name": "Second"}]),
        )
        clear_config_cache()
        changed = await async_client.patch(
            f"{BASE}/{auto['id']}", json={"target_id": "governed-2"}
        )
        assert changed.status_code == 200, changed.text
        listing = await async_client.get(f"{BASE}/{auto['id']}/runs")
        [stored] = listing.json()["runs"]
        assert stored["target_id"] == TARGET_ID
        assert stored["target_name"] == TARGET_NAME

    async def test_enable_with_removed_target_refused(
        self, async_client, targets_env, monkeypatch
    ):
        created = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        auto = created.json()
        monkeypatch.setenv("AUTOMATION_TARGETS", "[]")
        clear_config_cache()
        response = await async_client.patch(
            f"{BASE}/{auto['id']}", json={"state": "ACTIVE"}
        )
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_unavailable")]


class TestDispatchValidation:
    async def test_removed_target_fails_at_manual_dispatch(
        self, async_client, targets_env, monkeypatch
    ):
        created = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        auto = created.json()
        monkeypatch.setenv("AUTOMATION_TARGETS", "[]")
        clear_config_cache()
        response = await async_client.post(f"{BASE}/{auto['id']}/dispatch")
        assert response.status_code == 422
        assert _error_codes(response) == [("target_id", "target_unavailable")]

    async def test_one_in_flight_manual_run_per_target(self, async_client, targets_env):
        created = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        auto = created.json()
        first = await async_client.post(f"{BASE}/{auto['id']}/dispatch")
        assert first.status_code == 201, first.text
        second = await async_client.post(f"{BASE}/{auto['id']}/dispatch")
        assert second.status_code == 409

    async def test_no_secret_in_the_run_record(self, async_client, targets_env):
        created = await async_client.post(
            f"{BASE}/preset/prompt", json=_prompt_body("Prompt", target_id=TARGET_ID)
        )
        auto = created.json()
        dispatched = await async_client.post(f"{BASE}/{auto['id']}/dispatch")
        assert dispatched.status_code == 201, dispatched.text
        listing = await async_client.get(f"{BASE}/{auto['id']}/runs")
        assert listing.status_code == 200
        assert TARGET_SECRET not in listing.text
