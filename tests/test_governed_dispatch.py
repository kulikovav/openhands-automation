"""Governed dispatch: one conversation per run, reconciled by identity."""

import base64
import json
import logging
import uuid
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import httpx
import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from openhands.automation.backends.governed import (
    GOVERNED_START_GRACE_SECONDS,
    ConversationGoneError,
    ExecutionClaim,
    GovernedConversationBackend,
    TranscriptOutcome,
    cancel_governed_conversation,
    claim_execution,
    governed_conversation_id_for,
    release_conversation,
)
from openhands.automation.config import ServiceSettings, clear_config_cache
from openhands.automation.dispatcher import _execute_run, _requeue_transient_failure
from openhands.automation.models import (
    Automation,
    AutomationRun,
    AutomationRunStatus,
    TarballUpload,
    UploadStatus,
)
from openhands.automation.plugin_specs import (
    AutomationPluginSource,
    PluginSpecError,
    preflight_manifest,
)
from openhands.automation.targets import resolve_target
from openhands.automation.utils.tarball_validation import build_internal_url
from openhands.automation.utils.time import utcnow
from openhands.automation.watchdog import reconcile_governed_runs
from tests.private_git_server import PrivateGitServer, SecretStoreServer, make_bare_repo


TEST_USER_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
TEST_ORG_ID = uuid.UUID("87654321-4321-8765-4321-876543218765")
TARGET_ID = "governed-1"
TARGET_URL = "https://governed.example:8443"
TARGET_SECRET = "sk-target-secret-do-not-leak"
PROFILE_ID = "0a1b2c3d-0000-4000-8000-000000000042"
TARGET_ENTRY = {
    "id": TARGET_ID,
    "name": "Governed Sandbox",
    "url": TARGET_URL,
    "api_key": TARGET_SECRET,
    "agent_profile_id": PROFILE_ID,
    # Every governed dispatch test exercises one kind or another; the kind
    # gate itself is tested with an explicit prompt-only entry.
    "kinds": ["prompt", "script", "plugin", "custom_tarball"],
}

# Distinguishes "no override" from a scripted ``null`` JSON body.
_UNSET = object()


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Configuration is cached, so each test reads it fresh and leaves it clean."""
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def targets_env(monkeypatch):
    monkeypatch.setenv("AUTOMATION_TARGETS", json.dumps([TARGET_ENTRY]))
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def tarball_store(monkeypatch):
    """The file store the dispatcher reads an internal upload from."""
    store = MagicMock()
    monkeypatch.setattr("openhands.automation.storage.get_file_store", lambda: store)
    return store


async def store_upload(session_factory, data: bytes) -> str:
    """Record one completed internal upload and return its tarball_path."""
    upload_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(
            TarballUpload(
                id=upload_id,
                user_id=TEST_USER_ID,
                org_id=TEST_ORG_ID,
                name="bundle",
                status=UploadStatus.COMPLETED,
                storage_path=f"uploads/{upload_id}.tar.gz",
            )
        )
        await session.commit()
    return build_internal_url(upload_id)


class FakeGovernedServer:
    """The governed agent server, as the httpx transport sees it.

    Honors idempotent_conversation_create_v1: a repeat create with a known
    conversation id answers 200 and does not start a second conversation.
    """

    def __init__(self, session_factory, run_id=None):
        self.session_factory = session_factory
        self.run_id = run_id
        self.requests: list[tuple[str, str]] = []
        self.claim_at_post: list[dict | None] = []
        self.created: list[str] = []
        self.conversations: set[str] = set()
        self.deleted: list[str] = []
        self.released: list[str] = []
        self.release_status = 204
        self.delete_status = 200
        self.records: dict[str, tuple[int, dict]] = {}
        self.create_acks: list[int | str] = []
        self.create_error_body: object = _UNSET
        self.fail_create_with_read_timeout = False
        self.secrets: list[dict] = []
        self.secret_status = 200
        self.secret_queries: list[dict] = []
        self.secret_body: object = _UNSET
        self.capabilities: list[str] = ["conversation_repositories_v1"]
        self.payloads: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def posts(self, path: str) -> int:
        return sum(1 for m, p in self.requests if m == "POST" and p == path)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.requests.append((method, path))

        if method == "GET" and path == "/api/settings/secrets":
            self.secret_queries.append(dict(request.url.params))
            if self.secret_status >= 400:
                return httpx.Response(self.secret_status, json={"detail": "refused"})
            if self.secret_body is not _UNSET:
                return httpx.Response(200, json=self.secret_body)
            return httpx.Response(200, json={"secrets": self.secrets})

        if method == "GET" and path == "/server_info":
            return httpx.Response(200, json={"capabilities": self.capabilities})

        if method == "POST" and path == "/api/conversations":
            payload = json.loads(request.content)
            self.payloads.append(payload)
            conversation_id = payload["conversation_id"]
            self.created.append(conversation_id)
            async with self.session_factory() as session:
                claim = None
                if self.run_id is not None:
                    claim = await session.scalar(
                        select(AutomationRun.execution_claim).where(
                            AutomationRun.id == self.run_id
                        )
                    )
            self.claim_at_post.append(claim)
            if self.fail_create_with_read_timeout:
                raise httpx.ReadTimeout("connection dropped", request=request)
            replay = conversation_id in self.conversations
            self.conversations.add(conversation_id)
            if self.create_acks:
                ack = self.create_acks.pop(0)
                if ack == "mismatch":
                    return httpx.Response(201, json={"id": str(uuid.uuid4())})
                body = (
                    {"id": conversation_id}
                    if ack in (200, 201)
                    else (
                        self.create_error_body
                        if self.create_error_body is not _UNSET
                        else {"detail": "refused"}
                    )
                )
                return httpx.Response(ack, json=body)
            return httpx.Response(200 if replay else 201, json={"id": conversation_id})

        if method == "GET" and path.startswith("/api/conversations/"):
            conversation_id = path.rsplit("/", 1)[-1]
            if path.endswith("/runtime"):
                return httpx.Response(
                    200, json={"runtime_status": "available", "can_resume": True}
                )
            status_code, body = self.records.get(
                conversation_id,
                (200, {"id": conversation_id, "execution_status": "running"}),
            )
            return httpx.Response(status_code, json=body)

        if method == "POST" and path.endswith("/interrupt"):
            return httpx.Response(200, json={})

        if method == "DELETE" and path.endswith("/runtime"):
            conversation_id = path.split("/")[-2]
            self.released.append(conversation_id)
            return httpx.Response(self.release_status)

        if method == "DELETE" and path.startswith("/api/conversations/"):
            conversation_id = path.rsplit("/", 1)[-1]
            self.deleted.append(conversation_id)
            if self.delete_status < 400:
                self.conversations.discard(conversation_id)
            return httpx.Response(self.delete_status)

        return httpx.Response(404, json={"detail": "unexpected"})


async def make_run(
    session_factory,
    *,
    target_id: str | None = TARGET_ID,
    retain_transcript: bool = True,
    timeout_at=None,
    repos: list[Any] | None = None,
    preset_metadata: dict | None = _UNSET,
    plugin_spec: dict | None = None,
    tarball_path: str = "https://example.com/automation.tar.gz",
    entrypoint: str = "python main.py",
    setup_script_path: str | None = None,
):
    async with session_factory() as session:
        if preset_metadata is _UNSET:
            metadata: dict | None = {"preset_type": "prompt"}
            if repos is not None:
                metadata["repos"] = repos
        else:
            metadata = preset_metadata
        automation = Automation(
            id=uuid.uuid4(),
            user_id=TEST_USER_ID,
            created_by_user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            name="Governed",
            preset_metadata=metadata,
            plugin_spec=plugin_spec,
            prompt="Do the thing",
            trigger={"type": "cron", "schedule": "0 9 * * 1", "timezone": "UTC"},
            tarball_path=tarball_path,
            setup_script_path=setup_script_path,
            entrypoint=entrypoint,
            target_id=target_id,
            retain_transcript=retain_transcript,
        )
        run = AutomationRun(
            id=uuid.uuid4(),
            automation=automation,
            status=AutomationRunStatus.RUNNING,
            started_at=utcnow(),
            timeout_at=timeout_at or (utcnow() + timedelta(hours=1)),
            target_id=target_id,
            target_name="Governed Sandbox" if target_id else None,
        )
        session.add(run)
        await session.commit()
        return run


async def reload_run(session_factory, run_id) -> AutomationRun:
    async with session_factory() as session:
        return (
            (
                await session.execute(
                    select(AutomationRun)
                    .where(AutomationRun.id == run_id)
                    .options(selectinload(AutomationRun.automation))
                )
            )
            .scalars()
            .one()
        )


async def dispatch(run, session_factory, client):
    await _execute_run(run, ServiceSettings(), session_factory, client)


class TestDispatchOnce:
    async def test_one_dispatch_yields_one_post_and_one_conversation(
        self, async_session_factory, targets_env
    ):
        run = await make_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        expected = governed_conversation_id_for(run.id)
        assert server.created == [expected]
        assert server.posts("/api/conversations") == 1
        # The claim is committed before the POST: it is in the row at POST time.
        [claim] = server.claim_at_post
        assert claim is not None
        assert claim["conversation_id"] == expected
        assert claim["target_id"] == TARGET_ID
        stored = await reload_run(async_session_factory, run.id)
        assert stored.conversation_id == expected
        assert stored.execution_claim["conversation_id"] == expected

    async def test_duplicate_call_does_not_resend(
        self, async_session_factory, targets_env
    ):
        run = await make_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)
            second = await reload_run(async_session_factory, run.id)
            await dispatch(second, async_session_factory, client)

        assert server.posts("/api/conversations") == 1
        assert len(server.conversations) == 1

    async def test_ambiguous_ack_leaves_the_claim(
        self, async_session_factory, targets_env
    ):
        run = await make_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.fail_create_with_read_timeout = True

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        claim = ExecutionClaim.from_dict(stored.execution_claim)
        assert claim is not None
        assert stored.conversation_id is None
        first_claim = dict(stored.execution_claim)

        # A retry reuses the claim's conversation id; the idempotent create
        # starts no second conversation.
        server.fail_create_with_read_timeout = False
        async with server.client() as client:
            second = await reload_run(async_session_factory, run.id)
            await dispatch(second, async_session_factory, client)

        expected = governed_conversation_id_for(run.id)
        assert server.created == [expected, expected]
        assert len(server.conversations) == 1
        stored = await reload_run(async_session_factory, run.id)
        assert dict(stored.execution_claim) == first_claim

    async def test_removed_target_fails_at_dispatch_without_calls(
        self, async_session_factory, targets_env
    ):
        run = await make_run(async_session_factory, target_id="gone")
        server = FakeGovernedServer(async_session_factory, run.id)

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "target_unavailable"
        assert server.requests == []

    async def test_a_disallowed_kind_is_refused_at_dispatch_without_calls(
        self, async_session_factory, targets_env, monkeypatch
    ):
        """A target that runs prompts only refuses a triggered script run."""
        monkeypatch.setenv(
            "AUTOMATION_TARGETS",
            json.dumps([{**TARGET_ENTRY, "kinds": ["prompt"]}]),
        )
        clear_config_cache()
        run = await make_run(
            async_session_factory,
            preset_metadata={"template": {"id": "bundle", "version": "1.0.0"}},
            tarball_path="https://example.com/automation.tar.gz",
        )
        server = FakeGovernedServer(async_session_factory, run.id)

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "target_kind_unsupported"
        assert stored.execution_claim is None
        assert server.requests == []


class TestStartFailureDetail:
    """A refused start records the field names, never the target's body."""

    async def test_a_4xx_body_is_not_copied_into_the_run(
        self, async_session_factory, targets_env
    ):
        marker = "TARBALLMARKERNOTBASE64"
        run = await make_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.create_acks = [422]
        server.create_error_body = {
            "detail": [
                {
                    "loc": ["body", "bundle", "tarball_b64"],
                    "msg": f"The archive {marker} is not valid base64",
                    "type": "value_error",
                    "input": marker,
                }
            ]
        }

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "conversation_error"
        assert "bundle.tarball_b64" in (stored.error_detail or "")
        assert marker not in (stored.error_detail or "")
        assert marker not in json.dumps(stored.status_detail)


class TestConfiguredSecretsAndRepositories:
    """A governed start carries the target's secrets and its pre-clone repos."""

    async def test_the_start_carries_the_configured_secrets(
        self, async_session_factory, targets_env
    ):
        """The same LookupSecret references a direct conversation sends."""
        run = await make_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [
            {"name": "GITHUB_TOKEN", "description": "GitHub"},
            {"name": "GITLAB_TOKEN", "description": None},
        ]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        secrets = payload["secrets"]
        assert set(secrets) == {"GITHUB_TOKEN", "GITLAB_TOKEN"}
        assert secrets["GITHUB_TOKEN"]["kind"] == "LookupSecret"
        assert secrets["GITHUB_TOKEN"]["description"] == "GitHub"
        # The same auth header Canvas sends with each lookup, so an
        # out-of-process resolver can read the name.
        assert secrets["GITHUB_TOKEN"]["headers"] == {
            "X-Session-API-Key": TARGET_SECRET
        }
        assert urlsplit(secrets["GITHUB_TOKEN"]["url"]).path == (
            "/api/settings/secrets/GITHUB_TOKEN"
        )
        assert secrets["GITLAB_TOKEN"]["kind"] == "LookupSecret"
        assert urlsplit(secrets["GITLAB_TOKEN"]["url"]).path == (
            "/api/settings/secrets/GITLAB_TOKEN"
        )
        assert "description" not in secrets["GITLAB_TOKEN"]
        # The list is scoped to the target's agent profile, as the preset
        # runner's get_secrets(agent_profile_id=...) is.
        assert server.secret_queries == [{"agent_profile_id": PROFILE_ID}]

    async def test_the_start_preclones_a_repository_with_its_integration_secret(
        self, async_session_factory, targets_env
    ):
        """A repository whose provider token is configured is cloned at start."""
        repos = [
            {"url": "https://github.com/acme/api", "ref": "main"},
            {"url": "https://gitlab.com/acme/web"},
        ]
        run = await make_run(async_session_factory, repos=repos)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [{"name": "GITHUB_TOKEN", "description": None}]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["repositories"] == [
            {"url": "https://github.com/acme/api", "ref": "main"}
        ]

    async def test_a_repository_without_its_integration_secret_is_not_precloned(
        self, async_session_factory, targets_env
    ):
        """The same condition the app server applies: no connection, no clone."""
        run = await make_run(
            async_session_factory,
            repos=[{"url": "https://github.com/acme/api"}],
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [{"name": "GITLAB_TOKEN", "description": None}]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert "repositories" not in payload

    async def test_a_commit_ref_is_precloned(self, async_session_factory, targets_env):
        """The server clones a commit ref with a full clone plus a checkout."""
        commit = "0123456789abcdef0123456789abcdef01234567"
        run = await make_run(
            async_session_factory,
            repos=[{"url": "https://github.com/acme/api", "ref": commit}],
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [{"name": "GITHUB_TOKEN", "description": None}]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["repositories"] == [
            {"url": "https://github.com/acme/api", "ref": commit}
        ]

    async def test_a_string_form_repository_is_precloned(
        self, async_session_factory, targets_env
    ):
        """A bare URL is a repository form the server validates and clones."""
        run = await make_run(
            async_session_factory, repos=["https://github.com/acme/api"]
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [{"name": "GITHUB_TOKEN", "description": None}]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["repositories"] == [{"url": "https://github.com/acme/api"}]

    async def test_a_lowercase_token_name_is_precloned(
        self, async_session_factory, targets_env
    ):
        """The server's fetcher resolves either spelling of the provider token."""
        run = await make_run(
            async_session_factory,
            repos=[{"url": "https://github.com/acme/api"}],
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [{"name": "github_token", "description": None}]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["repositories"] == [{"url": "https://github.com/acme/api"}]

    async def test_ssh_and_file_urls_are_not_precloned(
        self, async_session_factory, targets_env
    ):
        """Neither URL form can carry a provider token, and no runner exists."""
        run = await make_run(
            async_session_factory,
            repos=[
                {"url": "git@github.com:acme/api.git"},
                {"url": "file:///tmp/acme/api"},
            ],
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [{"name": "GITHUB_TOKEN", "description": None}]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert "repositories" not in payload

    async def test_a_target_without_the_clone_capability_fails_the_run(
        self, async_session_factory, targets_env
    ):
        """A repository a pre-capability server would ignore is a visible failure."""
        run = await make_run(
            async_session_factory,
            repos=[{"url": "https://github.com/acme/api"}],
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secrets = [{"name": "GITHUB_TOKEN", "description": None}]
        server.capabilities = []

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "repositories_unsupported"
        assert stored.execution_claim is None
        assert server.posts("/api/conversations") == 0

    async def test_a_null_secret_list_starts_the_run_without_secrets(
        self, async_session_factory, targets_env
    ):
        """A 200 body that is not an object carries no secrets, not an error."""
        run = await make_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secret_body = None

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert "secrets" not in payload
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status != AutomationRunStatus.FAILED

    async def test_an_unreadable_secret_list_does_not_block_the_start(
        self, async_session_factory, targets_env
    ):
        """Secrets are best-effort, as in a preset run without them."""
        run = await make_run(
            async_session_factory,
            repos=[{"url": "https://github.com/acme/api"}],
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.secret_status = 503

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["conversation_id"] == governed_conversation_id_for(run.id)
        assert "secrets" not in payload
        assert "repositories" not in payload


BUNDLE_BYTES = b"not-a-real-tarball"


class TestGovernedBundles:
    """Script and custom-tarball runs ride one bundle in the start payload."""

    async def test_a_script_run_carries_one_bundle(
        self, async_session_factory, targets_env, tarball_store
    ):
        tarball_store.read.return_value = BUNDLE_BYTES
        tarball_path = await store_upload(async_session_factory, BUNDLE_BYTES)
        run = await make_run(
            async_session_factory,
            preset_metadata={
                "template": {
                    "id": "github-pr-reviewer",
                    "version": "1.0.0",
                    "config": {"label": "nightly"},
                }
            },
            tarball_path=tarball_path,
            entrypoint="python3 main.py",
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["automation_bundle_v1"]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["bundle"] == {
            "entrypoint": "python3 main.py",
            "setup_script_path": None,
            "config": {"label": "nightly"},
            "tarball_b64": base64.b64encode(BUNDLE_BYTES).decode("ascii"),
        }
        assert "plugins" not in payload
        stored = await reload_run(async_session_factory, run.id)
        assert stored.conversation_id == governed_conversation_id_for(run.id)

    async def test_a_custom_tarball_run_carries_one_bundle(
        self, async_session_factory, targets_env, tarball_store
    ):
        tarball_store.read.return_value = BUNDLE_BYTES
        tarball_path = await store_upload(async_session_factory, BUNDLE_BYTES)
        run = await make_run(
            async_session_factory,
            preset_metadata=None,
            tarball_path=tarball_path,
            entrypoint="bash run.sh",
            setup_script_path="setup.sh",
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["automation_bundle_v1"]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["bundle"] == {
            "entrypoint": "bash run.sh",
            "setup_script_path": "setup.sh",
            "config": {},
            "tarball_b64": base64.b64encode(BUNDLE_BYTES).decode("ascii"),
        }

    async def test_a_target_without_the_bundle_capability_fails_closed(
        self, async_session_factory, targets_env, tarball_store
    ):
        tarball_store.read.return_value = BUNDLE_BYTES
        tarball_path = await store_upload(async_session_factory, BUNDLE_BYTES)
        run = await make_run(
            async_session_factory,
            preset_metadata={"template": {"id": "bundle", "version": "1.0.0"}},
            tarball_path=tarball_path,
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = []

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "bundle_unsupported"
        assert stored.status_detail["transient"] is False
        assert stored.execution_claim is None
        assert server.posts("/api/conversations") == 0

    async def test_a_url_tarball_is_refused_before_the_claim(
        self, async_session_factory, targets_env
    ):
        run = await make_run(
            async_session_factory,
            preset_metadata={"template": {"id": "bundle", "version": "1.0.0"}},
            tarball_path="https://example.com/bundle.tar.gz",
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["automation_bundle_v1"]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "bundle_source_unsupported"
        assert stored.execution_claim is None
        assert server.posts("/api/conversations") == 0

    async def test_a_missing_upload_fails_before_the_claim(
        self, async_session_factory, targets_env
    ):
        run = await make_run(
            async_session_factory,
            preset_metadata={"template": {"id": "bundle", "version": "1.0.0"}},
            tarball_path=build_internal_url(uuid.uuid4()),
        )
        server = FakeGovernedServer(async_session_factory, run.id)

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "TarballNotFoundError"
        assert stored.execution_claim is None
        assert server.posts("/api/conversations") == 0

    async def test_bundle_bytes_never_reach_a_log_or_an_error(
        self, async_session_factory, targets_env, tarball_store, caplog
    ):
        caplog.set_level(logging.INFO)
        marker = "qa-bundle-bytes-must-not-leak"
        data = b"not-a-tarball" + marker.encode()
        tarball_store.read.return_value = data
        tarball_path = await store_upload(async_session_factory, data)
        run = await make_run(
            async_session_factory,
            preset_metadata={"template": {"id": "bundle", "version": "1.0.0"}},
            tarball_path=tarball_path,
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = []

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert marker not in caplog.text
        assert marker not in (stored.error_detail or "")
        assert marker not in json.dumps(stored.status_detail or {})

    async def test_a_duplicate_dispatch_does_not_reprepare_the_bundle(
        self, async_session_factory, targets_env, tarball_store
    ):
        tarball_store.read.return_value = BUNDLE_BYTES
        tarball_path = await store_upload(async_session_factory, BUNDLE_BYTES)
        run = await make_run(
            async_session_factory,
            preset_metadata={"template": {"id": "bundle", "version": "1.0.0"}},
            tarball_path=tarball_path,
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["automation_bundle_v1"]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)
            second = await reload_run(async_session_factory, run.id)
            await dispatch(second, async_session_factory, client)

        assert tarball_store.read.call_count == 1
        assert len(server.payloads) == 1


class TestGovernedPlugins:
    """A plugin run carries its sources and config in the plugins field."""

    async def test_a_plugin_preset_carries_its_sources(
        self, async_session_factory, targets_env
    ):
        plugins = [{"source": "github:owner/repo", "ref": "v1"}]
        run = await make_run(
            async_session_factory,
            preset_metadata={"preset_type": "plugin", "plugins": plugins},
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["conversation_plugins_v1"]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["plugins"] == plugins
        assert "bundle" not in payload

    async def test_a_plugin_spec_carries_its_sources_and_config(
        self, async_session_factory, targets_env
    ):
        plugin_spec = {
            "plugin": {"source": "github:owner/automation", "ref": "abc"},
            "config": {"mode": "fast"},
            "required_plugins": [
                {"source": "github:owner/required", "ref": "deadbeef"}
            ],
        }
        run = await make_run(
            async_session_factory,
            preset_metadata={"preset_type": "plugin_spec"},
            plugin_spec=plugin_spec,
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["conversation_plugins_v1"]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["plugins"] == [
            {
                "source": "github:owner/automation",
                "ref": "abc",
                "config": {"mode": "fast"},
            },
            {"source": "github:owner/required", "ref": "deadbeef"},
        ]

    async def test_plugin_variants_are_refused_before_the_claim(
        self, async_session_factory, targets_env
    ):
        run = await make_run(
            async_session_factory,
            preset_metadata={"preset_type": "plugin", "prompt": "hi"},
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["conversation_plugins_v1"]

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "plugin_variants_unsupported"
        assert stored.execution_claim is None
        assert server.posts("/api/conversations") == 0

    async def test_a_target_without_the_plugin_capability_fails_closed(
        self, async_session_factory, targets_env
    ):
        run = await make_run(
            async_session_factory,
            preset_metadata={
                "preset_type": "plugin",
                "plugins": [{"source": "github:owner/repo"}],
            },
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = []

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "plugins_unsupported"
        assert stored.status_detail["transient"] is False
        assert stored.execution_claim is None
        assert server.posts("/api/conversations") == 0

    async def test_a_plugin_spec_resolves_the_manifest_plugins(
        self, async_session_factory, targets_env, monkeypatch
    ):
        """A spec without pinned plugins pins the refs stock runs resolve."""
        run = await make_run(
            async_session_factory,
            preset_metadata={"preset_type": "plugin_spec"},
            plugin_spec={"plugin": {"source": "github:owner/automation", "ref": "abc"}},
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["conversation_plugins_v1"]

        async def fake_resolve(spec, **kwargs):  # noqa: ARG001
            return (
                AutomationPluginSource.model_validate(spec["plugin"]),
                None,
                "deadbeef",
                [{"source": "github:owner/required", "resolved_ref": "cafebabe"}],
            )

        monkeypatch.setattr(
            "openhands.automation.backends.governed.resolve_plugin_spec", fake_resolve
        )

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["plugins"] == [
            {"source": "github:owner/automation", "ref": "deadbeef"},
            {"source": "github:owner/required", "ref": "cafebabe"},
        ]

    async def test_a_plugin_spec_that_cannot_resolve_its_plugins_fails_closed(
        self, async_session_factory, targets_env, monkeypatch
    ):
        run = await make_run(
            async_session_factory,
            preset_metadata={"preset_type": "plugin_spec"},
            plugin_spec={"plugin": {"source": "github:owner/automation", "ref": "abc"}},
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["conversation_plugins_v1"]

        async def fail_resolve(spec, **kwargs):  # noqa: ARG001
            raise PluginSpecError("manifest unreachable")

        monkeypatch.setattr(
            "openhands.automation.backends.governed.resolve_plugin_spec", fail_resolve
        )

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "plugin_spec_unresolved"
        assert stored.execution_claim is None
        assert server.posts("/api/conversations") == 0


class TestGovernedPluginSecrets:
    """A private plugin source resolves through the target's own store."""

    SESSION_KEY = "fake-session-key"
    PLUGIN_TOKEN = "fake-git-token-0123456789"

    async def test_the_references_name_the_targets_secret_store(
        self, async_session_factory, targets_env
    ):
        run = await make_run(async_session_factory)
        backend = GovernedConversationBackend(run)

        secrets = await backend.plugin_source_secrets(
            httpx.AsyncClient(), ["PLUGIN_TOKEN"]
        )

        reference = secrets.preflight["PLUGIN_TOKEN"]
        assert reference.url == f"{TARGET_URL}/api/settings/secrets/PLUGIN_TOKEN"
        assert reference.headers == {"X-Session-API-Key": TARGET_SECRET}
        # The target resolves its own references when it fetches the plugin.
        assert secrets.fetch == {}

    async def test_a_private_source_resolves_through_the_target_store(
        self, async_session_factory, monkeypatch, tmp_path
    ):
        sha = make_bare_repo(
            tmp_path,
            "plugin",
            {
                ".plugin/plugin.json": json.dumps(
                    {
                        "name": "private-plugin",
                        "version": "0.1.0",
                        "description": "A plugin in a private repository",
                        "automation": {
                            "schema_version": "openhands.automation.plugin.v1",
                            "entrypoint": {
                                "runner": "script",
                                "command": "bash run.sh",
                            },
                        },
                    }
                ),
                "run.sh": "echo private plugin\n",
            },
        )
        with (
            PrivateGitServer(tmp_path, self.PLUGIN_TOKEN) as git_server,
            SecretStoreServer(
                self.SESSION_KEY, {"PLUGIN_TOKEN": self.PLUGIN_TOKEN}
            ) as store,
        ):
            monkeypatch.setenv(
                "AUTOMATION_TARGETS",
                json.dumps(
                    [
                        {
                            **TARGET_ENTRY,
                            "url": store.url,
                            "api_key": self.SESSION_KEY,
                        }
                    ]
                ),
            )
            clear_config_cache()
            run = await make_run(async_session_factory)
            backend = GovernedConversationBackend(run)
            host = git_server.url.removeprefix("http://")

            secrets = await backend.plugin_source_secrets(
                httpx.AsyncClient(), ["PLUGIN_TOKEN"]
            )
            resolved_ref, manifest = preflight_manifest(
                AutomationPluginSource(
                    source=f"http://x-access-token:${{PLUGIN_TOKEN}}@{host}/plugin.git"
                ),
                secrets=secrets.preflight,
            )

        assert resolved_ref == sha
        assert manifest.name == "private-plugin"
        assert store.requests == ["/api/settings/secrets/PLUGIN_TOKEN"]

    async def test_a_plugin_spec_is_resolved_with_the_target_lookup(
        self, async_session_factory, targets_env, monkeypatch
    ):
        run = await make_run(
            async_session_factory,
            preset_metadata={"preset_type": "plugin_spec"},
            plugin_spec={
                "plugin": {
                    "source": "https://x:${PLUGIN_TOKEN}@private.example/plugin.git",
                    "ref": "abc",
                }
            },
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.capabilities = ["conversation_plugins_v1"]
        seen = {}

        async def fake_resolve(spec, **kwargs):
            seen["reference"] = (await kwargs["secrets"](["PLUGIN_TOKEN"]))[
                "PLUGIN_TOKEN"
            ]
            return (
                AutomationPluginSource.model_validate(spec["plugin"]),
                None,
                "deadbeef",
                [],
            )

        monkeypatch.setattr(
            "openhands.automation.backends.governed.resolve_plugin_spec", fake_resolve
        )

        async with server.client() as client:
            await dispatch(run, async_session_factory, client)

        [payload] = server.payloads
        assert payload["plugins"] == [
            {
                "source": "https://x:${PLUGIN_TOKEN}@private.example/plugin.git",
                "ref": "deadbeef",
            }
        ]
        reference = seen["reference"]
        assert reference.url == f"{TARGET_URL}/api/settings/secrets/PLUGIN_TOKEN"
        assert reference.headers == {"X-Session-API-Key": TARGET_SECRET}


class TestReconcileByState:
    async def _claimed_run(self, session_factory, **kwargs):
        run = await make_run(session_factory, **kwargs)
        target = resolve_target(TARGET_ID)
        assert target is not None
        claim = await claim_execution(session_factory, run, target)
        assert claim is not None
        return run, claim

    async def test_finished_maps_to_completed(self, async_session_factory, targets_env):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "finished"},
        )

        async with server.client() as client:
            marked = await reconcile_governed_runs(async_session_factory, client=client)

        assert marked == 1
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.COMPLETED
        # A terminal run stops its runtime and keeps the conversation.
        assert ("DELETE", f"/api/conversations/{claim.conversation_id}/runtime") in (
            server.requests
        )
        assert ("DELETE", f"/api/conversations/{claim.conversation_id}") not in (
            server.requests
        )
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "transcript_retained"
        assert claim.conversation_id in server.conversations

    async def test_finished_with_retain_off_deletes_the_conversation(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(
            async_session_factory, retain_transcript=False
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "finished"},
        )

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.COMPLETED
        assert server.released == []
        assert server.deleted == [claim.conversation_id]
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "transcript_unavailable"
        assert claim.conversation_id not in server.conversations

    async def test_failed_release_records_the_outcome_and_schedules_retry(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)
        server.release_status = 500
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "finished"},
        )

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.COMPLETED
        # A release that failed is recorded, not dropped.
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "release_failed"
        # The janitor retries: nothing else stops a governed sandbox.
        assert stored.sandbox_cleanup_due_at is not None
        assert claim.conversation_id in server.conversations

    async def test_failed_delete_does_not_claim_an_unavailable_transcript(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(
            async_session_factory, retain_transcript=False
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)
        server.delete_status = 500
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "finished"},
        )

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.COMPLETED
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "release_failed"
        assert claim.conversation_id in server.conversations

    async def test_error_maps_to_failed_with_detail(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "error"},
        )

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "conversation_error"
        # The failure code survives: retention only labels completed runs.
        assert (
            "DELETE",
            f"/api/conversations/{claim.conversation_id}/runtime",
        ) in server.requests

    async def test_410_maps_to_runtime_missing(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.records[claim.conversation_id] = (410, {})

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "runtime_missing"

    async def test_missing_record_during_start_grace_stays_running(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.records[claim.conversation_id] = (404, {})

        async with server.client() as client:
            marked = await reconcile_governed_runs(async_session_factory, client=client)

        assert marked == 0
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.RUNNING
        assert ("DELETE", f"/api/conversations/{claim.conversation_id}") not in (
            server.requests
        )

    async def test_missing_record_after_start_grace_fails(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        stale_claim = dict(run.execution_claim or {})
        stale_claim["claimed_at"] = (
            utcnow() - timedelta(seconds=GOVERNED_START_GRACE_SECONDS + 1)
        ).isoformat()
        async with async_session_factory() as session:
            await session.execute(
                update(AutomationRun)
                .where(AutomationRun.id == run.id)
                .values(execution_claim=stale_claim)
            )
            await session.commit()

        server = FakeGovernedServer(async_session_factory, run.id)
        server.records[claim.conversation_id] = (404, {})

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail["code"] == "conversation_error"

    async def test_retain_transcript_true_keeps_the_record(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(
            async_session_factory, retain_transcript=True
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "finished"},
        )

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.COMPLETED
        assert stored.status_detail is not None
        assert stored.status_detail["code"] == "transcript_retained"
        assert server.released == [claim.conversation_id]
        assert server.deleted == []
        assert claim.conversation_id in server.conversations

    async def test_timeout_interrupts_and_releases_the_runtime(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(
            async_session_factory,
            timeout_at=utcnow() - timedelta(minutes=1),
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)

        async with server.client() as client:
            marked = await reconcile_governed_runs(async_session_factory, client=client)

        assert marked == 1
        paths = [p for _, p in server.requests]
        assert f"/api/conversations/{claim.conversation_id}/interrupt" in paths
        assert f"/api/conversations/{claim.conversation_id}/runtime" in paths
        assert f"/api/conversations/{claim.conversation_id}" not in paths
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.error_detail.startswith("Timed out:")
        # Retention never replaces the timeout's own status code.
        assert stored.status_detail is not None
        assert stored.status_detail["kind"] == "timeout"
        assert "code" not in stored.status_detail

    async def test_timeout_with_retain_off_deletes_the_conversation(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(
            async_session_factory,
            retain_transcript=False,
            timeout_at=utcnow() - timedelta(minutes=1),
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        assert server.released == []
        assert server.deleted == [claim.conversation_id]
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.status_detail is not None
        assert stored.status_detail["kind"] == "timeout"
        assert "code" not in stored.status_detail
        assert claim.conversation_id not in server.conversations

    async def test_watchdog_reconciles_and_never_requeues(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "finished"},
        )

        # A claimed RUNNING run is never requeued.
        assert (
            await _requeue_transient_failure(async_session_factory, run, None) is False
        )

        async with server.client() as client:
            await reconcile_governed_runs(async_session_factory, client=client)

        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.COMPLETED
        # Nothing resurrects a claimed run into the queue afterwards either.
        await _requeue_transient_failure(async_session_factory, stored, None)
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status != AutomationRunStatus.PENDING

    async def test_cancel_interrupts_and_releases_a_retained_transcript(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)

        async with server.client() as client:
            outcome = await cancel_governed_conversation(
                client, run, retain_transcript=True
            )

        assert outcome is TranscriptOutcome.RETAINED
        paths = [p for _, p in server.requests]
        assert paths == [
            f"/api/conversations/{claim.conversation_id}/interrupt",
            f"/api/conversations/{claim.conversation_id}/runtime",
        ]
        assert claim.conversation_id in server.conversations

    async def test_cancel_with_retain_off_deletes_the_conversation(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(
            async_session_factory, retain_transcript=False
        )
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)

        async with server.client() as client:
            outcome = await cancel_governed_conversation(
                client, run, retain_transcript=False
            )

        assert outcome is TranscriptOutcome.UNAVAILABLE
        paths = [p for _, p in server.requests]
        assert paths == [
            f"/api/conversations/{claim.conversation_id}/interrupt",
            f"/api/conversations/{claim.conversation_id}",
        ]
        assert claim.conversation_id not in server.conversations


class _FakeReleaseBackend:
    """A governed backend whose runtime release answers a scripted result."""

    def __init__(self, results: list[bool]):
        self.results = results
        self.calls = 0

    async def release_runtime(self, client, conversation_id: str) -> bool:  # noqa: ARG002
        self.calls += 1
        return self.results.pop(0)


class TestReleaseConversation:
    async def test_failed_release_is_retried(self):
        backend = _FakeReleaseBackend([False, True])
        claim = ExecutionClaim(
            target_id=TARGET_ID,
            target_url=TARGET_URL,
            conversation_id=str(uuid.uuid4()),
            claimed_at=utcnow().isoformat(),
        )

        async with httpx.AsyncClient() as client:
            outcome = await release_conversation(
                client, backend, claim, retain_transcript=True
            )

        assert outcome is TranscriptOutcome.RETAINED
        assert backend.calls == 2

    async def test_release_fails_after_three_attempts(self):
        backend = _FakeReleaseBackend([False, False, False])
        claim = ExecutionClaim(
            target_id=TARGET_ID,
            target_url=TARGET_URL,
            conversation_id=str(uuid.uuid4()),
            claimed_at=utcnow().isoformat(),
        )

        async with httpx.AsyncClient() as client:
            outcome = await release_conversation(
                client, backend, claim, retain_transcript=True
            )

        assert outcome is TranscriptOutcome.RELEASE_FAILED
        assert backend.calls == 3

    async def test_gone_conversation_is_not_retried(self):
        class GoneBackend:
            def __init__(self):
                self.calls = 0

            async def release_runtime(self, client, conversation_id: str) -> bool:  # noqa: ARG002
                self.calls += 1
                raise ConversationGoneError(conversation_id)

        backend = GoneBackend()
        claim = ExecutionClaim(
            target_id=TARGET_ID,
            target_url=TARGET_URL,
            conversation_id=str(uuid.uuid4()),
            claimed_at=utcnow().isoformat(),
        )

        async with httpx.AsyncClient() as client:
            outcome = await release_conversation(
                client, backend, claim, retain_transcript=True
            )

        assert outcome is TranscriptOutcome.UNAVAILABLE
        assert backend.calls == 1
