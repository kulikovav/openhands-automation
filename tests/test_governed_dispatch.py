"""Governed dispatch: one conversation per run, reconciled by identity."""

import json
import uuid
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from openhands.automation.backends.governed import (
    ExecutionClaim,
    cancel_governed_conversation,
    claim_execution,
    governed_conversation_id_for,
)
from openhands.automation.config import ServiceSettings, clear_config_cache
from openhands.automation.dispatcher import _execute_run, _requeue_transient_failure
from openhands.automation.models import (
    Automation,
    AutomationRun,
    AutomationRunStatus,
)
from openhands.automation.targets import resolve_target
from openhands.automation.utils.time import utcnow
from openhands.automation.watchdog import reconcile_governed_runs


TEST_USER_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
TEST_ORG_ID = uuid.UUID("87654321-4321-8765-4321-876543218765")
TARGET_ID = "governed-1"
TARGET_SECRET = "sk-target-secret-do-not-leak"
PROFILE_ID = "0a1b2c3d-0000-4000-8000-000000000042"
TARGET_ENTRY = {
    "id": TARGET_ID,
    "name": "Governed Sandbox",
    "url": "https://governed.example:8443",
    "api_key": TARGET_SECRET,
    "agent_profile_id": PROFILE_ID,
}


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
        self.records: dict[str, tuple[int, dict]] = {}
        self.create_acks: list[int | str] = []
        self.fail_create_with_read_timeout = False

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def posts(self, path: str) -> int:
        return sum(1 for m, p in self.requests if m == "POST" and p == path)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.requests.append((method, path))

        if method == "POST" and path == "/api/conversations":
            payload = json.loads(request.content)
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
                    else {"detail": "refused"}
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

        if method == "DELETE" and path.startswith("/api/conversations/"):
            self.conversations.discard(path.rsplit("/", 1)[-1])
            return httpx.Response(204)

        return httpx.Response(404, json={"detail": "unexpected"})


async def make_run(
    session_factory,
    *,
    target_id: str | None = TARGET_ID,
    keep_alive: bool | None = None,
    timeout_at=None,
):
    async with session_factory() as session:
        automation = Automation(
            id=uuid.uuid4(),
            user_id=TEST_USER_ID,
            created_by_user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            name="Governed",
            preset_metadata={"preset_type": "prompt"},
            prompt="Do the thing",
            trigger={"type": "cron", "schedule": "0 9 * * 1", "timezone": "UTC"},
            tarball_path="https://example.com/automation.tar.gz",
            entrypoint="python main.py",
            target_id=target_id,
            keep_alive=keep_alive,
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
        server.records[claim.conversation_id] = (
            200,
            {"id": claim.conversation_id, "execution_status": "finished"},
        )

        async with server.client() as client:
            marked = await reconcile_governed_runs(async_session_factory, client=client)

        assert marked == 1
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.COMPLETED
        # Released at terminal: the runtime is deleted.
        assert ("DELETE", f"/api/conversations/{claim.conversation_id}") in (
            server.requests
        )

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

    async def test_keep_alive_keeps_the_record(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory, keep_alive=True)
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
        assert ("DELETE", f"/api/conversations/{claim.conversation_id}") not in (
            server.requests
        )
        assert claim.conversation_id in server.conversations

    async def test_timeout_interrupts_and_deletes(
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
        assert f"/api/conversations/{claim.conversation_id}" in paths
        stored = await reload_run(async_session_factory, run.id)
        assert stored.status == AutomationRunStatus.FAILED
        assert stored.error_detail.startswith("Timed out:")

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

    async def test_cancel_interrupts_then_deletes(
        self, async_session_factory, targets_env
    ):
        run, claim = await self._claimed_run(async_session_factory)
        server = FakeGovernedServer(async_session_factory, run.id)
        server.conversations.add(claim.conversation_id)

        async with server.client() as client:
            await cancel_governed_conversation(client, run)

        paths = [p for _, p in server.requests]
        assert paths == [
            f"/api/conversations/{claim.conversation_id}/interrupt",
            f"/api/conversations/{claim.conversation_id}",
        ]
        assert claim.conversation_id not in server.conversations
