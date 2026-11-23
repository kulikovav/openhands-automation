"""Governed conversation execution for runs with an execution target.

A run on a governed target is dispatched as one Agent Server conversation on
the target's governed server. This backend never calls ``execute_in_context``,
never uploads a tarball and never starts host bash. The conversation UUID is
derived deterministically from the run id and recorded in the run's execution
claim before the start call, so a restart reuses the same identity and the
target's ``idempotent_conversation_create_v1`` dedupes a repeat start.

ponytail: the completion signal is the conversation's terminal state observed
by the control plane, polled every watchdog_interval_seconds (default 60), so
completion latency is one poll interval plus fetch time. The polling is
control-plane-side on purpose: the governed sandbox must not need egress to
the automation service. Upgrade path: a run-bound callback credential lets the
target post terminal state to /v1/runs/{id}/complete.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from openhands.automation.backends.base import (
    ExecutionBackend,
    ExecutionContext,
)
from openhands.automation.models import Automation, AutomationRun, AutomationRunStatus
from openhands.automation.plugin_specs import PluginSpecError
from openhands.automation.targets import AutomationTarget, resolve_target
from openhands.automation.utils.agent_server import (
    VerificationOutcome,
    VerificationResult,
)
from openhands.automation.utils.time import ensure_utc, utcnow


if TYPE_CHECKING:
    from openhands.automation.plugin_secrets import PluginSourceSecrets

logger = logging.getLogger("automation.governed")

# Pinned permanently: changing it makes every live run lose its conversation
# identity. Separate from subjects.CONVERSATION_NAMESPACE, so a governed run id
# can never collide with a subject-derived conversation id.
GOVERNED_RUN_NAMESPACE: Final[uuid.UUID] = uuid.UUID(
    "5b8d2f04-9c31-4a52-8f6e-1d7c0b3a9e21"
)

CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {"target_id", "target_url", "conversation_id", "claimed_at"}
)

# A governed start provisions a sandbox before the conversation record exists.
# A start whose response was lost (or that outlived the client timeout) is
# still in flight, so a 404 on the claimed conversation is not proof of failure
# during this window. The run timeout remains the outer bound.
GOVERNED_START_GRACE_SECONDS: Final[int] = 300

_RUNTIME_MISSING_STATUSES: Final[frozenset[str]] = frozenset(
    {"missing", "ownership_lost", "error"}
)
_CONVERSATION_FAILED_STATUSES: Final[frozenset[str]] = frozenset({"error", "stuck"})


def governed_conversation_id_for(run_id: uuid.UUID | str) -> str:
    """The one conversation UUID a governed run may ever use (uuid5)."""
    return str(uuid.uuid5(GOVERNED_RUN_NAMESPACE, str(run_id)))


@dataclass(frozen=True)
class ExecutionClaim:
    """The durable record of a governed run's intended conversation."""

    target_id: str
    target_url: str
    conversation_id: str
    claimed_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "target_id": self.target_id,
            "target_url": self.target_url,
            "conversation_id": self.conversation_id,
            "claimed_at": self.claimed_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> ExecutionClaim | None:
        if not isinstance(value, dict) or not CLAIM_KEYS <= set(value):
            return None
        return cls(
            target_id=str(value["target_id"]),
            target_url=str(value["target_url"]),
            conversation_id=str(value["conversation_id"]),
            claimed_at=str(value["claimed_at"]),
        )


async def claim_execution(
    session_factory: async_sessionmaker[AsyncSession],
    run: AutomationRun,
    target: AutomationTarget,
) -> ExecutionClaim | None:
    """Write the run's execution claim with one conditional UPDATE.

    The claim is committed before any start call. A second caller (a restart,
    a duplicate dispatch) gets the recorded claim back and reuses its
    conversation id; it never mints a new one. Returns None only when the run
    is no longer RUNNING and has no claim.
    """
    claim = ExecutionClaim(
        target_id=target.id,
        target_url=target.url,
        conversation_id=governed_conversation_id_for(run.id),
        claimed_at=utcnow().isoformat(),
    )
    async with session_factory() as session:
        result = await session.execute(
            update(AutomationRun)
            .where(
                AutomationRun.id == run.id,
                AutomationRun.status == AutomationRunStatus.RUNNING,
                AutomationRun.execution_claim.is_(None),
            )
            .values(execution_claim=claim.to_dict())
        )
        await session.commit()
        if result.rowcount:
            run.execution_claim = claim.to_dict()
            return claim

    async with session_factory() as session:
        recorded = await session.scalar(
            select(AutomationRun.execution_claim).where(AutomationRun.id == run.id)
        )
    existing = ExecutionClaim.from_dict(recorded)
    if existing is not None:
        run.execution_claim = recorded
    return existing


async def record_started_conversation(
    session_factory: async_sessionmaker[AsyncSession],
    run: AutomationRun,
    conversation_id: str,
) -> bool:
    """Record the confirmed conversation id, only while the run is RUNNING."""
    async with session_factory() as session:
        result = await session.execute(
            update(AutomationRun)
            .where(
                AutomationRun.id == run.id,
                AutomationRun.status == AutomationRunStatus.RUNNING,
            )
            .values(conversation_id=conversation_id)
        )
        await session.commit()
    if result.rowcount:
        run.conversation_id = conversation_id
    return bool(result.rowcount)


def build_initial_task(automation: Automation, run: AutomationRun) -> str:
    """The automation task, composed the way the prompt preset composes it.

    The preset's composition lives in its shipped script, which cannot be
    imported safely here, so the same text is composed from the automation
    prompt plus the triggering event payload.
    """
    # Imported lazily: conversations imports the turn helpers, which import
    # the backend registry this module is part of.
    from openhands.automation.conversations import COALESCED_TURNS_KEY

    prompt = automation.prompt or ""
    event_context = dict(run.event_payload or {})
    follow_up_turns = event_context.pop(COALESCED_TURNS_KEY, None)
    context_sections: list[str] = []
    if event_context:
        event_json = json.dumps(event_context, indent=2)
        context_sections.append(
            """## Event Payload

This automation was triggered by a webhook event. The full event payload is
included below; use it directly.

```json
"""
            + event_json
            + "```"
        )
    if follow_up_turns:
        follow_ups = "\n\n".join(str(turn) for turn in follow_up_turns)
        context_sections.append(
            """## Follow-up messages

More activity arrived on the same subject while this run was queued:

"""
            + follow_ups
        )
    if context_sections:
        context_block = "\n\n".join(context_sections)
        return f"""{context_block}

## Task

{prompt}"""
    return prompt


@dataclass(frozen=True)
class StartOutcome:
    """Result of the one idempotent conversation start call."""

    confirmed: bool = False
    ambiguous: bool = False
    error: str | None = None
    code: str | None = None
    status_code: int | None = None


@dataclass(frozen=True)
class ConversationState:
    """What one conversation-record read says about the run."""

    kind: str  # "running" | "completed" | "failed"
    code: str | None = None
    detail: str | None = None
    transient: bool = False


class TargetUnavailableError(RuntimeError):
    """The run's target is no longer configured."""


class GovernedConversationBackend(ExecutionBackend):
    """One Agent Server conversation per run, on the run's execution target.

    Selected by ``get_backend`` when the run's ``target_id`` is set. There is
    no sandbox, no tarball and no bash: the conversation is the execution.
    """

    def __init__(self, run: AutomationRun):
        self._run = run
        self.target: AutomationTarget | None = resolve_target(run.target_id)

    def _require_target(self) -> AutomationTarget:
        if self.target is None:
            raise TargetUnavailableError(
                f"Execution target {self._run.target_id!r} is not configured"
            )
        return self.target

    def _headers(self) -> dict[str, str]:
        return {"X-Session-API-Key": self._require_target().api_key}

    def _start_pending_within_grace(self) -> bool:
        """Whether a start is still in flight and its record may not exist yet.

        Only an unconfirmed start (the run's ``conversation_id`` is unset) has
        this window. Once the start is confirmed, a missing record is terminal.
        """
        if self._run.conversation_id is not None:
            return False
        claim = ExecutionClaim.from_dict(self._run.execution_claim)
        if claim is None:
            return False
        try:
            claimed_at = ensure_utc(datetime.fromisoformat(claim.claimed_at))
        except ValueError:
            return False
        age = (utcnow() - claimed_at).total_seconds()
        return age < GOVERNED_START_GRACE_SECONDS

    @property
    def is_local_mode(self) -> bool:
        return False

    def get_work_dir(self, run_id: str) -> str:  # noqa: ARG002
        """"""  # No workspace: governed runs never extract a tarball.
        return ""

    async def get_execution_context(
        self,
        client: httpx.AsyncClient,  # noqa: ARG002
    ) -> ExecutionContext:
        target = self._require_target()
        return ExecutionContext(agent_url=target.url, session_key=target.api_key)

    async def release_context(
        self,
        client: httpx.AsyncClient,  # noqa: ARG002
        ctx: ExecutionContext,  # noqa: ARG002
    ) -> None:
        # The conversation lifecycle is explicit: released at terminal by the
        # reconcile path (DELETE unless keep_alive), never implicitly here.
        return None

    async def get_api_key(self) -> str:
        return self._require_target().api_key

    async def plugin_source_secrets(
        self,
        client: httpx.AsyncClient,  # noqa: ARG002
        names: Collection[str],  # noqa: ARG002
    ) -> PluginSourceSecrets:
        raise PluginSpecError("Governed runs take no plugin sources")

    def build_env_vars(self) -> dict[str, str]:
        # No bash chain exists on a governed run.
        return {}

    async def start_conversation(
        self,
        client: httpx.AsyncClient,
        automation: Automation,
        run: AutomationRun,
        claim: ExecutionClaim,
    ) -> StartOutcome:
        """Start the run's conversation once, by its intended UUID.

        The governed server advertises ``idempotent_conversation_create_v1``:
        a repeat start with the same ``conversation_id`` answers 200 and does
        not resend the task. 201 and 200 are both a confirmed start.
        """
        target = self._require_target()
        preset = automation.preset_metadata or {}
        task = build_initial_task(automation, run)
        payload: dict[str, Any] = {
            "conversation_id": claim.conversation_id,
            "initial_message": {
                "role": "user",
                "content": [{"type": "text", "text": task}],
                "run": True,
            },
        }
        if target.agent_profile_id:
            payload["agent_profile_id"] = target.agent_profile_id
        # Only where the run already carries them: a plain prompt preset has
        # neither, and governed runs never get the finish-tool plugin.
        if preset.get("max_iterations") is not None:
            payload["max_iterations"] = preset["max_iterations"]
        if preset.get("plugins"):
            payload["plugins"] = preset["plugins"]

        try:
            response = await client.post(
                f"{target.url}/api/conversations",
                json=payload,
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            # The request may have reached the target before the failure: an
            # ambiguous ack keeps the claim and starts nothing else.
            return StartOutcome(ambiguous=True, error=str(exc) or type(exc).__name__)

        if response.status_code in (200, 201):
            body_id = str((response.json() or {}).get("id") or "")
            if body_id.lower() == claim.conversation_id.lower():
                return StartOutcome(confirmed=True, status_code=response.status_code)
            return StartOutcome(
                ambiguous=True,
                error="Start acknowledgement named no matching conversation id",
                status_code=response.status_code,
            )
        if response.status_code in (401, 403):
            return StartOutcome(
                error=(
                    "Target refused the conversation start "
                    f"(HTTP {response.status_code})"
                ),
                code="policy_denied",
                status_code=response.status_code,
            )
        if response.status_code in (408, 429) or response.status_code >= 500:
            return StartOutcome(
                ambiguous=True,
                error=f"Transient start failure (HTTP {response.status_code})",
                status_code=response.status_code,
            )
        return StartOutcome(
            error=(
                "Conversation start failed "
                f"(HTTP {response.status_code}): {response.text[:500]}"
            ),
            code="conversation_error",
            status_code=response.status_code,
        )

    async def fetch_conversation_state(
        self, client: httpx.AsyncClient, conversation_id: str
    ) -> ConversationState:
        """Read the conversation record directly from the target.

        Never through a proxied conversation read: only this read decides
        liveness and terminal state.
        """
        target = self._require_target()
        url = f"{target.url}/api/conversations/{conversation_id}"
        try:
            response = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            return ConversationState(
                kind="running", transient=True, detail=str(exc) or type(exc).__name__
            )

        if response.status_code == 410:
            return ConversationState(
                kind="failed",
                code="runtime_missing",
                detail="Conversation is archived or its runtime is gone (HTTP 410)",
            )
        if response.status_code == 404:
            if self._start_pending_within_grace():
                return ConversationState(
                    kind="running",
                    transient=True,
                    detail=(
                        "Conversation is not visible yet; the start may still "
                        "be provisioning its sandbox"
                    ),
                )
            return ConversationState(
                kind="failed",
                code="conversation_error",
                detail="Conversation record not found",
            )
        if response.status_code in (401, 403):
            return ConversationState(
                kind="failed",
                code="policy_denied",
                detail=(
                    "Target refused the conversation read "
                    f"(HTTP {response.status_code})"
                ),
            )
        if response.status_code in (408, 429) or response.status_code >= 500:
            return ConversationState(
                kind="running",
                transient=True,
                detail=f"Transient read failure (HTTP {response.status_code})",
            )
        if response.status_code != 200:
            return ConversationState(
                kind="failed",
                code="conversation_error",
                detail=f"Conversation read failed (HTTP {response.status_code})",
            )

        record = response.json() or {}
        execution_status = str(record.get("execution_status") or "")
        if execution_status == "finished":
            return ConversationState(kind="completed")
        if execution_status in _CONVERSATION_FAILED_STATUSES:
            return ConversationState(
                kind="failed",
                code="conversation_error",
                detail=f"Conversation ended in state {execution_status!r}",
            )
        if execution_status == "deleting":
            return ConversationState(
                kind="failed",
                code="runtime_missing",
                detail="Conversation record is being deleted",
            )

        runtime_info = record.get("runtime_info")
        if runtime_info is None:
            # The record does not carry runtime availability; ask the runtime
            # record itself before trusting the execution status.
            try:
                runtime_response = await client.get(
                    f"{url}/runtime", headers=self._headers()
                )
            except httpx.HTTPError as exc:
                return ConversationState(
                    kind="running",
                    transient=True,
                    detail=str(exc) or type(exc).__name__,
                )
            if runtime_response.status_code != 200:
                return ConversationState(
                    kind="running",
                    transient=True,
                    detail=f"Runtime read failed (HTTP {runtime_response.status_code})",
                )
            runtime_info = runtime_response.json() or {}
        runtime_status = str((runtime_info or {}).get("runtime_status") or "")
        if runtime_status in _RUNTIME_MISSING_STATUSES:
            return ConversationState(
                kind="failed",
                code="runtime_missing",
                detail=f"Execution runtime is {runtime_status!r}",
            )
        return ConversationState(kind="running")

    async def interrupt(self, client: httpx.AsyncClient, conversation_id: str) -> None:
        target = self._require_target()
        await client.post(
            f"{target.url}/api/conversations/{conversation_id}/interrupt",
            headers=self._headers(),
        )

    async def delete_conversation(
        self, client: httpx.AsyncClient, conversation_id: str
    ) -> None:
        target = self._require_target()
        await client.delete(
            f"{target.url}/api/conversations/{conversation_id}",
            headers=self._headers(),
        )

    async def verify_run(self, run_id: str) -> VerificationResult:  # noqa: ARG002
        """Contract completeness: the reconcile path owns governed liveness."""
        claim = ExecutionClaim.from_dict(self._run.execution_claim)
        if claim is None:
            return VerificationResult(
                outcome=VerificationOutcome.VERIFICATION_ERROR,
                detail="Run has no execution claim",
            )
        async with httpx.AsyncClient(timeout=30.0) as client:
            state = await self.fetch_conversation_state(client, claim.conversation_id)
        if state.kind == "completed":
            return VerificationResult(
                outcome=VerificationOutcome.COMPLETED, exit_code=0
            )
        if state.kind == "failed":
            return VerificationResult(
                outcome=VerificationOutcome.FAILED, error=state.detail
            )
        return VerificationResult(
            outcome=VerificationOutcome.STILL_RUNNING, detail=state.detail
        )

    async def cleanup_after_verification(self, run_id: str) -> None:  # noqa: ARG002
        claim = ExecutionClaim.from_dict(self._run.execution_claim)
        if claim is None:
            return
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self.delete_conversation(client, claim.conversation_id)

    async def stop_command(self, command_id: str) -> None:  # noqa: ARG002
        # No bash command exists; cancellation maps to interrupt + delete (see
        # cancel_governed_conversation).
        return None


async def release_conversation(
    client: httpx.AsyncClient,
    backend: GovernedConversationBackend,
    claim: ExecutionClaim,
    *,
    keep_alive: bool | None,
) -> None:
    """Release the runtime at terminal: DELETE unless keep_alive keeps it.

    A missing runtime is never recreated; the delete is best-effort.
    """
    if keep_alive is True:
        return
    try:
        await backend.delete_conversation(client, claim.conversation_id)
    except Exception as exc:
        logger.warning(
            "Could not release governed conversation %s: %s",
            claim.conversation_id,
            exc,
        )


async def cancel_governed_conversation(
    client: httpx.AsyncClient,
    run: AutomationRun,
) -> None:
    """Cancellation: interrupt the conversation, then delete it.

    Explicit cancellation deletes even when the automation keeps its runtime
    alive, exactly as a cancelled cloud run loses its sandbox.
    """
    claim = ExecutionClaim.from_dict(run.execution_claim)
    if claim is None:
        return
    backend = GovernedConversationBackend(run)
    try:
        await backend.interrupt(client, claim.conversation_id)
    except Exception as exc:
        logger.warning(
            "Could not interrupt governed conversation %s: %s",
            claim.conversation_id,
            exc,
        )
    try:
        await backend.delete_conversation(client, claim.conversation_id)
    except Exception as exc:
        logger.warning(
            "Could not delete governed conversation %s: %s",
            claim.conversation_id,
            exc,
        )
