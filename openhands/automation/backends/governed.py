"""Governed conversation execution for runs with an execution target.

A run on a governed target is dispatched as one Agent Server conversation on
the target's governed server. This backend never calls ``execute_in_context``
and never starts host bash. A script bundle rides the start payload for the
sandbox to unpack and run; nothing here extracts it. The conversation UUID is
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

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import quote

import httpx
from sqlalchemy import inspect, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from openhands.automation.backends.base import (
    ConversationStart,
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
from openhands.automation.utils.run_status_detail import (
    RunStatusDetailKind,
    RunStatusPhase,
    make_run_status_detail,
)
from openhands.automation.utils.time import ensure_utc, utcnow
from openhands.workspace import RepoSource


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

# The agent server's own settings routes, as a direct conversation addresses
# them: a secret name becomes a LookupSecret the server resolves in-process
# from the store it already holds. A relative URL is what Canvas sends, and it
# keeps resolution on the server that receives the start.
SECRETS_PATH: Final[str] = "/api/settings/secrets"
SECRETS_TIMEOUT_SECONDS: Final[float] = 30.0

# A server that predates the field accepts ``repositories`` (pydantic ignores
# extras) and never clones it, so the start only names repositories to a
# server that advertises this.
SERVER_INFO_PATH: Final[str] = "/server_info"
REPOSITORIES_CAPABILITY: Final[str] = "conversation_repositories_v1"
# A script bundle is only started on a server that can unpack and run it.
BUNDLE_CAPABILITY: Final[str] = "automation_bundle_v1"
# A plugin automation's sources and its config are only sent to a server that
# loads them. A server without it would run the task without the plugin.
PLUGINS_CAPABILITY: Final[str] = "conversation_plugins_v1"

_RUNTIME_MISSING_STATUSES: Final[frozenset[str]] = frozenset(
    {"missing", "ownership_lost", "error"}
)
_CONVERSATION_FAILED_STATUSES: Final[frozenset[str]] = frozenset({"error", "stuck"})

TRANSCRIPT_RETAINED: Final[str] = "transcript_retained"
TRANSCRIPT_UNAVAILABLE: Final[str] = "transcript_unavailable"


class TranscriptOutcome(StrEnum):
    """What a terminal governed release left behind for later review."""

    RETAINED = TRANSCRIPT_RETAINED
    UNAVAILABLE = TRANSCRIPT_UNAVAILABLE
    RELEASE_FAILED = "release_failed"

    @property
    def status_detail_code(self) -> str:
        """The status-detail code this release outcome records."""
        return self.value


_TRANSCRIPT_STATUS_DETAILS: Final[dict[TranscriptOutcome, str]] = {
    TranscriptOutcome.RETAINED: (
        "Runtime stopped; conversation and transcript retained"
    ),
    TranscriptOutcome.UNAVAILABLE: "Conversation and transcript deleted",
    TranscriptOutcome.RELEASE_FAILED: (
        "Runtime release failed; transcript state unknown"
    ),
}


def transcript_status_detail(outcome: TranscriptOutcome) -> dict[str, Any]:
    """The status detail for a terminal run's transcript outcome.

    A failed release is recorded as ``release_failed`` rather than dropped.
    The UI does not map that code: ``getAutomationRunGovernance`` returns null
    and the normal conversation link is shown.
    """
    return make_run_status_detail(
        phase=RunStatusPhase.CLEANUP,
        kind=RunStatusDetailKind.CLEANUP,
        detail=_TRANSCRIPT_STATUS_DETAILS[outcome],
        transient=False,
        source="automation_service",
        operation="release_conversation",
        code=outcome.status_detail_code,
    )


def governed_conversation_id_for(run_id: uuid.UUID | str) -> str:
    """The one conversation UUID a governed run may ever use (uuid5)."""
    return str(uuid.uuid5(GOVERNED_RUN_NAMESPACE, str(run_id)))


def _preclone_repositories(
    repos: list[Any] | None, secret_names: Collection[str]
) -> list[dict[str, Any]]:
    """The automation's repositories this target can clone with its secrets.

    The gate is the receiving agent server's own ``RepoSource`` acceptance: a
    string or a dict, a branch or tag, a commit ref (a full clone then a
    checkout), and a short ``owner/repo`` name. A governed run has no runner to
    clone what this drops, so a dropped repository leaves an empty workspace.
    The provider token must be in the target's secret store under either the
    upper- or lower-case spelling the server's fetcher resolves. An SSH or file
    URL carries no provider token, so it is dropped.
    """
    names = {name.lower() for name in secret_names}
    repositories: list[dict[str, Any]] = []
    for repo in repos or []:
        try:
            source = RepoSource.model_validate(repo)
            token_name = source.get_token_name()
        except ValueError:
            continue
        if source.url.startswith(("git@", "file://")):
            continue
        if token_name.lower() not in names:
            continue
        repositories.append(source.model_dump(mode="json", exclude_none=True))
    return repositories


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


class ConversationGoneError(Exception):
    """The governed conversation no longer exists on its target."""


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
        """"""  # No host workspace: the target's sandbox unpacks the bundle.
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
        # reconcile path according to retain_transcript, never implicitly here.
        return None

    async def get_api_key(self) -> str:
        return self._require_target().api_key

    async def plugin_source_secrets(
        self,
        client: httpx.AsyncClient,  # noqa: ARG002
        names: Collection[str],  # noqa: ARG002
    ) -> PluginSourceSecrets:
        raise PluginSpecError(
            "A governed run's plugin sources are fetched by its target"
        )

    def build_env_vars(self) -> dict[str, str]:
        # No bash chain exists on a governed run.
        return {}

    async def configured_secrets(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """The target's configured secrets, as a direct conversation sends them.

        ``GET /api/settings/secrets`` names them only, scoped to the target's
        agent profile, and each becomes a ``LookupSecret`` the server resolves
        in-process from its own store. Raw values therefore never transit this
        service. An unreadable or malformed list is empty, as a preset run
        without them.
        """
        target = self._require_target()
        params = (
            {"agent_profile_id": target.agent_profile_id}
            if target.agent_profile_id
            else None
        )
        try:
            response = await client.get(
                f"{target.url}{SECRETS_PATH}",
                params=params,
                headers=self._headers(),
                timeout=SECRETS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            # The reference a direct conversation sends: a relative URL the
            # receiving server resolves against its own base, and the session
            # key header an out-of-process resolver needs. A lookup URL names a
            # secret, never a value.
            body = response.json()
            items = body.get("secrets") if isinstance(body, dict) else None
            secrets: dict[str, Any] = {}
            for item in items or []:
                name = item["name"]
                secrets[name] = {
                    "kind": "LookupSecret",
                    "url": f"{SECRETS_PATH}/{quote(name, safe='')}",
                    "headers": self._headers(),
                    **(
                        {"description": item["description"]}
                        if item.get("description")
                        else {}
                    ),
                }
        except (
            httpx.HTTPError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Could not read the target's secrets, starting without them: %s", exc
            )
            return {}
        return secrets

    async def _advertises(self, client: httpx.AsyncClient, capability: str) -> bool:
        """Whether the target's build serves ``capability`` on /server_info."""
        target = self._require_target()
        try:
            response = await client.get(
                f"{target.url}{SERVER_INFO_PATH}",
                headers=self._headers(),
                timeout=SECRETS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            capabilities = response.json().get("capabilities")
        except (httpx.HTTPError, AttributeError, ValueError):
            return False
        return isinstance(capabilities, list) and capability in capabilities

    async def start_conversation(
        self,
        client: httpx.AsyncClient,
        automation: Automation,
        run: AutomationRun,
        claim: ExecutionClaim,
        start: ConversationStart,
    ) -> StartOutcome:
        """Start the run's conversation once, by its intended UUID.

        The governed server advertises ``idempotent_conversation_create_v1``:
        a repeat start with the same ``conversation_id`` answers 200 and does
        not resend the task. 201 and 200 are both a confirmed start.

        ``start`` carries the run's plugins, or the script bundle the sandbox
        unpacks. Each is sent only to a server that advertises it; a target
        without the capability fails the run rather than running it without
        the plugin or the script.
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
        # The configured secrets (by reference), and the repositories their
        # provider tokens can clone. A server from before the clone field
        # ignores it, so a run that names repositories is refused unless the
        # target advertises that it clones them.
        secrets = await self.configured_secrets(client)
        if secrets:
            payload["secrets"] = secrets
        repositories = _preclone_repositories(preset.get("repos"), secrets.keys())
        if repositories:
            if not await self._advertises(client, REPOSITORIES_CAPABILITY):
                return StartOutcome(
                    error=(
                        "Target cannot clone the run's repositories: it does "
                        f"not advertise {REPOSITORIES_CAPABILITY}"
                    ),
                    code="repositories_unsupported",
                )
            payload["repositories"] = repositories
        if start.plugins:
            if not await self._advertises(client, PLUGINS_CAPABILITY):
                return StartOutcome(
                    error=(
                        "Target cannot load the run's plugins: it does not "
                        f"advertise {PLUGINS_CAPABILITY}"
                    ),
                    code="plugins_unsupported",
                )
            payload["plugins"] = start.plugins
        if start.bundle is not None:
            if not await self._advertises(client, BUNDLE_CAPABILITY):
                return StartOutcome(
                    error=(
                        "Target cannot run the run's script bundle: it does "
                        f"not advertise {BUNDLE_CAPABILITY}"
                    ),
                    code="bundle_unsupported",
                )
            payload["bundle"] = {
                "entrypoint": start.bundle.entrypoint,
                "setup_script_path": start.bundle.setup_script_path,
                "config": start.bundle.config,
                "tarball_b64": base64.b64encode(start.bundle.tarball).decode("ascii"),
            }
        if target.agent_profile_id:
            payload["agent_profile_id"] = target.agent_profile_id
        # Only where the run already carries them: a plain prompt preset has
        # neither, and governed runs never get the finish-tool plugin.
        if preset.get("max_iterations") is not None:
            payload["max_iterations"] = preset["max_iterations"]

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
    ) -> bool:
        """Delete the conversation record and its events. True on success.

        Raises ConversationGoneError when the target answers 404, so a caller
        never retries a conversation that no longer exists.
        """
        target = self._require_target()
        response = await client.delete(
            f"{target.url}/api/conversations/{conversation_id}",
            headers=self._headers(),
        )
        if response.status_code == 404:
            logger.info("Governed conversation %s is already gone", conversation_id)
            raise ConversationGoneError(conversation_id)
        if not response.is_success:
            logger.warning(
                "Could not delete governed conversation %s (HTTP %s)",
                conversation_id,
                response.status_code,
            )
        return response.is_success

    async def release_runtime(
        self, client: httpx.AsyncClient, conversation_id: str
    ) -> bool:
        """Stop the conversation's sandbox, keeping the record and events.

        Raises ConversationGoneError when the target answers 404, so a caller
        never retries a conversation that no longer exists.
        """
        target = self._require_target()
        response = await client.delete(
            f"{target.url}/api/conversations/{conversation_id}/runtime",
            headers=self._headers(),
        )
        if response.status_code == 404:
            logger.info("Governed conversation %s is already gone", conversation_id)
            raise ConversationGoneError(conversation_id)
        if not response.is_success:
            logger.warning(
                "Could not stop the governed runtime of conversation %s (HTTP %s)",
                conversation_id,
                response.status_code,
            )
        return response.is_success

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

    async def cleanup_after_verification(self, run_id: str) -> bool | None:  # noqa: ARG002
        """Release the governed conversation, reporting whether to retry.

        Returns False when the release failed, so the caller keeps the
        deferred cleanup due and retries. A missing claim has nothing to
        release, and an unloaded parent falls back to retention because
        deleting a transcript is irreversible.
        """
        claim = ExecutionClaim.from_dict(self._run.execution_claim)
        if claim is None:
            return None
        retain_transcript = True
        if "automation" not in inspect(self._run).unloaded:
            automation = self._run.automation
            if automation is not None:
                retain_transcript = automation.retain_transcript
        async with httpx.AsyncClient(timeout=30.0) as client:
            outcome = await release_conversation(
                client, self, claim, retain_transcript=retain_transcript
            )
        return outcome is not TranscriptOutcome.RELEASE_FAILED

    async def stop_command(self, command_id: str) -> None:  # noqa: ARG002
        # No bash command exists; cancellation maps to interrupt and release
        # (see cancel_governed_conversation).
        return None


RELEASE_ATTEMPTS: Final[int] = 3
RELEASE_BACKOFF_SECONDS: Final[tuple[float, ...]] = (0.25, 0.5)


async def release_conversation(
    client: httpx.AsyncClient,
    backend: GovernedConversationBackend,
    claim: ExecutionClaim,
    *,
    retain_transcript: bool,
) -> TranscriptOutcome:
    """Stop a terminal governed conversation and settle its transcript.

    ``retain_transcript`` stops the sandbox but keeps the conversation record
    and its events; false deletes the conversation. A non-success response or
    a transport error is retried, then reported as ``RELEASE_FAILED``, never
    as a retained transcript.
    """
    for attempt in range(RELEASE_ATTEMPTS):
        try:
            if retain_transcript:
                if await backend.release_runtime(client, claim.conversation_id):
                    return TranscriptOutcome.RETAINED
            elif await backend.delete_conversation(client, claim.conversation_id):
                return TranscriptOutcome.UNAVAILABLE
        except ConversationGoneError:
            # A 404 is final: the conversation is gone, so retrying cannot help.
            return TranscriptOutcome.UNAVAILABLE
        except Exception as exc:
            logger.warning(
                "Could not release governed conversation %s (attempt %d/%d): %s",
                claim.conversation_id,
                attempt + 1,
                RELEASE_ATTEMPTS,
                exc,
            )
        if attempt + 1 < RELEASE_ATTEMPTS:
            await asyncio.sleep(RELEASE_BACKOFF_SECONDS[attempt])
    logger.warning(
        "Could not release governed conversation %s after %d attempts",
        claim.conversation_id,
        RELEASE_ATTEMPTS,
    )
    return TranscriptOutcome.RELEASE_FAILED


async def cancel_governed_conversation(
    client: httpx.AsyncClient,
    run: AutomationRun,
    *,
    retain_transcript: bool,
) -> TranscriptOutcome | None:
    """Cancellation: interrupt the conversation, then honor retention.

    The interrupt is best effort. With ``retain_transcript`` true, the
    runtime stops and the conversation and its transcript are kept;
    cancellation no longer deletes a retained transcript. False deletes the
    conversation. Returns None when the run has no execution claim.
    """
    claim = ExecutionClaim.from_dict(run.execution_claim)
    if claim is None:
        return None
    backend = GovernedConversationBackend(run)
    try:
        await backend.interrupt(client, claim.conversation_id)
    except Exception as exc:
        logger.warning(
            "Could not interrupt governed conversation %s: %s",
            claim.conversation_id,
            exc,
        )
    return await release_conversation(
        client, backend, claim, retain_transcript=retain_transcript
    )
