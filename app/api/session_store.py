"""The "Session State" component in the architecture diagram.

A live ``ScreeningOrchestrator`` holds an open cognition stream, an in-flight
speculative draft, and a Langfuse span recorder -- none of it is JSON
serialisable, and none of it should be: a screening call is bound to the
worker process that is actually talking to the candidate. So the source of
truth for an *active* session is a process-local registry, exactly like the
in-memory `HybridVectorStore` fallback already used for retrieval.

What a cache *can* usefully hold, across replicas, is the small serialisable
slice another process needs to answer "is this session alive, and where":
status, role, turn count, last-activity timestamp. When ``REDIS_URL`` is
configured that slice is mirrored to Redis on every turn; when it is not, the
mirror degrades to the same in-memory dict and every single-process behaviour
(including every test in this repo) is unaffected. This mirrors the exact
degradation pattern already used for Moss and Qdrant elsewhere in the code:
a missing dependency changes where data lives, never whether the call
succeeds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.orchestrator import ScreeningOrchestrator
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionStatus:
    session_id: str
    role: str
    status: str  # active | closed
    turns: int
    last_activity: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(
            {
                "session_id": self.session_id,
                "role": self.role,
                "status": self.status,
                "turns": self.turns,
                "last_activity": self.last_activity,
            }
        )


class SessionStore:
    """Process-local registry of live orchestrators, with an optional Redis mirror."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._orchestrators: dict[str, ScreeningOrchestrator] = {}
        self._lock = asyncio.Lock()
        self._redis: Any | None = None
        if self.settings.redis_url:
            self._connect_redis()

    def _connect_redis(self) -> None:
        try:  # pragma: no cover - requires the optional dependency + a live server
            import redis.asyncio as redis_asyncio

            self._redis = redis_asyncio.from_url(self.settings.redis_url, decode_responses=True)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Redis unavailable (%s); session state stays process-local.",
                type(exc).__name__,
            )
            self._redis = None

    @property
    def remote_enabled(self) -> bool:
        return self._redis is not None

    async def put(self, orchestrator: ScreeningOrchestrator) -> None:
        session_id = orchestrator.session.session_id
        async with self._lock:
            self._orchestrators[session_id] = orchestrator
        await self._mirror(orchestrator, status="active")

    async def get(self, session_id: str) -> ScreeningOrchestrator | None:
        async with self._lock:
            return self._orchestrators.get(session_id)

    async def touch(self, orchestrator: ScreeningOrchestrator) -> None:
        await self._mirror(orchestrator, status="active")

    async def close(self, session_id: str) -> None:
        async with self._lock:
            orchestrator = self._orchestrators.pop(session_id, None)
        if orchestrator is not None:
            await self._mirror(orchestrator, status="closed")

    async def _mirror(self, orchestrator: ScreeningOrchestrator, status: str) -> None:
        if self._redis is None:
            return
        record = SessionStatus(
            session_id=orchestrator.session.session_id,
            role=orchestrator.session.role.value,
            status=status,
            turns=orchestrator.flow.turns,
        )
        try:  # pragma: no cover - network path
            await self._redis.set(
                f"hrte:session:{record.session_id}",
                record.to_json(),
                ex=self.settings.session_ttl_seconds,
            )
        except Exception:  # pragma: no cover
            logger.warning("Failed to mirror session %s to Redis.", record.session_id)

    def __len__(self) -> int:
        return len(self._orchestrators)


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def reset_session_store() -> None:
    """Drop the process-wide store. Test isolation only -- mirrors reset_settings()."""
    global _store
    _store = None


__all__ = ["SessionStatus", "SessionStore", "get_session_store", "reset_session_store"]
