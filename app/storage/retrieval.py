"""The rubric retrieval layer, with Moss on the hot path.

Why retrieval is the latency story
----------------------------------
The 160ms turnaround budget allocates exactly 10ms to fetching the context the
interviewer needs -- which competency the candidate is currently demonstrating,
so the next question lands on the rubric instead of wandering. A conventional
vector database cannot participate in that budget honestly: a network hop to a
managed service is 15-40ms before the index has done any work at all, so the
"10ms" line item silently becomes 50ms and the whole budget is fiction.

`Moss <https://www.moss.dev/>`_ is a search *runtime* rather than a database.
It runs in-process -- browser, edge, device or cloud -- so a query is a function
call, not a round trip, and sub-10ms retrieval is achievable rather than
aspirational. That is why it sits on the hot path here.

Backends
--------
* :class:`MossRetriever` -- production. Sub-10ms in-process hybrid (semantic +
  keyword) search over the rubric corpus.
* :class:`EmbeddedRetriever` -- the offline fallback, backed by
  :class:`~app.storage.qdrant_client.HybridVectorStore`. Identical interface, no
  credentials, which is what lets CI exercise the full pipeline.

Both are selected by :func:`build_retriever` and are interchangeable at runtime,
so a Moss outage degrades retrieval quality rather than ending a live interview.

A naming caution
----------------
Two unrelated products called "Moss" appear in this system and conflating them
will waste someone an afternoon:

* **Moss (YC F25)** -- the retrieval runtime in *this* module.
* **MOSS-Speech** -- an open speech-to-speech model, wired up in
  :mod:`app.voice.moss_engine` and configured under ``HRTE_SPEECH_*``.

They share a name and nothing else.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from app.config import STAGE_BUDGETS_MS, Settings, Stage, get_settings
from app.schemas.roles import ROLE_RUBRICS, EngineeringRole
from app.storage.qdrant_client import HybridVectorStore

logger = logging.getLogger(__name__)

#: The stage budget this layer must live inside.
RETRIEVAL_BUDGET_MS: Final[float] = STAGE_BUDGETS_MS[Stage.VECTOR_MATCH]


@dataclass(frozen=True, slots=True)
class RubricHit:
    """One retrieved competency, resolved back to its rubric."""

    role: EngineeringRole
    competency_key: str
    competency_label: str
    score: float
    text: str = ""

    @property
    def document_id(self) -> str:
        return f"{self.role.value}::{self.competency_key}"


@dataclass(frozen=True, slots=True)
class RubricDocument:
    """One indexable unit: a single competency, not a whole rubric.

    Indexing per competency rather than per role is a retrieval-quality
    decision, not a packaging one. A whole-rubric document is four unrelated
    competencies concatenated; a candidate's answer matches one of them, and
    averaging it against the other three buries the signal. Splitting them keeps
    each document tight and lets the runtime return *which* competency matched,
    which is the thing the interviewer actually needs to know.
    """

    id: str
    text: str
    role: EngineeringRole
    competency_key: str
    competency_label: str

    def as_moss_document(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": {
                "role": self.role.value,
                "competency_key": self.competency_key,
                "competency_label": self.competency_label,
            },
        }


def rubric_documents() -> tuple[RubricDocument, ...]:
    """The full rubric corpus: 9 roles x their competencies."""
    docs: list[RubricDocument] = []
    for role in EngineeringRole:
        rubric = ROLE_RUBRICS[role]
        for competency in rubric.competencies:
            docs.append(
                RubricDocument(
                    id=f"{role.value}::{competency.key}",
                    text=(
                        f"{rubric.title}. {competency.label}. "
                        f"{' '.join(competency.signals)}. {competency.probe}"
                    ),
                    role=role,
                    competency_key=competency.key,
                    competency_label=competency.label,
                )
            )
    return tuple(docs)


@runtime_checkable
class RubricRetriever(Protocol):
    """Anything that can answer 'which competency is this answer evidence for?'."""

    @property
    def backend(self) -> str: ...

    @property
    def last_latency_ms(self) -> float: ...

    async def warm(self) -> None: ...

    async def search(self, evidence: str, limit: int = 3) -> list[RubricHit]: ...


class EmbeddedRetriever:
    """Offline fallback over the in-process hybrid index.

    Exact rather than approximate, because at rubric scale (a few dozen
    documents) an exact dot product is both faster than a graph traversal and
    trivially reproducible -- which is what CI needs.
    """

    def __init__(
        self, store: HybridVectorStore | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or HybridVectorStore(self.settings)
        self._last_latency_ms = 0.0
        self._by_id = {doc.id: doc for doc in rubric_documents()}

    @property
    def backend(self) -> str:
        return "embedded"

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    async def warm(self) -> None:
        self.store.ensure_collection()

    async def search(self, evidence: str, limit: int = 3) -> list[RubricHit]:
        if not evidence.strip():
            return []
        started = time.perf_counter()
        hits = self.store.search_rubrics(evidence, limit=limit)
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0

        out: list[RubricHit] = []
        for hit in hits:
            role_name = str(hit.payload.get("role", hit.id))
            if role_name not in EngineeringRole.__members__:
                continue
            role = EngineeringRole[role_name]
            label = str(hit.payload.get("competency_label", ""))
            key = _key_for_label(role, label)
            out.append(
                RubricHit(
                    role=role,
                    competency_key=key,
                    competency_label=label,
                    score=hit.score,
                )
            )
        return out


class MossRetriever:
    """Sub-10ms in-process retrieval over the rubric corpus, powered by Moss.

    Operational notes:

    * The index is built once at session open, during the greeting, so the first
      real turn never pays for indexing. The corpus is static -- it is the
      rubrics, which are version-controlled -- so this is pure warm-up.
    * ``load_index`` is attempted before ``create_index``: re-indexing a corpus
      that has not changed is wasted startup latency on every worker boot.
    * Every failure path degrades to :class:`EmbeddedRetriever`. A screening
      call in progress must never end because retrieval had a bad minute; worse
      context is recoverable, a dropped interview is not.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        fallback: RubricRetriever | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._fallback = fallback or EmbeddedRetriever(settings=self.settings)
        self._client: Any | None = None
        self._ready = False
        self._degraded = False
        self._last_latency_ms = 0.0
        self._documents = rubric_documents()
        self._by_id = {doc.id: doc for doc in self._documents}
        self._by_text = {doc.text: doc for doc in self._documents}

    @property
    def backend(self) -> str:
        return "embedded (moss degraded)" if self._degraded else "moss"

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def index_name(self) -> str:
        return self.settings.moss_index

    async def warm(self) -> None:
        # Moss owns the live rubric index. Do not initialise the archival
        # fallback when Moss is healthy: it adds needless startup work and
        # creates an accidental dependency on a second retrieval system.
        if not self.settings.moss_configured:
            self._degraded = True
            await self._fallback.warm()
            return
        try:  # pragma: no cover - requires the optional moss dependency
            from moss import MossClient  # type: ignore[import-not-found]

            self._client = MossClient(self.settings.moss_project_id, self.settings.moss_project_key)
            try:
                await self._client.load_index(self.index_name)
            except Exception:
                await self._client.create_index(
                    self.index_name,
                    [doc.as_moss_document() for doc in self._documents],
                )
                await self._client.load_index(self.index_name)
            self._ready = True
            self._degraded = False
            logger.info(
                "Moss index '%s' ready with %d rubric documents.",
                self.index_name,
                len(self._documents),
            )
        except Exception as exc:  # pragma: no cover - degradation path
            logger.warning(
                "Moss unavailable (%s); rubric retrieval falls back to the "
                "embedded index. Latency and the interview are unaffected; "
                "retrieval quality may be.",
                type(exc).__name__,
            )
            self._client = None
            self._ready = False
            self._degraded = True
            await self._fallback.warm()

    async def search(self, evidence: str, limit: int = 3) -> list[RubricHit]:
        if not evidence.strip():
            return []
        if not self._ready or self._client is None:
            hits = await self._fallback.search(evidence, limit)
            self._last_latency_ms = self._fallback.last_latency_ms
            return hits

        try:  # pragma: no cover - requires a live Moss project
            from moss import QueryOptions  # type: ignore[import-not-found]

            started = time.perf_counter()
            # Over-fetch: several competencies of one role often match, and we
            # want `limit` distinct roles back, not `limit` slices of one.
            result = await self._client.query(
                self.index_name, evidence, QueryOptions(top_k=max(limit * 3, 9))
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            # Prefer the runtime's own measurement when it reports one; it
            # excludes our own await scheduling overhead.
            self._last_latency_ms = float(getattr(result, "time_taken_ms", elapsed) or elapsed)
            return self._collapse(getattr(result, "docs", []) or [], limit)
        except Exception as exc:  # pragma: no cover - degradation path
            logger.warning("Moss query failed (%s); using the embedded index.", type(exc).__name__)
            self._degraded = True
            self._ready = False
            hits = await self._fallback.search(evidence, limit)
            self._last_latency_ms = self._fallback.last_latency_ms
            return hits

    def _collapse(self, docs: Sequence[Any], limit: int) -> list[RubricHit]:
        """Map Moss documents back to rubrics, best competency per role.

        Document identity is resolved defensively -- by ``id``, then by metadata,
        then by exact text -- because a retrieval SDK adding or renaming a field
        should degrade this to a slower lookup, not to a crash inside a live
        interview.
        """
        best: dict[EngineeringRole, RubricHit] = {}
        for doc in docs:
            resolved = self._resolve(doc)
            if resolved is None:
                continue
            document, score = resolved
            current = best.get(document.role)
            if current is None or score > current.score:
                best[document.role] = RubricHit(
                    role=document.role,
                    competency_key=document.competency_key,
                    competency_label=document.competency_label,
                    score=score,
                    text=document.text,
                )
        ordered = sorted(best.values(), key=lambda h: (-h.score, h.role.value))
        return ordered[:limit]

    def _resolve(self, doc: Any) -> tuple[RubricDocument, float] | None:
        score = float(getattr(doc, "score", 0.0) or 0.0)

        doc_id = getattr(doc, "id", None)
        if isinstance(doc_id, str) and doc_id in self._by_id:
            return self._by_id[doc_id], score

        metadata = getattr(doc, "metadata", None)
        if isinstance(metadata, dict):
            role_name = str(metadata.get("role", ""))
            key = str(metadata.get("competency_key", ""))
            candidate = self._by_id.get(f"{role_name}::{key}")
            if candidate is not None:
                return candidate, score

        text = getattr(doc, "text", None)
        if isinstance(text, str) and text in self._by_text:
            return self._by_text[text], score

        return None


def build_retriever(
    settings: Settings | None = None, store: HybridVectorStore | None = None
) -> RubricRetriever:
    """Select the retrieval backend from configuration.

    Moss when a project is configured, the embedded index otherwise. The return
    type is the protocol, so nothing downstream can accidentally depend on which
    one it got.
    """
    settings = settings or get_settings()
    fallback = EmbeddedRetriever(store=store, settings=settings)
    if settings.moss_configured:
        return MossRetriever(settings=settings, fallback=fallback)
    return fallback


def _key_for_label(role: EngineeringRole, label: str) -> str:
    for competency in ROLE_RUBRICS[role].competencies:
        if competency.label == label:
            return competency.key
    return ""


__all__ = [
    "RETRIEVAL_BUDGET_MS",
    "EmbeddedRetriever",
    "MossRetriever",
    "RubricDocument",
    "RubricHit",
    "RubricRetriever",
    "build_retriever",
    "rubric_documents",
]
