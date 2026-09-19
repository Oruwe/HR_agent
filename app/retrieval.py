"""Retrieval over the candidate pool, with Moss on the live path.

Why this exists
---------------
The analyst used to answer every question by putting the *entire* pool in the
prompt, capped at 200 records. That works for a demo and fails for the actual
job: a manager with 800 scraped profiles either gets a truncated pool or a
prompt nobody can afford. Worse, a model asked "who has the strongest
distributed systems evidence?" while holding 200 records reads them all with
equal attention and answers worse than one shown the six that matter.

So a question is now a *search* first and a generation second. Whatever the
retrieval layer returns is what the model reads.

Two backends, one protocol
--------------------------
* :class:`MossIndex` -- `Moss <https://usemoss.dev>`_, a semantic search
  runtime. Embeddings are computed by the runtime, so "who handled failure
  modes under load" finds a record that says "split brain during a partition"
  without either phrase sharing a word.
* :class:`LocalIndex` -- a TF-IDF cosine index in pure Python. No credentials,
  no network, no model download, deterministic, and fast enough at pool scale
  that it is what the test suite runs against.

The difference is real and this module does not paper over it. The local index
is *lexical*: it can only match words the record actually contains. It is a
working fallback, not an equivalent, and :func:`build_pool_index` reports
which one answered so ``/api/status`` and the dashboard can say so.

A note on the history here
--------------------------
An earlier version of this file talked about Moss in the present tense while
every Moss branch in it was unreachable -- written against a guessed API, no
credentials ever configured, silently falling through to the fallback on every
call. This one is written against the installed SDK's actual surface
(``moss`` 1.12: ``MossClient(project_id, project_key)``, ``create_index``,
``add_docs``, ``delete_docs``, ``load_index``, ``query``) and, until it has run
against a live project, says so rather than implying otherwise.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.config import Settings, get_settings
from app.security.pii_scrubber import scrub_text
from app.security.token_guard import redact_credentials

logger = logging.getLogger(__name__)

#: Words that appear in every record and carry no discriminating signal. Kept
#: short on purpose: an aggressive stop list throws away terms that are
#: genuinely rare in a *technical* corpus even if common in English.
_STOPWORD_TEXT = (
    "a an and are as at be been by for from has have in is it its of on or "
    "that the to was were will with"
)
_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split(" "))

#: Technical identifiers keep their punctuation: "c++", "node.js", "p99",
#: "gpt-4" are single terms, and splitting them loses the match entirely.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True, slots=True)
class PoolHit:
    """One retrieved candidate."""

    candidate_id: str
    score: float


@dataclass(frozen=True, slots=True)
class Retrieval:
    """What a search returned, and how."""

    hits: tuple[PoolHit, ...]
    backend: str
    latency_ms: float

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(h.candidate_id for h in self.hits)


def document_text(candidate: Any) -> str:
    """The indexable text for one candidate.

    Name and headline are repeated ahead of the record because they are the
    terms a manager actually types ("is Priya worth a call?"), and a single
    mention inside a 2KB blob is not enough for either backend to surface
    them. The rest is the scraped record as stored -- already redacted, since
    everything downstream of ``app.ingest`` is.
    """
    name = getattr(candidate, "name", "") or ""
    headline = getattr(candidate, "headline", "") or ""
    source = getattr(candidate, "source", None) or {}
    return f"{name}. {headline}. {name}. {_flatten(source)}".strip()


def _flatten(value: Any, depth: int = 0) -> str:
    """Render arbitrary scraped JSON to searchable text.

    Keys are included, not just values: a record whose only mention of
    security is the key ``"security_clearance"`` should still match a question
    about security.
    """
    if depth > 6:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten(v, depth + 1)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v, depth + 1) for v in value)
    if value is None or isinstance(value, bool):
        return ""
    return str(value)


@runtime_checkable
class PoolIndex(Protocol):
    """Anything that can answer 'which candidates are relevant to this?'."""

    @property
    def backend(self) -> str: ...

    async def sync(self, candidates: Sequence[Any]) -> int: ...

    async def search(self, query: str, candidates: Sequence[Any], limit: int) -> Retrieval: ...


# =============================================================================
# Local index
# =============================================================================


class LocalIndex:
    """TF-IDF cosine similarity over the pool, in process.

    Rebuilt per search from the rows it is handed rather than held as mutable
    state. At pool scale that costs microseconds, and it removes the entire
    class of bug where the index and the database disagree about who exists --
    which for a hiring tool means answering a question about a candidate the
    manager deleted.

    Sublinear term frequency (``1 + log tf``) and smoothed IDF, because a
    record that says "Kubernetes" nine times is not nine times as relevant as
    one that says it once, and a scraped skills array repeats everything.
    """

    @property
    def backend(self) -> str:
        return "local (lexical)"

    async def sync(self, candidates: Sequence[Any]) -> int:
        # Nothing to do: the index is derived from the rows at search time.
        return len(candidates)

    async def search(self, query: str, candidates: Sequence[Any], limit: int) -> Retrieval:
        started = time.perf_counter()
        terms = tokenize(query)
        rows = list(candidates)
        if not terms or not rows:
            return Retrieval((), self.backend, (time.perf_counter() - started) * 1000.0)

        docs = [Counter(tokenize(document_text(row))) for row in rows]
        n = len(docs)
        idf = {
            term: math.log((n + 1) / (1 + sum(1 for d in docs if term in d))) + 1.0
            for term in set(terms)
        }

        scored: list[PoolHit] = []
        for row, counts in zip(rows, docs, strict=True):
            if not counts:
                continue
            # Cosine needs the full document norm, not just the query terms.
            norm = math.sqrt(sum((1.0 + math.log(c)) ** 2 for c in counts.values()))
            if norm == 0.0:
                continue
            dot = sum(
                (1.0 + math.log(counts[term])) * idf[term] for term in set(terms) if term in counts
            )
            if dot > 0.0:
                scored.append(PoolHit(candidate_id=row.id, score=dot / norm))

        scored.sort(key=lambda h: -h.score)
        return Retrieval(
            hits=tuple(scored[:limit]),
            backend=self.backend,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


# =============================================================================
# Moss
# =============================================================================


@dataclass
class MossIndex:
    """Semantic retrieval over the pool via the Moss runtime.

    Every failure degrades to :class:`LocalIndex` rather than raising. A
    manager asking a question should get a lexically-matched answer and a
    dashboard that says retrieval is degraded, not a 500 -- and the same rule
    the model client follows applies here: a fallback is *counted*, because a
    configured backend that silently never answers is indistinguishable from a
    healthy one.
    """

    settings: Settings
    fallback: LocalIndex = field(default_factory=LocalIndex)
    _client: Any | None = field(default=None, init=False)
    _loaded: bool = field(default=False, init=False)
    _degraded: bool = field(default=False, init=False)
    _indexed_ids: set[str] = field(default_factory=set, init=False)

    @property
    def backend(self) -> str:
        if self._degraded:
            return "local (moss degraded)"
        return "moss (semantic)" if self._loaded else "moss (not yet indexed)"

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:  # pragma: no cover - requires the optional moss dependency
            from moss import MossClient  # type: ignore[import-not-found]

            self._client = MossClient(self.settings.moss_project_id, self.settings.moss_project_key)
        except Exception as exc:  # pragma: no cover - degradation path
            logger.warning(
                "Moss client unavailable (%s: %s); retrieval uses the local index.",
                type(exc).__name__,
                redact_credentials(str(exc))[:300],
            )
            self._client = None
            self._degraded = True
        return self._client

    def _documents(self, candidates: Sequence[Any]) -> list[Any]:
        """Build Moss documents. Scrubbed again at the boundary, because this
        is a network egress and re-scanning costs less than trusting a flag."""
        from moss import DocumentInfo  # type: ignore[import-not-found]

        out = []
        for row in candidates:
            out.append(
                DocumentInfo(
                    id=row.id,
                    text=scrub_text(redact_credentials(document_text(row)))[:20000],
                    metadata={"name": scrub_text(row.name or "")},
                )
            )
        return out

    async def sync(self, candidates: Sequence[Any]) -> int:
        """Make the remote index match the pool, then load it for querying."""
        client = self._ensure_client()
        if client is None:
            return await self.fallback.sync(candidates)

        name = self.settings.moss_index
        try:  # pragma: no cover - requires a live Moss project
            docs = self._documents(candidates)
            current = {row.id for row in candidates}

            if not self._indexed_ids:
                # First sync of this process. create_index is idempotent from
                # our side in the sense that we fall back to add_docs when the
                # index already exists -- we do not delete and rebuild, which
                # would drop another worker's documents.
                try:
                    await client.create_index(name, docs)
                except Exception:
                    if docs:
                        await client.add_docs(name, docs)
            else:
                new = [d for d in docs if d.id not in self._indexed_ids]
                if new:
                    await client.add_docs(name, new)
                gone = sorted(self._indexed_ids - current)
                if gone:
                    await client.delete_docs(name, gone)

            await client.load_index(name)
            self._indexed_ids = current
            self._loaded = True
            self._degraded = False
            logger.info("Moss index '%s' holds %d candidate records.", name, len(current))
            return len(current)
        except Exception as exc:  # pragma: no cover - degradation path
            record_retrieval_fallback()
            logger.warning(
                "Moss sync failed (%s: %s); retrieval uses the local index. Answers "
                "stay lexical until this recovers.",
                type(exc).__name__,
                redact_credentials(str(exc))[:300],
            )
            self._loaded = False
            self._degraded = True
            return await self.fallback.sync(candidates)

    async def search(self, query: str, candidates: Sequence[Any], limit: int) -> Retrieval:
        client = self._ensure_client()
        if client is None or not self._loaded:
            return await self.fallback.search(query, candidates, limit)

        try:  # pragma: no cover - requires a live Moss project
            from moss import QueryOptions  # type: ignore[import-not-found]

            known = {row.id for row in candidates}
            started = time.perf_counter()
            result = await client.query(
                self.settings.moss_index,
                scrub_text(query),
                QueryOptions(top_k=limit),
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            # Prefer the runtime's own measurement when it reports one: it
            # excludes our await-scheduling overhead.
            latency = float(getattr(result, "time_taken_ms", None) or elapsed)

            hits = []
            for doc in getattr(result, "docs", None) or []:
                doc_id = str(getattr(doc, "id", ""))
                # A document can outlive its row if a delete did not reach the
                # index. Dropping unknown ids here means a deleted candidate
                # can never reappear in an answer.
                if doc_id in known:
                    hits.append(PoolHit(doc_id, float(getattr(doc, "score", 0.0) or 0.0)))
            return Retrieval(tuple(hits[:limit]), self.backend, latency)
        except Exception as exc:  # pragma: no cover - degradation path
            record_retrieval_fallback()
            logger.warning(
                "Moss query failed (%s: %s); answering from the local index.",
                type(exc).__name__,
                redact_credentials(str(exc))[:300],
            )
            self._degraded = True
            return await self.fallback.search(query, candidates, limit)


# =============================================================================
# Degradation counter
# =============================================================================

_retrieval_fallbacks = 0


def record_retrieval_fallback() -> None:
    global _retrieval_fallbacks
    _retrieval_fallbacks += 1


def retrieval_fallback_count() -> int:
    return _retrieval_fallbacks


def reset_retrieval_fallbacks() -> None:
    """Test isolation only."""
    global _retrieval_fallbacks
    _retrieval_fallbacks = 0


def build_pool_index(settings: Settings | None = None) -> PoolIndex:
    settings = settings or get_settings()
    if settings.moss_configured:
        return MossIndex(settings)
    return LocalIndex()


__all__ = [
    "LocalIndex",
    "MossIndex",
    "PoolHit",
    "PoolIndex",
    "Retrieval",
    "build_pool_index",
    "document_text",
    "record_retrieval_fallback",
    "reset_retrieval_fallbacks",
    "retrieval_fallback_count",
    "tokenize",
]
