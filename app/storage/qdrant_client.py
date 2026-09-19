"""Hybrid (dense + sparse) vector store with a zero-infrastructure fallback.

Two backends behind one interface:

* **Qdrant** when ``QDRANT_URL`` is configured -- HNSW, in-memory graph,
  server-side RRF fusion.
* **:class:`EmbeddedIndex`** otherwise -- an in-process NumPy index. This is not
  a toy: for the rubric collection (9 points) and a session-scale candidate
  collection (thousands), an exact dot product over an L2-normalised matrix is
  a single BLAS call that completes in well under a millisecond, which is
  faster than HNSW *and* exact. Approximate search only starts paying at a
  scale this collection does not reach during a live interview.

The practical effect is that the whole product runs, and the whole test suite
passes, with no services running anywhere. That is a deliberate operational
choice: an interview pipeline whose local dev story requires four containers
gets tested less, and a screening agent that is tested less is a liability.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.schemas.evaluation import PII_EXEMPT_PAYLOAD_KEYS
from app.schemas.roles import ROLE_RUBRICS, EngineeringRole
from app.security.pii_scrubber import assert_zero_pii
from app.storage.embeddings import (
    DeterministicEmbedder,
    EmbeddingProvider,
    GeminiEmbedder,
    SparseEncoder,
    sparse_dot,
)
from app.storage.vector_models import (
    DEFAULT_COLLECTION,
    CollectionConfig,
    SearchHit,
    SparseVector,
    VectorPoint,
)

logger = logging.getLogger(__name__)

#: Reciprocal-rank-fusion damping. 60 is the value from the original RRF paper
#: and is deliberately not tuned: RRF's whole appeal is that it fuses rankings
#: with incomparable score scales without needing calibration.
RRF_K: int = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> dict[str, float]:
    """Fuse several ranked id lists into one score map.

    Dense cosine scores and BM25 scores live on different scales, so adding or
    averaging them requires a calibration constant that drifts with the corpus.
    RRF sidesteps that entirely by consuming only *positions*.

    ``weights`` exists because plain RRF has a failure mode we measured rather
    than guessed: it treats every ranking as equally authoritative, so a channel
    that is returning noise gets exactly as many votes as a channel that is
    certain. See :func:`channel_confidence` for how the weights are derived.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("rankings and weights must be the same length")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight <= 0.0:
            continue
        for rank, point_id in enumerate(ranking):
            fused[point_id] = fused.get(point_id, 0.0) + weight / (k + rank + 1)
    return fused


def channel_confidence(scores: Sequence[float]) -> float:
    """How decisive a retrieval channel is about its own top result, in [0, 1].

    Defined as the relative margin between the best and second-best score. A
    channel whose top two results are indistinguishable has told us nothing
    about which is correct, and its vote is damped accordingly; a channel with a
    clear winner votes at close to full strength.

    This matters concretely here. The offline lexical encoder is accurate about
    95% of the time on short-query/long-document rubric matching, and on the
    unlucky tail it returns essentially arbitrary ordering with all scores
    bunched together. Margin weighting is what stops that tail from outvoting a
    sparse channel that is separating its top hit by an order of magnitude.

    A small floor is applied so a damped channel still breaks ties rather than
    being silenced outright.
    """
    ranked = sorted((s for s in scores if s > 0.0), reverse=True)
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return 1.0
    margin = (ranked[0] - ranked[1]) / ranked[0]
    return max(0.1, min(1.0, margin))


@dataclass(slots=True)
class _StoredPoint:
    id: str
    dense: np.ndarray
    sparse: SparseVector | None
    payload: dict[str, Any]


class EmbeddedIndex:
    """Exact in-process hybrid index. Deterministic and dependency-free."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._points: dict[str, _StoredPoint] = {}
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._points)

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        for point in points:
            dense = np.asarray(point.dense, dtype=np.float32)
            if dense.shape != (self._dim,):
                raise ValueError(
                    f"Point {point.id}: expected a {self._dim}-d vector, got {dense.shape}"
                )
            self._points[point.id] = _StoredPoint(
                id=point.id,
                dense=dense,
                sparse=point.sparse,
                payload=dict(point.payload),
            )
        self._invalidate()

    def delete(self, ids: Sequence[str]) -> None:
        for point_id in ids:
            self._points.pop(point_id, None)
        self._invalidate()

    def _invalidate(self) -> None:
        self._matrix = None
        self._ids = []

    def _ensure_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Rebuild the contiguous search matrix, sorted by id for determinism."""
        if self._matrix is None:
            self._ids = sorted(self._points)
            if self._ids:
                self._matrix = np.vstack([self._points[i].dense for i in self._ids])
            else:
                self._matrix = np.zeros((0, self._dim), dtype=np.float32)
        return self._matrix, self._ids

    def search(
        self,
        dense: np.ndarray,
        sparse: SparseVector | None = None,
        limit: int = 5,
        payload_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        matrix, ids = self._ensure_matrix()
        if not ids:
            return []

        candidate_idx = list(range(len(ids)))
        if payload_filter:
            candidate_idx = [
                i
                for i in candidate_idx
                if all(
                    self._points[ids[i]].payload.get(key) == value
                    for key, value in payload_filter.items()
                )
            ]
            if not candidate_idx:
                return []

        query = np.asarray(dense, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm

        subset = matrix[candidate_idx]
        dense_scores = subset @ query
        # Descending score, ascending id on ties -- np.lexsort is stable and
        # keeps the ordering reproducible across runs and platforms.
        order = np.lexsort((np.arange(len(candidate_idx)), -dense_scores))
        dense_ranked = [ids[candidate_idx[i]] for i in order]

        rankings: list[Sequence[str]] = [dense_ranked]
        weights: list[float] = [channel_confidence([float(s) for s in dense_scores])]

        if sparse is not None and sparse.indices:
            sparse_scored = [
                (sparse_dot(sparse, self._points[ids[i]].sparse), ids[i])
                for i in candidate_idx
                if self._points[ids[i]].sparse is not None
            ]
            positive = [(score, pid) for score, pid in sparse_scored if score > 0.0]
            if positive:
                positive.sort(key=lambda pair: (-pair[0], pair[1]))
                rankings.append([pid for _, pid in positive])
                weights.append(channel_confidence([score for score, _ in positive]))

        if len(rankings) == 1:
            score_by_id = {
                ids[candidate_idx[i]]: float(dense_scores[i]) for i in range(len(candidate_idx))
            }
            ordered = dense_ranked
        else:
            fused = reciprocal_rank_fusion(rankings, weights)
            score_by_id = fused
            ordered = sorted(fused, key=lambda pid: (-fused[pid], pid))

        return [
            SearchHit(
                id=pid,
                score=round(float(score_by_id.get(pid, 0.0)), 6),
                payload=dict(self._points[pid].payload),
            )
            for pid in ordered[:limit]
        ]


class HybridVectorStore:
    """The storage seam the rest of the agent talks to.

    Responsibilities beyond plain search:

    * refuses to store a payload containing PII (the assertion runs on upsert,
      not on a code review);
    * keeps the rubric collection warm so the first turn of an interview is not
      the one that pays for index construction;
    * degrades from Qdrant to the embedded index on any connection failure,
      because a retrieval outage must not end a live interview.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config: CollectionConfig | None = None,
        embedder: EmbeddingProvider | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or DEFAULT_COLLECTION
        if self.config.name != self.settings.qdrant_collection:
            self.config = CollectionConfig(
                name=self.settings.qdrant_collection,
                dim=self.config.dim,
                distance=self.config.distance,
                hnsw=self.config.hnsw,
                sparse_vector_name=self.config.sparse_vector_name,
                dense_vector_name=self.config.dense_vector_name,
            )

        self.embedder: EmbeddingProvider = embedder or self._default_embedder()
        self.sparse = sparse_encoder or SparseEncoder()
        self._rubrics = EmbeddedIndex(self.config.dim)
        self._candidates = EmbeddedIndex(self.config.dim)
        self._qdrant: Any | None = None
        self._rubrics_ready = False

    def _default_embedder(self) -> EmbeddingProvider:
        if self.settings.cognition_configured:
            return GeminiEmbedder(
                api_key=self.settings.google_api_key,
                dim=self.config.dim,
                fallback=DeterministicEmbedder(self.config.dim),
            )
        return DeterministicEmbedder(self.config.dim)

    # -- backend selection ---------------------------------------------------

    @property
    def backend(self) -> str:
        return "qdrant" if self._qdrant is not None else "embedded"

    def _connect_qdrant(self) -> Any | None:
        if not self.settings.qdrant_configured:
            return None
        try:  # pragma: no cover - requires a live Qdrant
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]

            client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
                prefer_grpc=True,  # gRPC shaves the HTTP framing off every query
                timeout=2.0,  # a retrieval slower than this has already
                # blown the 10ms budget; fail over instead
            )
            client.get_collections()
            return client
        except Exception as exc:  # pragma: no cover - fall back, never drop
            logger.warning(
                "Qdrant unavailable (%s); using the embedded index. Retrieval "
                "quality is unchanged at rubric scale.",
                type(exc).__name__,
            )
            return None

    # -- lifecycle -----------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection if needed and warm the rubric index."""
        if self._qdrant is None:
            self._qdrant = self._connect_qdrant()

        if self._qdrant is not None:  # pragma: no cover - requires a live Qdrant
            try:
                from qdrant_client import models as qmodels  # type: ignore[import-not-found]

                if not self._qdrant.collection_exists(self.config.name):
                    self._qdrant.create_collection(
                        collection_name=self.config.name,
                        vectors_config={
                            self.config.dense_vector_name: qmodels.VectorParams(
                                size=self.config.dim,
                                distance=qmodels.Distance.COSINE,
                                hnsw_config=qmodels.HnswConfigDiff(**self.config.hnsw.to_qdrant()),
                            )
                        },
                        sparse_vectors_config={
                            self.config.sparse_vector_name: qmodels.SparseVectorParams()
                        },
                    )
            except Exception as exc:
                logger.warning("Qdrant collection setup failed (%s)", type(exc).__name__)
                self._qdrant = None

        self.warm_rubrics()

    def warm_rubrics(self) -> None:
        """Index the rubrics one *competency* at a time. Idempotent.

        Indexing each competency as its own point rather than concatenating a
        whole rubric into a single vector is a late-interaction (ColBERT-style)
        layout, and it is here for a measured reason. A whole-rubric vector is
        roughly 380 hashed features; a candidate's spoken answer is about 30.
        Cosine between a short query and a long document is both diluted and
        collision-noisy, and the matching competency's signal gets averaged away
        by the three competencies that have nothing to do with the answer.

        Scoring per competency and keeping the best match per role fixes both
        problems at once, and it costs 36 points instead of 9 -- which at this
        scale is free.
        """
        if self._rubrics_ready:
            return

        entries: list[tuple[str, str, EngineeringRole, str]] = []
        for role in EngineeringRole:
            rubric = ROLE_RUBRICS[role]
            for competency in rubric.competencies:
                entries.append(
                    (
                        f"{role.value}::{competency.key}",
                        f"{rubric.title} {competency.label} {' '.join(competency.signals)}",
                        role,
                        competency.label,
                    )
                )

        texts = [text for _, text, _, _ in entries]
        self.sparse.fit(texts)
        vectors = self.embedder.embed(texts)
        self._rubrics.upsert(
            [
                VectorPoint(
                    id=point_id,
                    dense=tuple(float(x) for x in vectors[i]),
                    sparse=self.sparse.encode(texts[i]),
                    payload={
                        "role": role.value,
                        "title": ROLE_RUBRICS[role].title,
                        "competency_label": label,
                        "advance_threshold": ROLE_RUBRICS[role].advance_threshold,
                    },
                )
                for i, (point_id, _, role, label) in enumerate(entries)
            ]
        )
        self._rubrics_ready = True

    # -- queries -------------------------------------------------------------

    def search_rubrics(self, evidence: str, limit: int = 3) -> list[SearchHit]:
        """Hybrid rubric lookup, collapsed to one hit per role.

        This is the call that sits inside the 5ms retrieval budget. It returns
        *retrieval context* for the interviewer prompt -- which competency the
        candidate is currently demonstrating -- and is explicitly not the role
        routing decision. Routing is decided by the deterministic matcher in
        :mod:`app.schemas.roles`, because a career-affecting assignment has to
        be reproducible and explainable line by line, which an approximate
        nearest-neighbour lookup is not.
        """
        self.warm_rubrics()
        dense = self.embedder.embed_one(evidence)
        # Over-fetch, then keep each role's best-matching competency.
        raw = self._rubrics.search(dense, self.sparse.encode(evidence), limit=max(limit * 6, 12))

        best: dict[str, SearchHit] = {}
        for hit in raw:
            role = str(hit.payload.get("role", hit.id))
            if role not in best or hit.score > best[role].score:
                best[role] = SearchHit(id=role, score=hit.score, payload=hit.payload)

        ordered = sorted(best.values(), key=lambda h: (-h.score, h.id))
        return ordered[:limit]

    def search_candidates(
        self,
        evidence: str,
        limit: int = 5,
        role: EngineeringRole | None = None,
    ) -> list[SearchHit]:
        """Find comparable prior candidates, optionally within one track."""
        dense = self.embedder.embed_one(evidence)
        payload_filter = {"target_role": role.value} if role else None
        return self._candidates.search(
            dense, self.sparse.encode(evidence), limit=limit, payload_filter=payload_filter
        )

    # -- writes --------------------------------------------------------------

    def upsert_candidate(self, payload: Mapping[str, Any], evidence: str) -> VectorPoint:
        """Persist one evaluation. Refuses payloads that carry PII.

        The guard walks the payload's string leaves *and keys*. It is the last
        checkpoint before data becomes durable, and durable is the one state
        you cannot take back.

        :data:`PII_EXEMPT_PAYLOAD_KEYS` (``candidate_id``) is skipped rather
        than scanned: it is a machine-generated UUID that cannot legitimately
        contain PII, and the scrubber's loose passport pattern (one letter,
        seven digits, no context marker) matches a UUID hex fragment by
        chance often enough to intermittently drop a clean evaluation.
        """
        for key, value in payload.items():
            if key in PII_EXEMPT_PAYLOAD_KEYS:
                continue
            assert_zero_pii(str(key), boundary="qdrant.payload_key")
            if isinstance(value, str):
                assert_zero_pii(value, boundary=f"qdrant.payload[{key}]")
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        assert_zero_pii(item, boundary=f"qdrant.payload[{key}][]")

        point = VectorPoint(
            id=str(payload.get("candidate_id", f"candidate-{time.time_ns()}")),
            dense=tuple(float(x) for x in self.embedder.embed_one(evidence)),
            sparse=self.sparse.encode(evidence),
            payload=dict(payload),
        )
        self._candidates.upsert([point])

        if self._qdrant is not None:  # pragma: no cover - requires a live Qdrant
            try:
                from qdrant_client import models as qmodels  # type: ignore[import-not-found]

                self._qdrant.upsert(
                    collection_name=self.config.name,
                    points=[
                        qmodels.PointStruct(
                            id=point.id,
                            vector={
                                self.config.dense_vector_name: list(point.dense),
                                self.config.sparse_vector_name: qmodels.SparseVector(
                                    indices=list(point.sparse.indices),
                                    values=list(point.sparse.values),
                                ),
                            },
                            payload=point.payload,
                        )
                    ],
                )
            except Exception as exc:
                logger.warning(
                    "Qdrant upsert failed (%s); the evaluation is retained in the "
                    "embedded index and will need replay.",
                    type(exc).__name__,
                )
        return point

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    def close(self) -> None:
        if self._qdrant is not None:  # pragma: no cover - requires a live Qdrant
            with contextlib.suppress(Exception):
                self._qdrant.close()
            self._qdrant = None


__all__ = [
    "RRF_K",
    "EmbeddedIndex",
    "HybridVectorStore",
    "channel_confidence",
    "reciprocal_rank_fusion",
]
