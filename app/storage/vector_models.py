"""Vector-store data contracts: collection config, points, and search hits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from app.config import EMBEDDING_DIM


class Distance(StrEnum):
    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"


@dataclass(frozen=True, slots=True)
class HnswConfig:
    """HNSW graph parameters.

    ``on_disk=False`` is not a default we drifted into -- it is the reason the
    10ms retrieval budget is achievable. An on-disk graph turns every hop into
    a potential page fault, and HNSW does ``ef`` hops per query. For a rubric
    collection measured in thousands of points the entire graph is a few MB of
    RAM, so there is nothing to trade.
    """

    m: int = 16
    ef_construct: int = 100
    full_scan_threshold: int = 10_000
    on_disk: bool = False

    def to_qdrant(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "ef_construct": self.ef_construct,
            "full_scan_threshold": self.full_scan_threshold,
            "on_disk": self.on_disk,
        }


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    name: str = "engineering_talent_rubrics"
    dim: int = EMBEDDING_DIM
    distance: Distance = Distance.COSINE
    hnsw: HnswConfig = field(default_factory=HnswConfig)
    #: Name of the sparse (BM25) vector, stored alongside the dense one.
    sparse_vector_name: str = "bm25"
    dense_vector_name: str = "dense"


DEFAULT_COLLECTION: Final[CollectionConfig] = CollectionConfig()


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Sparse BM25 vector in Qdrant's index/value form."""

    indices: tuple[int, ...] = ()
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("SparseVector indices and values must be the same length")

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """A stored point: id, dense vector, optional sparse vector, payload.

    ``payload`` is required to be free of identifiers; the store asserts this
    on upsert rather than trusting the caller.
    """

    id: str
    dense: tuple[float, ...]
    payload: Mapping[str, Any]
    sparse: SparseVector | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    payload: Mapping[str, Any]


__all__ = [
    "DEFAULT_COLLECTION",
    "CollectionConfig",
    "Distance",
    "HnswConfig",
    "SearchHit",
    "SparseVector",
    "VectorPoint",
]
