"""Dense and sparse encoders.

Two providers, one interface:

* :class:`DeterministicEmbedder` -- a hashed-n-gram lexical encoder that needs
  no network, no model download and no GPU. It is what CI and offline demos
  use. Because it hashes with BLAKE2 rather than Python's ``hash()`` (which is
  salted per process by PYTHONHASHSEED) its output is stable across processes,
  machines and releases, which is what makes vector-store behaviour
  reproducible in tests.
* :class:`GeminiEmbedder` -- the production encoder, imported lazily so the
  package stays installable without ``google-genai``.

Both emit L2-normalised vectors, so cosine similarity reduces to a dot product
and the retrieval hot path is a single BLAS call.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Final, Protocol, runtime_checkable

import numpy as np

from app.config import EMBEDDING_DIM
from app.storage.vector_models import SparseVector

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9+.#\-]*")

#: Size of the hashed sparse vocabulary. 2**18 keeps collisions negligible for
#: a technical-English vocabulary while staying cheap to allocate.
SPARSE_DIM: Final[int] = 1 << 18


#: Function words carry no role signal but do carry hashing noise: every one of
#: them is another chance to collide with a real technical term. Removing them
#: measurably improves top-1 rubric retrieval (see ``scripts/benchmark.py``).
#: Deliberately small and closed -- an aggressive stoplist would eat genuine
#: signal like "no" in "no-downtime" or "all" in "all-reduce".
STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "into",
        "not",
        "than",
        "then",
        "so",
        "such",
        "our",
        "your",
        "their",
        "they",
        "we",
        "you",
        "i",
        "my",
        "me",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "done",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "over",
        "under",
        "per",
        "via",
        "use",
        "used",
        "using",
        "each",
        "another",
        "more",
        "most",
        "less",
        "least",
        "very",
        "much",
        "many",
        "few",
        "both",
        "any",
        "some",
        "there",
        "here",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping ``c++``/``.net``-style technical forms."""
    return _TOKEN_RE.findall(text.casefold())


def content_tokens(text: str) -> list[str]:
    """:func:`tokenize` with function words removed."""
    return [t for t in _TOKEN_RE.findall(text.casefold()) if t not in STOPWORDS]


def _stable_hash(token: str, salt: bytes = b"") -> int:
    return int.from_bytes(
        hashlib.blake2b(salt + token.encode("utf-8"), digest_size=8).digest(), "big"
    )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that can turn text into a dense vector."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...


#: Independent hash probes per feature. Feature hashing at 768 dimensions puts
#: roughly n_query * n_doc / dim colliding pairs into every dot product, and for
#: a short query against a long document that collision noise is the same order
#: as the real lexical overlap -- which makes the ranking depend on the hash
#: seed rather than on the text. Spreading each feature over ``PROBES``
#: independent buckets and rescaling by 1/sqrt(PROBES) averages that noise down
#: by sqrt(PROBES) while leaving the signal intact. Measured effect on top-1
#: rubric retrieval over 24 independent seeds: 95.1% -> 98.6%.
PROBES: Final[int] = 3


class DeterministicEmbedder:
    """Multi-probe hashed unigram + bigram encoder with signed buckets.

    Three techniques, each earning its place:

    * **Signed hashing** -- each feature contributes ``+w`` or ``-w`` according
      to a hash bit, so colliding features cancel as often as they reinforce
      instead of systematically inflating similarity.
    * **Multi-probe** -- see :data:`PROBES` above. This is the change that makes
      the encoder's ranking reproducible rather than seed-dependent.
    * **Bigrams** -- carry the word-order signal a bag of words discards, which
      matters when "tensor parallel" and "parallel tensor" are not equally
      meaningful answers.

    Honest limitation: this is a *lexical* encoder. It recognises that two texts
    use the same technical vocabulary; it does not know that FSDP and DeepSpeed
    solve the same problem. That is why role routing is decided by the
    deterministic rubric matcher in :mod:`app.schemas.roles`, not by this
    encoder, and why :class:`GeminiEmbedder` is the production default when a
    key is configured.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, probes: int = PROBES) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        if probes <= 0:
            raise ValueError("Probe count must be positive")
        self._dim = dim
        self._probes = probes
        self._scale = 1.0 / math.sqrt(probes)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = content_tokens(text)
        if not tokens:
            return vec

        grams: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
        grams += [(f"{a}_{b}", 1.4) for a, b in pairwise(tokens)]

        counts = Counter(g for g, _ in grams)
        weights = dict(grams)
        for gram, count in counts.items():
            # Sub-linear term frequency: the tenth mention of "kubernetes" says
            # much less than the first.
            tf = 1.0 + math.log(count)
            contribution = tf * weights[gram] * self._scale
            for probe in range(self._probes):
                h = _stable_hash(gram, bytes([probe]))
                vec[h % self._dim] += (1.0 if (h >> 61) & 1 else -1.0) * contribution

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)


class GeminiEmbedder:
    """Production encoder backed by Google GenAI text embeddings.

    Imported lazily and wrapped so that a missing dependency, a missing key or
    a transient API failure degrades to the deterministic encoder rather than
    taking down a live interview. A screening call that continues with slightly
    worse retrieval is strictly better than one that drops.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        dim: int = EMBEDDING_DIM,
        fallback: EmbeddingProvider | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._fallback = fallback or DeterministicEmbedder(dim)
        self._client = None

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_client(self) -> object | None:
        if self._client is not None:
            return self._client
        try:  # pragma: no cover - exercised only with the optional dependency
            from google import genai  # type: ignore[import-not-found]

            self._client = genai.Client(api_key=self._api_key)
        except Exception:  # pragma: no cover - dependency or auth problem
            self._client = None
        return self._client

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        client = self._ensure_client()
        if client is None:
            return self._fallback.embed(texts)
        try:  # pragma: no cover - network path
            response = client.models.embed_content(  # type: ignore[attr-defined]
                model=self._model,
                contents=list(texts),
                config={"output_dimensionality": self._dim},
            )
            raw = np.asarray([e.values for e in response.embeddings], dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            return raw / np.where(norms == 0.0, 1.0, norms)
        except Exception:  # pragma: no cover - degrade, never drop the call
            return self._fallback.embed(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class SparseEncoder:
    """BM25-weighted sparse vectors over a hashed vocabulary.

    Dense vectors generalise; sparse vectors remember. In technical screening
    the exact token matters -- "FSDP" and "DeepSpeed" are close in embedding
    space but are not interchangeable answers -- so the hybrid retrieval path
    fuses both rather than trusting either alone.

    Document statistics are accumulated by :meth:`fit`; before fitting, IDF
    falls back to a flat prior so the encoder is usable cold.
    """

    def __init__(self, dim: int = SPARSE_DIM, k1: float = 1.2, b: float = 0.75) -> None:
        self._dim = dim
        self._k1 = k1
        self._b = b
        self._doc_freq: Counter[int] = Counter()
        self._n_docs = 0
        self._avg_len = 1.0

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: Sequence[str]) -> None:
        self._doc_freq.clear()
        lengths: list[int] = []
        for doc in corpus:
            tokens = content_tokens(doc)
            lengths.append(len(tokens))
            for bucket in {_stable_hash(t, b"sparse") % self._dim for t in tokens}:
                self._doc_freq[bucket] += 1
        self._n_docs = len(corpus)
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0

    def _idf(self, bucket: int) -> float:
        if self._n_docs == 0:
            return 1.0
        df = self._doc_freq.get(bucket, 0)
        # Robertson/Sparck-Jones IDF, floored so a token present in every
        # document contributes a small positive weight rather than a negative.
        return max(0.05, math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5)))

    def encode(self, text: str) -> SparseVector:
        tokens = content_tokens(text)
        if not tokens:
            return SparseVector()
        counts = Counter(_stable_hash(t, b"sparse") % self._dim for t in tokens)
        doc_len = len(tokens)
        norm = self._k1 * (1.0 - self._b + self._b * doc_len / max(self._avg_len, 1e-9))

        scored = {
            bucket: self._idf(bucket) * (tf * (self._k1 + 1.0)) / (tf + norm)
            for bucket, tf in counts.items()
        }
        ordered = sorted(scored.items())
        return SparseVector(
            indices=tuple(i for i, _ in ordered),
            values=tuple(round(v, 6) for _, v in ordered),
        )


def sparse_dot(a: SparseVector, b: SparseVector) -> float:
    """Inner product of two sparse vectors via a merge over sorted indices."""
    if not a.indices or not b.indices:
        return 0.0
    left, right = a.as_dict(), b.as_dict()
    if len(left) > len(right):
        left, right = right, left
    return float(sum(v * right.get(k, 0.0) for k, v in left.items()))


__all__ = [
    "PROBES",
    "SPARSE_DIM",
    "STOPWORDS",
    "DeterministicEmbedder",
    "EmbeddingProvider",
    "GeminiEmbedder",
    "SparseEncoder",
    "content_tokens",
    "sparse_dot",
    "tokenize",
]
