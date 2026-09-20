"""Retrieval over the candidate pool.

Two things are being defended here.

First, that retrieval actually *retrieves*: a question about security finds
the security engineer and not the mobile developer. A ranking layer that
returns the pool in arbitrary order is worse than no ranking layer, because
the model downstream will confidently answer from whatever it was handed.

Second, that Moss degrades rather than fails. The Moss code path cannot run
in CI -- it needs a live project -- so what is tested here is the contract
around it: that a broken client falls back, that the fallback is counted, and
that a candidate deleted from the pool can never come back in an answer even
if the remote index still holds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from app.config import Settings
from app.retrieval import (
    LocalIndex,
    MossIndex,
    build_pool_index,
    document_text,
    retrieval_fallback_count,
    tokenize,
)


@dataclass
class _Row:
    id: str
    name: str
    headline: str = ""
    source: dict[str, Any] = field(default_factory=dict)


POOL = [
    _Row(
        id="sec",
        name="Sofia Kowalski",
        headline="Security Engineer",
        source={
            "skills": ["threat modelling", "eBPF", "Go"],
            "detail": "Found and fixed an auth bypass in the partner API token scoping.",
        },
    ),
    _Row(
        id="mob",
        name="Ravi Menon",
        headline="Mobile Engineer",
        source={"skills": ["Kotlin", "Swift"], "detail": "Reduced cold start by 20%."},
    ),
    _Row(
        id="dist",
        name="Daniel Okafor",
        headline="Staff Backend Engineer",
        source={
            "skills": ["Go", "Kafka", "Postgres"],
            "detail": "Ledger at 40k req/s, exactly-once settlement, cut p99 900ms to 120ms.",
        },
    ),
    _Row(
        id="thin",
        name="Jonas Weber",
        headline="Full-stack Engineer",
        source={"detail": "Shipped a dashboard."},
    ),
]


# ---- Tokenizing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Node.js and C++", ["node.js", "c++"]),
        ("p99 latency", ["p99", "latency"]),
        ("GPT-4 fine-tuning", ["gpt-4", "fine-tuning"]),
    ],
)
def test_technical_identifiers_survive_tokenizing(text: str, expected: list[str]) -> None:
    """Splitting "node.js" into "node" and "js" loses the match entirely."""
    assert tokenize(text) == expected


def test_stopwords_are_dropped() -> None:
    assert "the" not in tokenize("the engineer and the manager")


# ---- Document text ----------------------------------------------------------


def test_document_text_includes_nested_values() -> None:
    text = document_text(POOL[0])
    assert "ebpf" in text.lower()
    assert "auth bypass" in text.lower()


def test_document_text_includes_keys_not_just_values() -> None:
    """A record whose only mention of security is the key
    `security_clearance` should still match a question about security."""
    row = _Row(id="x", name="X", source={"security_clearance": "yes"})
    assert "security_clearance" in document_text(row)


def test_document_text_survives_odd_shapes() -> None:
    row = _Row(id="x", name="X", source={"a": None, "b": True, "c": [[["deep"]]], "d": 3})
    text = document_text(row)
    assert "deep" in text
    assert "3" in text


def test_document_text_does_not_recurse_forever() -> None:
    payload: dict[str, Any] = {}
    node = payload
    for _ in range(50):
        node["next"] = {}
        node = node["next"]
    node["leaf"] = "bottom"
    assert isinstance(document_text(_Row(id="x", name="X", source=payload)), str)


# ---- The local index --------------------------------------------------------


def test_finds_the_right_candidate_for_a_topical_question(run) -> None:
    result = run(lambda: LocalIndex().search("who has security experience?", POOL, 2))
    assert result.ids[0] == "sec"


def test_ranks_the_distributed_systems_record_first(run) -> None:
    result = run(lambda: LocalIndex().search("kafka postgres throughput", POOL, 2))
    assert result.ids[0] == "dist"


def test_search_by_name(run) -> None:
    """The commonest question a manager actually types."""
    result = run(lambda: LocalIndex().search("tell me about Ravi Menon", POOL, 1))
    assert result.ids == ("mob",)


def test_respects_the_limit(run) -> None:
    result = run(lambda: LocalIndex().search("engineer", POOL, 2))
    assert len(result.hits) <= 2


def test_irrelevant_query_returns_nothing_rather_than_noise(run) -> None:
    """The analyst falls back to the head of the pool when retrieval is
    empty; it cannot do that if retrieval invents matches."""
    result = run(lambda: LocalIndex().search("zzzzz quantum basketweaving", POOL, 3))
    assert result.ids == ()


def test_empty_query_and_empty_pool_are_not_errors(run) -> None:
    assert run(lambda: LocalIndex().search("", POOL, 3)).ids == ()
    assert run(lambda: LocalIndex().search("security", [], 3)).ids == ()


def test_scores_descend(run) -> None:
    result = run(lambda: LocalIndex().search("go engineer kafka", POOL, 4))
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_a_repeated_term_does_not_dominate(run) -> None:
    """Sublinear term frequency: a scraped skills array repeats everything,
    and a record saying "Go" nine times is not nine times as relevant."""
    spammy = _Row(id="spam", name="Spam", source={"skills": ["Go"] * 60})
    result = run(lambda: LocalIndex().search("go kafka postgres ledger", [*POOL, spammy], 1))
    assert result.ids[0] == "dist"


def test_backend_is_named_honestly(run) -> None:
    """The local index is lexical, not semantic, and says so -- the
    dashboard shows this string to the manager."""
    result = run(lambda: LocalIndex().search("security", POOL, 1))
    assert "local" in result.backend
    assert "lexical" in result.backend


# ---- Backend selection ------------------------------------------------------


def test_no_credentials_means_the_local_index() -> None:
    assert isinstance(build_pool_index(Settings()), LocalIndex)


def test_both_credentials_are_required() -> None:
    """A half-configured deployment must not look configured."""
    assert isinstance(build_pool_index(Settings(moss_project_id="p")), LocalIndex)
    assert isinstance(build_pool_index(Settings(moss_project_key="k")), LocalIndex)


def test_credentials_select_moss() -> None:
    index = build_pool_index(Settings(moss_project_id="p", moss_project_key="k"))
    assert isinstance(index, MossIndex)


def test_moss_configured_is_not_a_claim_that_moss_works() -> None:
    settings = Settings(moss_project_id="p", moss_project_key="k")
    assert settings.moss_configured is True
    # ...but nothing has been indexed, so the backend does not claim semantic.
    assert "semantic" not in MossIndex(settings).backend


# ---- Moss degradation -------------------------------------------------------


class _BrokenClient:
    """Stands in for a Moss project that is unreachable."""

    async def create_index(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("moss is down")

    async def add_docs(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("moss is down")

    async def load_index(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("moss is down")

    async def query(self, *a: Any, **k: Any) -> None:
        raise RuntimeError("moss is down")


def _broken(settings: Settings) -> MossIndex:
    index = MossIndex(settings)
    index._client = _BrokenClient()
    return index


@pytest.fixture
def moss_settings() -> Settings:
    return Settings(moss_project_id="p", moss_project_key="k")


def test_a_failed_sync_falls_back_instead_of_raising(run, moss_settings: Settings) -> None:
    index = _broken(moss_settings)
    assert run(lambda: index.sync(POOL)) == len(POOL)
    assert index.degraded is True


def test_a_failed_sync_is_counted(run, moss_settings: Settings) -> None:
    """A configured backend that silently never answers is externally
    indistinguishable from a healthy one. This is the same rule the model
    client follows, for the same reason."""
    before = retrieval_fallback_count()
    run(lambda: _broken(moss_settings).sync(POOL))
    assert retrieval_fallback_count() > before


def test_search_still_works_when_moss_is_down(run, moss_settings: Settings) -> None:
    index = _broken(moss_settings)
    run(lambda: index.sync(POOL))
    result = run(lambda: index.search("who has security experience?", POOL, 2))
    assert result.ids[0] == "sec"


def test_a_degraded_backend_says_so(run, moss_settings: Settings) -> None:
    index = _broken(moss_settings)
    run(lambda: index.sync(POOL))
    assert "degraded" in index.backend


class _StaleClient:
    """A Moss index still holding a candidate the pool no longer has."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def create_index(self, *a: Any, **k: Any) -> None: ...
    async def add_docs(self, *a: Any, **k: Any) -> None: ...
    async def load_index(self, *a: Any, **k: Any) -> None: ...

    async def delete_docs(self, name: str, doc_ids: list[str]) -> None:
        self.deleted.extend(doc_ids)

    async def query(self, *a: Any, **k: Any) -> Any:
        class _Doc:
            def __init__(self, doc_id: str) -> None:
                self.id, self.score = doc_id, 0.9

        class _Result:
            docs: ClassVar[list[Any]] = [_Doc("ghost"), _Doc("sec")]
            time_taken_ms = 2.5

        return _Result()


def test_a_deleted_candidate_can_never_appear_in_an_answer(run, moss_settings: Settings) -> None:
    """The remote index can lag a delete. Resolving hits against the rows we
    were handed means a candidate the manager removed cannot come back."""
    index = MossIndex(moss_settings)
    index._client = _StaleClient()
    run(lambda: index.sync(POOL))
    result = run(lambda: index.search("security", POOL, 5))
    assert "ghost" not in result.ids
    assert "sec" in result.ids


def test_removed_candidates_are_deleted_from_the_remote_index(run, moss_settings: Settings) -> None:
    client = _StaleClient()
    index = MossIndex(moss_settings)
    index._client = client
    run(lambda: index.sync(POOL))
    run(lambda: index.sync([r for r in POOL if r.id != "mob"]))
    assert client.deleted == ["mob"]


def test_moss_reports_its_own_latency_measurement(run, moss_settings: Settings) -> None:
    index = MossIndex(moss_settings)
    index._client = _StaleClient()
    run(lambda: index.sync(POOL))
    assert run(lambda: index.search("security", POOL, 5)).latency_ms == pytest.approx(2.5)
