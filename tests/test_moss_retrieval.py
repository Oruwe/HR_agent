"""The Moss retrieval layer: budget, correctness, and graceful degradation.

The Moss SDK is an optional dependency, so a test suite that skipped whenever it
was absent would leave the integration completely unverified in CI -- which is
where it matters most. Instead a fake ``moss`` module is injected into
``sys.modules`` implementing the documented surface (``MossClient`` with async
``create_index`` / ``load_index`` / ``query``, and ``QueryOptions``). That
exercises the real code path: index construction, query dispatch, document
resolution, per-role collapsing, latency capture, and every failure branch.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from app.config import Settings
from app.schemas.roles import ROLE_RUBRICS, EngineeringRole
from app.storage.retrieval import (
    RETRIEVAL_BUDGET_MS,
    EmbeddedRetriever,
    MossRetriever,
    RubricHit,
    RubricRetriever,
    build_retriever,
    rubric_documents,
)
from app.telemetry.metrics import percentile
from tests.conftest import ROLE_PROFILES, run_async

# =============================================================================
# A fake Moss SDK matching the documented surface
# =============================================================================


@dataclass
class _FakeDoc:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeResult:
    docs: list[_FakeDoc]
    time_taken_ms: float = 1.8


class _FakeQueryOptions:
    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k


class _FakeMossClient:
    """Keyword-overlap stand-in with the real client's async shape."""

    #: Class-level switches let a test steer behaviour without a custom subclass.
    fail_load: ClassVar[bool] = True  # a fresh project has no index yet
    fail_create: ClassVar[bool] = False
    fail_query: ClassVar[bool] = False
    instances: ClassVar[list[_FakeMossClient]] = []

    def __init__(self, project_id: str, project_key: str) -> None:
        self.project_id = project_id
        self.project_key = project_key
        self.indexes: dict[str, list[dict[str, Any]]] = {}
        self.loaded: list[str] = []
        self.queries: list[tuple[str, str, int]] = []
        _FakeMossClient.instances.append(self)

    async def create_index(self, name: str, documents: list[dict[str, Any]]) -> None:
        if type(self).fail_create:
            raise RuntimeError("index creation refused")
        self.indexes[name] = list(documents)

    async def load_index(self, name: str) -> None:
        if name not in self.indexes and type(self).fail_load:
            raise RuntimeError("no such index")
        self.loaded.append(name)

    async def query(self, name: str, text: str, options: Any) -> _FakeResult:
        if type(self).fail_query:
            raise RuntimeError("query failed")
        top_k = getattr(options, "top_k", 10)
        self.queries.append((name, text, top_k))

        wanted = {w for w in text.casefold().split() if len(w) > 3}
        scored: list[_FakeDoc] = []
        for doc in self.indexes.get(name, []):
            tokens = {w for w in str(doc["text"]).casefold().split() if len(w) > 3}
            overlap = len(wanted & tokens)
            if overlap:
                scored.append(
                    _FakeDoc(
                        id=doc["id"],
                        text=doc["text"],
                        score=overlap / max(len(tokens), 1),
                        metadata=dict(doc.get("metadata", {})),
                    )
                )
        scored.sort(key=lambda d: (-d.score, d.id))
        return _FakeResult(docs=scored[:top_k])


@pytest.fixture
def fake_moss(monkeypatch: pytest.MonkeyPatch) -> type[_FakeMossClient]:
    module = types.ModuleType("moss")
    module.MossClient = _FakeMossClient  # type: ignore[attr-defined]
    module.QueryOptions = _FakeQueryOptions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "moss", module)

    _FakeMossClient.instances = []
    _FakeMossClient.fail_load = True
    _FakeMossClient.fail_create = False
    _FakeMossClient.fail_query = False
    return _FakeMossClient


@pytest.fixture
def moss_settings() -> Settings:
    return Settings(moss_project_id="proj_test", moss_project_key="key_test")


# =============================================================================
# Corpus shape
# =============================================================================


def test_corpus_has_one_document_per_competency() -> None:
    docs = rubric_documents()
    expected = sum(len(ROLE_RUBRICS[r].competencies) for r in EngineeringRole)
    assert len(docs) == expected == 36


def test_document_ids_are_unique_and_addressable() -> None:
    docs = rubric_documents()
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))
    for doc in docs:
        assert doc.id == f"{doc.role.value}::{doc.competency_key}"


def test_documents_carry_signals_and_the_probe() -> None:
    """Retrieval quality depends on the document containing what candidates say."""
    doc = next(d for d in rubric_documents() if d.id == "AI_ML_SYSTEMS_ENGINEER::inference_kernels")
    assert "vllm" in doc.text
    assert "flashattention" in doc.text
    assert "KV-cache" in doc.text  # the probe question


def test_moss_document_shape_matches_the_sdk() -> None:
    payload = rubric_documents()[0].as_moss_document()
    assert set(payload) == {"id", "text", "metadata"}
    assert isinstance(payload["id"], str)
    assert isinstance(payload["text"], str)
    assert payload["metadata"]["role"] in EngineeringRole.__members__


# =============================================================================
# Backend selection
# =============================================================================


def test_embedded_backend_is_chosen_without_credentials() -> None:
    retriever = build_retriever(Settings())
    assert isinstance(retriever, EmbeddedRetriever)
    assert retriever.backend == "embedded"


def test_moss_backend_is_chosen_when_configured(moss_settings: Settings) -> None:
    assert moss_settings.moss_configured
    assert isinstance(build_retriever(moss_settings), MossRetriever)


def test_partial_credentials_do_not_select_moss() -> None:
    """Half-configured is not configured; silently querying with no key is worse."""
    assert not Settings(moss_project_id="proj_only").moss_configured
    assert isinstance(build_retriever(Settings(moss_project_id="proj_only")), EmbeddedRetriever)


def test_both_backends_satisfy_the_protocol(moss_settings: Settings) -> None:
    assert isinstance(build_retriever(Settings()), RubricRetriever)
    assert isinstance(build_retriever(moss_settings), RubricRetriever)


# =============================================================================
# Moss path, driven through the fake SDK
# =============================================================================


def test_warm_creates_then_loads_the_index(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    async def go() -> MossRetriever:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        return retriever

    retriever = run_async(go)
    client = fake_moss.instances[-1]

    assert retriever.ready
    assert retriever.backend == "moss"
    assert client.indexes[moss_settings.moss_index]
    assert len(client.indexes[moss_settings.moss_index]) == 36
    assert moss_settings.moss_index in client.loaded


def test_existing_index_is_loaded_not_rebuilt(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    """Re-indexing a static corpus is pure startup latency on every worker boot."""
    fake_moss.fail_load = False  # pretend the index already exists

    async def go() -> _FakeMossClient:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        return fake_moss.instances[-1]

    client = run_async(go)
    assert client.loaded == [moss_settings.moss_index]
    assert client.indexes == {}, "an existing index was rebuilt"


@pytest.mark.parametrize("expected", list(EngineeringRole), ids=lambda r: r.value)
def test_moss_retrieval_routes_each_profile(
    fake_moss: type[_FakeMossClient], moss_settings: Settings, expected: EngineeringRole
) -> None:
    async def go() -> list[RubricHit]:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        return await retriever.search(ROLE_PROFILES[expected], limit=3)

    hits = run_async(go)
    assert hits, "Moss returned nothing"
    assert hits[0].role is expected


def test_results_are_collapsed_to_one_hit_per_role(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    """Several competencies of one role match; the caller wants distinct roles."""

    async def go() -> list[RubricHit]:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        return await retriever.search(
            ROLE_PROFILES[EngineeringRole.DEVOPS_SRE_CLOUD_ARCHITECT], limit=3
        )

    hits = run_async(go)
    assert len({h.role for h in hits}) == len(hits)
    assert len(hits) <= 3


def test_hits_identify_the_specific_competency(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    """The interviewer needs to know *which* competency matched, not just the role."""

    async def go() -> list[RubricHit]:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        return await retriever.search(
            "We used paged attention in vLLM and wrote Triton kernels for fused attention.",
            limit=1,
        )

    hits = run_async(go)
    assert hits[0].role is EngineeringRole.AI_ML_SYSTEMS_ENGINEER
    assert hits[0].competency_key == "inference_kernels"
    assert hits[0].competency_label
    assert hits[0].document_id == "AI_ML_SYSTEMS_ENGINEER::inference_kernels"


def test_runtime_reported_latency_is_preferred(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    """Moss reports its own timing, which excludes our await-scheduling overhead."""

    async def go() -> float:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        await retriever.search(ROLE_PROFILES[EngineeringRole.DATA_PLATFORM_ENGINEER])
        return retriever.last_latency_ms

    assert run_async(go) == pytest.approx(1.8)


def test_over_fetches_to_survive_collapsing(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    async def go() -> int:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        await retriever.search(ROLE_PROFILES[EngineeringRole.MOBILE_CORE_ENGINEER], limit=2)
        return fake_moss.instances[-1].queries[-1][2]

    assert run_async(go) >= 6, "top_k was not raised to survive per-role collapsing"


def test_empty_evidence_short_circuits(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    async def go() -> tuple[list[RubricHit], int]:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        hits = await retriever.search("   ")
        return hits, len(fake_moss.instances[-1].queries)

    hits, queries = run_async(go)
    assert hits == []
    assert queries == 0, "an empty query was dispatched"


# =============================================================================
# Degradation -- a retrieval problem must never end an interview
# =============================================================================


def test_missing_sdk_degrades_to_the_embedded_index(
    monkeypatch: pytest.MonkeyPatch, moss_settings: Settings
) -> None:
    monkeypatch.setitem(sys.modules, "moss", None)

    async def go() -> tuple[MossRetriever, list[RubricHit]]:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        hits = await retriever.search(ROLE_PROFILES[EngineeringRole.SECURITY_DEVSECOPS_ENGINEER])
        return retriever, hits

    retriever, hits = run_async(go)
    assert not retriever.ready
    assert "degraded" in retriever.backend
    assert hits and hits[0].role is EngineeringRole.SECURITY_DEVSECOPS_ENGINEER


def test_index_creation_failure_degrades(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    fake_moss.fail_create = True

    async def go() -> MossRetriever:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        return retriever

    retriever = run_async(go)
    assert not retriever.ready
    assert "degraded" in retriever.backend


def test_query_failure_mid_interview_falls_back_and_still_answers(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    """The failure that matters: Moss dies *during* a live screening call."""

    async def go() -> tuple[list[RubricHit], list[RubricHit], MossRetriever]:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        profile = ROLE_PROFILES[EngineeringRole.DISTRIBUTED_BACKEND_ENGINEER]
        healthy = await retriever.search(profile, limit=1)
        fake_moss.fail_query = True
        broken = await retriever.search(profile, limit=1)
        return healthy, broken, retriever

    healthy, broken, retriever = run_async(go)
    assert healthy[0].role is EngineeringRole.DISTRIBUTED_BACKEND_ENGINEER
    assert broken, "the interview lost its context entirely"
    assert broken[0].role is EngineeringRole.DISTRIBUTED_BACKEND_ENGINEER
    assert "degraded" in retriever.backend


def test_unknown_document_shape_is_skipped_not_fatal(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    """An SDK field rename must not crash a live interview."""
    retriever = MossRetriever(settings=moss_settings)
    hits = retriever._collapse([object(), _FakeDoc(id="nope::nope", text="x", score=1.0)], 3)
    assert hits == []


def test_document_resolves_by_metadata_when_id_is_absent(moss_settings: Settings) -> None:
    retriever = MossRetriever(settings=moss_settings)

    class _NoId:
        score = 0.9
        metadata: ClassVar[dict[str, str]] = {
            "role": "DATA_PLATFORM_ENGINEER",
            "competency_key": "stream_processing",
        }

    hits = retriever._collapse([_NoId()], 1)
    assert hits[0].role is EngineeringRole.DATA_PLATFORM_ENGINEER
    assert hits[0].competency_key == "stream_processing"


# =============================================================================
# The budget
# =============================================================================


def test_embedded_retrieval_is_within_the_10ms_budget() -> None:
    async def go() -> list[float]:
        retriever = EmbeddedRetriever()
        await retriever.warm()
        evidence = ROLE_PROFILES[EngineeringRole.AI_ML_SYSTEMS_ENGINEER]
        await retriever.search(evidence)  # discard the first, unwarmed call
        samples: list[float] = []
        for _ in range(40):
            await retriever.search(evidence, limit=3)
            samples.append(retriever.last_latency_ms)
        return samples

    samples = run_async(go)
    p95 = percentile(samples, 95)
    assert p95 <= RETRIEVAL_BUDGET_MS, (
        f"retrieval p95 {p95:.2f}ms exceeds the {RETRIEVAL_BUDGET_MS}ms budget"
    )


def test_moss_retrieval_is_within_the_10ms_budget(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    async def go() -> list[float]:
        retriever = MossRetriever(settings=moss_settings)
        await retriever.warm()
        evidence = ROLE_PROFILES[EngineeringRole.FRONTEND_PLATFORM_ENGINEER]
        samples: list[float] = []
        for _ in range(40):
            await retriever.search(evidence, limit=3)
            samples.append(retriever.last_latency_ms)
        return samples

    assert percentile(run_async(go), 95) <= RETRIEVAL_BUDGET_MS


def test_retrieval_budget_is_the_declared_stage_budget() -> None:
    from app.config import STAGE_BUDGETS_MS, Stage

    assert RETRIEVAL_BUDGET_MS == STAGE_BUDGETS_MS[Stage.VECTOR_MATCH] == 10.0


# =============================================================================
# Integration with the orchestrator hot path
# =============================================================================


def test_orchestrator_uses_the_configured_backend(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    from app.agent.cognition import MockCognition
    from app.agent.orchestrator import ScreeningOrchestrator, new_session

    async def go() -> ScreeningOrchestrator:
        session = new_session(ROLE_PROFILES[EngineeringRole.AI_ML_SYSTEMS_ENGINEER])
        orchestrator = ScreeningOrchestrator(
            session, settings=moss_settings, cognition=MockCognition(ttft_ms=1.0)
        )
        await orchestrator.open()
        orchestrator.observe_candidate("We tuned FSDP sharding and NCCL collectives.")
        await orchestrator.speculate()
        await orchestrator.cancel_speculation()
        return orchestrator

    orchestrator = run_async(go)
    assert orchestrator.retrieval_backend == "moss"
    assert orchestrator.retrieval_latencies_ms
    assert max(orchestrator.retrieval_latencies_ms) <= RETRIEVAL_BUDGET_MS


def test_retrieved_context_reaches_the_prompt(
    fake_moss: type[_FakeMossClient], moss_settings: Settings
) -> None:
    """Retrieval is only useful if the interviewer actually sees it."""
    from app.agent.cognition import MockCognition
    from app.agent.orchestrator import ScreeningOrchestrator, new_session

    async def go() -> str:
        session = new_session(ROLE_PROFILES[EngineeringRole.AI_ML_SYSTEMS_ENGINEER])
        orchestrator = ScreeningOrchestrator(
            session, settings=moss_settings, cognition=MockCognition(ttft_ms=1.0)
        )
        await orchestrator.open()
        orchestrator.observe_candidate("We cut KV-cache with paged attention in vLLM.")
        await orchestrator.speculate()
        context = orchestrator._pending_context
        await orchestrator.cancel_speculation()
        return context

    context = run_async(go)
    assert "evidence for" in context
    assert len(context) > 30
