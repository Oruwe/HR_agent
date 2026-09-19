"""Shared fixtures.

Two deliberate choices worth explaining:

**No pytest-asyncio.** Async tests are driven through the :func:`run_async`
helper instead. The plugin's event-loop fixtures are a recurring source of
deprecation warnings across pytest majors, and this suite runs under
``filterwarnings = ["error"]`` -- a warning is a failure here. An explicit
``asyncio.run`` per test is also easier to reason about when a barge-in test
deadlocks.

**Environment isolation.** Every test gets a clean, explicitly-set environment.
Settings are cached process-wide for hot-path reasons, so a test that mutated
the environment without clearing that cache would silently poison every test
after it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, TypeVar

import pytest

from app.config import Settings, reset_settings
from app.schemas.roles import EngineeringRole
from app.security.pii_scrubber import reset_redaction_key
from app.storage.qdrant_client import HybridVectorStore

T = TypeVar("T")

#: Every HRTE_* / provider variable the agent reads. Cleared before each test so
#: a developer's populated .env cannot change what CI asserts.
_MANAGED_ENV: tuple[str, ...] = (
    "HRTE_ENV",
    "HRTE_LOG_LEVEL",
    "HRTE_LATENCY_BUDGET_MS",
    "HRTE_ENDPOINT_SILENCE_MS",
    "HRTE_SPECULATIVE_SILENCE_MS",
    "HRTE_BARGE_IN_MS",
    "HRTE_ROOM_PREFIX",
    "HRTE_COGNITION_MODEL",
    "HRTE_COGNITION_TEMPERATURE",
    "HRTE_COGNITION_MAX_TOKENS",
    "HRTE_SPEECH_ENGINE",
    "HRTE_SPEECH_WS_URL",
    "HRTE_SPEECH_API_KEY",
    "HRTE_MOSS_INDEX",
    "HRTE_QDRANT_COLLECTION",
    "HRTE_PII_MODE",
    "HRTE_REDACTION_KEY",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "GOOGLE_API_KEY",
    "MOSS_PROJECT_ID",
    "MOSS_PROJECT_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "DEEPGRAM_API_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "HRTE_CORS_ORIGINS",
    "HRTE_SESSION_TTL_SECONDS",
    "HRTE_ADMIN_TOKEN",
    "HRTE_DATA_DIR",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in _MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)
    reset_settings()
    reset_redaction_key()
    yield
    reset_settings()
    reset_redaction_key()


@pytest.fixture
def settings() -> Settings:
    """Offline settings with the documented default timings."""
    return Settings()


def run_async(coro_fn: Callable[[], Awaitable[T]]) -> T:
    """Run an async callable to completion on a fresh event loop."""
    return asyncio.run(coro_fn())


@pytest.fixture
def run() -> Callable[[Callable[[], Awaitable[Any]]], Any]:
    return run_async


@pytest.fixture(scope="session")
def warm_store() -> HybridVectorStore:
    """A store with the rubric index already built.

    Session-scoped because warming is pure and idempotent; rebuilding it per
    test would add seconds to the suite for no isolation benefit. Tests that
    write candidates use their own instance.
    """
    store = HybridVectorStore()
    store.ensure_collection()
    return store


# =============================================================================
# Nine candidate profiles, one per rubric
# =============================================================================
#
# Written as prose an engineer would actually say, not keyword lists. A matcher
# that only works on bags of technology names would pass a keyword-stuffed
# fixture and fail in production on the first real transcript.

ROLE_PROFILES: dict[EngineeringRole, str] = {
    EngineeringRole.AI_ML_SYSTEMS_ENGINEER: """
        Six years on model training and serving infrastructure. Ran FSDP across
        512 A100s with DeepSpeed ZeRO-3 earlier on, retuned NCCL collectives and
        added gradient checkpointing to fit activations. On the serving side I
        moved us onto vLLM with paged attention and continuous batching, wrote
        fused Triton kernels for attention, and used FlashAttention where the
        shapes allowed. Tuned an HNSW index over 200 million embeddings, trading
        ANN recall against p99. Comfortable reading PyTorch C++ and ATen when a
        custom op misbehaves, and debugging GPU memory fragmentation.
    """,
    EngineeringRole.FRONTEND_PLATFORM_ENGINEER: """
        I own the browser runtime budget for a large retail site. Cut largest
        contentful paint from 4.1s to 1.3s by killing layout thrash and
        eliminating a long task that blocked the main thread. Built our
        micro-frontend layer on module federation with a shared dependency
        policy, and moved CSV parsing into a web worker. Wrote a Babel plugin
        and a codemod to migrate 400 components, and tune tree shaking in our
        Vite build. I profile with Lighthouse and the browser's own flame chart,
        and I understand DOM reconciliation well enough to know when the virtual
        DOM is not the problem.
    """,
    EngineeringRole.DISTRIBUTED_BACKEND_ENGINEER: """
        I work on payment infrastructure. Implemented Raft leader election and
        the quorum rules for our metadata store, and handled split brain during
        a partition. Everything writes through an outbox pattern with an
        idempotency key, so a timed-out request that gets retried never double
        charges. Moved a hot table from a B-tree to an LSM tree engine once the
        write amplification made sense, and I read the write-ahead log during
        incidents. Our services speak gRPC with protobuf and we evolve schemas
        backward compatible across forty deployments.
    """,
    EngineeringRole.DEVOPS_SRE_CLOUD_ARCHITECT: """
        Platform and reliability. I wrote a Kubernetes operator with a custom
        resource definition and a reconcile loop to replace a Helm chart that
        could not express our failover ordering. We run eBPF-based tracing with
        Cilium, and I have chased down an OOM killer event that looked like an
        application leak but was a cgroup limit. Delivery is GitOps through
        ArgoCD, Terraform state is isolated per environment with drift
        detection. I own our SLO and error budget policy and run multi-region
        failover drills with real RTO and RPO targets.
    """,
    EngineeringRole.FULL_STACK_PRODUCT_ENGINEER: """
        I ship product features end to end. We have end-to-end type safety with
        tRPC and Zod, so a backend field rename fails the frontend build rather
        than production. Built optimistic updates with rollback on error and
        stale-while-revalidate caching, and had to work through conflict
        resolution when two tabs edited the same record offline. I run expand
        contract migrations with a backfill and a dual write so schema changes
        land with zero downtime, and I manage API versioning with contract tests
        and a deprecation window before anything is retired.
    """,
    EngineeringRole.MOBILE_CORE_ENGINEER: """
        iOS and Android, mostly core infrastructure rather than features. Found
        a retain cycle that survived two releases by capturing weak self in a
        closure chain, proved it with the leaks instrument. I own our React
        Native bridge work, moved hot paths to a TurboModule over JSI to stop
        paying serialization overhead per call. Cut cold start by 40 percent
        using a startup trace, and dealt with Doze mode killing our WorkManager
        sync on Android 13. I profile with Instruments and Perfetto, and
        maintain the offline cache and its SQLite migrations.
    """,
    EngineeringRole.SECURITY_DEVSECOPS_ENGINEER: """
        Application security and build integrity. I ran the STRIDE threat model
        for our payments boundary and cut the blast radius by splitting trust
        boundaries. We moved to zero-trust with mTLS between workloads and
        short-lived credentials issued per workload identity, keys in an HSM
        with automated rotation. I fixed an SSRF in a webhook fetcher and a CSRF
        gap in an older form flow, and I run SAST and DAST in the pipeline with
        fuzzing on parsers. We publish SBOMs and sign artifacts with cosign,
        working toward SLSA level three provenance.
    """,
    EngineeringRole.DATA_PLATFORM_ENGINEER: """
        I own our streaming and lakehouse platform. Apache Flink jobs over Kafka
        with event time semantics, watermarks tuned for our late-arriving
        clickstream, and proper backpressure handling when consumer lag spikes.
        Migrated the warehouse to Apache Iceberg for partition pruning and time
        travel, with scheduled compaction jobs. On the batch side I fix data
        skew with salting and broadcast joins, and read the Spark shuffle
        metrics when one task in a stage runs long. Everything is orchestrated
        as idempotent pipelines with data lineage tracked through dbt.
    """,
    EngineeringRole.TECHNICAL_SOLUTIONS_ARCHITECT: """
        I work between engineering and the business on system design. Led a
        domain-driven design exercise that produced a context map and an
        anti-corruption layer between our billing and provisioning bounded
        contexts. I argue trade-offs explicitly: where CAP theorem forces a
        consistency choice, what the latency budget actually permits, and the
        total cost of ownership of build versus buy. Migrated a monolith with a
        strangler fig rather than a big-bang cutover. I run our RFC process and
        write the architecture decision records, including the data residency
        and compliance boundary constraints.
    """,
}

assert len(ROLE_PROFILES) == 9, "One profile per rubric is required"


@pytest.fixture
def role_profiles() -> dict[EngineeringRole, str]:
    return dict(ROLE_PROFILES)


# =============================================================================
# Resumes carrying planted PII
# =============================================================================

PII_RESUME = """
Priya Raman
priya.raman@example.com | +91 98765 43210 | alt +1 (555) 123-4567
Aadhaar: 3412 7856 9034 | PAN ABCDE1234F | Passport K1234567
42 MG Road, Indiranagar, Bengaluru 560038
Emergency contact reachable on 9876543210.

Senior ML Systems Engineer. Led FSDP training across 512 A100s, cut step time
34 percent. Wrote Triton kernels. Reduced KV-cache 40 percent with vLLM paged
attention. Tuned HNSW recall against p99 latency.
"""

INTERNATIONAL_PII = """
Kim Min-jun, RRN 900101-1234567, Seoul.
Tanaka Yuki, MyNumber: 1234 5678 9012, Tokyo.
John Smith, SSN: 123-45-6789, 221 Baker Street, Springfield, IL 62704.
Card on file 4111 1111 1111 1111. Residence at 12.9716, 77.5946.
"""

#: Technical prose containing digit patterns that must NOT be redacted. This is
#: the false-positive corpus -- a scrubber that eats "p99 under 45ms" destroys
#: the very evidence the rubric scores.
TECHNICAL_NO_PII = """
Cut p99 from 1200ms to 160ms across 12 regions at 1500 QPS.
Ran 512 A100s from 2019-2023 with 99.99 percent uptime.
Reduced KV-cache 40 percent; HNSW m=16 and ef_construct=100.
Scaled to 200000000 vectors. Error budget 0.05 percent over 30 days.
Version 1.0.0 of the spec, RFC 7519, HTTP 429 backoff at 2^n seconds.
"""


@pytest.fixture
def pii_resume() -> str:
    return PII_RESUME


@pytest.fixture
def international_pii() -> str:
    return INTERNATIONAL_PII


@pytest.fixture
def technical_no_pii() -> str:
    return TECHNICAL_NO_PII
