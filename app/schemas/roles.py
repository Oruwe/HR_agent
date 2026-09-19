"""The nine engineering role rubrics and the deterministic role matcher.

Why deterministic
-----------------
Role assignment is a decision that affects a person's career, so it has to be
explainable and reproducible: the same resume must route to the same role on
every run, on every replica, forever. That rules out asking an LLM "which role
is this?" as the primary mechanism. Instead we score evidence against a fixed,
version-controlled rubric using weighted, IDF-discounted signal matching, and
the LLM is used only for conversational probing -- never for the routing call.

The IDF discount is the part that does the real work. "kubernetes" appears in
several rubrics and therefore says little about which role a candidate fits;
"flashattention" appears in exactly one and is nearly decisive. Weighting each
signal by its inverse role frequency makes the matcher discriminative instead
of merely enthusiastic, and it does so without any training data.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType
from typing import Final

# =============================================================================
# Role identity
# =============================================================================


class EngineeringRole(StrEnum):
    """The nine screening tracks. Enum order is the deterministic tie-break."""

    AI_ML_SYSTEMS_ENGINEER = "AI_ML_SYSTEMS_ENGINEER"
    FRONTEND_PLATFORM_ENGINEER = "FRONTEND_PLATFORM_ENGINEER"
    DISTRIBUTED_BACKEND_ENGINEER = "DISTRIBUTED_BACKEND_ENGINEER"
    DEVOPS_SRE_CLOUD_ARCHITECT = "DEVOPS_SRE_CLOUD_ARCHITECT"
    FULL_STACK_PRODUCT_ENGINEER = "FULL_STACK_PRODUCT_ENGINEER"
    MOBILE_CORE_ENGINEER = "MOBILE_CORE_ENGINEER"
    SECURITY_DEVSECOPS_ENGINEER = "SECURITY_DEVSECOPS_ENGINEER"
    DATA_PLATFORM_ENGINEER = "DATA_PLATFORM_ENGINEER"
    TECHNICAL_SOLUTIONS_ARCHITECT = "TECHNICAL_SOLUTIONS_ARCHITECT"


@dataclass(frozen=True, slots=True)
class Competency:
    """One scorable dimension of a rubric.

    ``signals`` are lowercase surface forms searched for in the evidence text.
    They are intentionally written as the phrases engineers actually say out
    loud in an interview, not as canonical technology names.
    """

    key: str
    label: str
    weight: float
    signals: tuple[str, ...]
    #: The question the live interviewer asks to probe this dimension.
    probe: str

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"{self.key}: weight must be in (0, 1], got {self.weight}")
        if not self.signals:
            raise ValueError(f"{self.key}: needs at least one signal")


@dataclass(frozen=True, slots=True)
class EliminationCriterion:
    """A hard gate. Failing one caps the recommendation regardless of fit score.

    Elimination is deliberately tied to a specific competency key rather than
    to free-text judgement, so that a rejection can always be traced to a
    numbered rubric line during a fairness audit.
    """

    key: str
    description: str
    competency_key: str
    #: Score at or below which the criterion is considered failed.
    floor: float = 0.35


@dataclass(frozen=True, slots=True)
class RoleRubric:
    """The complete, versioned definition of one screening track."""

    role: EngineeringRole
    title: str
    summary: str
    competencies: tuple[Competency, ...]
    eliminations: tuple[EliminationCriterion, ...]
    #: Minimum weighted fit for an ADVANCE recommendation.
    advance_threshold: float = 0.62

    def __post_init__(self) -> None:
        keys = [c.key for c in self.competencies]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.role}: duplicate competency keys")
        for elim in self.eliminations:
            if elim.competency_key not in keys:
                raise ValueError(
                    f"{self.role}: elimination '{elim.key}' references unknown "
                    f"competency '{elim.competency_key}'"
                )

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.competencies)

    def competency(self, key: str) -> Competency:
        for c in self.competencies:
            if c.key == key:
                return c
        raise KeyError(f"{self.role}: no competency '{key}'")

    def all_signals(self) -> tuple[str, ...]:
        return tuple(s for c in self.competencies for s in c.signals)

    def rubric_text(self) -> str:
        """Flattened text used to build this rubric's dense embedding."""
        parts = [self.title, self.summary]
        parts.extend(f"{c.label}: {', '.join(c.signals)}" for c in self.competencies)
        return " | ".join(parts)


# =============================================================================
# The nine rubrics
# =============================================================================

_RUBRIC_LIST: Final[tuple[RoleRubric, ...]] = (
    RoleRubric(
        role=EngineeringRole.AI_ML_SYSTEMS_ENGINEER,
        title="AI / ML Systems Engineer",
        summary=(
            "Owns model training and inference infrastructure: distributed training "
            "topology, serving throughput, and GPU memory economics."
        ),
        competencies=(
            Competency(
                key="distributed_training",
                label="Distributed training topology",
                weight=1.0,
                signals=(
                    "fsdp",
                    "deepspeed",
                    "zero-3",
                    "tensor parallel",
                    "pipeline parallel",
                    "data parallel",
                    "nccl",
                    "gradient checkpointing",
                    "mixed precision",
                    "megatron",
                ),
                probe=(
                    "Walk me through how you decided between tensor and pipeline "
                    "parallelism on your largest training run, and what the "
                    "communication cost looked like."
                ),
            ),
            Competency(
                key="inference_kernels",
                label="Inference optimisation and kernels",
                weight=1.0,
                signals=(
                    "vllm",
                    "tgi",
                    "tensorrt",
                    "flashattention",
                    "flash attention",
                    "cuda kernel",
                    "triton kernel",
                    "paged attention",
                    "continuous batching",
                    "torch.compile",
                    "quantization",
                    "awq",
                    "gptq",
                ),
                probe=(
                    "Compute the KV-cache footprint for a 7B model at 8k context and "
                    "batch 32, and tell me where that number forces your hand."
                ),
            ),
            Competency(
                key="vector_indexing",
                label="Vector indexing and retrieval",
                weight=0.7,
                signals=("hnsw", "ivf-pq", "faiss", "embedding index", "ann recall", "reranker"),
                probe=(
                    "How did you trade recall against p99 latency when tuning your "
                    "ANN index parameters?"
                ),
            ),
            Competency(
                key="framework_internals",
                label="Framework internals",
                weight=0.6,
                signals=(
                    "pytorch c++",
                    "aten",
                    "autograd",
                    "custom op",
                    "cuda graph",
                    "memory fragmentation",
                    "gpu memory",
                ),
                probe="Describe a time you had to read framework source to fix a bug.",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="kv_cache_math",
                description=(
                    "Cannot explain KV-cache memory calculation or distinguish "
                    "pipeline from tensor parallelism."
                ),
                competency_key="inference_kernels",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.FRONTEND_PLATFORM_ENGINEER,
        title="Frontend Platform Engineer",
        summary=(
            "Owns the browser runtime budget: rendering performance, build tooling, "
            "and the composition model that many product teams build on."
        ),
        competencies=(
            Competency(
                key="render_performance",
                label="Rendering pipeline and Core Web Vitals",
                weight=1.0,
                signals=(
                    "core web vitals",
                    "largest contentful paint",
                    "lcp",
                    "cumulative layout shift",
                    "interaction to next paint",
                    "reflow",
                    "repaint",
                    "layout thrash",
                    "composite layer",
                    "critical rendering path",
                    "lighthouse",
                ),
                probe=(
                    "A list re-render is costing 40ms of layout. Walk me from the "
                    "flame chart to the fix."
                ),
            ),
            Competency(
                key="composition_architecture",
                label="Micro-frontend composition",
                weight=0.9,
                signals=(
                    "micro-frontend",
                    "module federation",
                    "single-spa",
                    "import map",
                    "shared dependency",
                    "island architecture",
                    "hydration",
                ),
                probe="How did you version shared dependencies across federated remotes?",
            ),
            Competency(
                key="runtime_concurrency",
                label="Browser concurrency",
                weight=0.8,
                signals=(
                    "web worker",
                    "service worker",
                    "offscreencanvas",
                    "requestidlecallback",
                    "main thread blocking",
                    "long task",
                    "wasm",
                ),
                probe="What did you move off the main thread, and how did you measure the win?",
            ),
            Competency(
                key="build_tooling",
                label="AST tooling and bundling",
                weight=0.7,
                signals=(
                    "babel plugin",
                    "ast transform",
                    "codemod",
                    "tree shaking",
                    "rollup",
                    "vite",
                    "esbuild",
                    "swc",
                    "source map",
                    "dom reconciliation",
                    "virtual dom",
                ),
                probe="Describe a codemod or compiler plugin you wrote and why.",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="library_only",
                description=(
                    "Relies purely on component libraries with no grasp of layout "
                    "reflow/repaint cost."
                ),
                competency_key="render_performance",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.DISTRIBUTED_BACKEND_ENGINEER,
        title="Distributed Backend Engineer",
        summary=(
            "Owns correctness under partition: consensus, durable state, and service "
            "contracts that survive partial failure."
        ),
        competencies=(
            Competency(
                key="consensus_replication",
                label="Consensus and replication",
                weight=1.0,
                signals=(
                    "raft",
                    "paxos",
                    "leader election",
                    "quorum",
                    "split brain",
                    "consensus protocol",
                    "linearizab",
                    "replication lag",
                ),
                probe="Why does Raft need a no-op entry at the start of a new term?",
            ),
            Competency(
                key="failure_semantics",
                label="Partition tolerance and idempotency",
                weight=1.0,
                signals=(
                    "partition tolerance",
                    "idempotent",
                    "idempotency key",
                    "exactly-once",
                    "at-least-once",
                    "retry storm",
                    "outbox pattern",
                    "two-phase commit",
                    "saga",
                    "fencing token",
                ),
                probe=(
                    "A payment write times out with no response. Describe the exact "
                    "mechanism that stops a double charge on retry."
                ),
            ),
            Competency(
                key="storage_internals",
                label="Database internals",
                weight=0.9,
                signals=(
                    "write-ahead log",
                    "wal",
                    "lsm tree",
                    "b-tree",
                    "compaction",
                    "mvcc",
                    "isolation level",
                    "index selectivity",
                    "vacuum",
                ),
                probe="When does an LSM tree beat a B-tree for your write pattern, and why?",
            ),
            Competency(
                key="service_contracts",
                label="Service contracts",
                weight=0.6,
                signals=(
                    "grpc",
                    "protobuf",
                    "event sourcing",
                    "cqrs",
                    "schema evolution",
                    "backward compatible",
                ),
                probe="How do you roll out a breaking protobuf change across 40 services?",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="reliable_network_fallacy",
                description=(
                    "Treats the network as reliable, or cannot articulate an "
                    "idempotent transaction mechanism."
                ),
                competency_key="failure_semantics",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.DEVOPS_SRE_CLOUD_ARCHITECT,
        title="DevOps & SRE Cloud Architect",
        summary=(
            "Owns the production control plane: declarative infrastructure, "
            "observability depth, and failover that has actually been rehearsed."
        ),
        competencies=(
            Competency(
                key="orchestration",
                label="Kubernetes and operators",
                weight=1.0,
                signals=(
                    "kubernetes operator",
                    "custom resource definition",
                    "crd",
                    "reconcile loop",
                    "admission webhook",
                    "statefulset",
                    "helm chart",
                    "pod disruption budget",
                ),
                probe="What made you write an operator instead of a Helm chart?",
            ),
            Competency(
                key="observability",
                label="Deep observability",
                weight=0.9,
                signals=(
                    "ebpf",
                    "cilium",
                    "distributed tracing",
                    "prometheus",
                    "cardinality explosion",
                    "flame graph",
                    "perf record",
                    "oom killer",
                    "cgroup",
                    "kernel tuning",
                ),
                probe=(
                    "A pod is OOM-killed but the app reports low heap. Take me "
                    "through your diagnosis."
                ),
            ),
            Competency(
                key="declarative_delivery",
                label="GitOps and infrastructure as code",
                weight=0.9,
                signals=(
                    "gitops",
                    "argocd",
                    "flux",
                    "terraform state",
                    "state isolation",
                    "remote backend",
                    "drift detection",
                    "immutable infrastructure",
                ),
                probe="How is your Terraform state partitioned, and what breaks if it is not?",
            ),
            Competency(
                key="reliability_algebra",
                label="SLO / SLI algebra and failover",
                weight=0.9,
                signals=(
                    "slo",
                    "sli",
                    "error budget",
                    "multi-region failover",
                    "rto",
                    "rpo",
                    "chaos engineering",
                    "canary deploy",
                    "blue-green",
                    "zero-downtime",
                ),
                probe="Derive the error budget for a 99.95% monthly SLO and tell me how you spend it.",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="manual_intervention",
                description=(
                    "Cannot diagnose Linux OOM-killer behaviour, or depends on "
                    "manual configuration intervention."
                ),
                competency_key="observability",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.FULL_STACK_PRODUCT_ENGINEER,
        title="Full Stack Product Engineer",
        summary=(
            "Owns a vertical slice end to end: type-safe contracts, responsive UI "
            "state, and schema changes that ship without downtime."
        ),
        competencies=(
            Competency(
                key="type_safety",
                label="End-to-end type safety",
                weight=1.0,
                signals=(
                    "end-to-end type safety",
                    "trpc",
                    "zod",
                    "openapi codegen",
                    "generated client",
                    "discriminated union",
                    "type-safe api",
                ),
                probe="How does a backend field rename reach the frontend build as an error?",
            ),
            Competency(
                key="ui_concurrency",
                label="Optimistic UI and state sync",
                weight=0.9,
                signals=(
                    "optimistic update",
                    "optimistic ui",
                    "rollback on error",
                    "stale-while-revalidate",
                    "cache invalidation",
                    "conflict resolution",
                    "react query",
                    "server component",
                ),
                probe=(
                    "Two tabs edit the same record offline. What does the user see "
                    "when both reconnect?"
                ),
            ),
            Competency(
                key="schema_evolution",
                label="Migrations under load",
                weight=0.9,
                signals=(
                    "expand contract migration",
                    "backfill",
                    "online schema change",
                    "zero downtime migration",
                    "dual write",
                    "shadow read",
                ),
                probe="Walk me through dropping a NOT NULL column on a hot table with no downtime.",
            ),
            Competency(
                key="api_versioning",
                label="API contract versioning",
                weight=0.7,
                signals=(
                    "api versioning",
                    "contract test",
                    "deprecation window",
                    "feature flag",
                    "sunset header",
                ),
                probe="How do you retire a v1 endpoint that a partner still calls?",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="state_anomalies",
                description=(
                    "Neglects state synchronisation anomalies or cannot isolate a "
                    "backend service bottleneck."
                ),
                competency_key="ui_concurrency",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.MOBILE_CORE_ENGINEER,
        title="Mobile Core Engineer",
        summary=(
            "Owns the on-device experience: memory lifetime, native bridge cost, "
            "and behaviour under OS-imposed execution limits."
        ),
        competencies=(
            Competency(
                key="memory_lifetime",
                label="Memory and retain cycles",
                weight=1.0,
                signals=(
                    "retain cycle",
                    "arc",
                    "weak self",
                    "memory leak instrument",
                    "garbage collection pause",
                    "bitmap recycling",
                    "leakcanary",
                ),
                probe="How did you find and prove a retain cycle in production?",
            ),
            Competency(
                key="bridge_mechanics",
                label="Native runtime bridge",
                weight=0.9,
                signals=(
                    "jni",
                    "react native bridge",
                    "turbomodule",
                    "jsi",
                    "platform channel",
                    "kotlin multiplatform",
                    "swift interop",
                    "serialization overhead",
                ),
                probe="What is the real cost of a bridge crossing, and how did you batch it away?",
            ),
            Competency(
                key="lifecycle_limits",
                label="Background execution limits",
                weight=0.9,
                signals=(
                    "background execution",
                    "doze mode",
                    "workmanager",
                    "background fetch",
                    "cold start",
                    "app startup trace",
                    "process death",
                    "foreground service",
                ),
                probe="Your sync job stops running on Android 13. Walk me through why.",
            ),
            Competency(
                key="device_profiling",
                label="Profiling and offline caching",
                weight=0.8,
                signals=(
                    "instruments",
                    "systrace",
                    "perfetto",
                    "time profiler",
                    "offline cache",
                    "sqlite migration",
                    "thread scheduling",
                ),
                probe="Which profiler trace told you where cold start was going?",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="no_profiling",
                description=(
                    "No profiling experience with Instruments/Systrace, or weak "
                    "understanding of thread scheduling."
                ),
                competency_key="device_profiling",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.SECURITY_DEVSECOPS_ENGINEER,
        title="Security & DevSecOps Engineer",
        summary=(
            "Owns the trust boundary: identity, threat models, and a build pipeline "
            "whose output can be attested."
        ),
        competencies=(
            Competency(
                key="zero_trust_identity",
                label="Zero-Trust identity",
                weight=1.0,
                signals=(
                    "zero-trust",
                    "zero trust",
                    "mtls",
                    "workload identity",
                    "oauth2",
                    "oidc",
                    "pkce",
                    "token binding",
                    "short-lived credential",
                    "hsm",
                    "key rotation",
                ),
                probe="Why is PKCE required for a public client even with a client secret?",
            ),
            Competency(
                key="threat_modeling",
                label="Threat modelling",
                weight=0.9,
                signals=(
                    "stride",
                    "threat model",
                    "attack surface",
                    "trust boundary",
                    "abuse case",
                    "blast radius",
                    "least privilege",
                ),
                probe="STRIDE this screening agent for me. Where is its worst exposure?",
            ),
            Competency(
                key="vulnerability_remediation",
                label="Vulnerability remediation",
                weight=0.9,
                signals=(
                    "ssrf",
                    "csrf",
                    "xss",
                    "sql injection",
                    "deserialization",
                    "sast",
                    "dast",
                    "fuzzing",
                    "penetration test",
                ),
                probe="Close an SSRF in a webhook fetcher. What exactly do you change?",
            ),
            Competency(
                key="supply_chain",
                label="Supply chain provenance",
                weight=0.8,
                signals=(
                    "slsa",
                    "sbom",
                    "sigstore",
                    "cosign",
                    "provenance attestation",
                    "reproducible build",
                    "dependency pinning",
                ),
                probe="What does SLSA level 3 actually require of your build system?",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="weak_credential_handling",
                description=(
                    "Recommends symmetric/reversible credential storage, or cannot "
                    "remediate SSRF/CSRF."
                ),
                competency_key="vulnerability_remediation",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.DATA_PLATFORM_ENGINEER,
        title="Data Platform Engineer",
        summary=(
            "Owns the movement and shape of data at rest and in flight: streaming "
            "semantics, table formats, and reproducible transformations."
        ),
        competencies=(
            Competency(
                key="stream_processing",
                label="Stream processing semantics",
                weight=1.0,
                signals=(
                    "apache flink",
                    "flink",
                    "kafka streams",
                    "watermark",
                    "event time",
                    "windowing",
                    "backpressure",
                    "consumer lag",
                    "checkpoint barrier",
                ),
                probe="Late events arrive an hour after the window closed. What happens, and why?",
            ),
            Competency(
                key="table_formats",
                label="Lakehouse table formats",
                weight=0.9,
                signals=(
                    "apache iceberg",
                    "iceberg",
                    "delta lake",
                    "apache hudi",
                    "parquet",
                    "time travel",
                    "schema evolution",
                    "partition pruning",
                    "compaction job",
                ),
                probe="Why does Iceberg's hidden partitioning change how you write queries?",
            ),
            Competency(
                key="batch_tuning",
                label="Distributed batch tuning",
                weight=0.9,
                signals=(
                    "spark shuffle",
                    "shuffle partition",
                    "data skew",
                    "salting",
                    "broadcast join",
                    "adaptive query execution",
                    "spill to disk",
                ),
                probe="One task in a 2000-task stage runs 40x longer. Diagnose it.",
            ),
            Competency(
                key="lineage_reproducibility",
                label="Lineage and reproducibility",
                weight=0.8,
                signals=(
                    "data lineage",
                    "openlineage",
                    "dbt",
                    "idempotent pipeline",
                    "change data capture",
                    "cdc",
                    "medallion architecture",
                    "airflow dag",
                ),
                probe="How do you re-run yesterday's pipeline and get byte-identical output?",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="no_backpressure",
                description=(
                    "Overlooks backpressure handling, or designs non-reproducible data mutations."
                ),
                competency_key="stream_processing",
            ),
        ),
    ),
    RoleRubric(
        role=EngineeringRole.TECHNICAL_SOLUTIONS_ARCHITECT,
        title="Technical Product / Solutions Architect",
        summary=(
            "Owns the decision record: bounded contexts, explicit trade-offs, and "
            "architecture justified by cost, compliance and latency, not fashion."
        ),
        competencies=(
            Competency(
                key="domain_modelling",
                label="Domain-Driven Design",
                weight=1.0,
                signals=(
                    "domain-driven design",
                    "bounded context",
                    "ubiquitous language",
                    "aggregate root",
                    "anti-corruption layer",
                    "context map",
                ),
                probe="Where did you draw a bounded context boundary, and what did it cost you?",
            ),
            Competency(
                key="tradeoff_reasoning",
                label="Explicit trade-off reasoning",
                weight=1.0,
                signals=(
                    "cap theorem",
                    "consistency trade-off",
                    "latency budget",
                    "cost-performance",
                    "total cost of ownership",
                    "tco",
                    "capacity planning",
                    "build vs buy",
                ),
                probe=(
                    "Name an architecture you rejected despite it being the popular "
                    "choice, and the number that decided it."
                ),
            ),
            Competency(
                key="integration_patterns",
                label="Enterprise integration",
                weight=0.8,
                signals=(
                    "enterprise integration pattern",
                    "message broker topology",
                    "strangler fig",
                    "anti-pattern review",
                    "migration strategy",
                    "reference architecture",
                ),
                probe="How did you migrate a monolith without a big-bang cutover?",
            ),
            Competency(
                key="governance",
                label="RFC and decision governance",
                weight=0.8,
                signals=(
                    "architecture decision record",
                    "adr",
                    "rfc process",
                    "design review board",
                    "stakeholder alignment",
                    "compliance boundary",
                    "data residency",
                ),
                probe="Walk me through an RFC you wrote that changed someone else's mind.",
            ),
        ),
        eliminations=(
            EliminationCriterion(
                key="hype_driven",
                description=(
                    "Chooses architecture by hype without defining cost, compliance "
                    "and latency boundaries."
                ),
                competency_key="tradeoff_reasoning",
            ),
        ),
    ),
)

#: Read-only registry. Insertion order == :class:`EngineeringRole` order.
ROLE_RUBRICS: Final[Mapping[EngineeringRole, RoleRubric]] = MappingProxyType(
    {r.role: r for r in _RUBRIC_LIST}
)

assert len(ROLE_RUBRICS) == 9, "The screening product is specified for exactly 9 tracks"
assert set(ROLE_RUBRICS) == set(EngineeringRole), "Every role must have a rubric"


def rubric_for(role: EngineeringRole) -> RoleRubric:
    return ROLE_RUBRICS[role]


# =============================================================================
# Deterministic matcher
# =============================================================================

_NORMALISE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9+.#\- ]+")
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation noise, collapse whitespace.

    Padded with single spaces so that word-boundary checks can be done with
    plain substring containment, which is markedly faster than a regex scan and
    matters when this runs inside the 10ms retrieval budget.
    """
    lowered = text.casefold()
    lowered = _NORMALISE_RE.sub(" ", lowered)
    return f" {_WS_RE.sub(' ', lowered).strip()} "


@lru_cache(maxsize=1)
def signal_idf() -> Mapping[str, float]:
    """Inverse role frequency for every signal across all nine rubrics.

    A signal claimed by one rubric is near-decisive; a signal claimed by six is
    almost noise. ``log(1 + N/df)`` gives a smooth discount with no zero terms.
    """
    n_roles = len(ROLE_RUBRICS)
    doc_freq: dict[str, int] = {}
    for rubric in ROLE_RUBRICS.values():
        for signal in set(rubric.all_signals()):
            doc_freq[signal] = doc_freq.get(signal, 0) + 1
    return MappingProxyType({sig: math.log1p(n_roles / df) for sig, df in sorted(doc_freq.items())})


def _signal_hit(haystack: str, signal: str) -> bool:
    """Containment with word-ish boundaries, without paying for a regex."""
    needle = signal.casefold()
    idx = haystack.find(needle)
    while idx != -1:
        before = haystack[idx - 1] if idx else " "
        after_idx = idx + len(needle)
        after = haystack[after_idx] if after_idx < len(haystack) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        idx = haystack.find(needle, idx + 1)
    return False


@dataclass(frozen=True, slots=True)
class CompetencyMatch:
    """Per-competency evidence score in [0, 1] plus the signals that fired."""

    key: str
    label: str
    score: float
    weight: float
    matched_signals: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoleMatch:
    """A role's weighted fit against one body of evidence."""

    role: EngineeringRole
    score: float
    competencies: tuple[CompetencyMatch, ...]

    @property
    def matched_signal_count(self) -> int:
        return sum(len(c.matched_signals) for c in self.competencies)

    def competency_scores(self) -> dict[str, float]:
        """Payload-shaped mapping, rounded for stable storage and diffing."""
        return {c.key: round(c.score, 4) for c in self.competencies}


def score_role(evidence: str, role: EngineeringRole) -> RoleMatch:
    """Score one body of evidence against one rubric. Pure and deterministic."""
    rubric = ROLE_RUBRICS[role]
    haystack = normalise(evidence)
    idf = signal_idf()

    matches: list[CompetencyMatch] = []
    weighted_total = 0.0
    for competency in rubric.competencies:
        possible = sum(idf[s] for s in competency.signals)
        hit_signals = tuple(s for s in competency.signals if _signal_hit(haystack, s))
        achieved = sum(idf[s] for s in hit_signals)
        score = (achieved / possible) if possible else 0.0
        # Saturating curve: hitting half a competency's signals is already
        # strong evidence, and a resume that keyword-stuffs all of them should
        # not out-score a candidate who demonstrated depth on a few.
        score = min(1.0, score**0.6) if score > 0 else 0.0
        matches.append(
            CompetencyMatch(
                key=competency.key,
                label=competency.label,
                score=score,
                weight=competency.weight,
                matched_signals=hit_signals,
            )
        )
        weighted_total += score * competency.weight

    return RoleMatch(
        role=role,
        score=weighted_total / rubric.total_weight if rubric.total_weight else 0.0,
        competencies=tuple(matches),
    )


def rank_roles(evidence: str) -> tuple[RoleMatch, ...]:
    """Score every rubric, best first.

    Ties break on matched-signal count and then on :class:`EngineeringRole`
    declaration order, so the ranking is total and never depends on dict
    iteration or floating-point noise.
    """
    order = {role: i for i, role in enumerate(EngineeringRole)}
    matches = [score_role(evidence, role) for role in EngineeringRole]
    matches.sort(key=lambda m: (-round(m.score, 9), -m.matched_signal_count, order[m.role]))
    return tuple(matches)


def best_role(evidence: str) -> RoleMatch:
    """The single best-fitting rubric for this evidence."""
    return rank_roles(evidence)[0]


def match_confidence(ranked: Sequence[RoleMatch]) -> float:
    """Separation between the top match and the runner-up, in [0, 1].

    A low value means the resume did not commit to a track -- the interviewer
    should ask a routing question before burning probe turns on the wrong
    rubric.
    """
    if not ranked:
        return 0.0
    if len(ranked) == 1 or ranked[0].score <= 0:
        return 1.0 if ranked[0].score > 0 else 0.0
    return max(0.0, min(1.0, (ranked[0].score - ranked[1].score) / ranked[0].score))


__all__ = [
    "ROLE_RUBRICS",
    "Competency",
    "CompetencyMatch",
    "EliminationCriterion",
    "EngineeringRole",
    "RoleMatch",
    "RoleRubric",
    "best_role",
    "match_confidence",
    "normalise",
    "rank_roles",
    "rubric_for",
    "score_role",
    "signal_idf",
]
