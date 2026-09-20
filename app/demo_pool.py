"""A sample candidate pool, shaped like something a scraper would emit.

Used by ``python -m app.main seed`` to populate a fresh deployment so the
dashboard has something real to rank. Deliberately messy and inconsistent --
different keys present on different records, free-text experience, some
records much thinner than others -- because that is what scraped data looks
like, and a demo pool where every record is tidy would hide exactly the
behaviour worth demonstrating: the analyst saying "there isn't enough here".

These are fictional people. The contact details are deliberately fake and
exist to show the PII scrubber redacting them on import.
"""

from __future__ import annotations

from typing import Any

DEMO_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "Priya Raman",
        "headline": "Senior ML Systems Engineer",
        "location": "Bengaluru, India",
        "years_experience": 6,
        "email": "priya.raman@example.com",
        "phone": "+91 98765 43210",
        "skills": ["PyTorch", "FSDP", "Triton", "vLLM", "CUDA", "NCCL", "Kubernetes"],
        "experience": [
            {
                "company": "Inference infrastructure startup",
                "role": "Senior ML Systems Engineer",
                "years": 3,
                "detail": (
                    "Led FSDP distributed training across 512 A100s. Retuned NCCL "
                    "collectives and added gradient checkpointing, cutting step time 34%. "
                    "Wrote fused Triton attention kernels; moved serving to vLLM with paged "
                    "attention, reducing KV-cache footprint 40%."
                ),
            },
            {
                "company": "Large consumer platform",
                "role": "ML Engineer",
                "years": 3,
                "detail": "Ranking models, feature pipelines, online/offline skew debugging.",
            },
        ],
        "github": "github.com/example-priya",
        "notes": "Tuned an HNSW index over 200M embeddings, trading recall against p99.",
    },
    {
        "name": "Daniel Okafor",
        "headline": "Staff Backend Engineer, distributed systems",
        "location": "Lagos, Nigeria",
        "years_experience": 9,
        "email": "d.okafor@example.com",
        "skills": ["Go", "gRPC", "Kafka", "Postgres", "Kubernetes", "Terraform"],
        "experience": [
            {
                "company": "Payments company",
                "role": "Staff Engineer",
                "years": 4,
                "detail": (
                    "Owned the ledger service: 40k req/s peak, exactly-once semantics on the "
                    "settlement path with idempotent Kafka consumers. Cut p99 from 900ms to "
                    "120ms by replacing a synchronous fan-out with a materialised read model. "
                    "Ran the migration live with zero downtime over six weeks."
                ),
            },
            {
                "company": "Logistics scale-up",
                "role": "Senior Backend Engineer",
                "years": 5,
                "detail": "Multi-region Postgres, schema migrations under load, on-call lead.",
            },
        ],
        "notes": "Writes detailed design docs; three referenced in public engineering blog.",
    },
    {
        "name": "Mei Lin Chen",
        "headline": "Frontend Platform Engineer",
        "location": "Singapore",
        "years_experience": 7,
        "skills": ["TypeScript", "React", "Vite", "Web Vitals", "Module Federation"],
        "experience": [
            {
                "company": "B2B SaaS",
                "role": "Frontend Platform Lead",
                "years": 4,
                "detail": (
                    "Rebuilt the design system in React with a strict TypeScript token "
                    "pipeline adopted by 6 product teams. Cut LCP from 4.1s to 1.3s via code "
                    "splitting, font preloading and removing a render-blocking analytics "
                    "bundle. Migrated to module federation so teams deploy independently."
                ),
            }
        ],
        "github": "github.com/example-meilin",
        "notes": "Maintains a popular open-source a11y linting plugin.",
    },
    {
        "name": "Tomás Herrera",
        "headline": "DevOps / SRE",
        "location": "Madrid, Spain",
        "years_experience": 5,
        "phone": "+34 600 123 456",
        "skills": ["AWS", "Terraform", "Prometheus", "ArgoCD", "Python"],
        "experience": [
            {
                "company": "Health-tech",
                "role": "SRE",
                "years": 3,
                "detail": (
                    "Cut cloud spend 38% by rightsizing and moving batch to spot with "
                    "checkpointing. Built the incident review process; MTTR fell from 4h to "
                    "45m over two quarters."
                ),
            }
        ],
        "notes": "Strong on reliability process. Less evidence of deep systems programming.",
    },
    {
        "name": "Aisha Bello",
        "headline": "Data Platform Engineer",
        "location": "Remote (UTC+1)",
        "years_experience": 8,
        "skills": ["Spark", "Flink", "dbt", "Airflow", "Iceberg", "Scala"],
        "experience": [
            {
                "company": "Retail analytics",
                "role": "Lead Data Engineer",
                "years": 5,
                "detail": (
                    "Moved a nightly Spark batch to Flink streaming, dropping end-to-end "
                    "freshness from 14h to under 3 minutes for 400M events/day. Introduced "
                    "Iceberg for schema evolution after two costly backfills."
                ),
            }
        ],
        "notes": "Mentored four engineers to senior; runs the internal data guild.",
    },
    {
        "name": "Jonas Weber",
        "headline": "Full-stack Engineer",
        "location": "Berlin, Germany",
        "years_experience": 4,
        "skills": ["Node.js", "React", "Postgres", "Docker"],
        "experience": [
            {
                "company": "Early-stage startup",
                "role": "Full-stack Engineer",
                "years": 2,
                "detail": "Built and shipped the customer dashboard and billing integration.",
            }
        ],
        "notes": "Generalist. Record is light on scale, measurement or trade-off detail.",
    },
    {
        "name": "Sofia Kowalski",
        "headline": "Security Engineer",
        "location": "Warsaw, Poland",
        "years_experience": 6,
        "skills": ["Threat modelling", "Go", "eBPF", "Kubernetes", "SAST"],
        "experience": [
            {
                "company": "Fintech",
                "role": "Product Security Engineer",
                "years": 4,
                "detail": (
                    "Found and fixed an auth bypass in the partner API affecting token "
                    "scoping. Built an eBPF-based runtime policy agent now running across "
                    "300 nodes. Led the SOC2 technical evidence work."
                ),
            }
        ],
        "notes": "Speaks at regional security meetups.",
    },
    {
        "name": "Ravi Menon",
        "headline": "Mobile Engineer",
        "location": "Chennai, India",
        "years_experience": 3,
        "skills": ["Kotlin", "Swift", "React Native"],
        "experience": [
            {
                "company": "Consumer app",
                "role": "Mobile Engineer",
                "years": 3,
                "detail": "Shipped features across Android and iOS. Reduced cold start by 20%.",
            }
        ],
    },
    {
        "name": "Elena Rossi",
        "headline": "Engineering Manager, Platform",
        "location": "Milan, Italy",
        "years_experience": 11,
        "email": "elena.rossi@example.com",
        "skills": ["Java", "Kubernetes", "Team leadership", "Architecture review"],
        "experience": [
            {
                "company": "Enterprise software",
                "role": "Engineering Manager",
                "years": 5,
                "detail": (
                    "Grew a platform team 4 -> 14 across two locations. Drove the monolith "
                    "decomposition that cut deploy lead time from 9 days to same-day. Still "
                    "reviews architecture; last hands-on commit within the year."
                ),
            }
        ],
        "notes": "Manager track. Depends whether the role needs an IC.",
    },
]


#: Baseline rankings for the demo pool, keyed by name.
#:
#: These are hand-written, not model output. They exist so a fresh deployment
#: shows a fully ranked board the moment it is seeded, instead of nine rows of
#: "--" while someone works out why their API key isn't set. ``seed`` applies
#: them first and then, if a model is configured, overwrites every one of them
#: with a real analysis pass -- so on a live deployment nothing here survives.
#:
#: They are scoped to the same question the analyst answers: is this record
#: worth a hiring manager's time. Each rationale cites evidence from the record
#: above it, and the two PASS verdicts are about thin records rather than bad
#: engineers, which is the distinction the demo is meant to show.
DEMO_BASELINE: dict[str, tuple[float, str, str]] = {
    "Daniel Okafor": (
        0.92,
        "INTERVIEW",
        "Owned a ledger at 40k req/s with exactly-once settlement, and cut p99 from "
        "900ms to 120ms by replacing synchronous fan-out with a materialised read "
        "model -- shipped live over six weeks with no downtime. Names the mechanism, "
        "not just the outcome.",
    ),
    "Priya Raman": (
        0.89,
        "INTERVIEW",
        "Distributed training across 512 A100s with a 34% step-time cut from NCCL "
        "retuning and gradient checkpointing, plus fused Triton kernels and a vLLM "
        "migration worth 40% of the KV-cache footprint. Unusually specific about "
        "where the wins came from.",
    ),
    "Aisha Bello": (
        0.84,
        "INTERVIEW",
        "Took end-to-end freshness from 14 hours to under 3 minutes on 400M events a "
        "day by moving Spark batch to Flink streaming, and adopted Iceberg after two "
        "costly backfills -- a decision with a stated reason behind it. Also mentored "
        "four engineers to senior.",
    ),
    "Sofia Kowalski": (
        0.78,
        "INTERVIEW",
        "Found and fixed an auth bypass in a partner API's token scoping, and built "
        "an eBPF runtime policy agent now running on 300 nodes. Real security "
        "engineering rather than tooling administration, though the record covers one "
        "employer.",
    ),
    "Mei Lin Chen": (
        0.71,
        "INTERVIEW",
        "Cut LCP from 4.1s to 1.3s through code splitting, font preloading and "
        "removing a render-blocking bundle, and built a design system six teams "
        "adopted. Depth is clear; breadth is one role, so the range beyond frontend "
        "platform work is untested here.",
    ),
    "Elena Rossi": (
        0.62,
        "MAYBE",
        "Grew a platform team from 4 to 14 and drove a monolith decomposition that "
        "took deploy lead time from 9 days to same-day, with hands-on commits inside "
        "the year. Strong record -- but it is a manager's record, so the fit depends "
        "entirely on whether this opening is an IC one.",
    ),
    "Tomás Herrera": (
        0.55,
        "MAYBE",
        "Cut cloud spend 38% via rightsizing and spot with checkpointing, and built "
        "an incident review process that moved MTTR from 4h to 45m. The evidence is "
        "operational and process-led; there is little here about systems programming "
        "depth, which may or may not matter for the role.",
    ),
    "Ravi Menon": (
        0.34,
        "PASS",
        "Three years across Android and iOS with one measured result, a 20% cold-start "
        "reduction, and no detail on how. The record is too thin to judge rather than "
        "clearly weak -- a short call would settle it if mobile is the opening.",
    ),
    "Jonas Weber": (
        0.30,
        "PASS",
        "Two years at one early-stage startup, shipping a customer dashboard and "
        "billing integration. No scale, measurement or trade-off detail anywhere in "
        "the record, so there is nothing here to assess against a senior bar.",
    ),
}


__all__ = ["DEMO_BASELINE", "DEMO_CANDIDATES"]
