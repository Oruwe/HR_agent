# Moss-first HR agent architecture

Moss is the primary live knowledge layer for the interviewer. It indexes the version-controlled engineering-role rubrics at competency granularity and returns the evidence most relevant to a candidate's latest answer.

## Live path

1. At session open, load the Moss index during the fixed greeting.
2. When endpoint speculation begins, query Moss against the sanitized candidate evidence.
3. Give the compact interviewer prompt the retrieved competency context and ask one short, role-relevant follow-up.
4. Stream the first response token to speech immediately.
5. Persist evaluation and telemetry after the live response starts.

This keeps retrieval local to the Moss runtime rather than placing a remote vector database in the candidate-facing path.

## Candidate recommendations

Moss guides the next question; deterministic rubric scoring combines resume evidence and live answers into an auditable recommendation: Strong hire, Consider, or Do not progress. A human reviewer owns all final hiring decisions.

## Resilience

The embedded index is an offline continuity fallback only. It is initialized only when Moss is absent or becomes unavailable. Qdrant is an optional cold archive for completed evaluation records and is never required to begin or conduct a Moss-backed interview.
