# Architecture

`hr-talent-evaluator` — sub-150ms voice screening agent, Moss on the retrieval hot path.

---

## 1. System overview

```mermaid
flowchart TB
    subgraph EDGE["Candidate device"]
        MIC["Microphone<br/>48kHz mono PCM16"]
        SPK["Speaker"]
    end

    subgraph TRANSPORT["Transport — LiveKit WebRTC"]
        SFU["LiveKit SFU<br/>20ms frames"]
    end

    subgraph WORKER["Screening worker (single process)"]
        direction TB
        RING["Dual-track ring buffer<br/>epoch-tagged, bounded 400ms"]
        VAD["VAD state machine<br/>Silero v5 / adaptive energy<br/>2 thresholds: 120ms | 250ms"]
        ORCH["Screening orchestrator<br/>speculation · turn accounting"]
        FLOW["Interview flow<br/>stage machine · probe selection"]
        SYNTH["Streaming synthesis<br/>clause chunking · abortable"]
        SCRUB["PII scrubber<br/>single-pass · offset-traceable"]
    end

    subgraph RETRIEVAL["Retrieval — hot path"]
        MOSS["Moss runtime<br/>in-process · no vector DB<br/>36 competency documents"]
        FALLBACK["Embedded index<br/>exact hybrid · fallback"]
    end

    subgraph COLD["Cold path — off the conversational loop"]
        COG["Streaming cognition<br/>temperature 0.2 · 150 tok"]
        QDRANT["Qdrant archive<br/>completed dossiers"]
        TRACE["Langfuse + OTel<br/>PII-gated spans"]
    end

    MIC -->|"20ms"| SFU --> RING --> VAD
    VAD -->|"SPECULATE @40ms"| ORCH
    VAD -->|"TURN_COMMIT @120ms"| ORCH
    VAD -->|"BARGE_IN @30ms"| RING

    ORCH -->|"&lt;5ms"| MOSS
    MOSS -.->|"any failure"| FALLBACK
    ORCH --> FLOW
    ORCH --> COG
    COG --> SYNTH --> RING
    RING -->|"paced publish"| SFU --> SPK

    ORCH --> SCRUB
    SCRUB --> COG
    SCRUB --> QDRANT
    SCRUB --> TRACE

    classDef hot fill:#1f6f4a,stroke:#0d3d28,color:#fff
    classDef cold fill:#2b4a6f,stroke:#16283d,color:#fff
    classDef sec fill:#7a3b2e,stroke:#40201a,color:#fff
    class MOSS,FALLBACK,VAD,RING,ORCH hot
    class COG,QDRANT,TRACE cold
    class SCRUB sec
```

**Green** is the conversational hot path, governed by the 150ms budget.
**Blue** is everything that must never block a turn.
**Red** is the security boundary every outbound payload crosses.

---

## 2. The speculation timeline

This is the mechanism that makes the budget reachable.

```mermaid
sequenceDiagram
    autonumber
    participant C as Candidate
    participant V as VAD
    participant O as Orchestrator
    participant M as Moss
    participant L as Cognition
    participant S as Synthesis

    C->>V: speech frames (20ms each)
    Note over V: CANDIDATE_SPEAKING

    C-->>V: stops speaking (t = 0)

    rect rgb(232, 245, 236)
    Note over V,S: Speculation window — paid for by silence the candidate is producing anyway
    V->>O: SPECULATE (t = 120ms)
    O->>M: which competency is this?
    M-->>O: hit in 0.6ms
    O->>L: begin streaming generation
    L-->>O: tokens buffering...
    end

    alt Candidate resumes talking
        C->>V: speech resumes
        V->>O: SPECULATION_CANCELLED
        O->>L: abort — draft discarded, cost is compute only
    else Silence holds
        V->>O: TURN_COMMIT (t = 250ms)
        Note over O: tokens already buffered
        O->>S: first clause, immediately
        S-->>C: first audio byte at commit + 12.5ms
    end
```

Without speculation the same turn costs **43ms** after commit: retrieval, then
time-to-first-token, then synthesis, all serialised behind the candidate's wait.
With it, **12.5ms** — a 71% reduction, asserted directly by
`test_speculation_is_what_buys_the_budget`.

---

## 3. Barge-in: epoch invalidation

The failure a naive implementation ships: cancel the producer, but the queue
between synthesiser and wire keeps draining, so the agent talks over the
candidate for another few hundred milliseconds.

```mermaid
flowchart LR
    subgraph T0["Agent speaking — epoch 0"]
        P0["Synthesiser<br/>generating faster<br/>than realtime"] -->|push| B0["Egress buffer<br/>epoch 0<br/>20 frames queued"]
        B0 -->|"1 frame / 20ms"| W0["Wire"]
    end

    subgraph T1["Candidate interrupts"]
        D["1. drain() → epoch := 1<br/>2. synthesiser.abort()<br/>3. transport.clear_egress()"]
    end

    subgraph T2["After — epoch 1"]
        P1["In-flight frame<br/>stamped epoch 0"] -->|"push refused"| B1["Egress buffer<br/>epoch 1<br/>empty"]
        B1 --> W1["Wire — silent"]
    end

    T0 --> T1 --> T2

    classDef bad fill:#7a3b2e,stroke:#40201a,color:#fff
    classDef good fill:#1f6f4a,stroke:#0d3d28,color:#fff
    class P1 bad
    class B1,W1 good
```

Order is load-bearing. The epoch is bumped **first**, retroactively invalidating
every frame already in flight; only then is the producer cancelled. Cancelling
first leaves a window where the synthesiser's final frame lands in a
freshly-cleared buffer and gets published after the interrupt.

Measured: **0.036ms** to drop 400ms of queued audio, zero frames published
afterwards, agent recovers the floor for the next turn.

---

## 4. Retrieval corpus layout

```mermaid
flowchart TB
    R["9 role rubrics<br/>version controlled"] --> SPLIT{"Index granularity"}

    SPLIT -->|"rejected"| WHOLE["9 documents<br/>one per role<br/>~380 features each"]
    SPLIT -->|"chosen"| PER["36 documents<br/>one per competency<br/>~20 features each"]

    WHOLE --> PROB["Answer matches 1 of 4<br/>competencies; averaged<br/>against 3 irrelevant ones<br/>→ signal buried"]
    PER --> GOOD["Tight documents.<br/>Returns WHICH competency<br/>matched — the thing the<br/>interviewer needs"]

    PER --> MOSS["Moss index<br/>hybrid semantic + keyword"]
    MOSS --> Q["query(evidence, top_k=9)"]
    Q --> COLLAPSE["Collapse: best competency per role"]
    COLLAPSE --> CTX["Interviewer context<br/>'reads as evidence for: …'"]

    classDef bad fill:#7a3b2e,stroke:#40201a,color:#fff
    classDef good fill:#1f6f4a,stroke:#0d3d28,color:#fff
    class WHOLE,PROB bad
    class PER,GOOD good
```

Over-fetching (`top_k = 3 × limit`) then collapsing per role is deliberate:
several competencies of the *same* role routinely match one answer, and the
caller wants distinct roles back, not three slices of one.

---

## 5. Two independent scoring channels

Role assignment affects a person's career, so it is never a single model call.

```mermaid
flowchart LR
    RESUME["Resume text<br/>(scrubbed)"] --> EV["Candidate evidence"]
    SPEECH["Candidate speech only<br/>agent turns excluded"] --> EV

    EV --> DET["Deterministic matcher<br/>IDF-weighted signals<br/>reproducible · auditable"]
    EV --> MOSSC["Moss retrieval<br/>semantic + keyword"]

    DET -->|"authoritative"| ROUTE["Role assignment"]
    MOSSC -->|"corroboration"| ROUTE

    LIVE["Interviewer scores<br/>via tool calls"] --> BLEND["Dossier blend"]
    DET --> BLEND
    BLEND --> GATES["Gate ordering<br/>1. INCOMPLETE beats rejection<br/>2. elimination beats aggregate<br/>3. threshold"]
    GATES --> REC["Recommendation"]
```

Agent speech is excluded from the evidence on purpose: it contains the rubric's
own vocabulary, so scoring against it would let the interviewer's questions
inflate the candidate's score — the classic self-confirming evaluation bug.

Where a competency was probed live, the interviewer's score wins: a demonstrated
explanation is stronger evidence than a mention on a resume. The matcher fills
the gaps so an unreached competency still carries its resume signal rather than
scoring zero.

---

## 6. Security boundary

```mermaid
flowchart LR
    IN["Resume · live transcript"] --> S["Single-pass scrubber"]

    S --> S1["1. Scan all patterns<br/>against ORIGINAL text"]
    S1 --> S2["2. Reject matches inside<br/>existing tokens<br/>→ structural idempotence"]
    S2 --> S3["3. Resolve overlaps<br/>by fixed precedence"]
    S3 --> S4["4. Splice once<br/>+ build offset map"]

    S4 --> G{"assert_zero_pii<br/>at every boundary"}
    G -->|pass| LLM["Cognition"]
    G -->|pass| VEC["Vector store"]
    G -->|pass| TEL["Telemetry"]
    G -->|pass| DISK["Dossier archive"]
    G -->|fail| EX["SecurityBreachException<br/>message never quotes<br/>the offending value"]

    classDef sec fill:#7a3b2e,stroke:#40201a,color:#fff
    class G,EX sec
```

The gate **re-scans** rather than trusting a flag, because the failure that
actually happens in production is text being concatenated back together after
scrubbing.

---

## 7. Latency budget allocation

```mermaid
pie showData
    title 150ms turnaround budget (ms)
    "cognition TTFT" : 45
    "speech synthesis" : 35
    "VAD endpoint" : 20
    "transport egress" : 20
    "transport ingress" : 15
    "audio ingestion" : 10
    "vector match (Moss)" : 5
```

`assert_budget_is_coherent()` runs at import and refuses to start the process if
these stop summing to 150 — the budget table is the one place an engineer is
tempted to "just add 5ms" under deadline.

---

## 8. Deployment topology

```mermaid
flowchart TB
    subgraph REGION["Per-region deployment"]
        LK["LiveKit SFU"]
        subgraph POD["Screening worker pod"]
            APP["hr-talent-evaluator<br/>+ Moss runtime in-process"]
        end
    end

    CAND["Candidates"] -->|WebRTC| LK --> POD
    POD -->|"streaming, cold path"| LLM["Inference provider"]
    POD -->|"async, off hot path"| ARCH["Qdrant archive"]
    POD -->|"async, scrubbed"| OBS["Langfuse / OTel"]

    classDef hot fill:#1f6f4a,stroke:#0d3d28,color:#fff
    class APP,LK hot
```

Moss ships **inside the worker pod**. There is no retrieval tier to scale, no
network hop to budget for, and no vector database to operate — which is the
entire reason the 5ms line item is real. Workers are stateless between
sessions and scale horizontally per concurrent interview.