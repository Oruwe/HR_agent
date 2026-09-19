# Product Requirements Document

**Product:** `hr-talent-evaluator` — voice-native technical screening agent
**Version:** 1.0.0
**Status:** Built, tested, benchmarked
**Submission:** YC Fall 2026 × Moss — The Zero Latency Builder Sprint, Track 1 (Real-Time Voice & Conversational AI)

---

## 1. Problem

### 1.1 The hiring funnel's worst bottleneck

A mid-size technology company running 40 engineering roles receives on the order
of 200 applications per role. The first technical filter — a 15-minute screening
call — costs roughly 25 minutes of a senior engineer's time once scheduling,
context-switching and write-up are counted.

That produces three failures that compound:

1. **Throughput collapse.** Nobody has 3,300 engineer-hours per hiring cycle, so
   most applicants are filtered on résumé keywords instead — the least
   predictive signal available.
2. **Inconsistency.** Two candidates for the same role get different questions
   from different interviewers on different days. Their scores are not
   comparable, which makes the entire funnel's output unauditable.
3. **Latency in the loop.** Screening backlogs push time-to-first-signal out by
   one to three weeks. Strong candidates accept other offers inside that window.

### 1.2 Why existing voice agents do not solve it

The naïve solution — point an LLM voice agent at the problem — fails on the
dimension candidates notice first.

A standard cascaded pipeline (wait for silence → transcribe → retrieve →
generate → synthesise) takes **1,500–4,000ms per turn**. Human conversational
floor-transfer is roughly **200ms**. An agent operating an order of magnitude
slower is not a slow interviewer; it is an uncanny one. Candidates talk over it,
repeat themselves, and disengage — and this is a conversation in which they are
already being judged.

Worse, the usual fix for "the agent asks generic questions" is retrieval, and
retrieval against a hosted vector database costs 15–40ms of network round trip
before the index does any work. So teams face a false choice: ask worse
questions, or feel slower.

---

## 2. Goals and non-goals

### 2.1 Goals

| # | Goal | Measure |
|---|---|---|
| G1 | Conversational presence | ≤150ms from endpoint commit to first audio byte, p95 |
| G2 | On-rubric questioning | Rubric context retrieved on every turn in <5ms |
| G3 | Comparable evaluations | Identical rubric, identical order, deterministic scores |
| G4 | Interruptible | Queued agent audio stops being heard within 15ms of barge-in |
| G5 | Zero PII egress | No identifier reaches model, store, trace, or disk |
| G6 | Auditable decisions | Every recommendation traces to a numbered rubric line |
| G7 | Operable | Runs with zero infrastructure for dev/CI; degrades, never drops |

### 2.2 Non-goals

- **Not an offer engine.** The agent never extends offers, discusses
  compensation, or commits to next steps. It produces a dossier; humans decide.
- **Not a replacement for a full technical interview.** It is a *screen*, sized
  at 15 minutes and 12 turns.
- **Not a coding assessment.** It cannot evaluate whiteboard diagrams or live
  coding, and says so in `EXPLAINABILITY.md`.
- **Not a résumé parser product.** Résumé text is one evidence channel, not the
  deliverable.

---

## 3. Users

| User | Needs | Success looks like |
|---|---|---|
| **Candidate** | To be evaluated fairly by something that feels present and does not waste their evening | Finishes the call without noticing latency; questions engaged with what they actually said |
| **Hiring manager** | A defensible shortlist, fast | Dossier with per-competency scores, cited evidence, explicit limitations |
| **Recruiter / ops** | Throughput without headcount | Screens run concurrently, 24/7, at consistent quality |
| **Compliance / legal** | No PII retention, explainable automated decisions | Zero-leak invariant enforced by tests; scores trace to rubric lines |

---

## 4. Solution

### 4.1 Core insight

The 1,500ms turnaround is **architectural, not physical**. Most of it is spent
waiting in sequence for work that could have started earlier.

Two changes recover almost all of it:

1. **Speculative turn-taking.** Start retrieving and generating at 40ms of
   silence; commit at 120ms. The 80ms window is paid for by silence the
   candidate is producing anyway. If they resume, discard the draft — the cost
   is compute, not latency.
2. **In-process retrieval.** Moss runs inside the worker, so fetching rubric
   context is a function call rather than a network hop. This is what makes the
   5ms retrieval line item real rather than aspirational.

### 4.2 Why Moss specifically

Retrieval answers one question per turn: *which rubric competency is this
candidate demonstrating right now?* The answer steers the next question.

- Under 5ms → the interviewer stays on-rubric at no perceptible cost.
- At 50ms (hosted vector DB) → either the budget breaks, or you stop retrieving
  and ask worse questions.

Moss is a **search runtime, not a database**, which removes the network hop
entirely and eliminates a service to operate. The rubric corpus — 36 documents,
one per competency — is static and version-controlled, indexed once at session
open behind the fixed greeting, so no turn ever pays for indexing.

### 4.3 Feature set

| Feature | Description |
|---|---|
| Live voice screening | LiveKit WebRTC, 20ms frames, 48kHz mono PCM16, real barge-in |
| Nine engineering tracks | AI/ML Systems, Frontend Platform, Distributed Backend, DevOps/SRE, Full Stack, Mobile Core, Security/DevSecOps, Data Platform, Solutions Architect |
| Deterministic routing | IDF-weighted rubric matching; 9/9 accuracy, reproducible |
| Adaptive probing | Per-competency probe selection, heaviest weight first, coverage-tracked |
| Structured tool calling | `record_candidate_competency`, `trigger_role_transition`, `terminate_screening_session` |
| PII redaction | Aadhaar, PAN, SSN, passport, Korean RRN, Japanese MyNumber, email, phone, address, coordinates, cards, credentials — each replaced with a typed token and a keyed BLAKE2b fingerprint |
| Evaluation dossier | Per-competency scores with evidence source, elimination flags, alternate role fits, recommendation |
| Observability | Per-stage spans, percentile reporting, `latency_exceeded` alarms, PII-gated payloads |

---

## 5. Requirements

### 5.1 Functional

| ID | Requirement | Status |
|---|---|---|
| F1 | Conduct bidirectional voice screening over WebRTC | ✅ |
| F2 | Detect end-of-turn and commit within the configured silence window | ✅ |
| F3 | Interrupt agent playback on candidate barge-in | ✅ |
| F4 | Route candidates to one of nine rubrics deterministically | ✅ |
| F5 | Retrieve the matching competency on every turn | ✅ |
| F6 | Score competencies from live evidence via tool calls | ✅ |
| F7 | Produce a dossier with recommendation and cited limitations | ✅ |
| F8 | Redact all PII before any egress | ✅ |
| F9 | Emit per-stage latency telemetry | ✅ |
| F10 | Operate fully offline with no credentials | ✅ |

### 5.2 Non-functional

| ID | Requirement | Target | Measured |
|---|---|---|---|
| N1 | Turnaround, p95 | ≤150 ms | **12.58 ms** |
| N2 | Rubric retrieval, p95 | ≤5 ms | **1.29 ms** |
| N3 | Barge-in drain | ≤15 ms | **0.036 ms** |
| N4 | Barge-in detection | ≤30 ms sustained speech | 40 ms (2 frames) |
| N5 | Unredacted PII at any boundary | 0 | **0** |
| N6 | False positives on technical prose | 0 | **0** |
| N7 | Role routing accuracy | 9/9 | **9/9** |
| N8 | Test suite warnings | 0 | **0** |

### 5.3 Constraints

- **C1 — Budget coherence.** Stage budgets must sum to exactly 150ms; enforced
  at import, not by convention.
- **C2 — No infrastructure for dev.** The full pipeline and full test suite run
  with zero services.
- **C3 — Graceful degradation.** No single dependency failure may end a live
  interview. Moss, Gemini, LiveKit, Qdrant and Langfuse each degrade
  independently.
- **C4 — Determinism.** Identical input produces byte-identical scrubbing and
  identical role assignment, across processes and machines.

---

## 6. Design decisions

Decisions worth defending, and what was rejected.

### D1 — Role assignment is not a model call
**Chosen:** IDF-weighted signal matching against version-controlled rubrics.
**Rejected:** asking the LLM "which role is this?"
**Why:** role assignment changes what a person is asked and how they are scored.
It must be reproducible and explainable line by line. A model's opinion is
neither, and it cannot be re-derived during a fairness audit two years later.

### D2 — Per-competency documents, not per-role
**Chosen:** 36 documents.
**Rejected:** 9 documents.
**Why:** a whole-rubric document is four unrelated competencies concatenated. A
candidate's answer matches one; averaging against the other three buries the
signal. Measured improvement in top-1 retrieval: 95.1% → 98.6% over 24
independent seeds.

### D3 — Single-pass scrubbing, not a `re.sub` cascade
**Why:** a cascade lets pass *N* match inside pass *N−1*'s output, and every
substitution destroys the offset mapping downstream span annotations depend on.
A structural guard rejects any match overlapping an existing replacement token,
making idempotence a property of the algorithm rather than of each pattern's
care. *This bug was found by a failing test — the generic passport rule was
matching the word `REDACTED` inside `<PHONE_REDACTED>`.*

### D4 — Epoch-tagged buffers for barge-in
**Rejected:** clearing the queue on interrupt.
**Why:** a producer that decided to push *before* the interrupt completes that
push afterwards, landing stale audio in a freshly-cleared buffer. Epoch tagging
makes such a frame invalid on arrival — no lock, no cancellation point, no race.

### D5 — Synthesis and publication are separate tasks
**Why:** a speech engine generates several times faster than realtime while the
wire consumes one frame per 20ms, so audio *will* queue between them — and that
queue is exactly what barge-in must discard. Collapsing them into one loop hides
the queue and ships interruption behaviour that was never actually tested.

### D6 — `INCOMPLETE` outranks `DO_NOT_ADVANCE`
**Why:** a call that dropped after one question must never be recorded as a
rejection. Rejecting a candidate because the network failed is the worst outcome
this system can produce.

### D7 — Agent speech excluded from candidate evidence
**Why:** agent turns contain the rubric's own vocabulary. Scoring against them
would let the interviewer's questions inflate the candidate's score — the
classic self-confirming evaluation bug.

### D8 — Keyed fingerprints, not plain digests
**Chosen:** keyed BLAKE2b (a MAC) over every redacted identifier, with an
ephemeral key when none is configured.
**Rejected:** a plain digest, or a constant default key.
**Why:** an unkeyed hash of a 10-digit phone number is not anonymised data —
the search space is 10¹⁰ and a laptop exhausts it in seconds, so the digest *is*
the number. Keying makes a guess unconfirmable. Defaulting to an ephemeral key
when none is set means the failure mode is "fingerprints stop linking", which is
visible, rather than "fingerprints are guessable", which is not.

### D9 — Fixed greeting, not a generated one
**Why:** the greeting is the only turn with no prior audio to hide latency
behind. Generating it would make the interview's first impression its slowest
response.

---

## 7. Metrics

### 7.1 Product

| Metric | Baseline (human screen) | Target |
|---|---|---|
| Cost per screen | ~25 engineer-minutes | <$1 compute |
| Time to first signal | 1–3 weeks | <24 hours |
| Screens per role | ~8 (budget-limited) | Unlimited |
| Score comparability | Interviewer-dependent | Identical rubric, identical order |

### 7.2 Technical (gates CI)

- Turnaround p95 ≤150ms, no turn beyond 2× budget.
- Retrieval p95 ≤5ms.
- Zero unredacted PII, zero false positives on the technical corpus.
- 9/9 deterministic role routing.
- 250 tests, zero warnings.

### 7.3 Fairness (operational)

- Elimination-flag rate per role, monitored for drift.
- `HOLD_FOR_HUMAN_REVIEW` rate — a rising rate means rubric gates are probing
  badly, not that candidates got worse.
- Turn count distribution — truncated interviews indicate transport problems.

---

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Provider TTFT exceeds the speculation window | Turnaround degrades toward provider latency | Speculation absorbs 80ms; budget and alarms make the overage visible rather than silent |
| Retrieval outage mid-call | Interviewer drifts off-rubric | Automatic fallback to the embedded index; tested explicitly for mid-call failure |
| Accented or noisy audio reduces VAD accuracy | Early cut-off or missed turns | Adaptive noise floor, zero-crossing gate, asymmetric hysteresis; documented in `EXPLAINABILITY.md` |
| Scrubber false positive destroys evidence | Competency scored unfairly low | Dedicated false-positive corpus in CI; zero tolerance |
| Candidate injects instructions into speech | Prompt injection | Guardrails treat candidate speech as data; routing is deterministic and not model-controlled |
| Automated decisions face regulatory scrutiny | Legal exposure | Every score traces to a numbered rubric line; no protected attributes are inputs; humans make all decisions |

---

## 9. Scope status

**In scope and delivered:** the nine rubrics, voice pipeline with barge-in, Moss
retrieval with fallback, PII redaction, deterministic evaluation, telemetry,
OpenGAP compliance, 250 tests, reproducible benchmark, offline demo.

**Deliberately out of scope for this build:**
- Multilingual screening (rubric signals are English).
- Video / whiteboard evaluation.
- ATS integrations (Greenhouse, Lever).
- Candidate-facing scheduling UI.
- Model fine-tuning — rubrics are version-controlled data, which is the point.

---

## 10. Appendix — measured results

```
Retrieval (rubric lookup)      p50 1.11ms   p95 1.29ms   p99 1.42ms   budget 5ms     PASS
Turnaround (with speculation)  p50 12.44ms  p95 12.58ms  p99 12.79ms  budget 150ms   PASS
Turnaround (sequential)        p50 43.30ms  p95 43.57ms  p99 43.67ms  budget 150ms   PASS

Speculation saves a median of 30.9ms per turn — 71% of the sequential cost.
```

Reproduce with `python scripts/benchmark.py --iterations 60`. No credentials
required.

**Measurement honesty:** these are measured against modelled component costs
(30ms cognition TTFT, 12ms to first audio frame) on the offline providers. Real
end-to-end turnaround adds whatever your inference provider's TTFT exceeds the
80ms speculation window. The claim being made is not that physics was defeated;
it is that **the orchestration contributes ~13ms rather than ~1,500ms**, and
every millisecond of retrieval moved off the network is a millisecond kept.
