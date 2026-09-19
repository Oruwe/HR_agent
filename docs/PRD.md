# Product Requirements Document

**Product:** `hr-talent-evaluator` — an AI hiring analyst over scraped candidate data
**Version:** 2.0.0
**Status:** Built, tested, deployed

---

## 1. Problem

A hiring manager with an open engineering role ends up with a folder of
scraped profiles — from a sourcing tool, a LinkedIn export, a scraper someone
on the team wrote. Two hundred records for one opening is normal.

What happens to them is the problem:

1. **They get read by keyword.** Nobody reads 200 profiles carefully, so they
   get filtered on whether the word "Kubernetes" appears — the least
   predictive signal available. A record describing a zero-downtime ledger
   migration and a record listing "distributed systems" as a skill look the
   same to a keyword filter.
2. **The reading is not comparable.** Profile 4 is read at 9am and profile 180
   at 6pm on a different day. The bar drifts. There is no artefact afterwards
   explaining why anyone was dropped.
3. **Thin records are indistinguishable from weak candidates.** A scraped
   profile with three lines on it is usually a scraping failure, not a bad
   engineer — but under time pressure both end up in the same pile.

### 1.1 Why "just use an LLM" isn't the whole answer

Pasting profiles into a chat window one at a time produces a per-candidate
opinion, not a ranking. Ranking is inherently comparative: "stronger than the
other three on distributed systems" is a statement a model can only make if it
was shown the other three. One-at-a-time scoring also drifts — the same record
scores differently depending on what preceded it.

And a model given a hiring question will confidently answer it whether or not
the record supports an answer. Without an explicit instruction to say "the
evidence is thin", it invents a reason, which is worse than silence because it
looks like signal.

---

## 2. Goals and non-goals

### 2.1 Goals

| # | Goal | Measure |
| --- | --- | --- |
| G1 | Rank a whole pool comparatively | One model call covers the pool; scores span the range |
| G2 | Every verdict cites evidence | Each rationale names a project, system, number or role from the record |
| G3 | Thin records are called thin | "Not enough here to judge" is an available and used verdict |
| G4 | Answer questions against the pool | Manager asks in plain language; answers are grounded in stored records |
| G5 | Accept any scraper's output | Import imposes no schema beyond "it's a JSON object" |
| G6 | Zero PII egress | No identifier reaches the model, the database, or the dashboard |
| G7 | Degradation is visible | A configured model that is failing is reported, not silently mocked |
| G8 | Works with nothing configured | No credentials ⇒ a functioning, honestly-labelled board |

### 2.2 Non-goals

- **Interviewing candidates.** There is no candidate-facing surface. An
  earlier version of this project was a voice interviewer; that is removed.
- **Scraping.** Data collection belongs to whatever tool the team already
  uses. This consumes its output.
- **Deciding.** The analyst is advisory. A human makes every hiring decision,
  and the product's language never implies otherwise.
- **ATS features.** No scheduling, no email, no offer workflow.

---

## 3. Users

One: the hiring manager or founder who owns an opening and has a pile of
profiles. They are not a recruiter, they do not have a sourcing team, and the
time they can spend on first-pass filtering is measured in minutes.

---

## 4. Requirements

### 4.1 Import

| # | Requirement |
| --- | --- |
| R1 | Accepts an array of arbitrary JSON objects |
| R2 | Stores each record verbatim, minus redactions — no field is dropped for being unrecognised |
| R3 | Lifts a name and a headline out of the blob for display, checking the keys a scraper is likely to use |
| R4 | A record with no recognisable name is kept as "Unknown candidate", never dropped |
| R5 | Redaction happens at import, in one place, before anything is stored |
| R6 | Reports how many records contained PII |

### 4.2 Ranking

| # | Requirement |
| --- | --- |
| R7 | The whole pool goes to the model in one call |
| R8 | Every candidate gets a score in [0, 1], a verdict, and a rationale |
| R9 | Verdicts are a fixed three-rung ladder: INTERVIEW / MAYBE / PASS |
| R10 | Malformed model output leaves the pool unranked rather than scored arbitrarily |
| R11 | Model output is scrubbed on the way back in — a model can echo PII from a record it was shown |
| R12 | Re-running updates rows in place; it never duplicates candidates |

The verdict ladder is deliberately short. Free-text verdicts cannot be
filtered, sorted or audited, and "strong hire" vs "hire" vs "leaning hire" is a
distinction nobody applies consistently across a pool.

### 4.3 Conversation

| # | Requirement |
| --- | --- |
| R13 | The manager asks a question; the answer is grounded in the stored pool |
| R14 | Prior turns are carried as history |
| R15 | The manager's own question is scrubbed — they will paste a resume into the box |
| R16 | An empty pool answers honestly rather than erroring |

### 4.4 Dashboard

| # | Requirement |
| --- | --- |
| R17 | Three columns: pool context and controls, ranked scores, the analyst |
| R18 | Candidates are ordered best-first, unscored last |
| R19 | Filter by verdict; search across name, role and rationale |
| R20 | Opening a candidate shows the stored record exactly as the model saw it |
| R21 | Model status — offline, live, or degraded — is visible without opening a console |
| R22 | Usable at phone width |

### 4.5 Operations

| # | Requirement |
| --- | --- |
| R23 | Every `/api` route is gated by an admin token when one is set |
| R24 | Health and readiness are never gated |
| R25 | A failing *configured* model is reported as `degraded`, distinctly from offline-by-choice |
| R26 | A fresh deployment can be populated from the UI, with no shell access |

---

## 5. Design decisions

### 5.1 No rubric engine

An earlier version scored candidates against nine hand-written engineering
rubrics with a deterministic matcher. It was removed. The rubric only ever
matched the roles someone had thought to write down, and a keyword matcher
dressed up as judgement is still a keyword matcher — it scored a record
mentioning "Kubernetes" above one describing the operator its author wrote.
Ranking is now the model's, in full, with the rationale shown so the manager
can disagree with it.

### 5.2 Schema-agnostic storage

The scraped record is stored as a JSON blob and rendered to the model with
`json.dumps`, not through a formatter. A formatter that knows about the fields
we happened to see first silently drops everything else the day the scraper
changes — and the scraper is not ours.

### 5.3 The offline provider is a real provider

With no credentials the mock returns well-formed rankings covering exactly the
candidates it was shown, scored by a transparent heuristic (record richness)
that every rationale names as such. This makes the product demonstrable with
zero infrastructure and lets the entire test suite run without network. It is
never presented as judgement.

### 5.4 Fallbacks are counted

`model_configured` means "a key is set", not "that key works". A deployment
whose every call 4xx's is externally indistinguishable from a healthy one —
this project lost hours to that twice, once to a retired model pin
(`gemini-2.0-flash`) and once to thinking tokens consuming the entire output
budget and returning an empty string. Every fallback increments a counter;
`/api/status` exposes it; the dashboard shows an amber pill.

### 5.5 One ingest path

`app/ingest.py` is the only way a record enters the database, used by both the
HTTP route and the CLI. Two ingest paths drift, and the drift shows up as
unredacted records nobody meant to store.

---

## 6. Out of scope for this version

- Per-role rankings (the analyst currently ranks against "worth the manager's
  time", not a specific job description)
- De-duplicating the same person across two scraped sources, though the keyed
  fingerprints make it possible
- Anything write-back to the source system
