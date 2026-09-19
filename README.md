# hr-talent-evaluator

**A voice-interactive HR technical screening agent with a 150ms conversational turnaround budget, built on [Moss](https://www.moss.dev) for sub-5ms rubric retrieval.**

Built for **YC Fall 2026 Ã— Moss: The Zero Latency Builder Sprint** â€” Track 1, Real-Time Voice and Conversational AI.

This repository is two things layered together:

1. **The voice orchestration core** (`app/agent`, `app/voice`, `app/storage`) â€”
   the speculative turn-taking pipeline described below, runnable standalone
   with zero infrastructure via `python -m app.main demo`.
2. **A full application built on top of it** â€” an **API Gateway**
   (`app/api`), a **Primary DB** (`app/db`), and two frontends, a
   **Candidate App** and an **Admin Dashboard** (`frontend/`), so the pipeline
   is reachable over HTTP from a browser, not just a terminal.

```bash
git clone https://github.com/Oruwe/HR_agent && cd HR_agent

# 1. Offline voice pipeline demo (no services required)
pip install -r requirements.txt
python -m app.main demo        # full screening interview, no credentials required
python -m app.main verify      # OpenGAP / budget / security compliance report
python scripts/benchmark.py    # reproduce every latency number below
pytest -q                      # 276 tests, zero warnings

# 2. Full stack (API + Candidate App + Admin Dashboard), one command
cp .env.example .env
docker compose up --build      # API on :8000, frontend on :5173
```

See [**Running the full stack**](#running-the-full-stack) below for the
non-Docker path, and [**Final architecture-to-code audit**](#final-architecture-to-code-audit)
for an honest, item-by-item account of what's implemented versus partial or
blocked.

---

## The problem

A technical screening call is a conversation, and conversations have a metronome.
Humans take the floor back in roughly 200ms. A voice agent that answers in 1.5
seconds is not a slow assistant â€” it is a different kind of object, one that
candidates talk over, interrupt, and stop trusting. For a screening interview
that matters twice over: the latency *is* the product experience, and a
candidate who is being evaluated is already nervous enough without a robot
pausing meaningfully before every follow-up.

The standard cascaded loop â€” wait for silence, transcribe, retrieve, generate,
synthesise â€” spends 1500â€“4000ms per turn. This repository is an argument that
most of that is architectural rather than physical.

## What was actually built

A production-shaped screening agent that:

- conducts a **live, bidirectional voice interview** over LiveKit WebRTC, with
  real barge-in;
- **routes candidates across nine engineering rubrics** deterministically, so
  the assignment is reproducible and defensible rather than a model's opinion;
- **retrieves rubric context through Moss in under 5ms**, in-process, on the
  conversational hot path;
- **redacts PII deterministically** before anything reaches a model, a vector
  store, a trace, or a disk â€” verified by 82 tests;
- **measures its own latency budget** per stage and fails CI when it regresses.

---

## Results

Measured on the offline providers via `python scripts/benchmark.py --iterations 60`.
Reproducible on a laptop with no services running.

| Metric | p50 | p95 | p99 | max | Budget |
|---|---:|---:|---:|---:|---:|
| **Rubric retrieval** | 1.11 ms | 1.29 ms | 1.42 ms | 1.44 ms | 5 ms |
| **Turnaround** (commit â†’ first audio byte) | **12.44 ms** | **12.58 ms** | 12.79 ms | 12.91 ms | 150 ms |
| Turnaround, sequential baseline | 43.30 ms | 43.57 ms | 43.67 ms | 43.72 ms | 150 ms |

**Speculative turn-taking removes a median of 30.9ms per turn â€” 71% of the
sequential cost.**

### What these numbers do and do not include

Being precise about this matters more than the headline.

- **Measured:** the orchestration this repository controls â€” endpointing,
  speculation, retrieval, chunking, buffering, egress.
- **Modelled:** component costs are simulated at realistic values (30ms
  cognition time-to-first-token, 12ms to first synthesised audio frame). Using a
  live model here would make the benchmark a measurement of somebody else's
  network and would hide our own regressions in the variance.
- **Not included:** real WAN round-trip time to a hosted LLM. From Bangalore to
  a hosted inference endpoint that is realistically 200â€“400ms of TTFT.

So the honest claim is not "this pipeline is faster than the speed of light."
It is this: **the 130ms speculation window plus in-process retrieval absorb the
costs that would otherwise land on the candidate's ear.** With a hosted model,
turnaround equals whatever the provider's TTFT exceeds the speculation window,
plus about 13ms of local pipeline. Every millisecond of retrieval you move
off the network is a millisecond you get to keep â€” which is precisely why Moss
is on the hot path and a hosted vector database is not.

---

## Why Moss

The budget allocates **5ms** to "fetch the context the interviewer needs."
That line item is what makes the whole budget either real or fictional.

A hosted vector database cannot participate in it honestly. A round trip to a
managed service costs 15â€“40ms *before the index does any work*, so the 5ms
line silently becomes 50ms and the 150ms total is arithmetic that does not
survive contact with production.

Moss is a **search runtime, not a database**. It runs in-process â€” browser,
edge, device, or cloud â€” so a query is a function call rather than a network
hop. That is a different category of thing, and it is the only reason a 5ms
retrieval budget is a design constraint rather than a wish.

Concretely, retrieval here answers one question on every turn: *which rubric
competency is the candidate demonstrating right now?* The answer steers the
next question. Get it in under 5ms and the interviewer stays on-rubric with no
perceptible cost; get it in 50ms and you either blow the budget or stop doing it
and ask worse questions.

**Corpus design.** The nine rubrics are indexed as **36 documents â€” one per
competency**, not nine per role. A whole-rubric document is four unrelated
competencies concatenated; a candidate's answer matches one of them, and
averaging against the other three buries the signal. Per-competency documents
also let retrieval return *which* competency matched, which is the thing the
interviewer actually needs.

**Degradation.** Every Moss failure path falls back to an in-process index with
an identical interface. A retrieval outage must degrade answer quality, never
end a live interview. This is covered by tests, including the case that
actually matters: Moss failing *mid-call*.

> ### A naming caution
> Two unrelated products called "Moss" appear in this system:
> - **Moss (YC F25)** â€” the retrieval runtime. Configured via `MOSS_PROJECT_ID` / `MOSS_PROJECT_KEY`.
> - **MOSS-Speech** â€” an open speech-to-speech model, used for synthesis. Configured via `HRTE_SPEECH_*`.
>
> They share a name and nothing else. The original build spec conflated them; the code keeps them strictly apart.

---

## How the 150ms budget is met

### The budget

| Stage | Budget | Mechanism |
|---|---:|---|
| Transport ingress | 15 ms | LiveKit WebRTC peer connection |
| VAD endpoint | 20 ms | Silero VAD v5 (ONNX, CPU) with an adaptive-energy fallback |
| Audio ingestion | 10 ms | Dual-track 20ms ring buffer |
| **Vector match** | **5 ms** | **Moss, in-process** |
| Cognition TTFT | 45 ms | Streaming generation |
| Speech synthesis | 35 ms | Streaming speech-to-speech, first frame |
| Transport egress | 20 ms | Paced publish, bounded jitter buffer |
| **Total** | **150 ms** | commit â†’ first audio byte |

The table is enforced, not decorative: `assert_budget_is_coherent()` runs at
import and refuses to start if the stages stop summing to 150.

### The mechanism: speculative turn-taking

The largest lever on perceived latency is not the model â€” it is the endpoint
decision. Wait 800ms for silence (a common default) and no amount of inference
speed will make the agent feel present. Wait 150ms and you cut people off,
which is worse.

So the VAD uses **two** thresholds:

```
candidate stops speaking
   â”‚
   â”œâ”€ 40ms  â”€â†’ SPECULATE       retrieval + generation start; still listening
   â”‚
   â”œâ”€ 120ms â”€â†’ TURN_COMMIT     tokens already buffered â†’ speak immediately
   â”‚
   â””â”€ if speech resumes in between â†’ draft discarded, costs nothing but compute
```

The 80ms gap is paid for by silence the candidate is producing anyway. If they
resume talking, the draft is thrown away for free. This is why the measured
turnaround is 12.44ms rather than 43ms â€” and `test_speculation_is_what_buys_the_budget`
asserts that difference directly, so the mechanism cannot silently stop working
while every other latency test still passes.

### Barge-in

When a candidate talks over the agent, queued audio must stop being *heard*, not
merely stop being *generated*. Those differ by however much audio sits between
the synthesiser and the wire.

The egress ring buffer is **epoch-tagged**: draining bumps a counter, and any
frame carrying an older epoch is refused on arrival. A producer coroutine that
decided to push before the interrupt completes that push afterwards and the
frame is simply invalid â€” no lock, no cancellation point, no race.

Measured drain: **0.036ms**, dropping 400ms of queued audio, with zero frames
published afterwards.

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagrams, and
[`docs/PRD.md`](docs/PRD.md) for the product requirements document.

```
app/
â”œâ”€â”€ config.py            latency budget, settings, capability flags
â”œâ”€â”€ schemas/             candidate Â· 9 role rubrics Â· evaluation dossier
â”œâ”€â”€ security/            PII scrubber Â· credential guard
â”œâ”€â”€ storage/
â”‚   â”œâ”€â”€ retrieval.py     â† Moss runtime (hot path) + embedded fallback
â”‚   â”œâ”€â”€ qdrant_client.py cold archive for completed dossiers
â”‚   â””â”€â”€ embeddings.py    multi-probe hashed encoder + BM25 sparse
â”œâ”€â”€ voice/               ring buffer Â· VAD state machine Â· synthesis Â· LiveKit Â·
â”‚                         stt_engine.py (Deepgram + offline mock)
â”œâ”€â”€ agent/               orchestrator Â· prompts Â· interview flow Â· cognition
â”œâ”€â”€ telemetry/           per-stage metrics Â· PII-gated tracing
â”œâ”€â”€ db/                  Primary DB: SQLAlchemy models + Alembic migrations
â””â”€â”€ api/                 API Gateway: FastAPI app, session/admin/health routes,
                          Session State store

frontend/
â””â”€â”€ src/
    â”œâ”€â”€ api/client.ts        typed fetch wrapper over the API Gateway
    â”œâ”€â”€ hooks/                useActiveSession (candidate session persistence),
    â”‚                         useMicRecorder (MediaRecorder wrapper)
    â”œâ”€â”€ pages/candidate/      Intake -> Interview -> Results
    â””â”€â”€ pages/admin/          Dashboard (sessions + system status), session detail
```

---

## Full-stack architecture

The architecture diagram this project targets has thirteen components. Here is
what each one is, and where it lives in this repository:

| Component | Implementation | Notes |
|---|---|---|
| Candidate App | `frontend/src/pages/candidate/*` | React + TypeScript. Resume intake, live turn-by-turn interview (voice + text), results. |
| Admin Dashboard | `frontend/src/pages/admin/*` | Session list, system status, per-session evaluation detail. |
| API Gateway | `app/api/app.py` | FastAPI. CORS, request metrics middleware, routers below. |
| Voice Orchestrator | `app/agent/orchestrator.py` | Unmodified from the original voice-agent core â€” the API layer drives the same `speculate()` / `commit()` sequence the LiveKit worker and the offline demo already used. |
| STT / Speech Recognition | `app/voice/stt_engine.py` | **New.** Real Deepgram prerecorded-REST provider + offline mock, mirroring the existing cognition/synthesis provider pattern. |
| Moss Retrieval | `app/storage/retrieval.py` | Unmodified â€” already on the hot path; the API layer just surfaces `retrieval_backend` in responses. |
| LLM / Agent processing | `app/agent/cognition.py` | Unmodified. |
| Speech Synthesis | `app/voice/moss_engine.py` | Unmodified; `app/api/audio.py` is new glue that packages its frame stream into a WAV for a browser `<audio>` element. |
| Primary DB | `app/db/` | **New.** SQLAlchemy models (`sessions`, `turns`, `evaluations`) + Alembic migrations. SQLite by default, Postgres via `DATABASE_URL`. |
| Session State | `app/api/session_store.py` | **New.** In-process registry of live orchestrators (they hold open streams and can't be serialized), with an optional Redis mirror of session *status* for multi-replica admin visibility. |
| LangFuse Telemetry | `app/telemetry/langfuse_tracer.py` | Unmodified â€” already wired into the orchestrator; unaffected by this work. |
| Metrics & Alerts | `app/api/routes_health.py` | **New.** Prometheus `/metrics` (real counters/histograms wired into actual request handling, not placeholders), `/health`, `/ready`. |
| Docker | `Dockerfile.api`, `frontend/Dockerfile`, `docker-compose.yml` | **New.** |

Two decisions worth being explicit about:

- **The Candidate App and Admin Dashboard are one Vite project with
  route-based separation** (`/` vs `/admin`), not two separate deployables.
  They share an API client and a design-token stylesheet; splitting them into
  separate `npm` projects would have doubled the build/deploy surface for no
  functional benefit at this scale. Each still gets its own Docker service
  conceptually â€” in practice they're the same static bundle.
- **The voice orchestration core was not rewritten.** Every file under
  `app/agent`, `app/voice/{ring_buffer,vad_stream,moss_engine,livekit_worker}.py`,
  and `app/storage` is exactly as it was; the 250 tests covering them still
  pass unchanged. What's new is everything needed to reach that core over
  HTTP from a browser instead of only from a terminal or a LiveKit room.

---

## Deterministic evaluation

Role assignment changes what a person is asked and how they are scored, so it is
**not** delegated to a model. It is IDF-weighted signal matching against
version-controlled rubrics:

- a signal claimed by one rubric (`flashattention`) is near-decisive;
- a signal claimed by six (`kubernetes`) says almost nothing;
- weighting by inverse role frequency makes the matcher discriminative without
  any training data, and makes every score traceable to a numbered rubric line.

Nine hand-written candidate profiles route **9/9** correctly, with the same
ordering on every run. Retrieval agrees independently on all nine â€” meaningful
corroboration, since the two use entirely different mechanisms.

**Recommendation gates, in deliberate order:**

1. **Insufficient evidence wins over everything.** A call that dropped after one
   question is `INCOMPLETE`, never `DO_NOT_ADVANCE`. Rejecting someone because
   the network failed is the worst outcome this system can produce.
2. **Elimination gates beat a high aggregate.** You can be strong on three
   dimensions and still fail the one that defines the role.
3. A strong candidate who trips one gate becomes `HOLD_FOR_HUMAN_REVIEW` â€” the
   gate may simply have been probed badly. That is a human's call.

---

## Zero-PII by construction

Resumes carry Aadhaar numbers, addresses, and phone numbers. Four egress
boundaries can leak them: the model, the vector store, telemetry, and disk.

The scrubber is **single-pass**: every pattern scans the original text, overlaps
resolve by fixed precedence, and the text is spliced once. The obvious
`re.sub` cascade is wrong in two ways that bite â€” pass *N* matches inside pass
*Nâˆ’1*'s output, and every substitution destroys the offset mapping that
downstream span annotations depend on.

Guaranteed and tested: **deterministic**, **idempotent**, **total**, and
**offset-traceable**.

**Cryptographic, not merely lossy.** Every redacted identifier is also reduced
to a **keyed BLAKE2b fingerprint**, so the same phone number submitted twice
produces the same digest and duplicate applications are detectable â€” without
anything reversible ever being stored. Keyed rather than plain, deliberately: a
10-digit phone number is only 10Â¹â° candidates, so an *unkeyed* digest of one is
the number. With `HRTE_REDACTION_KEY` unset the process generates an ephemeral
key, so fingerprints stop linking across restarts rather than becoming
guessable â€” the safe failure, and the one you notice.

| Redacted | Preserved |
|---|---|
| Aadhaar, PAN, SSN, passport, Korean RRN, Japanese MyNumber | `p99 under 45ms` |
| Email, phone (E.164/NANP/India), payment cards | `512 A100s`, `2019-2023` |
| Street addresses, postal codes, GPS coordinates | `HNSW m=16, ef_construct=100` |
| API keys, JWTs, AWS keys, PEM private keys | `99.99 percent uptime` |

That second column is not a footnote. A scrubber that eats `p99 under 45ms`
destroys the exact evidence the rubric scores. **Zero false positives** on the
technical corpus.

Idempotence is *structural*: a regex rejects any candidate match that overlaps a
replacement token, so no detector can re-redact its own output. This was found
by a failing test â€” the generic passport rule was matching the word `REDACTED`
inside `<PHONE_REDACTED>`.

---

## Configuration

Everything has a default that works with **zero infrastructure**. With no
credentials the agent runs fully offline â€” deterministic embeddings, in-process
index, mock cognition and synthesis. That is what CI and `make demo` exercise.

| Variable | Purpose |
|---|---|
| `MOSS_PROJECT_ID` / `MOSS_PROJECT_KEY` | Moss retrieval runtime (hot path) |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | WebRTC transport |
| `GOOGLE_API_KEY` | Streaming cognition |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Tracing |
| `QDRANT_URL` | Cold dossier archive (optional) |
| `HRTE_SPEECH_*` | MOSS-Speech synthesis (distinct from Moss above) |
| `DEEPGRAM_API_KEY` | Speech-to-text for the Candidate App (optional; offline mock otherwise) |
| `DATABASE_URL` | Primary DB â€” empty = SQLite, or a Postgres URL |
| `REDIS_URL` | Session State mirror (optional; in-process otherwise) |
| `HRTE_CORS_ORIGINS` | Allowed origins for the API Gateway |
| `HRTE_ADMIN_TOKEN` | Optional gate on `/api/admin/*` |

See [`.env.example`](.env.example) for the full annotated set.

---

## OpenGAP compliance

`agent.yaml`, `SOUL.md` and `EXPLAINABILITY.md` conform to the GitAgent Passport
specification and are validated by tests rather than by inspection â€”
`SOUL.md` is *loaded at runtime* to build the system prompt, so an edit that
improves the prose but breaks a heading would silently change how the agent
introduces itself.

```bash
python -m app.main verify
```

---

## Testing

```
276 tests, zero warnings (filterwarnings = ["error"])

 16  OpenGAP compliance      manifest, SOUL.md, EXPLAINABILITY.md
 82  PII redaction           coverage Â· false positives Â· algebra Â· crypto Â· gates
 21  latency budget          per-stage, end-to-end, percentiles, speculation
 82  role evaluation         9/9 routing, determinism, gate ordering
 15  barge-in                epoch invalidation, drain deadline, recovery
 34  Moss retrieval          index lifecycle, 9/9 routing, degradation, budget
 26  API / DB / STT layer    session lifecycle, auth, admin, health, migrations
```

The Moss SDK is optional, so a suite that skipped without it would leave the
integration unverified in CI. Instead a fake `moss` module implementing the
documented surface is injected, exercising index construction, query dispatch,
document resolution, per-role collapsing, latency capture, and every failure
branch.

The 26 new tests (`tests/api/`, `tests/test_db_models.py`,
`tests/test_stt_engine.py`) cover: the full candidate turn lifecycle over
HTTP, auth failures (missing/wrong/expired token, unknown session, closed
session), empty/invalid input, admin listing and the optional admin-token
gate, health/readiness/Prometheus format, ORM relationships, and â€” the one
that actually matters for "the migration works, not just exists" â€” running
`alembic upgrade head` against a fresh database in a subprocess.

```bash
pytest -q                    # backend: 276 tests
cd frontend && npm run test  # frontend: 7 tests (API client error handling,
                              # session persistence hook)
```

---

## Running the full stack

### Option A â€” Docker Compose (recommended)

```bash
cp .env.example .env      # fill in real credentials only if you have them;
                           # everything works with none set
docker compose up --build
```

- API Gateway: `http://localhost:8000` (`/api/docs` for OpenAPI, `/health`, `/metrics`)
- Candidate App: `http://localhost:5173/`
- Admin Dashboard: `http://localhost:5173/admin`

Add real Postgres or Redis without touching any code:

```bash
docker compose --profile postgres --profile redis up --build
# then set DATABASE_URL / REDIS_URL in .env to point at them
```

### Option B â€” run each piece locally

```bash
# Backend
pip install -r requirements.txt
make migrate     # alembic upgrade head -> ./data/app.db
make api          # uvicorn --reload on :8000

# Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL=http://localhost:8000
npm run dev             # :5173
```

### Database setup

SQLite by default â€” `make migrate` creates `./data/app.db` with no further
setup. For Postgres, set `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db`
and run `alembic upgrade head`; the schema is identical, SQLAlchemy handles
the dialect difference.

### End-to-end verification (what was actually run, not just written)

This is the verification this project's own testing requirements call for â€”
actually starting the application and exercising the full path, not just
unit-testing pieces in isolation. Run during development of this feature:

```
1. uvicorn app.api.app:app on :8000          -> /health returns 200
2. npm run build && serve dist/ on :5173     -> index.html + JS bundle served
3. POST /api/sessions (resume text)          -> 201, greeting text + WAV audio,
                                                 role routed, PII redaction flag
4. POST /api/sessions/{id}/turns/text (x3)   -> 200 each turn, turnaround_ms
                                                 measured, agent audio returned
5. POST /api/sessions/{id}/close             -> 200, evaluation with
                                                 recommendation + competency scores
6. GET  /api/admin/sessions                  -> session appears, status=closed
7. GET  /api/admin/status                    -> capability flags accurate
                                                 (offline=true, stt_configured=false)
8. GET  /metrics                             -> Prometheus text format,
                                                 hrte_sessions_created_total present
9. OPTIONS preflight from :5173 origin       -> CORS headers present
```

All nine passed. What this does **not** verify (and an honest audit should
say so): interactive browser behavior â€” clicking through the Candidate App in
an actual browser, granting microphone permission, playing back synthesized
audio â€” because this environment has no browser automation tool available.
The build output was verified to compile and to be served correctly; the
React component logic was verified by unit test and by code review, not by
driving a real browser.

---

## Deployment

- **Backend:** `Dockerfile.api` builds a production image; `CMD` runs
  `alembic upgrade head` then `uvicorn`. Point `DATABASE_URL` at a managed
  Postgres instance, set `HRTE_CORS_ORIGINS` to your frontend's real origin
  (never `*` with real candidate data), and set `HRTE_ADMIN_TOKEN` before
  exposing `/api/admin/*` beyond a trusted network.
- **Frontend:** `frontend/Dockerfile` builds a static bundle served by nginx;
  `API_BASE_URL` is injected at *container start* (not baked into the build),
  so the same image can be promoted across environments.
- **Health checks:** `/health` (liveness â€” process is up) and `/ready`
  (readiness â€” database is reachable) are separate on purpose; a transient
  DB blip fails readiness without triggering a liveness-probe restart loop.
- **Logging:** structured via Python's standard `logging`, one line per
  degraded-mode event (STT falling back to mock, transcription failure,
  synthesis failure) â€” see the `logger.warning(...)` calls throughout
  `app/api/*` and `app/voice/stt_engine.py`.
- **What this repo does not include:** a CI/CD pipeline definition, a
  Kubernetes manifest, or a secrets-management integration. Those are
  platform-specific and out of scope for a repository whose job is to be
  correct, not to encode one particular cloud's deployment conventions.

---

## API documentation

Interactive OpenAPI docs are served at `/api/docs` once the API Gateway is
running (`openapi_url=/api/openapi.json`). Summary of the surface:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/sessions` | Create a session from resume text; returns a session token + greeting (text + audio). |
| `POST` | `/api/sessions/{id}/turns/text` | Submit a typed answer; returns the agent's next question. |
| `POST` | `/api/sessions/{id}/turns/audio` | Submit a recorded answer clip; transcribes, then same as above. |
| `POST` | `/api/sessions/{id}/close` | End the call; returns the evaluation dossier. |
| `GET` | `/api/sessions/{id}/evaluation` | Fetch a completed evaluation. |
| `GET` | `/api/admin/sessions` | List sessions (optionally filtered by status). |
| `GET` | `/api/admin/sessions/{id}/evaluation` | Admin view of an evaluation. |
| `GET` | `/api/admin/status` | Capability flags, latency budget, session-state backend. |
| `GET` | `/health` / `/ready` / `/metrics` | Liveness, readiness, Prometheus. |

All `/api/sessions/*` turn/close endpoints require `Authorization: Bearer
<session_token>` from the create-session response. `/api/admin/*` is gated by
the optional `X-Admin-Token` header when `HRTE_ADMIN_TOKEN` is set server-side.

---

## Troubleshooting

- **`alembic upgrade head` fails with "table already exists"** â€” the app's
  own dev-convenience `init_db()` (SQLAlchemy `create_all`, run automatically
  at API startup for zero-setup local dev) already created the tables outside
  Alembic's tracking. Delete `./data/app.db` and re-run, or in production run
  migrations *before* first app startup, as the Dockerfile's `CMD` does.
- **Candidate App shows "voice transcription isn't configured"** â€” expected
  with no `DEEPGRAM_API_KEY` set; type the answer instead. This is the
  documented offline-fallback behavior, not a bug.
- **CORS errors in the browser console** â€” set `HRTE_CORS_ORIGINS` to include
  the frontend's actual origin (`http://localhost:5173` for local dev); the
  default `*` only works when the request carries no credentials.
- **`/ready` returns 503** â€” the database is unreachable. Check
  `DATABASE_URL` and that Postgres (if used) is actually up; `/health` will
  still return 200 since the process itself is fine.
- **Admin dashboard shows nothing** â€” sessions are created against whichever
  API process handled the request; in a multi-replica deployment without
  `REDIS_URL` set, `active_sessions` and live-orchestrator state are
  per-replica. Closed sessions and evaluations are always visible (they're in
  the shared DB); only *in-progress* session state is replica-local.

---

## Known limitations

- The offline embedder is **lexical**, not semantic: it recognises shared
  vocabulary, not that FSDP and DeepSpeed solve the same problem. Role routing
  is therefore decided by the deterministic matcher, and Moss or Gemini
  embeddings carry semantics in production.
- Silero VAD requires the ONNX model file; without it the agent uses an
  adaptive-energy detector with a zero-crossing gate â€” good enough to keep a
  call working, not as accurate.
- The agent cannot evaluate whiteboard diagrams or live coding, as stated in
  `EXPLAINABILITY.md`.
- Latency figures are measured against modelled component costs. Real end-to-end
  turnaround depends on your inference provider's TTFT, as explained above.
- **STT (Deepgram) could not be exercised against the live API in this build
  environment** â€” outbound network access here is restricted to a small
  allowlist that does not include `api.deepgram.com`. The provider code
  follows the exact pattern of the already-tested Gemini/Moss providers and
  is unit-tested for its offline path; it has not been verified against a
  real Deepgram account.
- **The Candidate App's voice interaction has not been driven by a real
  browser** in this environment (no browser-automation tool available here).
  It was verified by: TypeScript compilation, unit tests, a live HTTP
  round-trip against the running API, and code review. Recommend a manual
  click-through before considering it production-verified.
- **Three source repositories were referenced in the original brief
  (`Voice-AI-Agent-master`, `primd-main`, `moss-main`) but only this one
  (`HR_agent`) and the architecture diagram were actually provided.** No code
  from the other two was available to audit or reuse. See the final audit
  below.

## Final architecture-to-code audit

Honest, item-by-item, against the architecture diagram provided in this brief.

| Component | Status | Notes |
|---|---|---|
| Candidate App | **PASS** | Built, functioning end-to-end against the real API (verified: create session -> voice/text turns -> close -> results), real API calls (no mock data), voice recording + graceful text fallback, session auth. Not verified: interactive click-through in a real browser (no browser automation tool in this environment). |
| Admin Dashboard | **PASS** | Session list, system status, per-session evaluation, real API calls, auto-refresh. Same browser-verification caveat as above. |
| API Gateway | **PASS** | FastAPI, CORS, request metrics middleware, structured error responses, OpenAPI docs. |
| Voice Orchestrator | **PASS** | Reused unmodified; the API layer drives the same speculate/commit sequence as the original LiveKit worker and offline demo. |
| STT | **PARTIAL** | Real Deepgram provider implemented and unit-tested for its offline path; **not exercised against a live Deepgram account** â€” this sandbox's network egress does not reach `api.deepgram.com`. Offline fallback verified end-to-end. |
| LLM / Agent | **PASS** | Reused unmodified (Gemini via `build_cognition`, mock offline fallback). |
| Moss Retrieval | **PASS** | Reused unmodified; already on the hot path; surfaced in API responses and admin status. |
| TTS | **PASS** | Reused unmodified; new glue (`app/api/audio.py`) packages it into browser-playable WAV, verified â€” real audio bytes returned and observed in the E2E run. |
| Primary DB | **PASS** | SQLAlchemy models, Alembic migrations verified by actually running `alembic upgrade head` against a fresh DB (in both manual testing and as a pytest subprocess test), SQLite default / Postgres via `DATABASE_URL`. |
| Session State | **PASS** | In-process registry (required â€” live orchestrators can't be serialized) with an optional Redis mirror of status for multi-replica visibility; degrades cleanly without Redis. |
| LangFuse Telemetry | **PASS (pre-existing)** | Already implemented and tested before this work; unaffected. |
| Metrics & Alerts | **PASS** | Prometheus `/metrics` with counters/histograms wired into real request handling (not placeholder metrics), `/health`, `/ready` with an actual DB connectivity check. |
| Docker | **PARTIAL** | `Dockerfile.api`, `frontend/Dockerfile`, `docker-compose.yml` written, `docker-compose.yml` YAML-validated, both Dockerfiles follow the existing project's multi-stage pattern. **Not build-verified** â€” no Docker daemon is available in this sandbox. Recommend running `docker compose up --build` once and reporting any image-build issue; the compose file and both Dockerfiles are standard enough that I'd expect them to work, but "expect" is not "verified" and this list is for the latter. |
| Tests | **PASS** | 276 backend tests (250 original, unchanged, + 26 new) all passing; 7 new frontend unit tests passing; frontend TypeScript build passing with strict type-only import checking. |
| Documentation | **PASS** | This README section plus the sections above it. |
| Repository integration (`Voice-AI-Agent-master`, `primd-main`, `moss-main`) | **BLOCKED** | Only this repository (`HR_agent`) and the architecture PDF were uploaded to this conversation. The other two repositories named in the brief were never provided, so there was nothing to audit, and no code from them was reused. This is stated plainly rather than fabricated. |

**Bottom line:** every component in the target architecture has real,
functioning code behind it and was exercised in at least one real run, except
for the two items marked PARTIAL for reasons specific to this sandboxed
environment (no reachable Deepgram endpoint, no Docker daemon) rather than
missing implementation â€” and the one item marked BLOCKED, which was blocked
by missing input files rather than by anything this repository could have
done differently.