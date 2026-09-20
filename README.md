# hr-talent-evaluator

A hiring manager's dashboard. Scraped candidate records go in as JSON, an AI
analyst ranks them and explains why, and the manager can ask questions about
the pool in plain language.

```
  scraped JSON  ->  PII scrubbed  ->  stored  ->  ranked by the analyst  ->  dashboard
                                         |                                       |
                                         +--> indexed for search (Moss) <--+     |
                                                                           |     |
                                         manager asks a question  ---------+-----+
                                            -> search the pool
                                            -> answer from what was found
```

There is no candidate-facing side to this. Nobody is interviewed by it.

---

## Run it

```bash
pip install -r requirements.txt
python -m app.main seed          # loads and ranks a 9-candidate demo pool
make api                         # API on :8000
make frontend                    # dashboard on :5173
```

That works with **no credentials at all**. Without a `GOOGLE_API_KEY` the
dashboard reports "Offline analyst" and shows the demo pool's bundled
baseline rankings rather than live judgement — a real working board to click
through, honestly labelled as not being model output.

To turn the analyst on, get a free key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and:

```bash
export GOOGLE_API_KEY=...
python -m app.main seed          # now ranks with the model
```

## What the dashboard shows

Three columns, left to right:

| Column | What it holds |
| --- | --- |
| **Left** | Pool size, the verdict breakdown, re-analyse and import actions, filters and search, and the live model status |
| **Centre** | Every candidate ranked, with a score bar, a verdict, and the analyst's one-paragraph reason |
| **Right** | The analyst. Ask anything about the pool; answers are grounded in the records it can see |

Clicking a candidate opens their scraped record exactly as it was stored —
post-redaction, so what you see is what the model saw.

## Importing real data

The import route takes **any JSON shape**. The scraper owns the schema, not
this app:

```bash
curl -X POST localhost:8000/api/candidates/import \
  -H 'Content-Type: application/json' \
  -d '{"candidates": [{"name": "...", "whatever_your_scraper_emits": {...}}]}'
```

Or paste the array straight into "Import scraped JSON" in the left rail.

Two fields are lifted out of the blob into columns for display — a name and a
headline, looked for under the keys a scraper is likely to use (`name`,
`full_name`, `displayName`, `headline`, `title`, `position`, …). Everything
else is stored verbatim and read by the model as-is. A record with no
recognisable name is kept, not dropped, as "Unknown candidate".

`python -m app.main export` prints the demo pool in exactly the shape the
route expects, as a template.

## API

Every route is under `/api` and gated by `X-Admin-Token` when
`HRTE_ADMIN_TOKEN` is set.

| Route | Does |
| --- | --- |
| `POST /api/candidates/import` | Ingest scraped records |
| `POST /api/candidates/demo` | Load the bundled demo pool, pre-ranked (refuses if the pool isn't empty) |
| `GET /api/candidates` | The ranked pool: best first, unscored last |
| `GET /api/candidates/{id}` | One candidate plus their full stored record |
| `DELETE /api/candidates/{id}` | Remove one |
| `POST /api/analyze` | Score the whole pool in one comparative pass |
| `POST /api/chat` | Ask a question; returns the answer plus which records it read |
| `GET /api/status` | Pool counts, the model and retrieval backend in use, and whether either is failing |

`/health`, `/ready` and `/metrics` are unauthenticated, because a gated
health check makes the platform mark a healthy service down.

Interactive docs at `/api/docs`.

## Retrieval

A question is a **search first, a generation second**. The manager asks "who
has the strongest distributed systems evidence?", the pool is searched, and
only the records that matched go to the model.

This is not an optimisation, it is what makes the product work at size.
Putting the whole pool in the prompt caps you at a couple of hundred records,
and a model holding two hundred profiles reads them all with equal attention —
it answers worse than one shown the six that matter.

Two backends:

| | |
| --- | --- |
| **[Moss](https://usemoss.dev)** | Semantic. "Who handled failure under load" finds a record saying "split brain during a partition" without either phrase sharing a word. Needs `MOSS_PROJECT_ID` and `MOSS_PROJECT_KEY`. |
| **Local** | TF-IDF cosine in pure Python. No credentials, no network, deterministic, sub-millisecond at pool scale. **Lexical**: it only matches words the record actually contains. |

The difference is real and the product does not hide it. Every answer in the
dashboard carries a line saying what it read — *"Read 2 of 9 records · local
(lexical) · 0.43ms"* — with the matched candidates as clickable chips, and
`/api/status` reports the backend plus whether a configured Moss has started
failing.

Ranking is the exception: `POST /api/analyze` still sends the **whole** pool
in one call, deliberately. Ranking is comparative, and narrowing it would mean
scoring candidates against a subset of the field and calling the result a
ranking.

## Three things worth knowing

**"A key is set" is not "the key works".** A deployment whose every model
call 4xx's looks identical from the outside to a healthy one — this project
lost hours to exactly that, twice, once to a retired model pin and once to
thinking tokens eating the whole output budget. So every fallback to the
offline mock is counted, and `/api/status` reports `degraded: true` the
moment a *configured* model fails. The dashboard shows it as an amber pill.
Offline-by-choice and broken-in-production are different states and the UI
never conflates them. Retrieval is counted the same way, for the same reason.

**Candidates are never named to the model by their database id.** Records go
into the prompt as `C1`, `C2`, … and the mapping back never leaves the
process. Candidate ids are UUIDs, and a UUID run through the PII scrubber has
a ~5% chance of partial redaction — slices of one look like an Aadhaar number,
a passport or a payment card. The model then echoes the mangled id back, it
matches no row, and that candidate is silently left unscored. On a 200-record
pool that was ~10 people stuck at `--` on every run with nothing to say why.

**PII is redacted once, on the way in.** `app/ingest.py` is the only path a
record takes into the database, and it scrubs there. Everything downstream —
the database, the model prompt, the dashboard, the export — only ever sees
scrubbed text. That is the only arrangement that can be audited by reading a
single function. Identifiers are replaced with typed tokens plus a one-way
keyed BLAKE2b fingerprint, so two records belonging to the same person can
still be matched without the identifier being kept.

The scrubber is tuned hard against false positives: `p99 from 1200ms to
160ms`, `512 A100s`, `RFC 7519`, `ef_construct=100` all survive intact,
because a scrubber that eats the numbers eats the evidence the ranking is
built on.

## Configuration

Everything is optional. See `.env.example` for the annotated list.

| Variable | Default | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | The one that matters. Unset ⇒ offline analyst |
| `HRTE_MODEL` | `gemini-flash-latest` | A floating alias on purpose: pinned versions get retired |
| `HRTE_TEMPERATURE` | `0.4` | |
| `HRTE_MAX_OUTPUT_TOKENS` | `2048` | |
| `HRTE_THINKING_BUDGET` | `0` | Billed out of max output tokens. Raise both together or not at all |
| `MOSS_PROJECT_ID` / `MOSS_PROJECT_KEY` | — | Both needed. Unset ⇒ local lexical retrieval |
| `HRTE_MOSS_INDEX` | `candidate_pool` | Index name inside your Moss project |
| `HRTE_RETRIEVAL_TOP_K` | `12` | How many records one answer may read |
| `DATABASE_URL` | SQLite under `./data` | |
| `HRTE_ADMIN_TOKEN` | — | Set this on any public URL |
| `HRTE_REDACTION_KEY` | ephemeral | Set it so fingerprints survive a restart |
| `HRTE_CORS_ORIGINS` | `*` | Credentials only allowed with an explicit list |
| `HRTE_PII_MODE` | `strict` | |

## Deploying

`render.yaml` is a Render blueprint: API + dashboard + Postgres, all on free
plans. It generates an admin token and a redaction key for you; the only
value you fill in by hand is `GOOGLE_API_KEY`.

Locally, `docker compose up --build` brings up the same two images.

Either way, a fresh deployment starts empty. Click **Load demo pool** in the
left rail to populate it, or POST your scraper's output at the import route.

## Development

```bash
make test           # backend suite; warnings are errors
make lint           # ruff check + format check
make frontend-test  # frontend unit tests
make verify         # config + PII scrubber report
```

The test suite runs entirely offline — the mock analyst is a real provider
that produces well-formed rankings, not a stub that returns an apology, so
nothing here needs network or credentials.

## Layout

```
app/
  agent/analyst.py        the two calls: rank a pool, answer a question
  agent/cognition.py      Gemini + the offline mock behind one protocol
  retrieval.py            Moss + the local index behind one protocol
  api/routes_candidates.py  the whole HTTP surface
  security/pii_scrubber.py  redaction, fingerprinting, false-positive defence
  ingest.py               the single path a record takes into the database
  demo_pool.py            9 deliberately messy records + baseline rankings
frontend/src/
  pages/DashboardPage.tsx the dashboard
  api/client.ts           typed fetch wrapper
```
<<<<<<< HEAD

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
done differently.# HR Talent Evaluator

AI-powered voice interview assistant for technical hiring.

## Features

- Real-time voice conversation with candidates
- Automatic speech-to-text transcription
- AI-powered evaluation and feedback
- Configurable voice personalities
- Detailed analytics and reporting

## Deployment

### Render

1. Fork this repository
2. Create a new Render service
3. Connect your GitHub repository
4. Add required environment variables
5. Deploy!

Required environment variables:

- `LIVEKIT_URL` - LiveKit WebRTC server URL
- `LIVEKIT_API_KEY` - LiveKit API key
- `LIVEKIT_API_SECRET` - LiveKit API secret
- `GOOGLE_API_KEY` - Google Cloud API key
- `DEEPGRAM_API_KEY` - Deepgram API key
- `ELEVENLABS_API_KEY` - ElevenLabs API key
- `ELEVENLABS_AGENT_ID` - ElevenLabs agent ID
- `ELEVENLABS_VOICE_ID` - ElevenLabs voice ID
- `LANGFUSE_PUBLIC_KEY` - Langfuse public key
- `LANGFUSE_SECRET_KEY` - Langfuse secret key
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OpenTelemetry endpoint
- `MOSS_PROJECT_ID` - Moss project ID
- `MOSS_PROJECT_KEY` - Moss project key
- `QDRANT_URL` - Qdrant vector database URL
- `QDRANT_API_KEY` - Qdrant API key

## Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL
- Redis

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```
3. Set up environment variables
4. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   cd frontend && npm run dev
   ```

## License

MIT
=======
>>>>>>> a12cd5831229344cfe567d2f98949cf2442622d7
