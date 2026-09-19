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
