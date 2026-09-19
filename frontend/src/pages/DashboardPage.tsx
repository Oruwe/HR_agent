import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  analyzePool,
  askAnalyst,
  getCandidate,
  getStatus,
  importCandidates,
  listCandidates,
  loadDemoPool,
  setAdminToken,
} from "../api/client";
import type {
  CandidateDetail,
  CandidateRef,
  CandidateSummary,
  ChatMessage,
  StatusResponse,
} from "../api/client";

/** A chat turn plus, for the analyst's replies, what it was grounded in. */
interface Turn extends ChatMessage {
  sources?: CandidateRef[];
  backend?: string;
  poolSize?: number;
  retrievalMs?: number;
}

/** The whole product: scraped candidate records ranked by an AI analyst, with
 * the manager able to interrogate the pool in plain language.
 *
 * Three columns, left to right: pool context and controls, the ranked scores
 * (the thing the manager is actually here for, so it gets the centre and the
 * most space), and the analyst conversation.
 */

const VERDICT_META: Record<string, { label: string; className: string }> = {
  INTERVIEW: { label: "Interview", className: "verdict-interview" },
  MAYBE: { label: "Maybe", className: "verdict-maybe" },
  PASS: { label: "Pass", className: "verdict-pass" },
};

const SUGGESTED_QUESTIONS = [
  "Who are the top three and why?",
  "Who has the strongest distributed systems evidence?",
  "Which records are too thin to judge?",
  "Who would you drop first, and what would change your mind?",
];

const TOKEN_KEY = "hrte.admin_token";

function initialsFor(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

function timeAgo(epochSeconds: number | null): string {
  if (!epochSeconds) return "never";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

export default function DashboardPage() {
  const [token, setToken] = useState<string>(() => localStorage.getItem(TOKEN_KEY) || "");
  const [candidates, setCandidates] = useState<CandidateSummary[] | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const [filter, setFilter] = useState<string>("ALL");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<CandidateDetail | null>(null);
  const [showImport, setShowImport] = useState(false);
  const [importText, setImportText] = useState("");

  const [chat, setChat] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setAdminToken(token);
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  }, [token]);

  const refresh = useCallback(async () => {
    try {
      const [rows, st] = await Promise.all([listCandidates(), getStatus()]);
      setCandidates(rows);
      setStatus(st);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't load the candidate pool.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = setInterval(refresh, 20_000);
    return () => clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    // Only once there is something to scroll to. On mobile the columns are a
    // single stacked page, so scrolling an empty chat into view on mount
    // drags the whole dashboard down past its own header.
    if (chat.length === 0 && !thinking) return;
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [chat, thinking]);

  const counts = useMemo(() => {
    const rows = candidates ?? [];
    return {
      total: rows.length,
      interview: rows.filter((c) => c.recommendation === "INTERVIEW").length,
      maybe: rows.filter((c) => c.recommendation === "MAYBE").length,
      pass: rows.filter((c) => c.recommendation === "PASS").length,
      unscored: rows.filter((c) => c.score === null).length,
    };
  }, [candidates]);

  const visible = useMemo(() => {
    let rows = candidates ?? [];
    if (filter !== "ALL") rows = rows.filter((c) => c.recommendation === filter);
    const q = query.trim().toLowerCase();
    if (q) {
      rows = rows.filter(
        (c) =>
          c.name.toLowerCase().includes(q) ||
          c.headline.toLowerCase().includes(q) ||
          (c.rationale || "").toLowerCase().includes(q)
      );
    }
    return rows;
  }, [candidates, filter, query]);

  async function runAnalysis() {
    setBusy("Analysing the pool...");
    setError(null);
    try {
      const result = await analyzePool();
      await refresh();
      if (result.analyzed === 0) {
        setError("The analyst returned no rankings. The pool is unchanged.");
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The analysis run failed.");
    } finally {
      setBusy(null);
    }
  }

  async function runImport() {
    setBusy("Importing...");
    setError(null);
    try {
      const parsed = JSON.parse(importText);
      const records = Array.isArray(parsed) ? parsed : parsed.candidates;
      if (!Array.isArray(records) || records.length === 0) {
        throw new Error("Expected a JSON array of records, or {\"candidates\": [...]}.");
      }
      await importCandidates(records);
      setImportText("");
      setShowImport(false);
      await refresh();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "That JSON couldn't be imported."
      );
    } finally {
      setBusy(null);
    }
  }

  async function loadDemo() {
    setBusy("Loading the demo pool...");
    setError(null);
    try {
      await loadDemoPool();
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't load the demo pool.");
    } finally {
      setBusy(null);
    }
  }

  async function openCandidate(id: string) {
    try {
      setSelected(await getCandidate(id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't open that record.");
    }
  }

  async function send(question?: string) {
    const text = (question ?? draft).trim();
    if (!text || thinking) return;
    setDraft("");
    // Only role/content goes back as history -- the provenance is for the
    // reader, not for the model.
    const history = chat.slice(-10).map((m) => ({ role: m.role, content: m.content }));
    setChat((prev) => [...prev, { role: "user", content: text }]);
    setThinking(true);
    try {
      const result = await askAnalyst(text, history);
      setChat((prev) => [
        ...prev,
        {
          role: "model",
          content: result.reply,
          sources: result.sources,
          backend: result.retrieval_backend,
          poolSize: result.pool_size,
          retrievalMs: result.retrieval_ms,
        },
      ]);
    } catch (err) {
      setChat((prev) => [
        ...prev,
        {
          role: "model",
          content:
            err instanceof ApiError ? `I couldn't answer that: ${err.message}` : "Request failed.",
        },
      ]);
    } finally {
      setThinking(false);
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <div className="brand-name">Talent Pipeline</div>
            <div className="brand-sub">AI hiring analyst</div>
          </div>
        </div>
        <div className="topbar-right">
          {status && (
            <span className={`pill ${status.degraded ? "pill-alert" : status.offline ? "pill-mute" : "pill-ok"}`}>
              {status.degraded ? "Model degraded" : status.offline ? "Offline analyst" : status.model}
            </span>
          )}
          <input
            className="token-input"
            type="password"
            placeholder="Admin token"
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
        </div>
      </header>

      {status?.storage_ephemeral && (
        <div className="warn-banner">
          <b>This pool will not survive a restart.</b> The server is storing candidates in
          SQLite on its own container filesystem, which the platform discards on every
          deploy, restart and idle spin-down. Set <code>DATABASE_URL</code> to a Postgres
          instance to keep imported candidates.
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}
      {busy && <div className="busy-banner">{busy}</div>}

      <div className="columns">
        {/* ---- Left: pool context and controls ---------------------------- */}
        <aside className="col-left">
          <section className="card">
            <div className="card-title">Pool</div>
            <div className="stat-grid">
              <div className="stat">
                <div className="stat-value">{counts.total}</div>
                <div className="stat-label">Candidates</div>
              </div>
              <div className="stat">
                <div className="stat-value">{status?.analyzed ?? 0}</div>
                <div className="stat-label">Scored</div>
              </div>
            </div>
            <div className="dist-bar" aria-hidden>
              <span className="dist-interview" style={{ flexGrow: counts.interview || 0 }} />
              <span className="dist-maybe" style={{ flexGrow: counts.maybe || 0 }} />
              <span className="dist-pass" style={{ flexGrow: counts.pass || 0 }} />
              <span className="dist-none" style={{ flexGrow: counts.unscored || 0 }} />
            </div>
            <ul className="legend">
              <li>
                <span className="dot dot-interview" /> Interview <b>{counts.interview}</b>
              </li>
              <li>
                <span className="dot dot-maybe" /> Maybe <b>{counts.maybe}</b>
              </li>
              <li>
                <span className="dot dot-pass" /> Pass <b>{counts.pass}</b>
              </li>
              <li>
                <span className="dot dot-none" /> Unscored <b>{counts.unscored}</b>
              </li>
            </ul>
          </section>

          <section className="card">
            <div className="card-title">Actions</div>
            <button className="btn btn-primary" onClick={runAnalysis} disabled={!!busy}>
              Re-analyse pool
            </button>
            <button className="btn" onClick={() => setShowImport((v) => !v)} disabled={!!busy}>
              {showImport ? "Cancel import" : "Import scraped JSON"}
            </button>
            {counts.total === 0 && (
              <button className="btn" onClick={loadDemo} disabled={!!busy}>
                Load demo pool
              </button>
            )}
            {showImport && (
              <div className="import-panel">
                <textarea
                  className="import-text"
                  placeholder='[{"name": "...", "skills": [...]}, ...]'
                  value={importText}
                  onChange={(e) => setImportText(e.target.value)}
                  rows={7}
                />
                <button className="btn btn-primary" onClick={runImport} disabled={!importText.trim()}>
                  Import
                </button>
                <p className="hint">
                  Any JSON shape works. PII is redacted on import before anything is stored or
                  sent to the model.
                </p>
              </div>
            )}
          </section>

          <section className="card">
            <div className="card-title">Filter</div>
            <div className="chips">
              {["ALL", "INTERVIEW", "MAYBE", "PASS"].map((key) => (
                <button
                  key={key}
                  className={`chip ${filter === key ? "chip-active" : ""}`}
                  onClick={() => setFilter(key)}
                >
                  {key === "ALL" ? "All" : VERDICT_META[key].label}
                </button>
              ))}
            </div>
            <input
              className="search"
              placeholder="Search name, role, rationale..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </section>

          {status && (
            <section className="card card-muted">
              <div className="card-title">System</div>
              <dl className="kv">
                <dt>Model</dt>
                <dd>{status.model}</dd>
                <dt>Environment</dt>
                <dd>{status.environment}</dd>
                <dt>Fallbacks</dt>
                <dd className={status.degraded ? "alert-text" : ""}>{status.fallbacks}</dd>
                <dt>Retrieval</dt>
                <dd className={status.retrieval_degraded ? "alert-text" : ""}>
                  {status.retrieval_backend}
                </dd>
                <dt>Storage</dt>
                <dd className={status.storage_ephemeral ? "alert-text" : ""}>
                  {status.storage_backend}
                  {status.storage_ephemeral ? " (ephemeral)" : ""}
                </dd>
              </dl>
              {status.degraded && (
                <p className="hint alert-text">
                  Live model calls are failing and answers are coming from the offline mock.
                  Check GOOGLE_API_KEY and HRTE_MODEL.
                </p>
              )}
              {status.retrieval_degraded && (
                <p className="hint alert-text">
                  Moss is configured but failing, so retrieval has fallen back to the local
                  lexical index. Answers still work; they match words rather than meaning.
                </p>
              )}
              {!status.moss_configured && (
                <p className="hint">
                  Retrieval is lexical. Set MOSS_PROJECT_ID and MOSS_PROJECT_KEY for semantic
                  search over the pool.
                </p>
              )}
            </section>
          )}
        </aside>

        {/* ---- Centre: the scores ------------------------------------------ */}
        <main className="col-center">
          <div className="center-head">
            <h1>Candidate scores</h1>
            <span className="muted">
              {visible.length} of {counts.total} shown
              {status?.analyzed ? ` · last analysed ${timeAgo(
                Math.max(...(candidates ?? []).map((c) => c.analyzed_at ?? 0)) || null
              )}` : ""}
            </span>
          </div>

          {candidates === null && <div className="empty">Loading the pool...</div>}
          {candidates !== null && counts.total === 0 && (
            <div className="empty">
              No candidates yet. Import scraped JSON from the left, or load the demo pool to see
              the board working.
            </div>
          )}
          {candidates !== null && counts.total > 0 && visible.length === 0 && (
            <div className="empty">Nothing matches that filter.</div>
          )}

          <div className="candidate-list">
            {visible.map((c, i) => {
              const meta = VERDICT_META[c.recommendation || ""] ?? {
                label: "Unscored",
                className: "verdict-none",
              };
              const pct = Math.round((c.score ?? 0) * 100);
              return (
                <article
                  key={c.id}
                  className="candidate-card"
                  onClick={() => void openCandidate(c.id)}
                >
                  <div className="rank">{i + 1}</div>
                  <div className="avatar">{initialsFor(c.name)}</div>
                  <div className="candidate-body">
                    <div className="candidate-head">
                      <h2>{c.name}</h2>
                      <span className={`verdict ${meta.className}`}>{meta.label}</span>
                    </div>
                    <div className="candidate-role">{c.headline || "—"}</div>
                    <div className="score-row">
                      <div className="score-track">
                        <div className={`score-fill ${meta.className}-fill`} style={{ width: `${pct}%` }} />
                      </div>
                      <span className="score-num">{c.score === null ? "--" : pct}</span>
                    </div>
                    {c.rationale && <p className="rationale">{c.rationale}</p>}
                  </div>
                </article>
              );
            })}
          </div>
        </main>

        {/* ---- Right: the analyst ------------------------------------------ */}
        <aside className="col-right">
          <div className="chat-head">
            <div>
              <div className="card-title">Ask the analyst</div>
              <div className="muted small">
                {status?.retrieval_degraded
                  ? "Moss is failing; answers are falling back to lexical search"
                  : `Searches ${counts.total} records, then answers from what it finds`}
              </div>
            </div>
          </div>

          <div className="chat-log">
            {chat.length === 0 && (
              <div className="chat-intro">
                <p>Ask anything about the pool. For example:</p>
                <div className="suggestions">
                  {SUGGESTED_QUESTIONS.map((q) => (
                    <button key={q} className="suggestion" onClick={() => void send(q)}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {chat.map((m, i) => (
              <div key={i} className={`bubble-row ${m.role}`}>
                <div className="bubble">{m.content}</div>
                {m.role === "model" && m.sources && m.sources.length > 0 && (
                  <div className="sources">
                    <span className="sources-label">
                      Read {m.sources.length} of {m.poolSize} records
                      {m.backend ? ` · ${m.backend}` : ""}
                      {m.retrievalMs ? ` · ${m.retrievalMs}ms` : ""}
                    </span>
                    <span className="source-chips">
                      {m.sources.map((sc) => (
                        <button
                          key={sc.id}
                          className="source-chip"
                          onClick={() => void openCandidate(sc.id)}
                          title="Open this record"
                        >
                          {sc.name}
                        </button>
                      ))}
                    </span>
                  </div>
                )}
              </div>
            ))}
            {thinking && (
              <div className="bubble-row model">
                <div className="bubble bubble-thinking">Reading the pool...</div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <form
            className="chat-input"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <input
              placeholder="Ask about these candidates..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              disabled={thinking}
            />
            <button className="btn btn-primary" type="submit" disabled={thinking || !draft.trim()}>
              Ask
            </button>
          </form>
        </aside>
      </div>

      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div>
                <h2>{selected.name}</h2>
                <div className="muted">{selected.headline}</div>
              </div>
              <button className="btn" onClick={() => setSelected(null)}>
                Close
              </button>
            </div>
            {selected.rationale && <p className="modal-rationale">{selected.rationale}</p>}
            <div className="card-title">Scraped record</div>
            <pre className="record">{JSON.stringify(selected.source, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
