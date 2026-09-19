import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ApiError,
  getAdminEvaluation,
  getAdminTranscript,
  getSystemStatus,
  listSessions,
} from "../../api/client";
import type { EvaluationResponse, SessionSummary, SystemStatusResponse, TranscriptTurn } from "../../api/client";
import { useAdminToken } from "./useAdminToken";

/** The full manager-facing surface: a ranked candidate list on the left, the
 * selected candidate's scorecard and read-only conversation transcript on
 * the right. This intentionally has no path back into a live interview --
 * that's the Candidate App's job, not this one's. Everything here is a
 * completed or in-progress record being reviewed after the fact, never
 * something a manager participates in. */

const RECOMMENDATION_BADGE: Record<string, { label: string; className: string }> = {
  ADVANCE: { label: "Advance", className: "admin-badge-advance" },
  ADVANCE_WITH_RESERVATIONS: { label: "Advance (reservations)", className: "admin-badge-advance" },
  HOLD_FOR_HUMAN_REVIEW: { label: "Hold for review", className: "admin-badge-hold" },
  DO_NOT_ADVANCE: { label: "Do not advance", className: "admin-badge-reject" },
  INCOMPLETE: { label: "Incomplete", className: "admin-badge-neutral" },
};

function initialsFor(name: string): string {
  const parts = name.replace(/_/g, " ").trim().split(/\s+/);
  return parts
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

type SortMode = "score" | "recent";

function rankSessions(sessions: SessionSummary[], mode: SortMode): SessionSummary[] {
  const copy = [...sessions];
  if (mode === "score") {
    copy.sort((a, b) => (b.rubric_fit_index ?? -1) - (a.rubric_fit_index ?? -1));
  } else {
    copy.sort((a, b) => b.updated_at - a.updated_at);
  }
  return copy;
}

export default function DashboardPage() {
  const { adminToken, setAdminToken } = useAdminToken();
  const { sessionId: routeSessionId } = useParams<{ sessionId?: string }>();
  const navigate = useNavigate();

  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [sortMode, setSortMode] = useState<SortMode>("score");
  const [listError, setListError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(routeSessionId ?? null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [transcript, setTranscript] = useState<TranscriptTurn[] | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const ranked = useMemo(() => (sessions ? rankSessions(sessions, sortMode) : []), [sessions, sortMode]);

  // Poll the roster, same cadence the previous dashboard used.
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [sessionRows, systemStatus] = await Promise.all([
          listSessions(adminToken),
          getSystemStatus(adminToken),
        ]);
        if (cancelled) return;
        setSessions(sessionRows);
        setStatus(systemStatus);
        setListError(null);
      } catch (err) {
        if (!cancelled) {
          setListError(err instanceof ApiError ? err.message : "Couldn't load the candidate list.");
        }
      }
    }
    void load();
    const interval = setInterval(load, 10_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [adminToken]);

  // Default to the top-ranked candidate once the list arrives, if nothing is
  // selected yet (e.g. landing on plain /admin rather than a deep link).
  useEffect(() => {
    if (selectedId || !ranked.length) return;
    setSelectedId(ranked[0].session_id);
  }, [ranked, selectedId]);

  useEffect(() => {
    if (routeSessionId && routeSessionId !== selectedId) setSelectedId(routeSessionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeSessionId]);

  function selectCandidate(id: string) {
    setSelectedId(id);
    navigate(`/admin/sessions/${id}`, { replace: true });
  }

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    setEvaluation(null);
    setTranscript(null);
    const selected = sessions?.find((s) => s.session_id === selectedId);
    Promise.all([
      selected?.has_evaluation ? getAdminEvaluation(selectedId, adminToken) : Promise.resolve(null),
      getAdminTranscript(selectedId, adminToken),
    ])
      .then(([evalResult, transcriptResult]) => {
        if (cancelled) return;
        setEvaluation(evalResult);
        setTranscript(transcriptResult);
      })
      .catch((err) => {
        if (!cancelled) {
          setDetailError(err instanceof ApiError ? err.message : "Couldn't load this candidate.");
        }
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId, adminToken, sessions]);

  const selectedSummary = sessions?.find((s) => s.session_id === selectedId) ?? null;

  return (
    <div className="admin-shell">
      <header className="admin-topbar">
        <div className="admin-brand">
          <span className="admin-brand-mark" />
          <div>
            <div className="admin-brand-name">Talent Pipeline</div>
            <div className="admin-brand-sub">Screening review for hiring managers</div>
          </div>
        </div>
        <div className="admin-topbar-right">
          <input
            className="admin-token-input"
            type="password"
            placeholder="Admin token (optional)"
            value={adminToken ?? ""}
            onChange={(e) => setAdminToken(e.target.value)}
          />
          <Link to="/">Candidate app ↗</Link>
        </div>
      </header>

      {status && (
        <div className="admin-stats-row">
          <div className="admin-stat-card">
            <div className="admin-stat-card-label">Candidates</div>
            <div className="admin-stat-card-value">{sessions?.length ?? "--"}</div>
          </div>
          <div className="admin-stat-card">
            <div className="admin-stat-card-label">Active calls</div>
            <div className="admin-stat-card-value">{status.active_sessions}</div>
          </div>
          <div className="admin-stat-card">
            <div className="admin-stat-card-label">Cognition</div>
            <div className="admin-stat-card-value">{status.cognition_configured ? "Live" : "Offline"}</div>
          </div>
          <div className="admin-stat-card">
            <div className="admin-stat-card-label">Retrieval</div>
            <div className="admin-stat-card-value">{status.moss_configured ? "Moss" : "Embedded"}</div>
          </div>
        </div>
      )}

      {listError && <div className="admin-error-banner">{listError}</div>}

      <div className="admin-body">
        <aside className="admin-list-pane">
          <div className="admin-list-header">
            <h2>Candidates</h2>
            <div className="admin-sort-row">
              <button
                className={`admin-sort-btn ${sortMode === "score" ? "active" : ""}`}
                onClick={() => setSortMode("score")}
              >
                Top ranked
              </button>
              <button
                className={`admin-sort-btn ${sortMode === "recent" ? "active" : ""}`}
                onClick={() => setSortMode("recent")}
              >
                Most recent
              </button>
            </div>
          </div>

          {sessions === null && <div className="admin-loading">Loading candidates...</div>}
          {sessions && sessions.length === 0 && (
            <div className="admin-empty-state">No candidates have screened yet.</div>
          )}

          <div className="admin-candidate-list">
            {ranked.map((session, index) => (
              <button
                key={session.session_id}
                className={`admin-candidate-row ${session.session_id === selectedId ? "selected" : ""}`}
                onClick={() => selectCandidate(session.session_id)}
              >
                <span className="admin-candidate-rank mono">{sortMode === "score" ? index + 1 : ""}</span>
                <span className="admin-candidate-avatar">{initialsFor(session.sanitized_name)}</span>
                <span className="admin-candidate-main">
                  <span className="admin-candidate-name">{session.sanitized_name.replaceAll("_", " ")}</span>
                  <span className="admin-candidate-role">
                    {session.role.replaceAll("_", " ").toLowerCase()} · {session.status}
                  </span>
                </span>
                <span className="admin-candidate-score">
                  {session.rubric_fit_index !== null ? session.rubric_fit_index.toFixed(2) : "--"}
                </span>
              </button>
            ))}
          </div>
        </aside>

        <section className="admin-detail-pane">
          {!selectedSummary && !detailLoading && (
            <div className="admin-empty-state">Select a candidate to review their screening call.</div>
          )}

          {selectedSummary && (
            <>
              <div className="admin-detail-header">
                <div className="admin-detail-heading">
                  <h1>{selectedSummary.sanitized_name.replaceAll("_", " ")}</h1>
                  <p>
                    {selectedSummary.role.replaceAll("_", " ").toLowerCase()} ·{" "}
                    {selectedSummary.turns} turn{selectedSummary.turns === 1 ? "" : "s"} ·{" "}
                    {evaluation ? (
                      <span
                        className={`admin-badge ${
                          RECOMMENDATION_BADGE[evaluation.recommendation]?.className ?? "admin-badge-neutral"
                        }`}
                      >
                        {RECOMMENDATION_BADGE[evaluation.recommendation]?.label ?? evaluation.recommendation}
                      </span>
                    ) : (
                      <span className="admin-badge admin-badge-neutral">
                        {selectedSummary.status === "active" ? "Call in progress" : "Not yet evaluated"}
                      </span>
                    )}
                  </p>
                </div>

                {evaluation && (
                  <div className="admin-score-strip">
                    <div className="admin-score-strip-item">
                      <div className="admin-score-strip-value mono">{evaluation.rubric_fit_index.toFixed(2)}</div>
                      <div className="admin-score-strip-label">Fit index</div>
                    </div>
                    <div className="admin-score-strip-item">
                      <div className="admin-score-strip-value mono">
                        {evaluation.routing_confidence.toFixed(2)}
                      </div>
                      <div className="admin-score-strip-label">Routing confidence</div>
                    </div>
                    <div className="admin-score-strip-item">
                      <div className="admin-score-strip-value mono">
                        {Math.round(evaluation.latency_compliance * 100)}%
                      </div>
                      <div className="admin-score-strip-label">In-budget turns</div>
                    </div>
                  </div>
                )}
              </div>

              <div className="admin-detail-body">
                {detailError && <div className="admin-error-banner">{detailError}</div>}

                {evaluation && evaluation.competency_scores.length > 0 && (
                  <div>
                    <div className="admin-section-title">Competency scores</div>
                    <div className="admin-competency-list">
                      {evaluation.competency_scores.map((score) => (
                        <div key={score.key} className="admin-competency-row">
                          <div className="admin-competency-label">{score.label}</div>
                          <div className="admin-competency-track">
                            <div
                              className="admin-competency-fill"
                              style={{ width: `${Math.round(score.score * 100)}%` }}
                            />
                          </div>
                          <div className="admin-competency-value">{Math.round(score.score * 100)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {evaluation && evaluation.flagged_limitations.length > 0 && (
                  <div>
                    <div className="admin-section-title">Flagged for human review</div>
                    <ul className="admin-limitations">
                      {evaluation.flagged_limitations.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div>
                  <div className="admin-section-title">Conversation</div>
                  {detailLoading && <div className="admin-loading">Loading transcript...</div>}
                  {transcript && transcript.length === 0 && (
                    <div className="admin-empty-state">No turns recorded yet.</div>
                  )}
                  {transcript && transcript.length > 0 && (
                    <div className="admin-chat">
                      {transcript.map((turn, i) => (
                        <div key={i} className={`admin-chat-bubble-row ${turn.speaker}`}>
                          <div>
                            <div className="admin-chat-bubble">{turn.text}</div>
                            {/* Only the streaming voice path measures time-to-first-audio;
                                the HTTP turn endpoint leaves it at ~0, and printing
                                "0.0ms to first audio" there is worse than printing
                                nothing. */}
                            {turn.turnaround_ms !== null && turn.turnaround_ms >= 1 && (
                              <div className="admin-chat-meta mono">
                                {Math.round(turn.turnaround_ms)}ms to first audio
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
