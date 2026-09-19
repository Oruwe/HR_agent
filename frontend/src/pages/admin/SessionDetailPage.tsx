import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, getAdminEvaluation } from "../../api/client";
import type { EvaluationResponse } from "../../api/client";
import { useAdminToken } from "./useAdminToken";

export default function SessionDetailPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { adminToken } = useAdminToken();
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    getAdminEvaluation(sessionId, adminToken)
      .then(setEvaluation)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load this session."));
  }, [sessionId, adminToken]);

  return (
    <div className="admin-dashboard">
      <Link to="/admin" className="admin-back-link">
        ← All sessions
      </Link>

      {error && <div className="error-banner">{error}</div>}

      {evaluation && (
        <>
          <div className="panel">
            <h1>{evaluation.sanitized_name}</h1>
            <p>
              {evaluation.target_role.replaceAll("_", " ").toLowerCase()} · {evaluation.turns_completed} turns ·
              recommendation: <strong>{evaluation.recommendation.replaceAll("_", " ").toLowerCase()}</strong>
            </p>
          </div>

          <div className="admin-grid">
            <div className="panel admin-stat">
              <div className="admin-stat-label">Rubric fit index</div>
              <div className="admin-stat-value mono">{evaluation.rubric_fit_index.toFixed(3)}</div>
            </div>
            <div className="panel admin-stat">
              <div className="admin-stat-label">Routing confidence</div>
              <div className="admin-stat-value mono">{evaluation.routing_confidence.toFixed(3)}</div>
            </div>
            <div className="panel admin-stat">
              <div className="admin-stat-label">Latency compliance</div>
              <div className="admin-stat-value mono">{Math.round(evaluation.latency_compliance * 100)}%</div>
            </div>
          </div>

          {evaluation.competency_scores.length > 0 && (
            <div className="panel results-scores">
              <h2>Competency scores</h2>
              {evaluation.competency_scores.map((score) => (
                <div key={score.key} className="score-row">
                  <div className="score-label">{score.label}</div>
                  <div className="score-bar-track">
                    <div className="score-bar-fill" style={{ width: `${Math.round(score.score * 100)}%` }} />
                  </div>
                  <div className="mono score-value">{Math.round(score.score * 100)}</div>
                </div>
              ))}
            </div>
          )}

          {evaluation.flagged_limitations.length > 0 && (
            <div className="panel">
              <h2>Flagged limitations</h2>
              <ul className="results-limitations">
                {evaluation.flagged_limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}
