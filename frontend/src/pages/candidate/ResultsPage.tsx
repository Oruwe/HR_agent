import { useLocation, useNavigate } from "react-router-dom";
import type { EvaluationResponse } from "../../api/client";

const RECOMMENDATION_COPY: Record<string, { label: string; tone: string }> = {
  ADVANCE: { label: "Advancing to the next round", tone: "badge-ok" },
  ADVANCE_WITH_RESERVATIONS: { label: "Advancing, with follow-up areas", tone: "badge-ok" },
  HOLD_FOR_HUMAN_REVIEW: { label: "Held for human review", tone: "badge-live" },
  DO_NOT_ADVANCE: { label: "Not advancing this time", tone: "badge-warn" },
  INCOMPLETE: { label: "Call ended early -- incomplete", tone: "badge-live" },
};

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const evaluation = (location.state as { evaluation?: EvaluationResponse } | null)?.evaluation;

  if (!evaluation) {
    return (
      <div className="results-page">
        <div className="panel">
          <p>No results to show yet.</p>
          <button className="btn btn-primary" onClick={() => navigate("/")}>
            Start a new call
          </button>
        </div>
      </div>
    );
  }

  const copy = RECOMMENDATION_COPY[evaluation.recommendation] ?? {
    label: evaluation.recommendation,
    tone: "badge",
  };

  return (
    <div className="results-page">
      <div className="panel results-summary">
        <span className={`badge ${copy.tone}`}>{copy.label}</span>
        <h1>Thanks, {evaluation.sanitized_name.replace("_", " ")}.</h1>
        <p>
          You answered {evaluation.turns_completed} question{evaluation.turns_completed === 1 ? "" : "s"} for{" "}
          {evaluation.target_role.replaceAll("_", " ").toLowerCase()}.
        </p>
      </div>

      {evaluation.competency_scores.length > 0 && (
        <div className="panel results-scores">
          <h2>Competency signal</h2>
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
          <h2>Worth knowing</h2>
          <ul className="results-limitations">
            {evaluation.flagged_limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        </div>
      )}

      <button className="btn btn-primary" onClick={() => navigate("/")}>
        Start another call
      </button>
    </div>
  );
}
