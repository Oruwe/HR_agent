import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, getSystemStatus, listSessions } from "../../api/client";
import type { SessionSummary, SystemStatusResponse } from "../../api/client";
import { useAdminToken } from "./useAdminToken";

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return <span className={`badge ${ok ? "badge-ok" : "badge"}`}>{label}: {ok ? "configured" : "offline mode"}</span>;
}

export default function DashboardPage() {
  const { adminToken } = useAdminToken();
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [sessionRows, systemStatus] = await Promise.all([
          listSessions(adminToken),
          getSystemStatus(adminToken),
        ]);
        if (!cancelled) {
          setSessions(sessionRows);
          setStatus(systemStatus);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Couldn't load dashboard data.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    const interval = setInterval(load, 10_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [adminToken]);

  return (
    <div className="admin-dashboard">
      <div className="admin-grid">
        <div className="panel admin-stat">
          <div className="admin-stat-label">Active sessions</div>
          <div className="admin-stat-value mono">{status?.active_sessions ?? "--"}</div>
        </div>
        <div className="panel admin-stat">
          <div className="admin-stat-label">Latency budget</div>
          <div className="admin-stat-value mono">{status?.latency_budget_ms ?? "--"}ms</div>
        </div>
        <div className="panel admin-stat">
          <div className="admin-stat-label">Session state</div>
          <div className="admin-stat-value">{status?.session_state_backend ?? "--"}</div>
        </div>
        <div className="panel admin-stat">
          <div className="admin-stat-label">Database</div>
          <div className="admin-stat-value">{status?.database_url_scheme ?? "--"}</div>
        </div>
      </div>

      {status && (
        <div className="panel admin-capabilities">
          <h2>Pipeline configuration</h2>
          <div className="capability-pills">
            <StatusPill ok={status.cognition_configured} label="LLM cognition" />
            <StatusPill ok={status.moss_configured} label="Moss retrieval" />
            <StatusPill ok={status.qdrant_configured} label="Qdrant archive" />
            <StatusPill ok={status.transport_configured} label="LiveKit transport" />
            <StatusPill ok={status.stt_configured} label="Speech-to-text" />
            <StatusPill ok={status.telemetry_configured} label="Langfuse telemetry" />
          </div>
          <h3>Stage budget (160ms total)</h3>
          <div className="stage-budget-bars">
            {status.stage_budgets.map((stage) => (
              <div key={stage.stage} className="stage-budget-row">
                <div className="stage-budget-label">{stage.stage.replaceAll("_", " ")}</div>
                <div className="stage-budget-track">
                  <div
                    className="stage-budget-fill"
                    style={{ width: `${(stage.budget_ms / status.latency_budget_ms) * 100}%` }}
                  />
                </div>
                <div className="mono stage-budget-value">{stage.budget_ms}ms</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="panel admin-sessions">
        <h2>Sessions</h2>
        {error && <div className="error-banner">{error}</div>}
        {loading && !sessions && <p>Loading...</p>}
        {sessions && sessions.length === 0 && <p>No sessions yet.</p>}
        {sessions && sessions.length > 0 && (
          <table className="admin-table">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Role</th>
                <th>Status</th>
                <th>Retrieval</th>
                <th>Turns</th>
                <th>Evaluated</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => (
                <tr key={session.session_id}>
                  <td>{session.sanitized_name}</td>
                  <td>{session.role.replaceAll("_", " ").toLowerCase()}</td>
                  <td>
                    <span className={`badge ${session.status === "active" ? "badge-live" : "badge"}`}>
                      {session.status}
                    </span>
                  </td>
                  <td>{session.retrieval_backend}</td>
                  <td className="mono">{session.turns}</td>
                  <td>{session.has_evaluation ? "yes" : "--"}</td>
                  <td>
                    {session.has_evaluation && (
                      <Link className="btn" to={`/admin/sessions/${session.session_id}`}>
                        View
                      </Link>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
