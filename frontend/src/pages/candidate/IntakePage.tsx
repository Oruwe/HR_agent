import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, createSession } from "../../api/client";
import type { EngineeringRole } from "../../api/client";
import { useActiveSession } from "../../hooks/useActiveSession";

const ROLES: { value: EngineeringRole; label: string }[] = [
  { value: "AI_ML_SYSTEMS_ENGINEER", label: "AI / ML Systems Engineer" },
  { value: "BACKEND_DISTRIBUTED_SYSTEMS_ENGINEER", label: "Backend / Distributed Systems" },
  { value: "FRONTEND_PLATFORM_ENGINEER", label: "Frontend Platform Engineer" },
  { value: "DEVOPS_SRE_ENGINEER", label: "DevOps / SRE" },
  { value: "DATA_ENGINEER", label: "Data Engineer" },
  { value: "SECURITY_ENGINEER", label: "Security Engineer" },
  { value: "MOBILE_ENGINEER", label: "Mobile Engineer" },
  { value: "QA_TEST_AUTOMATION_ENGINEER", label: "QA / Test Automation" },
  { value: "EMBEDDED_SYSTEMS_ENGINEER", label: "Embedded Systems Engineer" },
];

export default function IntakePage() {
  const [resumeText, setResumeText] = useState("");
  const [role, setRole] = useState<EngineeringRole | "">("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setSession } = useActiveSession();
  const navigate = useNavigate();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (resumeText.trim().length < 20) {
      setError("Paste a bit more background -- a couple of sentences about what you've built.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await createSession(resumeText, role || undefined);
      setSession({
        sessionId: result.session_id,
        token: result.session_token,
        role: result.role,
        sanitizedName: result.sanitized_name,
      });
      navigate("/interview", {
        state: {
          greetingText: result.greeting_text,
          greetingAudioB64: result.greeting_audio_b64,
          redacted: result.redacted,
          retrievalBackend: result.retrieval_backend,
        },
      });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong starting the call.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="intake-page">
      <div className="intake-hero">
        <h1>Start your screening call</h1>
        <p>
          Paste a short background -- a resume, a summary, or a few sentences about what you've
          built. The call routes you to matching questions and starts immediately after.
        </p>
      </div>

      <form className="panel intake-form" onSubmit={handleSubmit}>
        <div>
          <label className="field-label" htmlFor="resume">
            Your background
          </label>
          <textarea
            id="resume"
            rows={9}
            placeholder="e.g. Backend engineer, four years, mostly Go and Postgres. Built the payments retry queue at my last job, migrated a monolith to services..."
            value={resumeText}
            onChange={(event) => setResumeText(event.target.value)}
            required
          />
        </div>

        <div>
          <label className="field-label" htmlFor="role">
            Target role (optional -- the agent can route you automatically)
          </label>
          <select id="role" value={role} onChange={(event) => setRole(event.target.value as EngineeringRole)}>
            <option value="">Let the agent decide</option>
            {ROLES.map((r) => (
              <option key={r.value} value={r.value}>
                {r.label}
              </option>
            ))}
          </select>
        </div>

        {error && <div className="error-banner">{error}</div>}

        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? "Connecting..." : "Begin call"}
        </button>

        <p className="intake-footnote">
          Any names, emails, phone numbers, or addresses in your text are redacted before anything
          is stored or scored.
        </p>
      </form>
    </div>
  );
}
