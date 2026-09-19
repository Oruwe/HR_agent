/** Thin fetch wrapper over the API Gateway. No SDK, no codegen -- the surface
 * is small enough that a typed wrapper by hand stays honest with what the
 * backend actually returns (see app/api/schemas.py, the source of truth).
 */

const RUNTIME_ENV = (typeof window !== "undefined" && (window as any).__ENV__) || {};
const BASE_URL = (RUNTIME_ENV.VITE_API_BASE_URL || (import.meta.env.VITE_API_BASE_URL as string | undefined) || "").replace(
  /\/$/,
  ""
);

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.headers || {}),
      },
    });
  } catch {
    throw new ApiError(0, "Could not reach the server. Check your connection and try again.");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.error || body.detail || detail;
    } catch {
      /* body wasn't JSON; fall back to statusText */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

// ---- Types (mirrors app/api/schemas.py) ------------------------------------

export type EngineeringRole =
  | "AI_ML_SYSTEMS_ENGINEER"
  | "BACKEND_DISTRIBUTED_SYSTEMS_ENGINEER"
  | "FRONTEND_PLATFORM_ENGINEER"
  | "DEVOPS_SRE_ENGINEER"
  | "DATA_ENGINEER"
  | "SECURITY_ENGINEER"
  | "MOBILE_ENGINEER"
  | "QA_TEST_AUTOMATION_ENGINEER"
  | "EMBEDDED_SYSTEMS_ENGINEER";

export interface CreateSessionResponse {
  session_id: string;
  session_token: string;
  sanitized_name: string;
  role: EngineeringRole;
  redacted: boolean;
  greeting_text: string;
  greeting_audio_b64: string | null;
  retrieval_backend: string;
}

export interface TurnResponse {
  agent_text: string;
  audio_b64: string | null;
  turnaround_ms: number;
  within_budget: boolean;
  speculation_hit: boolean;
  turns_completed: number;
  finished: boolean;
  stt_used: boolean;
  stt_provider: string | null;
}

export interface CompetencyScoreOut {
  key: string;
  label: string;
  score: number;
  source: string;
  rationale: string;
}

export interface EvaluationResponse {
  candidate_id: string;
  sanitized_name: string;
  session_id: string;
  target_role: EngineeringRole;
  rubric_fit_index: number;
  routing_confidence: number;
  recommendation: string;
  competency_scores: CompetencyScoreOut[];
  flagged_limitations: string[];
  turns_completed: number;
  latency_compliance: number;
}

export interface SessionSummary {
  session_id: string;
  sanitized_name: string;
  role: string;
  status: string;
  retrieval_backend: string;
  turns: number;
  created_at: number;
  updated_at: number;
  has_evaluation: boolean;
  rubric_fit_index: number | null;
  recommendation: string | null;
}

export interface TranscriptTurn {
  speaker: "candidate" | "agent" | string;
  text: string;
  offset_ms: number;
  turnaround_ms: number | null;
}

export interface LatencyStageOut {
  stage: string;
  budget_ms: number;
  p95_ms: number | null;
}

export interface SystemStatusResponse {
  environment: string;
  offline: boolean;
  cognition_configured: boolean;
  moss_configured: boolean;
  qdrant_configured: boolean;
  transport_configured: boolean;
  telemetry_configured: boolean;
  stt_configured: boolean;
  session_state_backend: string;
  database_url_scheme: string;
  active_sessions: number;
  latency_budget_ms: number;
  stage_budgets: LatencyStageOut[];
}

// ---- Candidate-facing calls -------------------------------------------------

export function createSession(resumeText: string, targetRole?: EngineeringRole) {
  return request<CreateSessionResponse>("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_text: resumeText, target_role: targetRole || null }),
  });
}

export function postTextTurn(sessionId: string, token: string, text: string) {
  return request<TurnResponse>(`/api/sessions/${sessionId}/turns/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ text }),
  });
}

export async function postAudioTurn(sessionId: string, token: string, blob: Blob) {
  const form = new FormData();
  form.append("file", blob, "answer.webm");
  return request<TurnResponse>(`/api/sessions/${sessionId}/turns/audio`, {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });
}

export function closeSession(sessionId: string, token: string) {
  return request<EvaluationResponse>(`/api/sessions/${sessionId}/close`, {
    method: "POST",
    headers: authHeaders(token),
  });
}

export function getEvaluation(sessionId: string) {
  return request<EvaluationResponse>(`/api/sessions/${sessionId}/evaluation`);
}

// ---- Admin calls -------------------------------------------------------------

export function listSessions(adminToken?: string) {
  return request<SessionSummary[]>("/api/admin/sessions", {
    headers: adminToken ? { "X-Admin-Token": adminToken } : undefined,
  });
}

export function getSystemStatus(adminToken?: string) {
  return request<SystemStatusResponse>("/api/admin/status", {
    headers: adminToken ? { "X-Admin-Token": adminToken } : undefined,
  });
}

export function getAdminEvaluation(sessionId: string, adminToken?: string) {
  return request<EvaluationResponse>(`/api/admin/sessions/${sessionId}/evaluation`, {
    headers: adminToken ? { "X-Admin-Token": adminToken } : undefined,
  });
}

export function getAdminTranscript(sessionId: string, adminToken?: string) {
  return request<TranscriptTurn[]>(`/api/admin/sessions/${sessionId}/transcript`, {
    headers: adminToken ? { "X-Admin-Token": adminToken } : undefined,
  });
}
