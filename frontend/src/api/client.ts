/** Thin fetch wrapper over the API. No SDK, no codegen -- the surface is
 * small enough that a hand-written typed wrapper stays honest with what the
 * backend actually returns (see app/api/schemas.py, the source of truth).
 */

const RUNTIME_ENV = (typeof window !== "undefined" && (window as any).__ENV__) || {};
const BASE_URL = (
  RUNTIME_ENV.VITE_API_BASE_URL ||
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
  ""
).replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

let adminToken = "";

export function setAdminToken(token: string) {
  adminToken = token;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(adminToken ? { "X-Admin-Token": adminToken } : {}),
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

// ---- Types (mirrors app/api/schemas.py) ------------------------------------

export type Verdict = "INTERVIEW" | "MAYBE" | "PASS";

export interface CandidateSummary {
  id: string;
  name: string;
  headline: string;
  score: number | null;
  recommendation: Verdict | string | null;
  rationale: string | null;
  imported_at: number;
  analyzed_at: number | null;
}

export interface CandidateDetail extends CandidateSummary {
  source: Record<string, unknown>;
}

export interface ImportResponse {
  imported: number;
  redacted: number;
  total_in_pool: number;
}

export interface AnalyzeResponse {
  analyzed: number;
  skipped: number;
  offline: boolean;
}

export interface ChatMessage {
  role: "user" | "model";
  content: string;
}

export interface ChatResponse {
  reply: string;
  candidates_considered: number;
}

export interface StatusResponse {
  environment: string;
  offline: boolean;
  model_configured: boolean;
  model: string;
  degraded: boolean;
  fallbacks: number;
  candidates: number;
  analyzed: number;
}

// ---- Calls -------------------------------------------------------------------

export function listCandidates() {
  return request<CandidateSummary[]>("/api/candidates");
}

export function getCandidate(id: string) {
  return request<CandidateDetail>(`/api/candidates/${id}`);
}

export function deleteCandidate(id: string) {
  return request<void>(`/api/candidates/${id}`, { method: "DELETE" });
}

export function importCandidates(candidates: Record<string, unknown>[]) {
  return request<ImportResponse>("/api/candidates/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidates }),
  });
}

/** Load the bundled demo pool, pre-ranked. Fails with 409 if the pool isn't empty. */
export function loadDemoPool() {
  return request<ImportResponse>("/api/candidates/demo", { method: "POST" });
}

export function analyzePool() {
  return request<AnalyzeResponse>("/api/analyze", { method: "POST" });
}

export function askAnalyst(message: string, history: ChatMessage[]) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
}

export function getStatus() {
  return request<StatusResponse>("/api/status");
}
