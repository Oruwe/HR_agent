import { describe, expect, it, vi, afterEach } from "vitest";
import { ApiError, createSession, postTextTurn } from "../client";

describe("api client", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("throws ApiError with the server's error message on a non-2xx response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      json: async () => ({ error: "validation_error", detail: "resume_text too short" }),
    }) as unknown as typeof fetch;

    await expect(createSession("hi")).rejects.toMatchObject(
      new ApiError(422, "validation_error")
    );
  });

  it("throws a network ApiError (status 0) when fetch itself rejects", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;

    let caught: unknown;
    try {
      await postTextTurn("session-1", "token-1", "hello");
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(ApiError);
    expect((caught as ApiError).status).toBe(0);
  });

  it("parses a successful session creation response", async () => {
    const body = {
      session_id: "s1",
      session_token: "t1",
      sanitized_name: "Candidate_001",
      role: "BACKEND_DISTRIBUTED_SYSTEMS_ENGINEER",
      redacted: false,
      greeting_text: "Hi there.",
      greeting_audio_b64: null,
      retrieval_backend: "embedded fallback",
    };
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: async () => body,
    }) as unknown as typeof fetch;

    const result = await createSession("A long enough resume for validation.");
    expect(result.session_id).toBe("s1");
    expect(result.role).toBe("BACKEND_DISTRIBUTED_SYSTEMS_ENGINEER");
  });
});
