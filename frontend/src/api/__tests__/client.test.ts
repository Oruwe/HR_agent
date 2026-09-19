import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  analyzePool,
  askAnalyst,
  importCandidates,
  listCandidates,
  setAdminToken,
} from "../client";

function mockFetch(impl: unknown) {
  globalThis.fetch = impl as unknown as typeof fetch;
}

describe("api client", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    setAdminToken("");
    vi.restoreAllMocks();
  });

  it("surfaces the server's error message on a non-2xx response", async () => {
    mockFetch(
      vi.fn().mockResolvedValue({
        ok: false,
        status: 401,
        statusText: "Unauthorized",
        json: async () => ({ error: "unauthorized", detail: "Admin token required." }),
      })
    );

    await expect(analyzePool()).rejects.toMatchObject(new ApiError(401, "unauthorized"));
  });

  it("falls back to statusText when the error body isn't JSON", async () => {
    mockFetch(
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        statusText: "Bad Gateway",
        json: async () => {
          throw new Error("not json");
        },
      })
    );

    await expect(listCandidates()).rejects.toMatchObject(new ApiError(502, "Bad Gateway"));
  });

  it("raises a network ApiError (status 0) when fetch itself rejects", async () => {
    mockFetch(vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    let caught: unknown;
    try {
      await listCandidates();
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(ApiError);
    expect((caught as ApiError).status).toBe(0);
  });

  it("parses a ranked candidate list", async () => {
    mockFetch(
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => [
          {
            id: "c1",
            name: "Elena Rossi",
            headline: "Staff Engineer",
            score: 0.91,
            recommendation: "INTERVIEW",
            rationale: "Owned the sharding migration end to end.",
            imported_at: 1,
            analyzed_at: 2,
          },
        ],
      })
    );

    const rows = await listCandidates();
    expect(rows).toHaveLength(1);
    expect(rows[0].recommendation).toBe("INTERVIEW");
    expect(rows[0].score).toBeCloseTo(0.91);
  });

  it("sends the admin token only once one has been set", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ imported: 1, redacted: 0, total_in_pool: 1 }),
    });
    mockFetch(fetchMock);

    await importCandidates([{ name: "Someone" }]);
    expect(fetchMock.mock.calls[0][1].headers).not.toHaveProperty("X-Admin-Token");

    setAdminToken("s3cret");
    await importCandidates([{ name: "Someone" }]);
    expect(fetchMock.mock.calls[1][1].headers["X-Admin-Token"]).toBe("s3cret");
  });

  it("posts the question and trimmed history to the analyst", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ reply: "Elena, then Priya.", candidates_considered: 9 }),
    });
    mockFetch(fetchMock);

    const result = await askAnalyst("Who are the top two?", [
      { role: "user", content: "hi" },
      { role: "model", content: "hello" },
    ]);

    expect(result.candidates_considered).toBe(9);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.message).toBe("Who are the top two?");
    expect(body.history).toHaveLength(2);
  });
});
