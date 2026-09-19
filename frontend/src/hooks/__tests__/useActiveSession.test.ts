import { renderHook, act } from "@testing-library/react";
import { describe, expect, it, beforeEach } from "vitest";
import { useActiveSession } from "../useActiveSession";

describe("useActiveSession", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it("starts with no session when sessionStorage is empty", () => {
    const { result } = renderHook(() => useActiveSession());
    expect(result.current.session).toBeNull();
  });

  it("persists a session to sessionStorage and reflects it in state", () => {
    const { result } = renderHook(() => useActiveSession());
    act(() => {
      result.current.setSession({
        sessionId: "s1",
        token: "t1",
        role: "BACKEND_DISTRIBUTED_SYSTEMS_ENGINEER",
        sanitizedName: "Candidate_001",
      });
    });
    expect(result.current.session?.sessionId).toBe("s1");
    expect(JSON.parse(sessionStorage.getItem("hrte.active_session")!).sessionId).toBe("s1");
  });

  it("clears sessionStorage when set to null", () => {
    const { result } = renderHook(() => useActiveSession());
    act(() => {
      result.current.setSession({
        sessionId: "s1",
        token: "t1",
        role: "r",
        sanitizedName: "n",
      });
    });
    act(() => {
      result.current.setSession(null);
    });
    expect(result.current.session).toBeNull();
    expect(sessionStorage.getItem("hrte.active_session")).toBeNull();
  });

  it("a fresh hook instance reads a session already in sessionStorage", () => {
    sessionStorage.setItem(
      "hrte.active_session",
      JSON.stringify({ sessionId: "s2", token: "t2", role: "r", sanitizedName: "n" })
    );
    const { result } = renderHook(() => useActiveSession());
    expect(result.current.session?.sessionId).toBe("s2");
  });
});
