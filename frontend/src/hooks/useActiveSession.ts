import { useCallback, useState } from "react";

/** The candidate call has no login -- session handling means one thing here:
 * remembering this browser tab's session id + bearer token across a reload
 * (e.g. an accidental refresh mid-interview) without persisting it anywhere
 * durable. sessionStorage is the right tool: it survives a reload, dies with
 * the tab, and never leaks a still-active token onto a shared machine the
 * way localStorage would.
 */

const STORAGE_KEY = "hrte.active_session";

export interface StoredSession {
  sessionId: string;
  token: string;
  role: string;
  sanitizedName: string;
}

function read(): StoredSession | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StoredSession) : null;
  } catch {
    return null;
  }
}

export function useActiveSession() {
  const [session, setSessionState] = useState<StoredSession | null>(() => read());

  const setSession = useCallback((next: StoredSession | null) => {
    setSessionState(next);
    try {
      if (next) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      /* storage unavailable (private mode, quota) -- the app still works, it
         just won't survive a reload; nothing to surface to the candidate. */
    }
  }, []);

  return { session, setSession };
}
