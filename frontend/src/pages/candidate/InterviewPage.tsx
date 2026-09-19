import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ApiError, closeSession, postAudioTurn, postTextTurn } from "../../api/client";
import type { TurnResponse } from "../../api/client";
import { useActiveSession } from "../../hooks/useActiveSession";
import { useMicRecorder } from "../../hooks/useMicRecorder";

interface TranscriptEntry {
  id: string;
  speaker: "agent" | "candidate";
  text: string;
  turnaroundMs?: number;
  withinBudget?: boolean;
  sttFallback?: boolean;
}

interface LocationState {
  greetingText?: string;
  greetingAudioB64?: string | null;
  redacted?: boolean;
  retrievalBackend?: string;
}

function playBase64Wav(b64: string | null | undefined) {
  if (!b64) return;
  const audio = new Audio(`data:audio/wav;base64,${b64}`);
  void audio.play().catch(() => {
    /* autoplay can be blocked before any user gesture; the text transcript
       is still authoritative, so this is a silent, non-fatal no-op */
  });
}

export default function InterviewPage() {
  const { session, setSession } = useActiveSession();
  const navigate = useNavigate();
  const location = useLocation();
  const state = (location.state as LocationState) || {};

  const [transcript, setTranscript] = useState<TranscriptEntry[]>(() =>
    state.greetingText
      ? [{ id: "greeting", speaker: "agent", text: state.greetingText }]
      : []
  );
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [finished, setFinished] = useState(false);
  const [closing, setClosing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastTurn, setLastTurn] = useState<TurnResponse | null>(null);
  const mic = useMicRecorder();
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!session) navigate("/", { replace: true });
  }, [session, navigate]);

  useEffect(() => {
    if (state.greetingAudioB64) playBase64Wav(state.greetingAudioB64);
    // Only run once, when the page first mounts from the intake redirect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [transcript]);

  const applyTurnResult = useCallback((candidateText: string, result: TurnResponse, sttFallback: boolean) => {
    setTranscript((prev) => [
      ...prev,
      { id: `${Date.now()}-c`, speaker: "candidate", text: candidateText, sttFallback },
      ...(result.agent_text
        ? [
            {
              id: `${Date.now()}-a`,
              speaker: "agent" as const,
              text: result.agent_text,
              turnaroundMs: result.turnaround_ms,
              withinBudget: result.within_budget,
            },
          ]
        : []),
    ]);
    setLastTurn(result);
    playBase64Wav(result.audio_b64);
    if (result.finished) setFinished(true);
  }, []);

  async function submitText() {
    if (!session || !draft.trim() || busy) return;
    const text = draft.trim();
    setDraft("");
    setBusy(true);
    setError(null);
    try {
      const result = await postTextTurn(session.sessionId, session.token, text);
      applyTurnResult(text, result, false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "That answer didn't go through. Try again.");
      setDraft(text);
    } finally {
      setBusy(false);
    }
  }

  async function handleMicClick() {
    if (!session) return;
    if (mic.status === "recording") {
      const blob = await mic.stop();
      if (!blob) return;
      setBusy(true);
      setError(null);
      try {
        const result = await postAudioTurn(session.sessionId, session.token, blob);
        if (!result.stt_used) {
          setError(
            "Voice transcription isn't configured in this environment -- type your answer below instead."
          );
        } else {
          applyTurnResult("(spoken answer)", result, false);
        }
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Couldn't process that recording. Try typing instead.");
      } finally {
        setBusy(false);
      }
    } else {
      await mic.start();
    }
  }

  async function handleEndCall() {
    if (!session || closing) return;
    setClosing(true);
    setError(null);
    try {
      const evaluation = await closeSession(session.sessionId, session.token);
      setSession(null);
      navigate("/results", { state: { evaluation } });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't close the call. Try again.");
      setClosing(false);
    }
  }

  if (!session) return null;

  return (
    <div className="interview-page">
      <div className="interview-meta panel">
        <div>
          <span className="badge badge-live">
            <span className="brand-mark" /> live
          </span>{" "}
          <span className="interview-role">{session.role.replaceAll("_", " ").toLowerCase()}</span>
        </div>
        {lastTurn && (
          <div className="mono interview-latency">
            <span className={lastTurn.within_budget ? "badge badge-ok" : "badge badge-warn"}>
              {lastTurn.turnaround_ms.toFixed(0)}ms {lastTurn.within_budget ? "within budget" : "over budget"}
            </span>
          </div>
        )}
      </div>

      <div className="transcript panel" ref={scrollRef}>
        {transcript.map((entry) => (
          <div key={entry.id} className={`transcript-row transcript-${entry.speaker}`}>
            <div className="transcript-speaker">{entry.speaker === "agent" ? "Interviewer" : "You"}</div>
            <div className="transcript-bubble">
              {entry.text}
              {entry.turnaroundMs !== undefined && (
                <div className="transcript-turnaround mono">{entry.turnaroundMs.toFixed(0)}ms</div>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="transcript-row transcript-agent">
            <div className="transcript-speaker">Interviewer</div>
            <div className="transcript-bubble transcript-thinking">thinking…</div>
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {finished ? (
        <div className="panel interview-finished">
          <p>That's the full set of questions for this role. Ready to see how it went?</p>
          <button className="btn btn-primary" onClick={handleEndCall} disabled={closing}>
            {closing ? "Scoring..." : "Finish and see results"}
          </button>
        </div>
      ) : (
        <div className="interview-controls panel">
          <button
            className={`btn mic-btn ${mic.status === "recording" ? "mic-btn-active" : ""}`}
            onClick={handleMicClick}
            disabled={busy || mic.status === "unsupported" || mic.status === "processing"}
            type="button"
          >
            {mic.status === "recording" ? "Stop recording" : "Hold to answer by voice"}
          </button>
          {mic.error && <p className="mic-error">{mic.error}</p>}

          <div className="interview-text-fallback">
            <textarea
              rows={2}
              placeholder="Or type your answer here..."
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitText();
                }
              }}
              disabled={busy}
            />
            <button className="btn btn-primary" onClick={submitText} disabled={busy || !draft.trim()}>
              Send
            </button>
          </div>

          <button className="btn btn-danger end-call-btn" onClick={handleEndCall} disabled={closing}>
            {closing ? "Ending..." : "End call early"}
          </button>
        </div>
      )}
    </div>
  );
}
