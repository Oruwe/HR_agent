import { useCallback, useRef, useState } from "react";

export type RecorderStatus = "idle" | "recording" | "processing" | "unsupported" | "denied";

/** Wraps MediaRecorder for one answer clip at a time. Kept deliberately
 * simple -- push-to-talk, not a live stream -- because the backend's STT
 * endpoint (app/api/routes_sessions.py::post_audio_turn) takes one complete
 * clip per turn, not a socket.
 */
export function useMicRecorder() {
  const [status, setStatus] = useState<RecorderStatus>(
    typeof window !== "undefined" && "MediaRecorder" in window ? "idle" : "unsupported"
  );
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const start = useCallback(async () => {
    setError(null);
    if (status === "unsupported") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.start();
      recorderRef.current = recorder;
      setStatus("recording");
    } catch {
      setStatus("denied");
      setError("Microphone access was denied or unavailable. You can type your answer instead.");
    }
  }, [status]);

  const stop = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = recorderRef.current;
      if (!recorder) {
        resolve(null);
        return;
      }
      setStatus("processing");
      recorder.onstop = () => {
        const blob = chunksRef.current.length
          ? new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" })
          : null;
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        recorderRef.current = null;
        setStatus("idle");
        resolve(blob);
      };
      recorder.stop();
    });
  }, []);

  return { status, error, start, stop };
}
