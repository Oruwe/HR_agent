"""Voice activity detection and the turn-taking state machine.

Where the latency actually comes from
-------------------------------------
The single largest lever on perceived conversational latency is not the model.
It is the endpoint decision: how long you wait after the candidate stops making
sound before you accept that they are done. Wait 800ms (the common default) and
no amount of inference speed will make the agent feel present. Wait 150ms and
you will cut people off mid-sentence, which is worse.

This module resolves that with *speculative turn-taking*. Two thresholds, not
one:

* at ``speculative_silence_ms`` (default 40ms) we start generating a response,
  while still listening;
* at ``endpoint_silence_ms`` (default 120ms) we commit and start speaking.

If the candidate resumes during the gap, the speculative turn is abandoned and
costs nothing but some compute. If they do not, retrieval and time-to-first-
token have already been paid for by the time we commit -- so the 80ms of
speculation window is subtracted from the user-perceived turnaround. That is
what makes the 150ms budget reachable at all; without it the budget would have
to fit *inside* the endpoint silence, which is not physically possible with a
network round trip involved.

Barge-in is the same machinery run backwards: while the agent is speaking,
30ms of sustained candidate energy drains the output buffer immediately.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from app.config import FRAME_DURATION_MS, Settings, get_settings
from app.voice.ring_buffer import AudioFrame

logger = logging.getLogger(__name__)


class VadState(StrEnum):
    """Turn-taking states. Names are stable -- they appear in traces."""

    IDLE = "IDLE"
    CANDIDATE_SPEAKING = "CANDIDATE_SPEAKING"
    SILENCE_PROBING = "SILENCE_PROBING"
    AGENT_TURNTABLE_FIRE = "AGENT_TURNTABLE_FIRE"
    AGENT_SPEAKING = "AGENT_SPEAKING"


class VadEventType(StrEnum):
    SPEECH_START = "speech_start"
    #: Speculation threshold crossed: begin generating, do not speak yet.
    SPECULATE = "speculate"
    #: Candidate resumed during the speculation window: discard the draft turn.
    SPECULATION_CANCELLED = "speculation_cancelled"
    #: Endpoint confirmed: commit the turn and start speaking.
    TURN_COMMIT = "turn_commit"
    #: Candidate spoke over the agent: drain output now.
    BARGE_IN = "barge_in"


@dataclass(frozen=True, slots=True)
class VadEvent:
    type: VadEventType
    state: VadState
    timestamp_ms: float
    #: Silence accumulated at the moment the event fired.
    silence_ms: float = 0.0
    #: Speech accumulated at the moment the event fired.
    speech_ms: float = 0.0


@runtime_checkable
class VoiceDetector(Protocol):
    """Anything that can answer 'is this 20ms of frame voiced?'."""

    def is_speech(self, frame: AudioFrame) -> bool: ...

    def reset(self) -> None: ...


class EnergyVad:
    """Adaptive-threshold energy VAD. The dependency-free default.

    Two refinements over a fixed RMS threshold, both of which matter in a real
    call:

    * **Adaptive noise floor.** The threshold tracks the quietest recent frames,
      so a candidate on a noisy street and one in a quiet room both work without
      configuration. The floor adapts quickly upward and slowly downward, which
      is the correct asymmetry: mistaking noise for speech costs an interruption,
      while mistaking speech for noise costs a missed turn.
    * **Zero-crossing gate.** Broadband noise has a far higher zero-crossing
      rate than voiced speech. Rejecting high-ZCR frames suppresses keyboard
      clatter and fricative-like hiss that pure energy detection would accept.

    It is not as accurate as Silero, and it is not meant to be. It is the
    fallback that keeps the agent working when the ONNX runtime is unavailable,
    and it is what makes the latency tests hermetic.
    """

    def __init__(
        self,
        floor_alpha: float = 0.05,
        margin_db: float = 9.0,
        min_rms: float = 0.006,
        max_zcr: float = 0.34,
    ) -> None:
        self._floor_alpha = floor_alpha
        self._margin = 10.0 ** (margin_db / 20.0)
        self._min_rms = min_rms
        self._max_zcr = max_zcr
        self._noise_floor = min_rms

    def reset(self) -> None:
        self._noise_floor = self._min_rms

    def is_speech(self, frame: AudioFrame) -> bool:
        rms = frame.rms()
        threshold = max(self._min_rms, self._noise_floor * self._margin)

        if rms < threshold:
            # Track the floor only on frames we believe are noise, otherwise
            # sustained speech would drag the floor up and gate itself off.
            self._noise_floor = (
                1.0 - self._floor_alpha
            ) * self._noise_floor + self._floor_alpha * rms
            return False

        signal = frame.as_float32()
        if len(signal) < 2:
            return False
        zcr = float(np.mean(np.abs(np.diff(np.signbit(signal))))) if len(signal) > 1 else 0.0
        return zcr <= self._max_zcr


class SileroVad:
    """Silero VAD v5 over ONNX Runtime, with automatic degradation.

    Silero expects 16kHz in 512-sample windows; LiveKit delivers 48kHz in
    960-sample frames. We decimate by 3 rather than resampling properly -- at
    this frame size the aliasing is well below the decision threshold and a
    polyphase resampler would cost more than the accuracy it buys inside a 25ms
    budget.

    If the model or the runtime is missing, this transparently becomes an
    :class:`EnergyVad`. A screening call must not fail to start because a model
    file did not download.
    """

    SAMPLE_RATE = 16_000
    WINDOW = 512

    def __init__(self, model_path: str | None = None, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._session: object | None = None
        self._state: np.ndarray | None = None
        self._tail = np.zeros(0, dtype=np.float32)
        self._fallback = EnergyVad()
        self._available = False
        if model_path:
            self._try_load(model_path)

    @property
    def available(self) -> bool:
        return self._available

    def _try_load(self, model_path: str) -> None:
        try:  # pragma: no cover - requires the optional onnxruntime dependency
            import onnxruntime as ort  # type: ignore[import-not-found]

            options = ort.SessionOptions()
            # One thread. VAD runs per 20ms frame; thread-pool handoff costs
            # more than the convolution it would parallelise.
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._available = True
        except Exception as exc:  # pragma: no cover - degradation path
            logger.warning(
                "Silero VAD unavailable (%s); falling back to the adaptive energy "
                "detector. Endpointing accuracy is reduced but the call proceeds.",
                type(exc).__name__,
            )
            self._session = None
            self._available = False

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)
        if self._state is not None:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._fallback.reset()

    def is_speech(self, frame: AudioFrame) -> bool:
        if not self._available or self._session is None:
            return self._fallback.is_speech(frame)
        try:  # pragma: no cover - requires the optional dependency
            decimated = frame.as_float32()[::3]  # 48kHz -> 16kHz
            buffer = np.concatenate([self._tail, decimated])
            probability = 0.0
            while len(buffer) >= self.WINDOW:
                window = buffer[: self.WINDOW].reshape(1, -1)
                out, self._state = self._session.run(  # type: ignore[attr-defined]
                    None,
                    {
                        "input": window,
                        "state": self._state,
                        "sr": np.array(self.SAMPLE_RATE, dtype=np.int64),
                    },
                )
                probability = max(probability, float(out[0][0]))
                buffer = buffer[self.WINDOW :]
            self._tail = buffer
            return probability >= self._threshold
        except Exception:  # pragma: no cover - degrade mid-call
            self._available = False
            return self._fallback.is_speech(frame)


@dataclass(slots=True)
class VadStream:
    """The turn-taking state machine. One instance per screening session.

    Hysteresis is deliberate and asymmetric. Entering speech needs two
    consecutive voiced frames (40ms) so a door slam does not open a turn;
    leaving speech is immediate, because the silence timer is what provides the
    debounce on that side and adding more would directly add latency.
    """

    settings: Settings = field(default_factory=get_settings)
    detector: VoiceDetector = field(default_factory=EnergyVad)
    state: VadState = VadState.IDLE

    _silence_ms: float = field(default=0.0, init=False)
    _speech_ms: float = field(default=0.0, init=False)
    _agent_speech_ms: float = field(default=0.0, init=False)
    _voiced_run: int = field(default=0, init=False)
    _speculated: bool = field(default=False, init=False)
    _history: deque[bool] = field(default_factory=lambda: deque(maxlen=64), init=False)
    _clock_ms: float = field(default=0.0, init=False)

    #: Consecutive voiced frames required to open a turn (2 frames == 40ms).
    speech_onset_frames: int = 2

    @property
    def agent_is_speaking(self) -> bool:
        return self.state is VadState.AGENT_SPEAKING

    @property
    def silence_ms(self) -> float:
        return self._silence_ms

    @property
    def speech_ms(self) -> float:
        return self._speech_ms

    @property
    def speculating(self) -> bool:
        return self._speculated

    def begin_agent_turn(self) -> None:
        """Called when agent audio starts flowing; arms barge-in detection."""
        self.state = VadState.AGENT_SPEAKING
        self._agent_speech_ms = 0.0
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._voiced_run = 0
        self._speculated = False

    def end_agent_turn(self) -> None:
        """Called when agent audio finishes; hands the floor back."""
        self.state = VadState.IDLE
        self._agent_speech_ms = 0.0
        self._voiced_run = 0

    def reset(self) -> None:
        self.state = VadState.IDLE
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._agent_speech_ms = 0.0
        self._voiced_run = 0
        self._speculated = False
        self._history.clear()
        self.detector.reset()

    def process(self, frame: AudioFrame) -> VadEvent | None:
        """Feed one 20ms frame. Returns an event when the turn state changes.

        Exactly one event per frame at most. Callers treat the return value as
        the only control signal; the state machine never mutates anything
        outside itself, which keeps it trivially unit-testable.
        """
        duration = frame.duration_ms or FRAME_DURATION_MS
        self._clock_ms = frame.timestamp_ms + duration
        voiced = self.detector.is_speech(frame)
        self._history.append(voiced)

        if self.state is VadState.AGENT_SPEAKING:
            return self._process_during_agent_turn(voiced, duration)
        return self._process_during_candidate_turn(voiced, duration)

    # -- barge-in ------------------------------------------------------------

    def _process_during_agent_turn(self, voiced: bool, duration: float) -> VadEvent | None:
        if not voiced:
            # Decay rather than reset: a candidate's speech is not perfectly
            # continuous, and a single unvoiced frame mid-word should not
            # restart the barge-in timer from zero.
            self._agent_speech_ms = max(0.0, self._agent_speech_ms - duration)
            return None

        self._agent_speech_ms += duration
        if self._agent_speech_ms >= self.settings.barge_in_ms:
            self.state = VadState.CANDIDATE_SPEAKING
            self._speech_ms = self._agent_speech_ms
            self._silence_ms = 0.0
            self._agent_speech_ms = 0.0
            self._voiced_run = self.speech_onset_frames
            return VadEvent(
                type=VadEventType.BARGE_IN,
                state=self.state,
                timestamp_ms=self._clock_ms,
                speech_ms=self._speech_ms,
            )
        return None

    # -- normal turn-taking --------------------------------------------------

    def _process_during_candidate_turn(self, voiced: bool, duration: float) -> VadEvent | None:
        if voiced:
            return self._on_voiced(duration)
        return self._on_silence(duration)

    def _on_voiced(self, duration: float) -> VadEvent | None:
        self._voiced_run += 1
        self._speech_ms += duration
        was_probing = self.state is VadState.SILENCE_PROBING
        had_speculated = self._speculated
        self._silence_ms = 0.0

        if self._voiced_run < self.speech_onset_frames and self.state is VadState.IDLE:
            return None  # not yet convinced this is speech

        previous = self.state
        self.state = VadState.CANDIDATE_SPEAKING

        if was_probing and had_speculated:
            # The candidate was only drawing breath. Throw the draft away --
            # this is the case speculation is designed to make cheap.
            self._speculated = False
            return VadEvent(
                type=VadEventType.SPECULATION_CANCELLED,
                state=self.state,
                timestamp_ms=self._clock_ms,
                speech_ms=self._speech_ms,
            )

        self._speculated = False
        if previous in (
            VadState.IDLE,
            VadState.SILENCE_PROBING,
            # Speech arriving after commit but before the agent has made a sound:
            # the candidate was still thinking. Reopen their turn.
            VadState.AGENT_TURNTABLE_FIRE,
        ):
            return VadEvent(
                type=VadEventType.SPEECH_START,
                state=self.state,
                timestamp_ms=self._clock_ms,
                speech_ms=self._speech_ms,
            )
        return None

    def _on_silence(self, duration: float) -> VadEvent | None:
        if self.state is VadState.IDLE:
            return None
        if self.state is VadState.AGENT_TURNTABLE_FIRE:
            # The turn is already committed and the orchestrator owns the floor.
            # Continuing to accumulate silence here would re-fire commit every
            # frame until the first audio byte goes out, which is how a single
            # pause turns into a stutter of duplicate generations.
            return None

        self._voiced_run = 0
        self._silence_ms += duration
        if self.state is VadState.CANDIDATE_SPEAKING:
            self.state = VadState.SILENCE_PROBING

        if not self._speculated and self._silence_ms >= self.settings.speculative_silence_ms:
            self._speculated = True
            return VadEvent(
                type=VadEventType.SPECULATE,
                state=self.state,
                timestamp_ms=self._clock_ms,
                silence_ms=self._silence_ms,
                speech_ms=self._speech_ms,
            )

        if self._silence_ms >= self.settings.endpoint_silence_ms:
            self.state = VadState.AGENT_TURNTABLE_FIRE
            event = VadEvent(
                type=VadEventType.TURN_COMMIT,
                state=self.state,
                timestamp_ms=self._clock_ms,
                silence_ms=self._silence_ms,
                speech_ms=self._speech_ms,
            )
            self._speech_ms = 0.0
            self._speculated = False
            return event

        return None


def build_detector(
    settings: Settings | None = None, model_path: str | None = None
) -> VoiceDetector:
    """Pick the best detector available, without ever failing to return one."""
    settings = settings or get_settings()
    if model_path:
        silero = SileroVad(model_path)
        if silero.available:
            return silero
    return EnergyVad()


__all__ = [
    "EnergyVad",
    "SileroVad",
    "VadEvent",
    "VadEventType",
    "VadState",
    "VadStream",
    "VoiceDetector",
    "build_detector",
]
