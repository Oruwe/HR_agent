"""Turn a stream of :class:`app.voice.ring_buffer.AudioFrame` into a WAV blob.

The internal pipeline speaks raw 20ms PCM16 frames (see ``app/voice/
ring_buffer.py``); a browser ``<audio>`` element needs a container it can
decode. WAV is the simplest one that needs no encoder dependency -- a 44-byte
header plus the concatenated PCM.
"""

from __future__ import annotations

import base64
import io
import wave
from collections.abc import AsyncIterator

import numpy as np

from app.voice.ring_buffer import AudioFrame


async def frames_to_wav_b64(frames: AsyncIterator[AudioFrame]) -> str | None:
    chunks: list[np.ndarray] = []
    sample_rate = 48_000
    async for frame in frames:
        chunks.append(frame.samples)
        sample_rate = frame.sample_rate
    if not chunks:
        return None
    pcm = np.concatenate(chunks).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # int16
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = ["frames_to_wav_b64"]
