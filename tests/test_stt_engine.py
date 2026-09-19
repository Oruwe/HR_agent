"""Tests for app.voice.stt_engine: provider selection and offline behavior."""

from __future__ import annotations

from app.config import Settings
from app.voice.stt_engine import (
    DeepgramTranscriber,
    MockTranscriber,
    build_transcriber,
)


def test_build_transcriber_defaults_to_mock_offline() -> None:
    transcriber = build_transcriber(Settings())
    assert isinstance(transcriber, MockTranscriber)


def test_build_transcriber_picks_deepgram_when_key_present() -> None:
    transcriber = build_transcriber(Settings(deepgram_api_key="dg-test-key"))
    assert isinstance(transcriber, DeepgramTranscriber)


def test_mock_transcriber_returns_placeholder_not_fabricated_text(run) -> None:
    transcriber = MockTranscriber()
    result = run(lambda: transcriber.transcribe(b"\x00\x01" * 500, "audio/webm"))
    assert result.provider == "mock"
    assert "not configured" in result.text


def test_mock_transcriber_handles_empty_audio(run) -> None:
    transcriber = MockTranscriber()
    result = run(lambda: transcriber.transcribe(b"", "audio/webm"))
    assert result.text == ""
    assert result.provider == "mock"
