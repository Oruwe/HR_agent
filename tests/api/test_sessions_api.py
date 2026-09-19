"""End-to-end tests over the HTTP API: the same lifecycle the Candidate App
drives (create -> turns -> close), plus the failure modes the brief calls out
explicitly (auth, empty input, closed sessions).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

RESUME = (
    "Senior ML systems engineer. Ran FSDP training across many GPUs, tuned "
    "NCCL collectives, wrote Triton kernels, worked with vLLM and paged "
    "attention. Tuned HNSW index parameters for a large vector store."
)


def test_create_session_returns_greeting_and_token(api_client: TestClient) -> None:
    response = api_client.post("/api/sessions", json={"resume_text": RESUME})
    assert response.status_code == 201
    body = response.json()
    assert body["session_id"]
    assert body["session_token"]
    assert body["greeting_text"]
    assert body["sanitized_name"].startswith("Candidate_")
    assert body["role"]


def test_create_session_redacts_pii_before_routing(api_client: TestClient) -> None:
    resume = RESUME + " Contact me at jane@example.com or +91 98765 43210."
    response = api_client.post("/api/sessions", json={"resume_text": resume})
    assert response.status_code == 201
    assert response.json()["redacted"] is True


def test_full_turn_lifecycle(create_session, api_client: TestClient) -> None:
    session = create_session(RESUME)
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]

    for answer in (
        "I led FSDP training across a large GPU cluster and retuned NCCL collectives.",
        "We used tensor parallelism inside a node and pipeline parallelism across nodes.",
        "KV-cache size is two times layers times heads times head dim times sequence "
        "length times batch, in bytes per element.",
    ):
        response = api_client.post(
            f"/api/sessions/{sid}/turns/text", json={"text": answer}, headers=headers
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["turnaround_ms"] >= 0
        assert isinstance(payload["within_budget"], bool)

    response = api_client.post(f"/api/sessions/{sid}/close", headers=headers)
    assert response.status_code == 200
    evaluation = response.json()
    assert evaluation["session_id"] == sid
    assert evaluation["recommendation"] in {
        "ADVANCE",
        "ADVANCE_WITH_RESERVATIONS",
        "HOLD_FOR_HUMAN_REVIEW",
        "DO_NOT_ADVANCE",
        "INCOMPLETE",
    }
    assert evaluation["turns_completed"] >= 3


def test_turn_without_token_is_rejected(create_session, api_client: TestClient) -> None:
    session = create_session()
    response = api_client.post(
        f"/api/sessions/{session['session_id']}/turns/text", json={"text": "hello"}
    )
    assert response.status_code == 401


def test_turn_with_wrong_token_is_rejected(create_session, api_client: TestClient) -> None:
    session = create_session()
    headers = {"Authorization": "Bearer not-the-real-token"}
    response = api_client.post(
        f"/api/sessions/{session['session_id']}/turns/text", json={"text": "hello"}, headers=headers
    )
    assert response.status_code == 401


def test_turn_on_unknown_session_is_404(api_client: TestClient) -> None:
    headers = {"Authorization": "Bearer whatever"}
    response = api_client.post(
        "/api/sessions/does-not-exist/turns/text", json={"text": "hello"}, headers=headers
    )
    assert response.status_code == 404


def test_empty_turn_text_is_rejected(create_session, api_client: TestClient) -> None:
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    response = api_client.post(
        f"/api/sessions/{session['session_id']}/turns/text", json={"text": "   "}, headers=headers
    )
    assert response.status_code == 422


def test_turn_after_close_is_rejected(create_session, api_client: TestClient) -> None:
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]

    close_response = api_client.post(f"/api/sessions/{sid}/close", headers=headers)
    assert close_response.status_code == 200

    turn_response = api_client.post(
        f"/api/sessions/{sid}/turns/text", json={"text": "hello"}, headers=headers
    )
    assert turn_response.status_code == 409


def test_get_evaluation_before_close_is_404(create_session, api_client: TestClient) -> None:
    session = create_session()
    response = api_client.get(f"/api/sessions/{session['session_id']}/evaluation")
    assert response.status_code == 404


def test_get_evaluation_after_close(create_session, api_client: TestClient) -> None:
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]
    api_client.post(f"/api/sessions/{sid}/close", headers=headers)

    response = api_client.get(f"/api/sessions/{sid}/evaluation")
    assert response.status_code == 200
    assert response.json()["session_id"] == sid


def test_empty_audio_upload_is_rejected(create_session, api_client: TestClient) -> None:
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]
    response = api_client.post(
        f"/api/sessions/{sid}/turns/audio",
        headers=headers,
        files={"file": ("answer.webm", b"", "audio/webm")},
    )
    assert response.status_code == 422


def test_audio_turn_falls_back_gracefully_offline(create_session, api_client: TestClient) -> None:
    """No DEEPGRAM_API_KEY configured -> MockTranscriber -> stt_used is False,
    and the interview is not ended by a failed transcription."""
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]
    response = api_client.post(
        f"/api/sessions/{sid}/turns/audio",
        headers=headers,
        files={"file": ("answer.webm", b"\x00\x01\x02\x03" * 100, "audio/webm")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stt_used"] is False
    assert body["stt_provider"] == "mock"
