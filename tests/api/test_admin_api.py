"""Admin dashboard API tests + health/readiness/metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import TOTAL_TURNAROUND_BUDGET_MS


def test_health_ok(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_database_check(api_client: TestClient) -> None:
    response = api_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["checks"]["database"] is True


def test_metrics_endpoint_exposes_prometheus_format(api_client: TestClient, create_session) -> None:
    create_session()
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert "hrte_sessions_created_total" in response.text
    assert "hrte_http_requests_total" in response.text


def test_admin_lists_sessions(create_session, api_client: TestClient) -> None:
    create_session()
    create_session()
    response = api_client.get("/api/admin/sessions")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {row["status"] for row in body} == {"active"}


def test_admin_status_reports_capability_flags(api_client: TestClient) -> None:
    response = api_client.get("/api/admin/status")
    assert response.status_code == 200
    body = response.json()
    assert body["offline"] is True  # no credentials in the test environment
    assert body["stt_configured"] is False
    assert body["session_state_backend"] == "in-process"
    assert len(body["stage_budgets"]) == 7
    assert sum(s["budget_ms"] for s in body["stage_budgets"]) == TOTAL_TURNAROUND_BUDGET_MS


def test_admin_session_evaluation_matches_candidate_facing_one(
    create_session, api_client: TestClient
) -> None:
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]
    api_client.post(f"/api/sessions/{sid}/close", headers=headers)

    admin_response = api_client.get(f"/api/admin/sessions/{sid}/evaluation")
    candidate_response = api_client.get(f"/api/sessions/{sid}/evaluation", headers=headers)
    assert admin_response.status_code == candidate_response.status_code == 200
    assert admin_response.json()["recommendation"] == candidate_response.json()["recommendation"]


def test_admin_transcript_returns_ordered_turns(create_session, api_client: TestClient) -> None:
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]
    api_client.post(
        f"/api/sessions/{sid}/turns/text", json={"text": "hello there"}, headers=headers
    )

    response = api_client.get(f"/api/admin/sessions/{sid}/transcript")
    assert response.status_code == 200
    turns = response.json()
    assert [t["speaker"] for t in turns] == ["agent", "candidate", "agent"]
    assert turns[1]["text"] == "hello there"
    assert turns == sorted(turns, key=lambda t: t["offset_ms"])


def test_admin_transcript_on_unknown_session_is_404(api_client: TestClient) -> None:
    response = api_client.get("/api/admin/sessions/does-not-exist/transcript")
    assert response.status_code == 404


def test_session_summary_carries_ranking_fields_only_after_close(
    create_session, api_client: TestClient
) -> None:
    session = create_session()
    headers = {"Authorization": f"Bearer {session['session_token']}"}
    sid = session["session_id"]

    row = next(s for s in api_client.get("/api/admin/sessions").json() if s["session_id"] == sid)
    assert row["rubric_fit_index"] is None
    assert row["recommendation"] is None

    api_client.post(f"/api/sessions/{sid}/close", headers=headers)

    row = next(s for s in api_client.get("/api/admin/sessions").json() if s["session_id"] == sid)
    assert isinstance(row["rubric_fit_index"], float)
    assert row["recommendation"] is not None


def test_admin_token_gate(create_session, api_client: TestClient) -> None:
    import os

    os.environ["HRTE_ADMIN_TOKEN"] = "secret-token"
    try:
        response = api_client.get("/api/admin/sessions")
        assert response.status_code == 401

        response = api_client.get("/api/admin/sessions", headers={"X-Admin-Token": "secret-token"})
        assert response.status_code == 200
    finally:
        del os.environ["HRTE_ADMIN_TOKEN"]
