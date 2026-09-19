"""Deterministic role assignment and the evaluation policy.

Role assignment changes what a person is asked and how they are scored, so the
property under test is not merely "accurate" but "reproducible and
explainable". Accuracy is checked against nine hand-written profiles; the
stronger assertions are the ones about determinism, tie-breaking, and the gate
ordering that decides a recommendation.
"""

from __future__ import annotations

import pytest

from app.schemas.candidate import CandidateProfile, ScreeningSession, pseudonym_for
from app.schemas.evaluation import (
    CandidateEvaluation,
    CompetencyScore,
    EvidenceSource,
    Recommendation,
    RoleFit,
    derive_recommendation,
)
from app.schemas.roles import (
    ROLE_RUBRICS,
    EngineeringRole,
    best_role,
    match_confidence,
    normalise,
    rank_roles,
    rubric_for,
    score_role,
    signal_idf,
)
from app.storage.qdrant_client import HybridVectorStore
from tests.conftest import ROLE_PROFILES

# =============================================================================
# Rubric integrity
# =============================================================================


def test_there_are_exactly_nine_rubrics() -> None:
    assert len(ROLE_RUBRICS) == 9
    assert set(ROLE_RUBRICS) == set(EngineeringRole)


@pytest.mark.parametrize("role", list(EngineeringRole))
def test_each_rubric_is_well_formed(role: EngineeringRole) -> None:
    rubric = ROLE_RUBRICS[role]
    assert rubric.competencies, f"{role} has no competencies"
    assert rubric.eliminations, f"{role} has no elimination gate"
    assert 0.0 < rubric.advance_threshold < 1.0
    keys = [c.key for c in rubric.competencies]
    assert len(keys) == len(set(keys))
    for competency in rubric.competencies:
        assert competency.signals
        assert competency.probe.endswith(("?", ".")), f"{competency.key} probe is not a question"
        assert all(s == s.lower() for s in competency.signals), (
            f"{competency.key} has a non-lowercase signal; matching is case-folded"
        )


@pytest.mark.parametrize("role", list(EngineeringRole))
def test_elimination_criteria_reference_real_competencies(role: EngineeringRole) -> None:
    rubric = ROLE_RUBRICS[role]
    keys = {c.key for c in rubric.competencies}
    for elimination in rubric.eliminations:
        assert elimination.competency_key in keys


def test_idf_discriminates_between_shared_and_unique_signals() -> None:
    """The mechanism that makes the matcher discriminative rather than eager."""
    idf = signal_idf()
    unique = idf["flashattention"]  # claimed by one rubric
    shared = idf["schema evolution"]  # claimed by more than one
    assert unique > shared, "IDF is not discounting shared signals"
    assert all(v > 0 for v in idf.values())


# =============================================================================
# The nine scenarios
# =============================================================================


@pytest.mark.parametrize("expected", list(EngineeringRole), ids=lambda r: r.value)
def test_each_profile_routes_to_its_own_rubric(expected: EngineeringRole) -> None:
    match = best_role(ROLE_PROFILES[expected])
    assert match.role is expected, f"{expected.value} profile routed to {match.role.value}"


@pytest.mark.parametrize("expected", list(EngineeringRole), ids=lambda r: r.value)
def test_routing_is_confident(expected: EngineeringRole) -> None:
    ranked = rank_roles(ROLE_PROFILES[expected])
    confidence = match_confidence(ranked)
    assert confidence >= 0.2, (
        f"{expected.value} separated from the runner-up by only {confidence:.0%}"
    )
    assert ranked[0].score > ranked[1].score


@pytest.mark.parametrize("expected", list(EngineeringRole), ids=lambda r: r.value)
def test_routing_is_deterministic_across_repeated_runs(expected: EngineeringRole) -> None:
    profile = ROLE_PROFILES[expected]
    baseline = rank_roles(profile)
    for _ in range(25):
        again = rank_roles(profile)
        assert [m.role for m in again] == [m.role for m in baseline]
        assert [round(m.score, 10) for m in again] == [round(m.score, 10) for m in baseline]


def test_ranking_is_a_total_order() -> None:
    ranked = rank_roles(ROLE_PROFILES[EngineeringRole.DATA_PLATFORM_ENGINEER])
    assert len(ranked) == 9
    assert len({m.role for m in ranked}) == 9
    scores = [m.score for m in ranked]
    assert scores == sorted(scores, reverse=True)


def test_empty_evidence_still_produces_a_stable_ranking() -> None:
    """No evidence must not mean a crash or an arbitrary assignment."""
    ranked = rank_roles("")
    assert len(ranked) == 9
    assert all(m.score == 0.0 for m in ranked)
    assert [m.role for m in ranked] == list(EngineeringRole), (
        "zero-score ties must fall back to declaration order"
    )
    assert match_confidence(ranked) == 0.0


def test_unrelated_text_scores_low_everywhere() -> None:
    ranked = rank_roles("I enjoy baking sourdough and restoring bicycles on weekends.")
    assert ranked[0].score < 0.15


def test_matched_signals_are_reported_as_evidence() -> None:
    """A score with no traceable evidence cannot be defended in an audit."""
    match = score_role(
        ROLE_PROFILES[EngineeringRole.AI_ML_SYSTEMS_ENGINEER],
        EngineeringRole.AI_ML_SYSTEMS_ENGINEER,
    )
    assert match.matched_signal_count > 0
    hit = {c.key: c.matched_signals for c in match.competencies}
    assert "fsdp" in hit["distributed_training"]
    assert "vllm" in hit["inference_kernels"]


def test_normalise_is_boundary_safe() -> None:
    """Substring matching must not fire on a fragment inside a longer word."""
    assert (
        not score_role("We used scaffolding.", EngineeringRole.AI_ML_SYSTEMS_ENGINEER)
        .competencies[0]
        .matched_signals
    )
    assert " fsdp " in normalise("Our FSDP setup")


# =============================================================================
# Vector retrieval agrees with the deterministic matcher
# =============================================================================


@pytest.mark.parametrize("expected", list(EngineeringRole), ids=lambda r: r.value)
def test_hybrid_retrieval_agrees_with_the_matcher(
    warm_store: HybridVectorStore, expected: EngineeringRole
) -> None:
    """Independent evidence: retrieval and the matcher should not disagree.

    They use different mechanisms -- hashed embeddings plus BM25 fusion versus
    IDF-weighted signal matching -- so agreement across all nine is meaningful
    corroboration rather than a tautology.
    """
    hits = warm_store.search_rubrics(ROLE_PROFILES[expected], limit=1)
    assert hits, "retrieval returned nothing"
    assert hits[0].id == expected.value


# =============================================================================
# Recommendation policy
# =============================================================================


def _passing_scores(role: EngineeringRole, value: float = 0.85) -> dict[str, float]:
    return {c.key: value for c in rubric_for(role).competencies}


def test_strong_candidate_advances() -> None:
    role = EngineeringRole.AI_ML_SYSTEMS_ENGINEER
    recommendation, flags = derive_recommendation(
        role, 0.88, _passing_scores(role), turns_completed=8
    )
    assert recommendation is Recommendation.ADVANCE
    assert not flags


def test_incomplete_beats_rejection_when_evidence_is_thin() -> None:
    """A dropped call must never be recorded as a rejection."""
    role = EngineeringRole.AI_ML_SYSTEMS_ENGINEER
    recommendation, _ = derive_recommendation(role, 0.1, {}, turns_completed=1)
    assert recommendation is Recommendation.INCOMPLETE


def test_elimination_gate_overrides_a_high_aggregate() -> None:
    role = EngineeringRole.AI_ML_SYSTEMS_ENGINEER
    scores = _passing_scores(role)
    scores["inference_kernels"] = 0.05  # the gated competency
    recommendation, flags = derive_recommendation(role, 0.80, scores, turns_completed=8)
    assert flags, "the elimination gate did not fire"
    assert flags[0].competency_key == "inference_kernels"
    assert recommendation is Recommendation.HOLD_FOR_HUMAN_REVIEW, (
        "a strong candidate who tripped one gate is a human's call, not a machine's"
    )


def test_weak_candidate_who_trips_a_gate_is_not_advanced() -> None:
    role = EngineeringRole.AI_ML_SYSTEMS_ENGINEER
    scores = {c.key: 0.1 for c in rubric_for(role).competencies}
    recommendation, flags = derive_recommendation(role, 0.12, scores, turns_completed=8)
    assert flags
    assert recommendation is Recommendation.DO_NOT_ADVANCE


def test_borderline_candidate_advances_with_reservations() -> None:
    role = EngineeringRole.AI_ML_SYSTEMS_ENGINEER
    threshold = rubric_for(role).advance_threshold
    scores = _passing_scores(role, 0.6)
    recommendation, _ = derive_recommendation(role, threshold * 0.8, scores, turns_completed=8)
    assert recommendation is Recommendation.ADVANCE_WITH_RESERVATIONS


@pytest.mark.parametrize("role", list(EngineeringRole))
def test_recommendation_is_deterministic(role: EngineeringRole) -> None:
    scores = _passing_scores(role, 0.7)
    outcomes = {derive_recommendation(role, 0.7, scores, turns_completed=6)[0] for _ in range(20)}
    assert len(outcomes) == 1


# =============================================================================
# Dossier shape and privacy
# =============================================================================


def test_evaluation_payload_matches_the_stored_schema() -> None:
    role = EngineeringRole.AI_ML_SYSTEMS_ENGINEER
    evaluation = CandidateEvaluation(
        candidate_id="11111111-2222-3333-4444-555555555555",
        sanitized_name="Candidate_491",
        session_id="s-1",
        target_role=role,
        rubric_fit_index=0.89,
        competency_scores=[
            CompetencyScore(
                key="distributed_training", score=0.92, source=EvidenceSource.LIVE_INTERVIEW
            ),
            CompetencyScore(key="inference_kernels", score=0.88),
        ],
        flagged_limitations=["Lacks deep C++ kernel authoring experience"],
        recommendation=Recommendation.ADVANCE,
    )
    payload = evaluation.to_payload()
    assert set(payload) == {
        "candidate_id",
        "sanitized_name",
        "target_role",
        "competency_scores",
        "rubric_fit_index",
        "flagged_limitations",
        "interview_timestamp",
    }
    assert payload["target_role"] == role.value
    assert payload["competency_scores"]["distributed_training"] == 0.92
    assert isinstance(payload["interview_timestamp"], int)


def test_evaluation_rationales_are_scrubbed() -> None:
    score = CompetencyScore(
        key="inference_kernels",
        score=0.9,
        rationale="Candidate said to call 9876543210 for references.",
    )
    assert "9876543210" not in score.rationale
    assert "<PHONE_REDACTED>" in score.rationale


def test_flagged_limitations_are_scrubbed() -> None:
    evaluation = CandidateEvaluation(
        candidate_id="c",
        sanitized_name="Candidate_001",
        session_id="s",
        target_role=EngineeringRole.AI_ML_SYSTEMS_ENGINEER,
        rubric_fit_index=0.5,
        flagged_limitations=["Reachable only at priya@example.com"],
    )
    assert "priya@example.com" not in evaluation.flagged_limitations[0]


def test_candidate_profile_scrubs_on_ingest() -> None:
    profile = CandidateProfile.from_resume(
        "Priya | priya@example.com | +91 98765 43210 | Aadhaar 3412 7856 9034 | FSDP, vLLM"
    )
    assert "priya@example.com" not in profile.resume_text
    assert "3412" not in profile.resume_text
    assert profile.redaction_counts == {"aadhaar": 1, "email": 1, "phone": 1}
    assert "FSDP" in profile.resume_text, "technical evidence must survive"


def test_pseudonym_is_stable_and_process_independent() -> None:
    assert pseudonym_for("abc") == pseudonym_for("abc")
    assert pseudonym_for("abc") != pseudonym_for("abd")
    assert pseudonym_for("abc").startswith("Candidate_")


def test_session_evidence_excludes_agent_speech() -> None:
    """Otherwise the interviewer's own vocabulary inflates the candidate's score."""
    from app.schemas.candidate import Speaker, TranscriptTurn

    profile = CandidateProfile.from_resume("Backend engineer.")
    session = ScreeningSession(candidate=profile, role=EngineeringRole.DISTRIBUTED_BACKEND_ENGINEER)
    session.record(TranscriptTurn.sanitised(Speaker.AGENT, "Tell me about Raft and Paxos.", 0.0))
    session.record(TranscriptTurn.sanitised(Speaker.CANDIDATE, "I used an outbox pattern.", 1.0))

    evidence = session.candidate_evidence()
    assert "outbox" in evidence
    assert "Paxos" not in evidence and "Raft" not in evidence


def test_role_fit_round_trips_from_a_match() -> None:
    match = best_role(ROLE_PROFILES[EngineeringRole.MOBILE_CORE_ENGINEER])
    fit = RoleFit.from_match(match)
    assert fit.role is EngineeringRole.MOBILE_CORE_ENGINEER
    assert 0.0 <= fit.fit_index <= 1.0
    assert set(fit.competency_scores) == {c.key for c in rubric_for(fit.role).competencies}
