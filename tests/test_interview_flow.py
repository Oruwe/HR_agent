"""The interview stage machine and probe selection.

This module previously had no direct test coverage at all, which is exactly
how a severe production bug shipped undetected: the fast live path
(`Settings.fast_path`, the production default) strips tool schemas from the
model call, so `record_candidate_competency` -- the only caller of
`InterviewFlow.record()` -- never fires there. Without the `fast_path` guard
in `uncovered()`, `next_probe()` returns the same highest-weight competency
forever, the stage machine can never leave COMPETENCY_PROBE, and a deployed
interviewer asks the same question on every turn for the whole interview.
"""

from __future__ import annotations

from app.agent.interview_flow import COVERAGE_THRESHOLD, InterviewFlow, InterviewStage
from app.schemas.roles import EngineeringRole, rubric_for

ROLE = EngineeringRole.AI_ML_SYSTEMS_ENGINEER
COMPETENCY_COUNT = len(rubric_for(ROLE).competencies)


def _drive_one_full_pass(flow: InterviewFlow) -> list[str]:
    """Simulate the orchestrator's per-turn sequence: directive() then advance().

    Mirrors ScreeningOrchestrator._finish_turn, which calls flow.advance() once
    per completed turn after the directive (and any tool calls) for that turn
    have already been produced.
    """
    probed: list[str] = []
    flow.advance()  # GREETING -> COMPETENCY_PROBE, as the orchestrator does on turn 1
    for _ in range(COMPETENCY_COUNT):
        probe = flow.next_probe()
        assert probe is not None, "ran out of competencies before the loop finished"
        flow.directive()  # appends probe.key to flow.asked, exactly as production does
        probed.append(probe.key)
        flow.advance()
    return probed


def test_fast_path_advances_through_every_competency_without_live_scoring() -> None:
    """The bug this test pins: no tool calls ever arrive on the fast path."""
    flow = InterviewFlow(role=ROLE, fast_path=True)
    probed = _drive_one_full_pass(flow)

    assert len(set(probed)) == COMPETENCY_COUNT, (
        f"the interviewer asked about the same competency more than once: {probed}"
    )
    assert flow.stage is InterviewStage.TRADEOFF_DRILLDOWN, (
        "fast-path interview never left COMPETENCY_PROBE -- it would loop for the rest of the call"
    )


def test_fast_path_never_reprobes_the_same_competency_twice_in_a_row() -> None:
    """Reproduces the exact user-visible symptom: two consecutive identical asks."""
    flow = InterviewFlow(role=ROLE, fast_path=True)
    flow.advance()

    first = flow.next_probe()
    assert first is not None
    flow.directive()

    second = flow.next_probe()
    assert second is not None
    assert second.key != first.key, "the same competency was selected two turns running"


def test_non_fast_path_still_reprobes_an_unscored_competency() -> None:
    """Off the fast path, a live model score is expected every turn, so a
    competency nobody has scored yet must stay eligible -- the original
    "half-answered is worse than unasked" design this flag must not break."""
    flow = InterviewFlow(role=ROLE, fast_path=False)
    flow.advance()

    first = flow.next_probe()
    assert first is not None
    flow.directive()

    second = flow.next_probe()
    assert second is not None
    assert second.key == first.key, "non-fast-path flow stopped re-probing an unscored competency"


def test_fast_path_still_respects_real_coverage_when_it_is_recorded() -> None:
    """A genuinely scored competency is skipped either way -- `asked` is a
    fallback for the missing signal, not a replacement for a real one."""
    flow = InterviewFlow(role=ROLE, fast_path=True)
    flow.advance()

    probe = flow.next_probe()
    assert probe is not None
    flow.record(probe.key, COVERAGE_THRESHOLD)
    flow.directive()

    assert probe.key in flow.covered()
    assert probe.key not in {c.key for c in flow.uncovered()}
