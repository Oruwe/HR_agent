"""Interview stage machine and probe selection.

The flow exists so that the *shape* of a screening is deterministic even though
the words are not. Two candidates for the same role get the same competencies
probed, in the same weight order, with the same coverage bar -- which is what
makes their scores comparable. The model chooses phrasing; it does not choose
what gets asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.schemas.roles import Competency, EngineeringRole, rubric_for


class InterviewStage(StrEnum):
    GREETING = "greeting"
    COMPETENCY_PROBE = "competency_probe"
    TRADEOFF_DRILLDOWN = "tradeoff_drilldown"
    CANDIDATE_QUESTIONS = "candidate_questions"
    CLOSING = "closing"
    TERMINATED = "terminated"


#: Evidence level at which a competency counts as established well enough to
#: move on. Below this we keep probing; a half-answered competency is worse than
#: an unasked one because it looks like coverage in the dossier.
COVERAGE_THRESHOLD: float = 0.55


@dataclass(slots=True)
class InterviewFlow:
    """Tracks coverage and decides what to ask next.

    ``max_turns`` is a real constraint, not a safety valve. A screening call has
    a budget of the candidate's attention, and an agent that probes indefinitely
    because no competency ever quite clears the bar is a worse experience than
    one that stops and says so.
    """

    role: EngineeringRole
    max_turns: int = 12
    stage: InterviewStage = InterviewStage.GREETING
    turns: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)
    termination_reason: str = ""
    #: The fast live path (app.config.Settings.fast_path) strips tool schemas
    #: from the model call, so record_candidate_competency -- the only caller
    #: of `record()` -- never fires there and `scores` never populates. Without
    #: this flag, `uncovered()` would never shrink: the interviewer would probe
    #: the same highest-weight competency forever and the stage machine could
    #: never leave COMPETENCY_PROBE. On the fast path, "asked" is therefore
    #: treated as "handled enough to move past" for probe selection; off the
    #: fast path, a live score from the model is still required, so a
    #: half-answered competency keeps getting re-probed as originally designed.
    fast_path: bool = False

    @property
    def rubric(self):
        return rubric_for(self.role)

    @property
    def finished(self) -> bool:
        return self.stage in (InterviewStage.CLOSING, InterviewStage.TERMINATED)

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.turns)

    def covered(self) -> list[str]:
        return sorted(k for k, v in self.scores.items() if v >= COVERAGE_THRESHOLD)

    def uncovered(self) -> list[Competency]:
        """Competencies still needing evidence, heaviest first.

        The tie-break on ``key`` is what makes two runs of the same interview
        ask questions in the same order -- without it, equal-weight
        competencies would be ordered by dict insertion.
        """
        done = set(self.covered())
        if self.fast_path:
            done |= set(self.asked)
        pending = [c for c in self.rubric.competencies if c.key not in done]
        return sorted(pending, key=lambda c: (-c.weight, c.key))

    def coverage_ratio(self) -> float:
        total = self.rubric.total_weight
        if total <= 0:
            return 0.0
        achieved = sum(
            c.weight
            for c in self.rubric.competencies
            if self.scores.get(c.key, 0.0) >= COVERAGE_THRESHOLD
        )
        return achieved / total

    def record(self, key: str, score: float) -> None:
        """Record evidence, keeping the strongest observation for a competency.

        Max rather than last-write-wins: a candidate who explains something well
        and then fumbles a follow-up has still demonstrated it. Averaging would
        penalise them for the interviewer's choice to keep digging.
        """
        clamped = max(0.0, min(1.0, score))
        self.scores[key] = max(self.scores.get(key, 0.0), clamped)

    def next_probe(self) -> Competency | None:
        pending = self.uncovered()
        return pending[0] if pending else None

    def advance(self) -> InterviewStage:
        """Move the stage machine one step. Called once per completed turn."""
        self.turns += 1

        if self.stage is InterviewStage.GREETING:
            self.stage = InterviewStage.COMPETENCY_PROBE
            return self.stage

        if self.turns >= self.max_turns:
            self.stage = InterviewStage.CLOSING
            return self.stage

        if self.stage is InterviewStage.COMPETENCY_PROBE:
            if not self.uncovered():
                self.stage = InterviewStage.TRADEOFF_DRILLDOWN
            return self.stage

        if self.stage is InterviewStage.TRADEOFF_DRILLDOWN:
            self.stage = InterviewStage.CANDIDATE_QUESTIONS
            return self.stage

        if self.stage is InterviewStage.CANDIDATE_QUESTIONS:
            self.stage = InterviewStage.CLOSING
            return self.stage

        return self.stage

    def terminate(self, reason: str) -> None:
        self.stage = InterviewStage.TERMINATED
        self.termination_reason = reason

    def switch_role(self, role: EngineeringRole, reason: str = "") -> None:
        """Re-route to a different rubric mid-interview.

        Scores are dropped deliberately. They were measured against a different
        rubric's competency keys, and carrying them across would silently credit
        the candidate for demonstrating something nobody asked about.
        """
        self.role = role
        self.scores.clear()
        self.asked.clear()
        self.stage = InterviewStage.COMPETENCY_PROBE

    def directive(self) -> str:
        """The per-turn instruction handed to the model alongside the prompt."""
        if self.stage is InterviewStage.GREETING:
            return "Open the interview and invite them to describe a demanding system."
        if self.stage is InterviewStage.COMPETENCY_PROBE:
            probe = self.next_probe()
            if probe is None:
                return "Move to a trade-off drill-down on what they described."
            self.asked.append(probe.key)
            return (
                f"Probe '{probe.key}'. Reference question, rephrase in your own "
                f"words and in reaction to what they just said: {probe.probe}"
            )
        if self.stage is InterviewStage.TRADEOFF_DRILLDOWN:
            return (
                "Push on one architectural trade-off they made. Ask what they "
                "gave up and what the number was that decided it."
            )
        if self.stage is InterviewStage.CANDIDATE_QUESTIONS:
            return "Invite one question about the role or the engineering team."
        return "Close the interview warmly without promising an outcome."


__all__ = ["COVERAGE_THRESHOLD", "InterviewFlow", "InterviewStage"]
