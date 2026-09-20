"""The analyst: what it sends, what it accepts back, and what it refuses.

The parsing tests matter more than they look. A model that wraps its JSON in
a fence or a sentence of preamble is normal behaviour, not an error, and a
parser that gives up on those would leave a pool silently unranked. The
rendering tests matter for the opposite reason: this is the last point where
a record can leak PII, and it is a single function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agent.analyst import (
    VERDICTS,
    Analyst,
    build_pool_context,
    parse_rankings,
    render_candidate,
)
from app.agent.cognition import MockCognition
from app.config import Settings


@dataclass
class _Row:
    id: str
    name: str
    source: dict[str, Any]


def _pool(n: int = 3) -> list[_Row]:
    return [
        _Row(
            id=f"c{i}",
            name=f"Candidate {i}",
            source={"headline": f"Engineer {i}", "detail": "shipped things " * (i + 1)},
        )
        for i in range(n)
    ]


# ---- Parsing ----------------------------------------------------------------


def test_parses_a_clean_ranking_payload() -> None:
    rankings = parse_rankings(
        '{"rankings": [{"id": "c1", "score": 0.8, "verdict": "INTERVIEW", '
        '"rationale": "Owned the ledger migration."}]}'
    )
    assert len(rankings) == 1
    assert rankings[0].candidate_id == "c1"
    assert rankings[0].score == pytest.approx(0.8)
    assert rankings[0].verdict == "INTERVIEW"


def test_parses_json_wrapped_in_a_code_fence() -> None:
    payload = '```json\n{"rankings": [{"id": "c1", "score": 0.5, "verdict": "MAYBE"}]}\n```'
    assert len(parse_rankings(payload)) == 1


def test_parses_json_with_prose_around_it() -> None:
    payload = (
        "Here is my assessment of the pool.\n"
        '{"rankings": [{"id": "c1", "score": 0.5, "verdict": "PASS"}]}\n'
        "Let me know if you want more detail."
    )
    assert parse_rankings(payload)[0].verdict == "PASS"


@pytest.mark.parametrize("payload", ["", "no json here at all", "{not valid json}", "{}"])
def test_unparseable_payloads_yield_no_rankings(payload: str) -> None:
    """Better an unranked pool than a pool scored from garbage."""
    assert parse_rankings(payload) == []


def test_rows_without_an_id_are_dropped() -> None:
    payload = '{"rankings": [{"score": 0.9, "verdict": "INTERVIEW"}, {"id": "c2", "score": 0.1}]}'
    rankings = parse_rankings(payload)
    assert [r.candidate_id for r in rankings] == ["c2"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(2.5, 1.0), (-4, 0.0), ("0.7", 0.7), (None, 0.0), ("not a number", 0.0)],
)
def test_scores_are_clamped_into_range(raw: Any, expected: float) -> None:
    payload = {"rankings": [{"id": "c1", "score": raw, "verdict": "MAYBE"}]}
    import json

    assert parse_rankings(json.dumps(payload))[0].score == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("interview", "INTERVIEW"),
        ("Pass", "PASS"),
        ("strong hire", "MAYBE"),
        ("", "MAYBE"),
        (None, "MAYBE"),
    ],
)
def test_verdicts_are_normalised_to_the_ladder(raw: Any, expected: str) -> None:
    import json

    payload = json.dumps({"rankings": [{"id": "c1", "score": 0.5, "verdict": raw}]})
    verdict = parse_rankings(payload)[0].verdict
    assert verdict == expected
    assert verdict in VERDICTS


def test_rationale_is_scrubbed_on_the_way_back_in() -> None:
    """A model can echo PII out of a record it was shown."""
    payload = (
        '{"rankings": [{"id": "c1", "score": 0.9, "verdict": "INTERVIEW", '
        '"rationale": "Reach them at priya.raman@example.com."}]}'
    )
    assert "priya.raman@example.com" not in parse_rankings(payload)[0].rationale


# ---- Rendering --------------------------------------------------------------


def test_render_keeps_unknown_keys() -> None:
    """The scraper owns the schema; a renderer that only knows our fields
    would silently drop whatever it had not seen before."""
    rendered = render_candidate("c1", "Ada", {"weird_key": "weird value", "nested": {"a": [1]}})
    assert "weird_key" in rendered
    assert "weird value" in rendered
    assert "nested" in rendered


def test_pool_context_is_scrubbed() -> None:
    rows = [_Row(id="c1", name="Priya", source={"email": "priya.raman@example.com"})]
    context = build_pool_context(rows)
    assert "priya.raman@example.com" not in context


def test_pool_context_redacts_credentials() -> None:
    rows = [_Row(id="c1", name="X", source={"note": "key sk-ant-api03-AAAABBBBCCCCDDDD"})]
    assert "sk-ant-api03-AAAABBBBCCCCDDDD" not in build_pool_context(rows)


def test_empty_pool_says_so_rather_than_rendering_nothing() -> None:
    assert "empty" in build_pool_context([]).lower()


def test_pool_context_respects_the_limit() -> None:
    context = build_pool_context(_pool(10), limit=3)
    assert context.count("### Candidate ") == 3


# ---- The two calls ----------------------------------------------------------


def test_rank_covers_every_candidate_offline(run) -> None:
    """The offline mock is a working provider, not an apology: a deployment
    with no credentials still ranks its pool."""
    analyst = Analyst(Settings(), cognition=MockCognition())
    rankings = run(lambda: analyst.rank(_pool(4)))
    assert {r.candidate_id for r in rankings} == {"c0", "c1", "c2", "c3"}


def test_rank_does_not_flatten_the_pool_to_one_score(run) -> None:
    """Every candidate landing on the same score is the failure mode that
    makes a ranking useless, and it has happened here before."""
    analyst = Analyst(Settings(), cognition=MockCognition())
    scores = {r.score for r in run(lambda: analyst.rank(_pool(5)))}
    assert len(scores) > 1


def test_rank_on_an_empty_pool_makes_no_model_call(run) -> None:
    mock = MockCognition()
    assert run(lambda: Analyst(Settings(), cognition=mock).rank([])) == []
    assert mock.calls == 0


def test_rank_returns_nothing_when_the_model_is_unparseable(run) -> None:
    analyst = Analyst(Settings(), cognition=MockCognition(responses=["I'd rather not."]))
    assert run(lambda: analyst.rank(_pool(2))) == []


def test_answer_reaches_the_model_with_the_pool_attached(run) -> None:
    mock = MockCognition(responses=["Daniel, on the ledger work."])
    analyst = Analyst(Settings(), cognition=mock)
    answer = run(lambda: analyst.answer("Who is strongest?", _pool(2)))
    assert "Daniel" in answer.reply
    assert mock.calls == 1


def test_answer_reports_which_records_it_read(run) -> None:
    """An answer grounded in 6 of 800 is a different claim from one that saw
    everything, and the manager is shown which they are reading."""
    analyst = Analyst(Settings(), cognition=MockCognition(responses=["ok"]))
    answer = run(lambda: analyst.answer("engineer 1", _pool(5)))
    assert answer.considered
    assert set(answer.considered) <= {f"c{i}" for i in range(5)}
    assert answer.backend


def test_answer_narrows_the_prompt_to_the_retrieved_records(run) -> None:
    """The point of the whole retrieval layer: the model must not be handed
    the entire pool when only a few records are relevant."""
    captured: list[str] = []

    class _Capturing(MockCognition):
        async def stream(self, system_prompt, messages):
            captured.append(system_prompt)
            async for chunk in super().stream(system_prompt, messages):
                yield chunk

    rows = [
        _Row(id="sec", name="Sofia", source={"detail": "auth bypass, eBPF, threat modelling"}),
        _Row(id="mob", name="Ravi", source={"detail": "kotlin swift cold start"}),
    ]
    analyst = Analyst(Settings(), cognition=_Capturing())
    answer = run(lambda: analyst.answer("who found an auth bypass?", rows))

    assert answer.considered == ("sec",)
    assert "Sofia" in captured[0]
    assert "Ravi" not in captured[0]


def test_an_unmatched_question_still_sees_candidates(run) -> None:
    """ "I found nothing" is the wrong answer to "who should I hire?" when
    there are candidates sitting right there."""
    analyst = Analyst(Settings(), cognition=MockCognition(responses=["ok"]))
    answer = run(lambda: analyst.answer("zzzzz quantum basketweaving", _pool(3)))
    assert len(answer.considered) == 3


def test_retrieval_respects_the_configured_ceiling(run) -> None:
    settings = Settings(retrieval_top_k=2)
    analyst = Analyst(settings, cognition=MockCognition(responses=["ok"]))
    answer = run(lambda: analyst.answer("engineer", _pool(9)))
    assert len(answer.considered) <= 2


def test_the_prompt_says_the_pool_was_narrowed(run) -> None:
    """A model told it is holding the whole pool will answer "nobody" with
    confidence it has not earned."""
    captured: list[str] = []

    class _Capturing(MockCognition):
        async def stream(self, system_prompt, messages):
            captured.append(system_prompt)
            async for chunk in super().stream(system_prompt, messages):
                yield chunk

    analyst = Analyst(Settings(retrieval_top_k=2), cognition=_Capturing())
    run(lambda: analyst.answer("engineer", _pool(9)))
    assert "most relevant" in captured[0]
    assert "pool of 9" in captured[0]


def test_answer_scrubs_the_managers_question(run) -> None:
    """The manager can paste a resume into the chat box."""
    captured: list[str] = []

    class _Capturing(MockCognition):
        async def stream(self, system_prompt, messages):
            captured.extend(m.content for m in messages)
            async for chunk in super().stream(system_prompt, messages):
                yield chunk

    analyst = Analyst(Settings(), cognition=_Capturing())
    run(lambda: analyst.answer("Compare to priya.raman@example.com", _pool(1)))
    assert not any("priya.raman@example.com" in c for c in captured)


def test_ranking_still_sees_the_whole_pool(run) -> None:
    """Retrieval narrows *questions*, never the ranking pass. Ranking is
    comparative -- narrowing it would mean scoring candidates against a
    subset of the field and calling the result a ranking."""
    captured: list[str] = []

    class _Capturing(MockCognition):
        async def stream(self, system_prompt, messages):
            captured.extend(m.content for m in messages)
            async for chunk in super().stream(system_prompt, messages):
                yield chunk

    analyst = Analyst(Settings(retrieval_top_k=2), cognition=_Capturing())
    run(lambda: analyst.rank(_pool(7)))
    assert captured[0].count("### Candidate ") == 7


# ---- Handles ----------------------------------------------------------------


def test_prompt_handles_are_not_database_ids() -> None:
    """A UUID rendered into the prompt gets partially redacted by the PII
    scrubber about 5% of the time -- slices of one look like an Aadhaar
    number, a passport or a card. The model echoes the mangled id back, it
    matches no row, and that candidate is silently left unscored."""
    rows = [_Row(id="8f14e45f-ceea-467a-9ba3-2d1c0a5e1f77", name="Ada", source={})]
    context = build_pool_context(rows)
    assert "8f14e45f" not in context
    assert "### Candidate C1 (Ada)" in context


def test_no_uuid_survives_rendering_to_be_redacted(run) -> None:
    """The regression, at the rate it actually occurs: every candidate must
    come back ranked, whatever uuid4 produced."""
    import uuid

    rows = [
        _Row(id=str(uuid.uuid4()), name=f"C{i}", source={"detail": "shipped things " * (i + 1)})
        for i in range(60)
    ]
    analyst = Analyst(Settings(), cognition=MockCognition())
    rankings = run(lambda: analyst.rank(rows))
    assert {r.candidate_id for r in rankings} == {r.id for r in rows}


def test_handles_map_back_to_the_right_row(run) -> None:
    """Off-by-one here would apply C3's verdict to C4 -- the worst possible
    failure for a hiring tool, and an invisible one."""
    payload = (
        '{"rankings": ['
        '{"id": "C1", "score": 0.1, "verdict": "PASS", "rationale": "first"},'
        '{"id": "C3", "score": 0.9, "verdict": "INTERVIEW", "rationale": "third"}]}'
    )
    rows = _pool(3)
    analyst = Analyst(Settings(), cognition=MockCognition(responses=[payload]))
    by_id = {r.candidate_id: r for r in run(lambda: analyst.rank(rows))}
    assert by_id[rows[0].id].rationale == "first"
    assert by_id[rows[2].id].rationale == "third"
    assert rows[1].id not in by_id


def test_a_hallucinated_handle_is_dropped_not_misapplied(run) -> None:
    """Better an unscored candidate than someone else's verdict on them."""
    payload = (
        '{"rankings": [{"id": "C99", "score": 0.9, "verdict": "INTERVIEW", "rationale": "nobody"}]}'
    )
    analyst = Analyst(Settings(), cognition=MockCognition(responses=[payload]))
    assert run(lambda: analyst.rank(_pool(3))) == []
