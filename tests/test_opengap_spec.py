"""OpenGAP registry compliance.

The registry validator parses these three files strictly and rejects the whole
submission on any deviation, so these tests assert the letter of the
specification rather than the spirit of it. They are also a regression guard on
a subtler problem: SOUL.md is loaded at runtime to build the system prompt, so
an edit that improves the prose but breaks a heading silently changes how the
agent introduces itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_YAML = ROOT / "agent.yaml"
SOUL_MD = ROOT / "SOUL.md"
EXPLAINABILITY_MD = ROOT / "EXPLAINABILITY.md"

REQUIRED_AGENT_KEYS = ["spec_version", "name", "version", "description"]
REQUIRED_SOUL_HEADINGS = ["# Identity", "# Behavior", "# Boundaries"]
REQUIRED_EXPLAIN_HEADINGS = [
    "## Decision Reasoning",
    "## Data Inputs",
    "## Known Limitations",
]

#: A sentence ends at ., ! or ? followed by whitespace or end-of-text.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def count_sentences(text: str) -> int:
    return len(_SENTENCE_END.findall(text.strip()))


# =============================================================================
# agent.yaml
# =============================================================================


@pytest.fixture(scope="module")
def agent_manifest() -> dict[str, object]:
    return yaml.safe_load(AGENT_YAML.read_text(encoding="utf-8"))


def test_agent_yaml_exists() -> None:
    assert AGENT_YAML.is_file(), "agent.yaml must sit in the repository root"


def test_agent_yaml_has_exactly_four_keys(agent_manifest: dict[str, object]) -> None:
    assert list(agent_manifest) == REQUIRED_AGENT_KEYS


def test_agent_yaml_values_are_all_scalar_strings(agent_manifest: dict[str, object]) -> None:
    for key, value in agent_manifest.items():
        assert isinstance(value, str), f"{key} must be a scalar string, got {type(value).__name__}"


def test_agent_yaml_has_no_nested_structures(agent_manifest: dict[str, object]) -> None:
    for key, value in agent_manifest.items():
        assert not isinstance(value, (dict, list, tuple, set)), f"{key} must not be a collection"


def test_agent_yaml_field_values(agent_manifest: dict[str, object]) -> None:
    assert agent_manifest["spec_version"] == "0.1.0"
    assert agent_manifest["name"] == "hr-talent-evaluator"
    assert agent_manifest["version"] == "1.0.0"
    assert agent_manifest["description"].startswith("Autonomous voice-enabled HR screening agent")


def test_agent_version_matches_package() -> None:
    from app import __version__

    manifest = yaml.safe_load(AGENT_YAML.read_text(encoding="utf-8"))
    assert manifest["version"] == __version__, (
        "agent.yaml and app.__version__ have drifted; the registry would publish "
        "a version the code does not claim."
    )


# =============================================================================
# SOUL.md
# =============================================================================


@pytest.fixture(scope="module")
def soul_text() -> str:
    return SOUL_MD.read_text(encoding="utf-8")


def test_soul_has_required_headings(soul_text: str) -> None:
    headings = [line.strip() for line in soul_text.splitlines() if line.startswith("# ")]
    assert headings == REQUIRED_SOUL_HEADINGS


def test_soul_sections_have_prose(soul_text: str) -> None:
    sections = _split_sections(soul_text, level="# ")
    for heading, body in sections.items():
        assert len(body.split()) >= 25, f"{heading} needs substantive explanatory prose"
        assert count_sentences(body) >= 2, f"{heading} needs at least two sentences"


def test_soul_boundaries_forbid_offers_and_raw_pii(soul_text: str) -> None:
    boundaries = _split_sections(soul_text, level="# ")["# Boundaries"].lower()
    assert "never offer employment" in boundaries
    assert "personally identifiable information" in boundaries


def test_soul_declares_cryptographic_scrubbing(soul_text: str) -> None:
    """The declared boundary must name the mechanism, not just the intent."""
    boundaries = _split_sections(soul_text, level="# ")["# Boundaries"].lower()
    for claim in ("cryptographically scrub", "memory persistence", "telemetry", "zero-leak"):
        assert claim in boundaries, f"SOUL.md Boundaries no longer states: {claim}"


def test_soul_cryptographic_claim_is_backed_by_code(soul_text: str) -> None:
    """SOUL.md is the published identity AND is loaded into the system prompt.

    A cryptographic guarantee asserted there and absent from the code would be
    exactly the drift this suite exists to catch, so the claim is checked
    against the implementation rather than taken on trust.
    """
    from app.security.pii_scrubber import fingerprint, scrub

    boundaries = _split_sections(soul_text, level="# ")["# Boundaries"].lower()
    assert "blake2b" in boundaries

    finding = scrub("Reach me on +91 98765 43210").findings[0]
    assert finding.fingerprint, "no fingerprint was produced"
    assert finding.fingerprint == fingerprint("+91 98765 43210")
    assert "98765" not in finding.fingerprint
    assert "one-way" in boundaries and "non-reversible" in boundaries


def test_soul_is_loadable_by_the_prompt_builder(soul_text: str) -> None:
    from app.agent.conversation_prompts import load_soul

    assert load_soul() == soul_text.strip()


# =============================================================================
# EXPLAINABILITY.md
# =============================================================================


@pytest.fixture(scope="module")
def explain_text() -> str:
    return EXPLAINABILITY_MD.read_text(encoding="utf-8")


def test_explainability_has_exactly_three_headings(explain_text: str) -> None:
    headings = [line.strip() for line in explain_text.splitlines() if line.startswith("## ")]
    assert headings == REQUIRED_EXPLAIN_HEADINGS


def test_explainability_sections_have_exactly_two_sentences(explain_text: str) -> None:
    sections = _split_sections(explain_text, level="## ")
    assert len(sections) == 3
    for heading, body in sections.items():
        assert count_sentences(body) == 2, (
            f"{heading} must contain exactly two sentences, found {count_sentences(body)}"
        )


def test_explainability_sections_are_non_trivial(explain_text: str) -> None:
    for heading, body in _split_sections(explain_text, level="## ").items():
        assert len(body.split()) >= 20, f"{heading} is too thin to be an explanation"


def test_explainability_describes_the_real_pipeline(explain_text: str) -> None:
    """Guards against the document drifting away from the implementation."""
    lowered = explain_text.lower()
    for term in ("livekit", "vad", "competenc", "schema", "audio"):
        assert term in lowered, f"EXPLAINABILITY.md no longer mentions {term}"


# =============================================================================
# helpers
# =============================================================================


def _split_sections(text: str, level: str) -> dict[str, str]:
    """Split markdown into {heading: body} for headings at exactly ``level``."""
    sections: dict[str, str] = {}
    heading: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith(level) and not line.startswith(level + "#"):
            if heading is not None:
                sections[heading] = "\n".join(body).strip()
            heading = line.strip()
            body = []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        sections[heading] = "\n".join(body).strip()
    return sections
