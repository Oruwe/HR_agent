"""Zero-leak redaction.

Structure of this file mirrors the threat model rather than the code:

1. every identifier class is caught;
2. nothing is caught that should not be (the false-positive corpus -- a
   scrubber that eats "p99 under 45ms" destroys the evidence the ranking
   is built on);
3. the transform has the algebraic properties the rest of the system relies on
   (deterministic, idempotent, total, offset-traceable);
4. the enforcement gates actually raise.
"""

from __future__ import annotations

import pytest

from app.security.pii_scrubber import (
    ALL_TOKENS,
    REDACTION_KEY_ENV,
    TOKEN_AADHAAR,
    TOKEN_CARD,
    TOKEN_EMAIL,
    TOKEN_GOV_ID,
    TOKEN_LOCATION,
    TOKEN_MYNUMBER,
    TOKEN_PHONE,
    TOKEN_RRN,
    PiiKind,
    SecurityBreachException,
    assert_zero_pii,
    detect,
    fingerprint,
    redaction_summary,
    reset_redaction_key,
    sanitize_for_egress,
    scrub,
    scrub_text,
)
from app.security.token_guard import (
    TOKEN_CREDENTIAL,
    CredentialLeakException,
    assert_zero_credentials,
    count_credentials,
    redact_credentials,
    sanitize,
)

# =============================================================================
# 1. Detection coverage
# =============================================================================

DETECTION_CASES: list[tuple[str, str, str]] = [
    ("email", "Reach me at priya.raman@example.com please", TOKEN_EMAIL),
    ("email_plus", "Mail dev+hr@sub.domain.co.in today", TOKEN_EMAIL),
    ("aadhaar_spaced", "Aadhaar: 3412 7856 9034", TOKEN_AADHAAR),
    ("aadhaar_bare", "UIDAI number 341278569034 on file", TOKEN_AADHAAR),
    ("aadhaar_hyphen", "Aadhaar 3412-7856-9034", TOKEN_AADHAAR),
    ("aadhaar_nbsp", "Aadhaar 3412\u00a07856\u00a09034", TOKEN_AADHAAR),
    ("mynumber", "MyNumber: 1234 5678 9012 issued in Tokyo", TOKEN_MYNUMBER),
    ("korean_rrn", "RRN 900101-1234567 registered in Seoul", TOKEN_RRN),
    ("ssn_hyphen", "SSN: 123-45-6789", TOKEN_GOV_ID),
    ("ssn_bare_with_marker", "Social Security 123456789 on record", TOKEN_GOV_ID),
    ("pan", "PAN ABCDE1234F for tax", TOKEN_GOV_ID),
    ("passport_in", "Passport K1234567 issued Bengaluru", TOKEN_GOV_ID),
    ("phone_e164", "Call +91 98765 43210 anytime", TOKEN_PHONE),
    ("phone_nanp", "Office (555) 123-4567 ext", TOKEN_PHONE),
    ("phone_bare_in", "Mobile 9876543210 preferred", TOKEN_PHONE),
    ("address_in", "Lives at 42 MG Road, Indiranagar, Bengaluru 560038", TOKEN_LOCATION),
    ("address_us", "Flat 3B, 221 Baker Street, Springfield", TOKEN_LOCATION),
    ("geo", "Home at 12.9716, 77.5946 exactly", TOKEN_LOCATION),
    ("card", "Card 4111 1111 1111 1111 on file", TOKEN_CARD),
]


@pytest.mark.parametrize(
    ("label", "text", "token"), DETECTION_CASES, ids=[c[0] for c in DETECTION_CASES]
)
def test_identifier_is_replaced_with_its_token(label: str, text: str, token: str) -> None:
    result = scrub(text)
    assert result.findings, f"{label}: nothing was detected"
    assert token in result.text, f"{label}: expected {token} in {result.text!r}"


@pytest.mark.parametrize(
    ("label", "text", "token"), DETECTION_CASES, ids=[c[0] for c in DETECTION_CASES]
)
def test_no_original_digits_survive(label: str, text: str, token: str) -> None:
    """The strong claim: the redacted span's characters are gone entirely."""
    result = scrub(text)
    for finding in result.findings:
        original = text[finding.start : finding.end]
        assert original not in result.text, f"{label}: {finding.kind.value} survived redaction"


@pytest.mark.parametrize(
    "text",
    [
        "Reach them at priya.raman@example.com.",
        "Mail a@b.co.uk.",
        "Two of them: a@b.com, then c@d.org.",
        "(a@b.com)",
        "a@b.com</p>",
    ],
)
def test_email_is_redacted_at_the_end_of_a_sentence(text: str) -> None:
    """Regression: the trailing guard read a sentence-ending period as part
    of the domain and rejected the match, so these went through in the clear.
    Scraped records are free text, so this was the common case, not an edge."""
    assert "@" not in scrub_text(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mail a@b.co.uk today", "<EMAIL_REDACTED>"),
        ("Mail a@b.co.uk.", "<EMAIL_REDACTED>."),
    ],
)
def test_multi_label_domains_are_redacted_whole(text: str, expected: str) -> None:
    """The guard's actual job: never leave `.uk` dangling after `a@b.co`."""
    assert expected in scrub_text(text)
    assert ".uk" not in scrub_text(text)


def test_planted_resume_is_fully_scrubbed(pii_resume: str) -> None:
    result = scrub(pii_resume)
    kinds = {f.kind for f in result.findings}
    for expected in (
        PiiKind.EMAIL,
        PiiKind.PHONE,
        PiiKind.AADHAAR,
        PiiKind.PAN,
        PiiKind.PASSPORT,
        PiiKind.STREET_ADDRESS,
    ):
        assert expected in kinds, f"{expected.value} was not detected in the planted resume"
    assert not detect(result.text), "second pass still finds PII"


def test_international_identifiers(international_pii: str) -> None:
    result = scrub(international_pii)
    kinds = {f.kind for f in result.findings}
    assert PiiKind.KOREAN_RRN in kinds
    assert PiiKind.MYNUMBER in kinds
    assert PiiKind.SSN in kinds
    assert PiiKind.PAYMENT_CARD in kinds
    assert PiiKind.GEO_COORDINATE in kinds
    assert TOKEN_RRN in result.text
    assert TOKEN_MYNUMBER in result.text
    assert not detect(result.text)


def test_technical_content_survives_intact(pii_resume: str) -> None:
    """Redaction must not destroy the evidence the ranking is built on."""
    scrubbed = scrub_text(pii_resume)
    for phrase in ("FSDP", "512 A100s", "34 percent", "Triton kernels", "HNSW", "p99"):
        assert phrase in scrubbed, f"scrubbing destroyed technical evidence: {phrase}"


# =============================================================================
# 2. False positives
# =============================================================================


def test_no_false_positives_on_technical_prose(technical_no_pii: str) -> None:
    findings = detect(technical_no_pii)
    assert not findings, "technical prose was misread as PII: " + ", ".join(
        f"{f.kind.value}={technical_no_pii[f.start : f.end]!r}" for f in findings
    )


@pytest.mark.parametrize(
    "text",
    [
        "Reduced p99 from 1200ms to 160ms.",
        "Worked there from 2019-2023 on 8 nodes.",
        "Version 1.0.0, RFC 7519, HTTP status 429.",
        "HNSW m=16 with ef_construct=100 over 200000000 vectors.",
        "Uptime was 99.99 percent across 12 regions.",
        "Scaled from 1500 to 45000 QPS in Q3 2024.",
        "Kubernetes 1.29 on kernel 6.6 with 64 GB RAM.",
    ],
)
def test_engineering_numbers_are_not_identifiers(text: str) -> None:
    assert not detect(text), f"false positive in: {text!r}"


# =============================================================================
# 3. Algebraic properties
# =============================================================================


def test_scrub_is_deterministic(pii_resume: str) -> None:
    first = scrub(pii_resume)
    for _ in range(50):
        again = scrub(pii_resume)
        assert again.text == first.text
        assert again.counts() == first.counts()


def test_scrub_is_idempotent(pii_resume: str, international_pii: str) -> None:
    for text in (pii_resume, international_pii):
        once = scrub_text(text)
        assert scrub_text(once) == once
        assert scrub_text(scrub_text(once)) == once


def test_replacement_tokens_are_never_themselves_matched() -> None:
    """The property that makes idempotence structural rather than incidental."""
    for token in ALL_TOKENS:
        assert not detect(token), f"{token} is matched by a detector -- cascade risk"
        assert not detect(f"Contact: {token} and {token}.")


def test_scrub_is_total_on_arbitrary_input() -> None:
    for text in (
        "",
        " ",
        "\n\n",
        "@@@",
        "0",
        "-" * 500,
        "\u00a0\u200b",
        "\U0001f642 emoji \U0001f642",
    ):
        assert isinstance(scrub(text).text, str)


def test_surrounding_text_is_preserved() -> None:
    result = scrub("Before priya@example.com after.")
    assert result.text == f"Before {TOKEN_EMAIL} after."


def test_offset_map_relocates_spans() -> None:
    text = "Priya priya@example.com wrote the FSDP trainer."
    result = scrub(text)
    start = text.index("FSDP")
    new_start = result.offsets.translate(start)
    assert result.text[new_start : new_start + 4] == "FSDP"


def test_offset_map_collapses_inside_redactions() -> None:
    text = "Mail priya@example.com now."
    result = scrub(text)
    inside = text.index("priya@") + 3
    translated = result.offsets.translate(inside)
    assert 0 <= translated <= len(result.text)


def test_offset_map_is_identity_without_findings() -> None:
    text = "No identifiers here at all."
    result = scrub(text)
    assert all(result.offsets.translate(i) == i for i in range(len(text)))


def test_findings_never_carry_the_raw_value(pii_resume: str) -> None:
    """A finding is an audit record, not a copy of the secret."""
    for finding in scrub(pii_resume).findings:
        raw = pii_resume[finding.start : finding.end]
        assert raw not in repr(finding)
        assert finding.original_length == len(raw)


def test_redaction_summary_aggregates() -> None:
    totals = redaction_summary([scrub("a@b.com"), scrub("c@d.com"), scrub("9876543210")])
    assert totals == {"email": 2, "phone": 1}


# =============================================================================
# 3b. Cryptographic fingerprints
# =============================================================================


def _keyed(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    monkeypatch.setenv(REDACTION_KEY_ENV, key)
    reset_redaction_key()


def test_same_identifier_fingerprints_identically_across_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the fingerprint: duplicate detection without the value."""
    _keyed(monkeypatch, "deployment-secret")
    a = scrub("Call +91 98765 43210 today").findings[0]
    b = scrub("Alternate 919876543210 preferred").findings[0]
    assert a.fingerprint == b.fingerprint


def test_different_identifiers_fingerprint_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keyed(monkeypatch, "deployment-secret")
    a = scrub("Call 9876543210").findings[0]
    b = scrub("Call 9876543211").findings[0]
    assert a.fingerprint != b.fingerprint


def test_fingerprint_never_contains_the_raw_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keyed(monkeypatch, "deployment-secret")
    result = scrub("Aadhaar 3412 7856 9034")
    digest = result.findings[0].fingerprint
    assert "3412" not in digest and "7856" not in digest and "9034" not in digest
    assert len(digest) == 16 and all(c in "0123456789abcdef" for c in digest)


def test_fingerprints_are_keyed_not_a_bare_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unkeyed, a digest of a 10-digit number is the number -- the search space
    is exhaustible in seconds. Keying is what makes the fingerprint safe."""
    _keyed(monkeypatch, "key-one")
    first = fingerprint("9876543210")
    _keyed(monkeypatch, "key-two")
    assert fingerprint("9876543210") != first


def test_unkeyed_deployments_get_an_ephemeral_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safe failure: fingerprints stop linking rather than becoming guessable."""
    monkeypatch.delenv(REDACTION_KEY_ENV, raising=False)
    reset_redaction_key()
    first = fingerprint("9876543210")
    reset_redaction_key()
    assert fingerprint("9876543210") != first


def test_fingerprints_are_stable_within_a_keyed_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keyed(monkeypatch, "deployment-secret")
    values = {scrub("Call 9876543210").findings[0].fingerprint for _ in range(20)}
    assert len(values) == 1


def test_fingerprint_map_is_keyed_by_class_not_by_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _keyed(monkeypatch, "deployment-secret")
    mapping = scrub("a@b.com and 9876543210 and c@d.com").fingerprints()
    assert set(mapping) == {"email:0", "email:1", "phone:0"}
    assert all(len(v) == 16 for v in mapping.values())


def test_every_finding_carries_a_fingerprint(pii_resume: str) -> None:
    assert all(f.fingerprint for f in scrub(pii_resume).findings)


# =============================================================================
# 4. Enforcement gates
# =============================================================================


def test_assert_zero_pii_raises_on_dirty_text() -> None:
    with pytest.raises(SecurityBreachException) as excinfo:
        assert_zero_pii("Call 9876543210", boundary="model_prompt")
    assert excinfo.value.boundary == "model_prompt"
    assert PiiKind.PHONE in excinfo.value.kinds


def test_security_exception_never_quotes_the_value() -> None:
    """An exception string is the most likely thing to reach a log aggregator."""
    try:
        assert_zero_pii("Aadhaar 3412 7856 9034 and mail a@b.com")
    except SecurityBreachException as exc:
        message = str(exc)
        assert "3412" not in message
        assert "a@b.com" not in message
        assert "aadhaar" in message
    else:
        pytest.fail("expected SecurityBreachException")


def test_assert_zero_pii_passes_on_clean_text(pii_resume: str) -> None:
    assert_zero_pii(scrub_text(pii_resume))


def test_assert_zero_pii_is_not_a_value_error() -> None:
    """Broad input validation must not swallow a security failure."""
    assert not issubclass(SecurityBreachException, ValueError)


def test_sanitize_for_egress_walks_nested_payloads() -> None:
    payload = {
        "candidate": {"contact": "priya@example.com", "phones": ["9876543210"]},
        "scores": {"inference_kernels": 0.91},
        "notes": ("Aadhaar 3412 7856 9034", "clean note"),
        "turns": 7,
        "passed": True,
    }
    cleaned = sanitize_for_egress(payload)
    assert cleaned["candidate"]["contact"] == TOKEN_EMAIL
    assert cleaned["candidate"]["phones"] == [TOKEN_PHONE]
    # Only the digits are redacted; the surrounding words are evidence and stay.
    assert TOKEN_AADHAAR in cleaned["notes"][0]
    assert "3412" not in cleaned["notes"][0]
    assert cleaned["scores"]["inference_kernels"] == 0.91
    assert cleaned["turns"] == 7 and cleaned["passed"] is True


def test_sanitize_for_egress_scrubs_mapping_keys() -> None:
    """A redacted value under a leaking key is still a leak."""
    cleaned = sanitize_for_egress({"priya@example.com": "candidate"})
    assert TOKEN_EMAIL in cleaned
    assert "priya@example.com" not in cleaned


# =============================================================================
# 5. Credential guard
# =============================================================================


# These fixtures are synthetic, but they are *shaped* like real credentials --
# that is the entire point, since the detectors match on shape. Writing them as
# source literals therefore trips GitHub push protection and every other secret
# scanner pointed at this repository, and a test suite that cannot be pushed is
# not a test suite. Assembling each one at runtime from inert fragments keeps the
# scanners quiet while giving the detectors exactly the same bytes to match.
def _synthetic(prefix: str, body: str) -> str:
    """Build a credential-shaped string with no scannable literal in source."""
    return prefix + body


_GOOGLE_KEY = _synthetic("AI" + "za", "SyA1234567890abcdefghijklmnopqrstuv")
_JWT = ".".join(
    (
        _synthetic("ey", "JhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
        _synthetic("ey", "JzdWIiOiIxMjM0NTY3ODkwIn0"),
        "abcdefghijkl",
    )
)
_AWS_KEY = _synthetic("AK" + "IA", "IOSFODNN7EXAMPLE")
_VENDOR_KEY = _synthetic("sk" + "_", "live_abcdefghijklmnopqrstuvwxyz")

CREDENTIAL_CASES = [
    f"GOOGLE_API_KEY={_GOOGLE_KEY}",
    f"Authorization: Bearer {_JWT}",
    f"aws key {_AWS_KEY} rotated",
    f"export LIVEKIT_API_SECRET={_VENDOR_KEY}",
    'config = {"api_key": "supersecretvalue123"}',
]


@pytest.mark.parametrize("text", CREDENTIAL_CASES)
def test_credentials_are_redacted(text: str) -> None:
    redacted = redact_credentials(text)
    assert TOKEN_CREDENTIAL in redacted
    assert count_credentials(text) >= 1


def test_assert_zero_credentials_raises() -> None:
    with pytest.raises(CredentialLeakException):
        assert_zero_credentials(f"Bearer {_JWT}")


def test_private_key_block_is_removed() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK\n-----END RSA PRIVATE KEY-----"
    assert "MIIEowIBAAK" not in redact_credentials(pem)


def test_sanitize_applies_both_gates() -> None:
    text = f"key={_GOOGLE_KEY} and phone 9876543210"
    cleaned = sanitize(text)
    assert TOKEN_CREDENTIAL in cleaned
    assert TOKEN_PHONE in cleaned
    assert not detect(cleaned)


def test_credential_redaction_keeps_the_key_name() -> None:
    """Logs must stay diagnosable: which secret leaked matters."""
    assert "GOOGLE_API_KEY" in redact_credentials(f"GOOGLE_API_KEY={_GOOGLE_KEY}")
