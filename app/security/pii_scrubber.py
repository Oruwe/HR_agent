"""Deterministic, single-pass PII redaction with offset tracking.

Threat model
------------
Candidate resumes and live interview transcripts routinely carry national ID
numbers, contact details and home addresses. Any of it can reach four egress
boundaries: the LLM provider, the vector store, the telemetry backend, and
disk. The invariant this module enforces is that **none of those boundaries
ever sees an unredacted identifier**, and that the enforcement is deterministic
-- the same input always produces byte-identical output, because a
probabilistic redactor cannot be audited.

Why a single pass
-----------------
The obvious implementation is a cascade of ``re.sub`` calls. It is wrong in two
ways that matter at this scale:

1. **Cascade artifacts.** Pass N can match inside the replacement token emitted
   by pass N-1, or across the seam it left behind, producing corrupted output
   that is hard to reason about and impossible to prove correct.
2. **Lost provenance.** Each ``re.sub`` destroys the mapping between original
   and redacted offsets, so downstream span annotations (competency evidence
   spans, diarisation offsets, Langfuse observation ranges) silently drift.

Instead we scan the *original* text once with every compiled pattern, resolve
overlapping candidate matches by a fixed precedence, and splice a single time.
That makes the transform total, order-independent within a priority class, and
offset-traceable via :class:`OffsetMap`.

Deliberate non-goal: **checksum verification.** Aadhaar carries a Verhoeff
check digit and we could use it to suppress false positives. We do not. The
operating invariant is "never output, persist, or verify identity digits", and
a checksum routine is a verification oracle. We accept a slightly higher false
positive rate in exchange for never computing over the digits at all.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from re import Pattern
from typing import Any, Final

# =============================================================================
# Replacement tokens
# =============================================================================

#: Bracketed national-ID tokens are mandated verbatim by the screening spec;
#: the angle-bracket tokens are the typed generic class. Neither form contains a
#: digit or an ``@``, which is what makes the scrubber idempotent: no pattern in
#: this module can match its own output.
TOKEN_AADHAAR: Final[str] = "[Aadhaar Redacted]"
TOKEN_RRN: Final[str] = "[RRN Omitted]"
TOKEN_MYNUMBER: Final[str] = "[MyNumber Redacted]"
TOKEN_EMAIL: Final[str] = "<EMAIL_REDACTED>"
TOKEN_PHONE: Final[str] = "<PHONE_REDACTED>"
TOKEN_GOV_ID: Final[str] = "<GOV_ID_REDACTED>"
TOKEN_LOCATION: Final[str] = "<LOCATION_REDACTED>"
TOKEN_CARD: Final[str] = "<CARD_REDACTED>"

ALL_TOKENS: Final[frozenset[str]] = frozenset(
    {
        TOKEN_AADHAAR,
        TOKEN_RRN,
        TOKEN_MYNUMBER,
        TOKEN_EMAIL,
        TOKEN_PHONE,
        TOKEN_GOV_ID,
        TOKEN_LOCATION,
        TOKEN_CARD,
    }
)


#: Matches any replacement token this system (or its credential guard) emits:
#: ``<THING_REDACTED>``, ``[Aadhaar Redacted]``, ``[RRN Omitted]``. Candidate
#: matches overlapping one of these are discarded before overlap resolution.
#:
#: This is what makes idempotence *structural* rather than a property each
#: individual pattern has to be careful enough to preserve. Without it, any
#: detector loose enough to match an uppercase word -- the generic passport
#: rule, for one -- happily re-redacts the word "REDACTED" inside a token it
#: emitted on the previous pass.
_PROTECTED_TOKEN_RE: Final[Pattern[str]] = re.compile(
    r"<[A-Z][A-Z_]*_(?:REDACTED|OMITTED)>|\[[A-Za-z][A-Za-z ]*(?:Redacted|Omitted)\]"
)


# =============================================================================
# Cryptographic redaction fingerprints
# =============================================================================

#: Environment variable holding the deployment's redaction key.
REDACTION_KEY_ENV: Final[str] = "HRTE_REDACTION_KEY"

#: Characters stripped before fingerprinting, so that "+91 98765 43210" and
#: "+919876543210" are recognised as the same identifier.
_FINGERPRINT_NOISE: Final[Pattern[str]] = re.compile(r"[\s\-().+/]")


@lru_cache(maxsize=1)
def _redaction_key() -> bytes:
    """The keying material for redaction fingerprints.

    When ``HRTE_REDACTION_KEY`` is set, fingerprints are stable across processes
    and restarts, which is what makes cross-session duplicate detection possible.

    When it is **not** set we generate an ephemeral per-process key rather than
    falling back to a constant. That choice is deliberate and worth stating,
    because the alternative is a trap: an unkeyed digest of a phone number is
    not anonymous. The search space is about ten billion, which a laptop
    exhausts in seconds, so a published "hash" of a mobile number is the number.
    An ephemeral key makes fingerprints useless to an attacker who obtains the
    stored data, at the cost of making them useless across restarts too -- the
    safe failure, and the one you notice.
    """
    configured = os.environ.get(REDACTION_KEY_ENV, "").strip()
    if configured:
        return hashlib.blake2b(configured.encode("utf-8"), digest_size=32).digest()
    return secrets.token_bytes(32)


def reset_redaction_key() -> None:
    """Drop the cached key so the next call re-reads the environment."""
    _redaction_key.cache_clear()


def fingerprint(value: str) -> str:
    """A keyed, one-way BLAKE2b digest of an identifier.

    This is what makes redaction *cryptographic* rather than merely lossy. The
    raw value is destroyed, but two occurrences of the same identifier -- the
    same phone number on two applications, the same Aadhaar across a duplicate
    submission -- produce the same fingerprint, so duplicates are detectable
    without anything reversible ever being stored.

    Keyed rather than plain: BLAKE2b in keyed mode is a MAC, so without the key
    an attacker cannot confirm a guess even for a small search space. That is
    the whole difference between a fingerprint and a thin disguise.
    """
    normalised = _FINGERPRINT_NOISE.sub("", value).casefold()
    return hashlib.blake2b(
        normalised.encode("utf-8"), key=_redaction_key(), digest_size=8
    ).hexdigest()


class PiiKind(StrEnum):
    """Classification of a redacted span. Safe to log -- carries no digits."""

    EMAIL = "email"
    PHONE = "phone"
    AADHAAR = "aadhaar"
    MYNUMBER = "mynumber"
    KOREAN_RRN = "korean_rrn"
    SSN = "ssn"
    PAN = "pan"
    PASSPORT = "passport"
    PAYMENT_CARD = "payment_card"
    STREET_ADDRESS = "street_address"
    POSTAL_CODE = "postal_code"
    GEO_COORDINATE = "geo_coordinate"


class SecurityBreachException(Exception):
    """Raised when unredacted PII is detected at an egress boundary.

    This is intentionally not a subclass of ``ValueError``: a caller doing
    broad input validation must not swallow it by accident.
    """

    def __init__(self, kinds: Sequence[PiiKind], boundary: str = "egress") -> None:
        self.kinds = tuple(kinds)
        self.boundary = boundary
        # The message never quotes the offending text -- an exception string is
        # the single most likely thing to end up in a log aggregator.
        listed = ", ".join(sorted({k.value for k in self.kinds}))
        super().__init__(
            f"Unredacted PII detected at the {boundary} boundary: {listed}. "
            f"{len(self.kinds)} finding(s). Route the payload through "
            f"pii_scrubber.scrub() before egress."
        )


# =============================================================================
# Context disambiguation
# =============================================================================

#: A bare 12-digit run is Aadhaar by default (the strict reading of the
#: invariant). Japanese MyNumber is also 12 digits, so it is only selected when
#: one of these markers sits within CONTEXT_WINDOW characters of the match.
_MYNUMBER_MARKERS: Final[tuple[str, ...]] = (
    "mynumber",
    "my number",
    "my-number",
    "individual number",
    "個人番号",
    "マイナンバー",
)
_AADHAAR_MARKERS: Final[tuple[str, ...]] = ("aadhaar", "aadhar", "uidai", "आधार")
CONTEXT_WINDOW: Final[int] = 48


def _context(text: str, start: int, end: int) -> str:
    lo = max(0, start - CONTEXT_WINDOW)
    hi = min(len(text), end + CONTEXT_WINDOW)
    return text[lo:hi].casefold()


def _classify_twelve_digit(text: str, start: int, end: int) -> tuple[PiiKind, str]:
    """Disambiguate a 12-digit run between Aadhaar and Japanese MyNumber."""
    ctx = _context(text, start, end)
    if any(marker in ctx for marker in _MYNUMBER_MARKERS):
        return PiiKind.MYNUMBER, TOKEN_MYNUMBER
    if any(marker in ctx for marker in _AADHAAR_MARKERS):
        return PiiKind.AADHAAR, TOKEN_AADHAAR
    # Strict default per invariant 1: any 12-digit number is treated as Aadhaar.
    return PiiKind.AADHAAR, TOKEN_AADHAAR


# =============================================================================
# Pattern table
# =============================================================================


@dataclass(frozen=True, slots=True)
class PiiPattern:
    """One compiled detector.

    ``priority`` resolves overlaps: a higher-priority match wins the span
    outright, regardless of which detector happened to be registered first.
    Ties break on match length, then on start offset, so resolution is total
    and stable.
    """

    kind: PiiKind
    regex: Pattern[str]
    token: str
    priority: int
    group: int = 0
    #: Optional case-folded markers that must appear near the match for it to
    #: count. Used for detectors that are otherwise too loose to run unguarded.
    requires_context: tuple[str, ...] = ()

    def resolve(self, text: str, start: int, end: int) -> tuple[PiiKind, str]:
        """Hook for detectors whose classification depends on context."""
        if self.kind is PiiKind.AADHAAR:
            return _classify_twelve_digit(text, start, end)
        return self.kind, self.token


# --- building blocks ---------------------------------------------------------
# `_SEP` is the set of separators that appear inside grouped identifiers.
# The non-breaking space is written as an escape rather than a literal: a
# literal U+00A0 in source is invisible, and resumes pasted out of Word are
# full of them.
_SEP = r"[ \u00a0.\-]"

_STREET_SUFFIX = (
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Highway|Hwy"
    r"|Cross|Main|Block|Sector|Phase|Nagar|Layout|Colony|Marg|Extension|Circle"
    r"|Court|Ct|Place|Pl|Terrace|Way|Park|Gardens?|Apartments?|Apt|Flat|Residency"
    r"|Towers?|Heights?)"
)

#: Compiled once, at module import. Nothing in the hot path recompiles a regex:
#: the 10ms vector-match budget leaves no room for it.
PATTERNS: Final[tuple[PiiPattern, ...]] = (
    # -- e-mail -------------------------------------------------------------
    PiiPattern(
        kind=PiiKind.EMAIL,
        regex=re.compile(
            r"(?<![\w.+-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}(?![\w.-])"
        ),
        token=TOKEN_EMAIL,
        priority=100,
    ),
    # -- payment card (13-19 digits) ----------------------------------------
    # Registered above the 12-digit rule so a card is never sliced into an
    # "Aadhaar" prefix plus a dangling remainder.
    PiiPattern(
        kind=PiiKind.PAYMENT_CARD,
        regex=re.compile(rf"(?<!\d)(?:\d{{4}}{_SEP}?){{3}}\d{{1,7}}(?!\d)"),
        token=TOKEN_CARD,
        priority=96,
    ),
    # -- South Korean RRN: YYMMDD-Gxxxxxx (13 digits) ------------------------
    PiiPattern(
        kind=PiiKind.KOREAN_RRN,
        regex=re.compile(rf"(?<!\d)\d{{6}}{_SEP}?[1-8]\d{{6}}(?!\d)"),
        token=TOKEN_RRN,
        priority=95,
    ),
    # -- Aadhaar / MyNumber: exactly 12 digits, optionally 4-4-4 grouped -----
    PiiPattern(
        kind=PiiKind.AADHAAR,
        regex=re.compile(rf"(?<!\d)\d{{4}}{_SEP}?\d{{4}}{_SEP}?\d{{4}}(?!\d)"),
        token=TOKEN_AADHAAR,
        priority=90,
    ),
    # -- US SSN --------------------------------------------------------------
    PiiPattern(
        kind=PiiKind.SSN,
        regex=re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"),
        token=TOKEN_GOV_ID,
        priority=88,
    ),
    # Unhyphenated SSN is indistinguishable from a 9-digit reference number,
    # so it only counts with an explicit marker nearby.
    PiiPattern(
        kind=PiiKind.SSN,
        regex=re.compile(r"(?<!\d)\d{9}(?!\d)"),
        token=TOKEN_GOV_ID,
        priority=84,
        requires_context=("ssn", "social security", "social-security"),
    ),
    # -- Indian PAN: AAAAA9999A ---------------------------------------------
    PiiPattern(
        kind=PiiKind.PAN,
        regex=re.compile(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])"),
        token=TOKEN_GOV_ID,
        priority=82,
    ),
    # -- Passport ------------------------------------------------------------
    # Indian format (1 letter + 7 digits) is distinctive enough to run bare.
    PiiPattern(
        kind=PiiKind.PASSPORT,
        regex=re.compile(r"(?<![A-Z0-9])[A-PR-WYa-prwy]\d{7}(?![A-Za-z0-9])"),
        token=TOKEN_GOV_ID,
        priority=80,
    ),
    # Generic 8-9 alphanumeric passport books need a marker to avoid eating
    # build IDs, commit SHAs and ticket numbers.
    PiiPattern(
        kind=PiiKind.PASSPORT,
        # The digit lookahead matters: without it this rule matches any
        # 8-9 letter uppercase word near the term "passport".
        regex=re.compile(r"(?<![A-Za-z0-9])(?=[A-Z0-9]*\d)[A-Z0-9]{8,9}(?![A-Za-z0-9])"),
        token=TOKEN_GOV_ID,
        priority=76,
        requires_context=("passport", "travel document"),
    ),
    # -- Telephone -----------------------------------------------------------
    # E.164 with an explicit country code.
    PiiPattern(
        kind=PiiKind.PHONE,
        regex=re.compile(
            rf"(?<![\w+])\+\d{{1,3}}{_SEP}?(?:\(\d{{1,4}}\){_SEP}?)?\d{{2,5}}"
            rf"(?:{_SEP}?\d{{2,5}}){{1,3}}(?!\d)"
        ),
        token=TOKEN_PHONE,
        priority=72,
    ),
    # NANP with separators or parentheses.
    PiiPattern(
        kind=PiiKind.PHONE,
        regex=re.compile(r"(?<![\w+])\(?[2-9]\d{2}\)?[ .\-]\d{3}[ .\-]\d{4}(?!\d)"),
        token=TOKEN_PHONE,
        priority=70,
    ),
    # Bare Indian mobile: 10 digits beginning 6-9.
    PiiPattern(
        kind=PiiKind.PHONE,
        regex=re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),
        token=TOKEN_PHONE,
        priority=68,
    ),
    # -- Geographic coordinates ---------------------------------------------
    PiiPattern(
        kind=PiiKind.GEO_COORDINATE,
        regex=re.compile(
            # Trailing guard is (?!\d), not (?![\w.]): a coordinate pair at the
            # end of a sentence is followed by a full stop, and the stricter
            # guard silently refused to match exactly that case.
            r"(?<![\w.])-?(?:1[0-7]\d|\d{1,2})\.\d{4,}\s*,\s*-?(?:1[0-7]\d|\d{1,2})\.\d{4,}(?!\d)"
        ),
        token=TOKEN_LOCATION,
        priority=66,
    ),
    # -- Street address ------------------------------------------------------
    # Anchored on a house/plot number followed by up to four capitalised tokens
    # and a street-type suffix, optionally trailing into ", City, ST 560001".
    PiiPattern(
        kind=PiiKind.STREET_ADDRESS,
        regex=re.compile(
            rf"(?<![\w#])(?:(?:Flat|Apt\.?|Apartment|House|Door|Plot|No\.?|#)\s*"
            rf"[\w/\-]{{1,8}},?\s+)?"
            rf"\d{{1,5}}(?:[/\-][\w]{{1,4}})?,?\s+"
            rf"(?:[A-Z][\w.'\-]{{1,20}}\s+){{0,4}}"
            rf"{_STREET_SUFFIX}\b\.?"
            rf"(?:,\s*[A-Z][\w.'\-]{{1,20}}(?:\s+[A-Z][\w.'\-]{{1,20}}){{0,2}}){{0,3}}"
            rf"(?:,?\s*(?:[A-Z]{{2}}\s+)?\d{{5,6}}(?:-\d{{4}})?)?",
            re.UNICODE,
        ),
        token=TOKEN_LOCATION,
        priority=60,
    ),
    # -- Postal code, marker-gated ------------------------------------------
    PiiPattern(
        kind=PiiKind.POSTAL_CODE,
        regex=re.compile(r"(?<!\d)\d{5,6}(?:-\d{4})?(?!\d)"),
        token=TOKEN_LOCATION,
        priority=50,
        requires_context=("pin code", "pincode", "pin:", "zip", "postal", "post code"),
    ),
)


# =============================================================================
# Offset tracking
# =============================================================================


@dataclass(frozen=True, slots=True)
class OffsetMap:
    """Maps offsets in the original text to offsets in the scrubbed text.

    Redaction changes string length, so any span computed against the original
    (a competency evidence quote, a diarisation range, a Langfuse observation
    window) needs translation before it can be applied to the scrubbed text.
    Without this, annotations drift by the cumulative redaction delta and
    silently point at the wrong words.
    """

    #: Start offsets of each redacted span in the ORIGINAL text, ascending.
    _starts: tuple[int, ...] = ()
    #: Cumulative length delta (scrubbed - original) applied AFTER each span.
    _deltas: tuple[int, ...] = ()
    #: End offsets in the ORIGINAL text, parallel to ``_starts``.
    _ends: tuple[int, ...] = ()
    #: End offsets in the SCRUBBED text, parallel to ``_starts``.
    _new_ends: tuple[int, ...] = ()

    def translate(self, index: int) -> int:
        """Translate an original-text offset to its scrubbed-text counterpart.

        An offset that falls *inside* a redacted span collapses to the end of
        the replacement token: the original characters no longer exist, and
        pointing at the token is the only honest answer.
        """
        if not self._starts:
            return index
        i = bisect_right(self._starts, index) - 1
        if i < 0:
            return index
        if index < self._ends[i]:
            return self._new_ends[i]
        return index + self._deltas[i]

    def translate_span(self, start: int, end: int) -> tuple[int, int]:
        lo = self.translate(start)
        hi = self.translate(end)
        return (lo, max(lo, hi))


# =============================================================================
# Findings and results
# =============================================================================


@dataclass(frozen=True, slots=True)
class PiiFinding:
    """A single redaction. Carries offsets and a class -- never the raw value.

    ``original_length`` is the only quantitative trace we keep, because
    reconstructing an identifier from a length is not possible and the value is
    genuinely useful when tuning detector precision.
    """

    kind: PiiKind
    start: int
    end: int
    token: str
    original_length: int
    #: Keyed BLAKE2b digest of the removed value. One-way and non-reversible;
    #: safe to persist, and the only trace of the identifier that survives.
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid finding span: [{self.start}, {self.end})")


@dataclass(frozen=True, slots=True)
class ScrubResult:
    """Outcome of one scrub: safe text, an audit trail, and offset provenance."""

    text: str
    findings: tuple[PiiFinding, ...] = ()
    offsets: OffsetMap = field(default_factory=OffsetMap)

    @property
    def clean(self) -> bool:
        """True when the input carried no detectable PII."""
        return not self.findings

    @property
    def kinds(self) -> tuple[PiiKind, ...]:
        return tuple(f.kind for f in self.findings)

    def fingerprints(self) -> dict[str, str]:
        """Map each redaction to its keyed digest, for duplicate detection.

        Keyed by ``kind:index`` rather than by value, because the value is
        exactly what must not appear here.
        """
        out: dict[str, str] = {}
        seen: dict[str, int] = {}
        for f in self.findings:
            n = seen.get(f.kind.value, 0)
            seen[f.kind.value] = n + 1
            out[f"{f.kind.value}:{n}"] = f.fingerprint
        return out

    def counts(self) -> dict[str, int]:
        """Per-class redaction counts. Safe to ship to telemetry verbatim."""
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.kind.value] = out.get(f.kind.value, 0) + 1
        return dict(sorted(out.items()))

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.text


# =============================================================================
# Core algorithm
# =============================================================================


def _candidates(text: str) -> list[tuple[int, int, PiiPattern]]:
    """Collect every pattern hit against the ORIGINAL text, unfiltered."""
    hits: list[tuple[int, int, PiiPattern]] = []
    folded = text.casefold()
    protected = [m.span() for m in _PROTECTED_TOKEN_RE.finditer(text)]
    for pattern in PATTERNS:
        for match in pattern.regex.finditer(text):
            start, end = match.span(pattern.group)
            if end <= start:
                continue
            if any(start < p_end and p_start < end for p_start, p_end in protected):
                continue  # inside a replacement token from a previous pass
            if pattern.requires_context:
                lo = max(0, start - CONTEXT_WINDOW)
                hi = min(len(text), end + CONTEXT_WINDOW)
                window = folded[lo:hi]
                if not any(marker in window for marker in pattern.requires_context):
                    continue
            hits.append((start, end, pattern))
    return hits


def _resolve_overlaps(
    hits: Sequence[tuple[int, int, PiiPattern]],
) -> list[tuple[int, int, PiiPattern]]:
    """Pick a non-overlapping subset by (priority, length, position).

    Greedy-by-position would be wrong: a low-priority phone match starting one
    character earlier would shadow the high-priority RRN that overlaps it. We
    therefore order by priority first and accept greedily, which makes the
    outcome independent of pattern registration order within a priority class.
    """
    ordered = sorted(hits, key=lambda h: (-h[2].priority, -(h[1] - h[0]), h[0]))
    accepted: list[tuple[int, int, PiiPattern]] = []
    for start, end, pattern in ordered:
        if any(start < a_end and a_start < end for a_start, a_end, _ in accepted):
            continue
        accepted.append((start, end, pattern))
    accepted.sort(key=lambda h: h[0])
    return accepted


def scrub(text: str) -> ScrubResult:
    """Redact every detectable identifier in ``text`` in a single pass.

    Properties this function guarantees, all covered by the test suite:

    * **Deterministic** -- no randomness, no dict-ordering dependence, no clock.
    * **Idempotent** -- ``scrub(scrub(t).text).text == scrub(t).text``, because
      no replacement token can match any pattern in the table.
    * **Total** -- never raises on arbitrary input; use :func:`assert_zero_pii`
      when you want a hard failure instead.
    * **Offset-traceable** -- the returned :class:`OffsetMap` relocates any span
      from the original text into the scrubbed text.
    """
    if not text:
        return ScrubResult(text="")

    accepted = _resolve_overlaps(_candidates(text))
    if not accepted:
        return ScrubResult(text=text)

    pieces: list[str] = []
    findings: list[PiiFinding] = []
    starts: list[int] = []
    ends: list[int] = []
    new_ends: list[int] = []
    deltas: list[int] = []

    cursor = 0
    delta = 0
    for start, end, pattern in accepted:
        kind, token = pattern.resolve(text, start, end)
        pieces.append(text[cursor:start])
        pieces.append(token)
        findings.append(
            PiiFinding(
                kind=kind,
                start=start,
                end=end,
                token=token,
                original_length=end - start,
                fingerprint=fingerprint(text[start:end]),
            )
        )
        delta += len(token) - (end - start)
        starts.append(start)
        ends.append(end)
        new_ends.append(end + delta)
        deltas.append(delta)
        cursor = end
    pieces.append(text[cursor:])

    return ScrubResult(
        text="".join(pieces),
        findings=tuple(findings),
        offsets=OffsetMap(
            _starts=tuple(starts),
            _deltas=tuple(deltas),
            _ends=tuple(ends),
            _new_ends=tuple(new_ends),
        ),
    )


def scrub_text(text: str) -> str:
    """Convenience wrapper returning only the safe string."""
    return scrub(text).text


def detect(text: str) -> tuple[PiiFinding, ...]:
    """Report what *would* be redacted, without rewriting the text."""
    return scrub(text).findings


def assert_zero_pii(text: str, *, boundary: str = "egress") -> None:
    """Hard gate. Raises :class:`SecurityBreachException` if ``text`` has PII.

    Call this immediately before a network write, a disk write, or a vector
    upsert. It re-scans rather than trusting a flag, so it also catches text
    that was concatenated back together after scrubbing -- the failure mode
    that actually happens in production.
    """
    findings = detect(text)
    if findings:
        raise SecurityBreachException([f.kind for f in findings], boundary=boundary)


def sanitize_for_egress(payload: Any, *, boundary: str = "egress") -> Any:
    """Recursively scrub any JSON-shaped payload.

    Mapping *keys* are scrubbed too. That is not paranoia: telemetry payloads
    routinely key metadata by candidate e-mail, and a redacted value under a
    leaking key is still a leak.
    """
    if isinstance(payload, str):
        return scrub_text(payload)
    if isinstance(payload, Mapping):
        return {
            sanitize_for_egress(k, boundary=boundary): sanitize_for_egress(v, boundary=boundary)
            for k, v in payload.items()
        }
    if isinstance(payload, (list, tuple, set, frozenset)):
        cleaned = [sanitize_for_egress(v, boundary=boundary) for v in payload]
        return type(payload)(cleaned) if isinstance(payload, (list, tuple)) else cleaned
    # int/float/bool/None and anything else opaque passes through untouched.
    return payload


def redaction_summary(results: Iterable[ScrubResult]) -> dict[str, int]:
    """Aggregate counts across many scrubs, for the end-of-session audit line."""
    totals: dict[str, int] = {}
    for result in results:
        for kind, count in result.counts().items():
            totals[kind] = totals.get(kind, 0) + count
    return dict(sorted(totals.items()))


__all__ = [
    "ALL_TOKENS",
    "CONTEXT_WINDOW",
    "PATTERNS",
    "REDACTION_KEY_ENV",
    "TOKEN_AADHAAR",
    "TOKEN_CARD",
    "TOKEN_EMAIL",
    "TOKEN_GOV_ID",
    "TOKEN_LOCATION",
    "TOKEN_MYNUMBER",
    "TOKEN_PHONE",
    "TOKEN_RRN",
    "OffsetMap",
    "PiiFinding",
    "PiiKind",
    "PiiPattern",
    "ScrubResult",
    "SecurityBreachException",
    "assert_zero_pii",
    "detect",
    "fingerprint",
    "redaction_summary",
    "reset_redaction_key",
    "sanitize_for_egress",
    "scrub",
    "scrub_text",
]
