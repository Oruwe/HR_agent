"""Credential redaction and the egress decorator that enforces both gates.

:mod:`app.security.pii_scrubber` protects the *candidate*. This module protects
the *operator*: API keys, bearer tokens, LiveKit room grants and private keys
leak through exactly the same channels (prompt echoes, trace payloads, error
messages), and a screening transcript is a surprisingly common place to find
one pasted by accident.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Awaitable, Callable
from re import Pattern
from typing import Any, Final, TypeVar

from app.config import PiiMode, get_settings
from app.security.pii_scrubber import (
    SecurityBreachException,
    detect,
    sanitize_for_egress,
)

TOKEN_CREDENTIAL: Final[str] = "<CREDENTIAL_REDACTED>"

#: Ordered so that longer, structurally distinctive secrets match before the
#: generic high-entropy fallbacks.
CREDENTIAL_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    # PEM private key blocks, including the body.
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    # JSON Web Tokens (LiveKit room grants are JWTs).
    re.compile(r"(?<![\w.])eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}(?![\w.])"),
    # Google / Gemini API keys.
    re.compile(r"(?<![\w-])AIza[0-9A-Za-z_\-]{35}(?![\w-])"),
    # AWS access key IDs.
    re.compile(r"(?<![\w-])(?:AKIA|ASIA)[0-9A-Z]{16}(?![\w-])"),
    # Vendor-prefixed secrets: sk-..., lf_pk_..., lk_api_..., ghp_..., etc.
    re.compile(r"(?<![\w-])(?:sk|pk|rk|lf|lk|ghp|gho|xoxb|xoxp)[_-][0-9A-Za-z_\-]{16,}(?![\w-])"),
    # Authorization headers, keeping the scheme so the log stays diagnosable.
    re.compile(r"(?i)\b(?:bearer|basic|token)\s+[0-9A-Za-z._\-+/=]{12,}"),
    # key=value / "key": "value" assignments for secret-shaped names.
    re.compile(
        r"(?i)\b(?P<name>[\w.\-]*(?:api[_-]?key|secret|password|passwd|token|credential)"
        r"[\w.\-]*)[\"']?\s*[:=]\s*[\"']?(?P<value>[^\s\"',;}]{8,})[\"']?"
    ),
)


class CredentialLeakException(Exception):
    """Raised when a live credential is detected at an egress boundary."""

    def __init__(self, count: int, boundary: str = "egress") -> None:
        self.count = count
        self.boundary = boundary
        super().__init__(
            f"{count} credential-shaped value(s) detected at the {boundary} "
            f"boundary. Route the payload through redact_credentials() first."
        )


def _replace(match: re.Match[str]) -> str:
    """Keep the key name when there is one, so logs stay diagnosable."""
    groups = match.groupdict()
    if groups.get("name") and groups.get("value"):
        return f"{groups['name']}={TOKEN_CREDENTIAL}"
    return TOKEN_CREDENTIAL


def redact_credentials(text: str) -> str:
    """Replace every credential-shaped value with :data:`TOKEN_CREDENTIAL`."""
    if not text:
        return text
    for pattern in CREDENTIAL_PATTERNS:
        text = pattern.sub(_replace, text)
    return text


def count_credentials(text: str) -> int:
    """How many credential-shaped values are present. Never returns the values."""
    if not text:
        return 0
    # Count against progressively-redacted text so overlapping patterns (a JWT
    # inside an Authorization header) are not double counted.
    total = 0
    working = text
    for pattern in CREDENTIAL_PATTERNS:
        working, n = pattern.subn(_replace, working)
        total += n
    return total


def assert_zero_credentials(text: str, *, boundary: str = "egress") -> None:
    n = count_credentials(text)
    if n:
        raise CredentialLeakException(n, boundary=boundary)


def sanitize(text: str) -> str:
    """Both gates, in the order that matters.

    Credentials first: a JWT body is base64 and can contain digit runs that the
    PII table would otherwise slice into an 'Aadhaar', destroying the structure
    that makes the credential detectable at all.
    """
    return sanitize_for_egress(redact_credentials(text))


F = TypeVar("F", bound=Callable[..., Any])


def guard_egress(boundary: str) -> Callable[[F], F]:
    """Decorate a function whose first positional argument crosses a boundary.

    In ``strict`` mode a violation raises. In ``permissive`` mode the payload is
    scrubbed in place and the call proceeds -- appropriate for a staging rollout
    where a hard failure would take down a live interview, never for production.

    Works on both sync and async callables.
    """

    def decorator(func: F) -> F:
        def _prepare(payload: Any) -> Any:
            if isinstance(payload, str):
                cleaned = redact_credentials(payload)
                findings = detect(cleaned)
                if findings and get_settings().pii_mode is PiiMode.STRICT:
                    raise SecurityBreachException([f.kind for f in findings], boundary=boundary)
                return sanitize_for_egress(cleaned)
            return sanitize_for_egress(payload, boundary=boundary)

        if _is_async(func):

            @functools.wraps(func)
            async def async_wrapper(payload: Any, *args: Any, **kwargs: Any) -> Any:
                return await func(_prepare(payload), *args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(payload: Any, *args: Any, **kwargs: Any) -> Any:
            return func(_prepare(payload), *args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _is_async(func: Callable[..., Any]) -> bool:
    import inspect

    return inspect.iscoroutinefunction(func) or isinstance(func, Awaitable)


__all__ = [
    "CREDENTIAL_PATTERNS",
    "TOKEN_CREDENTIAL",
    "CredentialLeakException",
    "assert_zero_credentials",
    "count_credentials",
    "guard_egress",
    "redact_credentials",
    "sanitize",
]
