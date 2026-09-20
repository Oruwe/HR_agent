"""Shared fixtures.

Two deliberate choices worth explaining:

**No pytest-asyncio.** Async tests are driven through the :func:`run_async`
helper instead. The plugin's event-loop fixtures are a recurring source of
deprecation warnings across pytest majors, and this suite runs under
``filterwarnings = ["error"]`` -- a warning is a failure here. An explicit
``asyncio.run`` per test is also easier to reason about.

**Environment isolation.** Every test gets a clean, explicitly-set
environment. Settings are cached process-wide, so a test that mutated the
environment without clearing that cache would silently poison every test
after it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, TypeVar

import pytest

from app.agent.cognition import reset_fallbacks
from app.config import Settings, reset_settings
from app.retrieval import reset_retrieval_fallbacks
from app.security.pii_scrubber import reset_redaction_key

T = TypeVar("T")

#: Every variable the app reads. Cleared before each test so a developer's
#: populated .env cannot change what CI asserts.
_MANAGED_ENV: tuple[str, ...] = (
    "HRTE_ENV",
    "HRTE_MODEL",
    "HRTE_TEMPERATURE",
    "HRTE_MAX_OUTPUT_TOKENS",
    "HRTE_THINKING_BUDGET",
    "HRTE_PII_MODE",
    "HRTE_MOSS_INDEX",
    "HRTE_RETRIEVAL_TOP_K",
    "MOSS_PROJECT_ID",
    "MOSS_PROJECT_KEY",
    "HRTE_REDACTION_KEY",
    "HRTE_CORS_ORIGINS",
    "HRTE_ADMIN_TOKEN",
    "GOOGLE_API_KEY",
    "DATABASE_URL",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in _MANAGED_ENV:
        monkeypatch.delenv(key, raising=False)
    reset_settings()
    reset_redaction_key()
    reset_fallbacks()
    reset_retrieval_fallbacks()
    yield
    reset_settings()
    reset_redaction_key()
    reset_fallbacks()
    reset_retrieval_fallbacks()


@pytest.fixture
def settings() -> Settings:
    """Offline settings: no key, so the mock provider is used."""
    return Settings()


def run_async(coro_fn: Callable[[], Awaitable[T]]) -> T:
    """Run an async callable to completion on a fresh event loop."""
    return asyncio.run(coro_fn())


@pytest.fixture
def run() -> Callable[[Callable[[], Awaitable[Any]]], Any]:
    return run_async


# =============================================================================
# Resumes carrying planted PII
# =============================================================================

PII_RESUME = """
Priya Raman
priya.raman@example.com | +91 98765 43210 | alt +1 (555) 123-4567
Aadhaar: 3412 7856 9034 | PAN ABCDE1234F | Passport K1234567
42 MG Road, Indiranagar, Bengaluru 560038
Emergency contact reachable on 9876543210.

Senior ML Systems Engineer. Led FSDP training across 512 A100s, cut step time
34 percent. Wrote Triton kernels. Reduced KV-cache 40 percent with vLLM paged
attention. Tuned HNSW recall against p99 latency.
"""

INTERNATIONAL_PII = """
Kim Min-jun, RRN 900101-1234567, Seoul.
Tanaka Yuki, MyNumber: 1234 5678 9012, Tokyo.
John Smith, SSN: 123-45-6789, 221 Baker Street, Springfield, IL 62704.
Card on file 4111 1111 1111 1111. Residence at 12.9716, 77.5946.
"""

#: Technical prose containing digit patterns that must NOT be redacted. This is
#: the false-positive corpus -- a scrubber that eats "p99 under 45ms" destroys
#: the very evidence the analyst reasons from.
TECHNICAL_NO_PII = """
Cut p99 from 1200ms to 160ms across 12 regions at 1500 QPS.
Ran 512 A100s from 2019-2023 with 99.99 percent uptime.
Reduced KV-cache 40 percent; HNSW m=16 and ef_construct=100.
Scaled to 200000000 vectors. Error budget 0.05 percent over 30 days.
Version 1.0.0 of the spec, RFC 7519, HTTP 429 backoff at 2^n seconds.
"""


@pytest.fixture
def pii_resume() -> str:
    return PII_RESUME


@pytest.fixture
def international_pii() -> str:
    return INTERNATIONAL_PII


@pytest.fixture
def technical_no_pii() -> str:
    return TECHNICAL_NO_PII
