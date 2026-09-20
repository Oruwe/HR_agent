"""The single path scraped records take into the database.

One function, used by both the HTTP import route and the ``seed`` command, so
there is exactly one place where a record is scrubbed and turned into a row.
Two ingest paths that drift apart is how a deployment ends up with unredacted
records nobody meant to store.
"""

from __future__ import annotations

from typing import Any

from app.db.models import Candidate
from app.security.pii_scrubber import sanitize_for_egress, scrub

#: Keys a scraper is likely to use for the two fields worth promoting out of
#: the blob into columns, checked in order. Anything unmatched simply stays in
#: `source` -- the schema is the scraper's to choose, not ours.
NAME_KEYS = ("name", "full_name", "fullName", "candidate_name", "displayName", "username")
HEADLINE_KEYS = ("headline", "title", "current_role", "currentRole", "position", "summary")


def first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_candidate(record: dict[str, Any]) -> tuple[Candidate, bool]:
    """Scrub one scraped record into a row. Returns (row, contained_pii).

    Redaction happens once, here, on the way in. Everything downstream -- the
    database, the model prompt, the dashboard, the export -- then only ever
    sees scrubbed text, which is the only arrangement that can be audited by
    reading a single function.
    """
    name = scrub(first_string(record, NAME_KEYS) or "Unknown candidate").text[:200]
    headline = scrub(first_string(record, HEADLINE_KEYS)).text[:400]
    contained_pii = bool(scrub(str(record)).findings)
    row = Candidate(
        name=name,
        headline=headline,
        source=sanitize_for_egress(record, boundary="candidate_import"),
    )
    return row, contained_pii


__all__ = ["HEADLINE_KEYS", "NAME_KEYS", "build_candidate", "first_string"]
