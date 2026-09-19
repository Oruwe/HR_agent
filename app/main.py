"""Entry point: seed a demo pool, analyse it, or report configuration.

``python -m app.main seed`` loads a sample scraped pool and ranks it, which
is the fastest way to see the product doing its job with no credentials and
no scraper wired up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence

from sqlalchemy import select

from app.agent.analyst import Analyst
from app.api.routes_candidates import apply_baseline
from app.config import Settings, get_settings
from app.db.engine import SessionLocal, init_db
from app.db.models import Candidate
from app.demo_pool import DEMO_CANDIDATES
from app.ingest import build_candidate
from app.security.pii_scrubber import scrub


def _print_header(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def _import(records: Sequence[dict]) -> int:
    """Insert records through the same ingest path the HTTP route uses."""
    with SessionLocal() as db:
        for record in records:
            row, _ = build_candidate(record)
            db.add(row)
        db.commit()
    return len(records)


async def run_seed(settings: Settings, *, analyze: bool = True) -> int:
    init_db()
    _print_header("SEEDING THE DEMO POOL")
    count = _import(DEMO_CANDIDATES)
    print(f"Imported : {count} scraped candidate records")

    with SessionLocal() as db:
        rows = db.execute(select(Candidate)).scalars().all()

        # The bundled baseline goes on first so the board is never empty, then
        # a configured model overwrites it with its own judgement. Offline, the
        # baseline is what stays -- and the dashboard says so, because a score
        # that looks like analysis but isn't is worse than no score.
        applied = apply_baseline(rows)
        db.commit()
        print(f"Baseline : {applied} bundled rankings applied")

        if analyze and not settings.offline:
            print(f"Analyst  : {settings.model} -- running a live pass")
            rankings = await Analyst(settings).rank(rows)
            by_id = {r.candidate_id: r for r in rankings}
            now = int(time.time())
            for row in rows:
                ranking = by_id.get(row.id)
                if ranking is None:
                    continue
                row.score, row.recommendation = ranking.score, ranking.verdict
                row.rationale, row.analyzed_at = ranking.rationale, now
            db.commit()
            print(f"Analyst  : {len(rankings)} of {len(rows)} re-scored by the model")
        elif analyze:
            print("Analyst  : no GOOGLE_API_KEY set -- showing the bundled baseline only")

        ranked = sorted(rows, key=lambda c: -(c.score or 0.0))
        _print_header("RANKED POOL")
        for i, row in enumerate(ranked, 1):
            verdict = row.recommendation or "--"
            score = f"{row.score:.2f}" if row.score is not None else " -- "
            print(f"{i:>2}. {score}  {verdict:<10} {row.name:<22} {row.headline[:34]}")
            if row.rationale:
                print(f"      {row.rationale[:150]}")
    return 0


def run_verify(settings: Settings) -> int:
    _print_header("CONFIGURATION")
    print(f"  environment      : {settings.environment}")
    print(f"  model            : {settings.model}")
    print(f"  model configured : {settings.model_configured}")
    print(f"  offline          : {settings.offline}")
    print(f"  pii mode         : {settings.pii_mode}")

    _print_header("PII SCRUBBER")
    probe = "Reach Priya at priya@example.com or +91 98765 43210, Aadhaar 3412 7856 9034."
    result = scrub(probe)
    print(f"  in  : {probe}")
    print(f"  out : {result.text}")
    print(f"  redactions: {result.counts()}")
    ok = not scrub(result.text).findings
    print(f"\n{'=' * 74}\nOVERALL: {'PASS' if ok else 'FAIL'}\n{'=' * 74}")
    return 0 if ok else 1


def run_export(settings: Settings) -> int:
    """Print the demo pool as JSON, i.e. the shape the import route expects."""
    print(json.dumps({"candidates": DEMO_CANDIDATES}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("seed", "verify", "export"),
        help="seed: load and rank the demo pool; verify: config + PII report; "
        "export: print the demo pool as import-ready JSON",
    )
    parser.add_argument(
        "--no-analyze", action="store_true", help="seed only, skip the ranking pass"
    )
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "seed":
        return asyncio.run(run_seed(settings, analyze=not args.no_analyze))
    if args.command == "export":
        return run_export(settings)
    return run_verify(settings)


if __name__ == "__main__":
    sys.exit(main())
