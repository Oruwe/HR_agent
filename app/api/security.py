"""Per-session bearer tokens.

The Candidate App has no user account -- a screening call is anonymous by
design (see ``app/schemas/candidate.py``: no legal name, no login). What it
needs instead is proof that a given HTTP request belongs to the session it
claims: without that, session ID alone would let anyone who intercepts one
URL read or continue someone else's interview.

So: a random 256-bit token is minted at session creation and returned exactly
once, in the create-session response. Every subsequent call for that session
must present it as ``Authorization: Bearer <token>``. Only its SHA-256 hash is
ever persisted, so a database read (or leak) cannot be replayed as a live
token -- the same asymmetry the codebase already applies to candidate PII via
keyed BLAKE2b fingerprints.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from fastapi import Header, HTTPException, status


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_token(token), expected_hash)


def extract_bearer_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header; expected 'Bearer <session_token>'.",
        )
    return authorization.split(" ", 1)[1].strip()


__all__ = ["extract_bearer_token", "hash_token", "new_token", "verify_token"]
