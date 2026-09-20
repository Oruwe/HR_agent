"""Legacy voice-interview schema (historical placeholder).

Revision ID: 6741599e8157
Revises:
Create Date: 2026-09-19

This revision no longer does anything. It exists so that alembic can still
*locate* it.

The product replaced its entire schema when it stopped being a voice
interviewer, and the migration that created the old sessions/turns/evaluations
tables was deleted along with the code that used them. Any database that had
already run it still carries `alembic_version = '6741599e8157'`, and alembic
refuses to move a database whose recorded revision is not in the script
directory::

    ERROR [alembic.util.messaging] Can't locate revision identified by '6741599e8157'
    FAILED: Can't locate revision identified by '6741599e8157'

That exits non-zero, and the API container's command is
``alembic upgrade head && uvicorn ...`` -- so the whole service fails to start
on a database that was merely one version behind. Reproduced against a
database stamped with that revision: exit code 255, no server.

Keeping the id resolvable is the conventional fix for a squashed history, and
it is better than the alternatives: rewriting ``alembic_version`` at startup
is action-at-a-distance on someone's data, and telling operators to hand-patch
a table is a runbook step nobody finds at 2am.

``upgrade`` is deliberately a no-op rather than a recreation of the old
tables. A fresh database should never grow them -- nothing reads them any
more. The old tables are also deliberately *not* dropped: on a database that
has them they hold real interview records, and a migration that silently
destroys data to tidy up is a worse bug than the untidiness. Drop them by hand
once you are sure you do not want them.
"""

from __future__ import annotations

revision = "6741599e8157"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op. See the module docstring."""


def downgrade() -> None:
    """No-op. See the module docstring."""
