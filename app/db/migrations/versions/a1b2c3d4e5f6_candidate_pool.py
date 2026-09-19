"""Initial schema: the candidate pool.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-09-19

Replaces the previous sessions/turns/evaluations schema. That schema existed
to record live voice interviews this product no longer conducts -- candidate
records now arrive as scraped JSON from an external tool, so there is one
table and the scraped payload lives in it verbatim.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, server_default="Unknown candidate"),
        sa.Column("headline", sa.String(400), nullable=False, server_default=""),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("imported_at", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("recommendation", sa.String(32), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("analyzed_at", sa.Integer(), nullable=True),
    )
    op.create_index("ix_candidates_score", "candidates", ["score"])


def downgrade() -> None:
    op.drop_index("ix_candidates_score", table_name="candidates")
    op.drop_table("candidates")
