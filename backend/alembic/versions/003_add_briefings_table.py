"""Add briefings table with tenant partitioning.

Revision ID: 003
Revises: 002
Create Date: 2024-01-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create briefings table with tenant partitioning
    op.create_table(
        "briefings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("content_md_uri", sa.String(500), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("generated_by", sa.dialects.postgresql.JSONB, nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()"), onupdate=sa.text("now()")),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "title", "version", name="uq_briefings_tenant_title_version"),
        postgresql_partition_by="LIST (tenant_id)",
    )

    # Create indexes
    op.create_index("ix_briefings_tenant_id", "briefings", ["tenant_id"])
    op.create_index("ix_briefings_tenant_id_is_current", "briefings", ["tenant_id", "is_current"])
    op.create_index("ix_briefings_tenant_id_created_at_desc", "briefings", ["tenant_id", sa.text("created_at DESC")])

    # Create default partition for briefings (catches any tenant not explicitly partitioned)
    op.execute("CREATE TABLE briefings_default PARTITION OF briefings FOR VALUES IN (DEFAULT)")

    # Add RLS policy (will be enforced when RLS is enabled on the table)
    # Note: RLS is already enabled on the parent table via the base migration


def downgrade() -> None:
    op.drop_table("briefings")