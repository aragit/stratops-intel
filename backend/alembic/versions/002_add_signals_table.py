"""Add signals table for ingestion.

Revision ID: 002
Revises: 001
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op  # type: ignore[attr-defined]
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Create signals table
    op.create_table(
        "signals",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("source_url", sa.String(2048), nullable=True),
        sa.Column("content_uri", sa.String(500), nullable=False),
        sa.Column("fingerprint", sa.String(128), nullable=False),
        sa.Column("structured_payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # Create unique constraint on tenant_id + fingerprint for dedup
    op.create_unique_constraint("uq_signals_tenant_fingerprint", "signals", ["tenant_id", "fingerprint"])

    # Create indexes
    op.create_index("ix_signals_tenant_id", "signals", ["tenant_id"])
    op.create_index("ix_signals_source_type", "signals", ["source_type"])
    op.create_index("ix_signals_collected_at", "signals", ["collected_at"])

    # Enable Row-Level Security
    conn.execute(sa.text("ALTER TABLE signals ENABLE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE signals FORCE ROW LEVEL SECURITY;"))

    # Create RLS policy
    conn.execute(sa.text("""
        CREATE POLICY tenant_isolation ON signals
        FOR ALL
        USING (tenant_id = current_setting('app.current_tenant')::UUID);
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop policy
    conn.execute(sa.text("DROP POLICY IF EXISTS tenant_isolation ON signals;"))

    # Disable RLS
    conn.execute(sa.text("ALTER TABLE signals DISABLE ROW LEVEL SECURITY;"))

    # Drop constraints and indexes
    op.drop_constraint("uq_signals_tenant_fingerprint", "signals", type_="unique")
    op.drop_index("ix_signals_tenant_id", "signals")
    op.drop_index("ix_signals_source_type", "signals")
    op.drop_index("ix_signals_collected_at", "signals")

    # Drop table
    op.drop_table("signals")