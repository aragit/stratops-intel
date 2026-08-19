"""Initial schema with multi-tenant PostgreSQL RLS setup.

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Create required extensions
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # Create set_tenant_context helper function for RLS
    conn.execute(sa.text("""
        CREATE OR REPLACE FUNCTION set_tenant_context(tenant_uuid UUID)
        RETURNS VOID AS $$
        BEGIN
            PERFORM set_config('app.current_tenant', tenant_uuid::TEXT, false);
        END;
        $$ LANGUAGE plpgsql;
    """))

    # Create tables
    op.create_table(
        "tenants",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("tier", sa.String(50), nullable=False, server_default="free"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="member"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.True_()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("key_hash", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.True_()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "tenant_configs",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # Create unique constraints and indexes
    op.create_unique_constraint("uq_users_tenant_email", "users", ["tenant_id", "email"])
    op.create_unique_constraint("uq_api_keys_key_hash", "api_keys", ["key_hash"])
    op.create_unique_constraint("uq_api_keys_tenant_name", "api_keys", ["tenant_id", "name"])
    op.create_unique_constraint("uq_tenant_configs_tenant_id", "tenant_configs", ["tenant_id"])

    # Enable Row-Level Security on all tables
    conn.execute(sa.text("ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE users ENABLE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE tenant_configs ENABLE ROW LEVEL SECURITY;"))

    # Force RLS (no BYPASSRLS privilege can skip)
    conn.execute(sa.text("ALTER TABLE tenants FORCE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE users FORCE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE api_keys FORCE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE tenant_configs FORCE ROW LEVEL SECURITY;"))

    # Create RLS policies
    conn.execute(sa.text("""
        CREATE POLICY tenant_isolation ON tenants
        FOR ALL
        USING (id = current_setting('app.current_tenant')::UUID);
    """))

    conn.execute(sa.text("""
        CREATE POLICY tenant_isolation ON users
        FOR ALL
        USING (tenant_id = current_setting('app.current_tenant')::UUID);
    """))

    conn.execute(sa.text("""
        CREATE POLICY tenant_isolation ON api_keys
        FOR ALL
        USING (tenant_id = current_setting('app.current_tenant')::UUID);
    """))

    conn.execute(sa.text("""
        CREATE POLICY tenant_isolation ON tenant_configs
        FOR ALL
        USING (tenant_id = current_setting('app.current_tenant')::UUID);
    """))

    # Create additional indexes for performance
    op.create_index("idx_users_tenant_email", "users", ["tenant_id", "email"])
    op.create_index("idx_api_keys_key_hash", "api_keys", ["key_hash"])
    op.create_index("idx_api_keys_tenant_name", "api_keys", ["tenant_id", "name"])


def downgrade() -> None:
    conn = op.get_bind()

    # Drop policies
    conn.execute(sa.text("DROP POLICY IF EXISTS tenant_isolation ON tenant_configs;"))
    conn.execute(sa.text("DROP POLICY IF EXISTS tenant_isolation ON api_keys;"))
    conn.execute(sa.text("DROP POLICY IF EXISTS tenant_isolation ON users;"))
    conn.execute(sa.text("DROP POLICY IF EXISTS tenant_isolation ON tenants;"))

    # Disable RLS
    conn.execute(sa.text("ALTER TABLE tenant_configs DISABLE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE api_keys DISABLE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE users DISABLE ROW LEVEL SECURITY;"))
    conn.execute(sa.text("ALTER TABLE tenants DISABLE ROW LEVEL SECURITY;"))

    # Drop constraints and indexes
    op.drop_constraint("uq_users_tenant_email", "users", type_="unique")
    op.drop_constraint("uq_api_keys_key_hash", "api_keys", type_="unique")
    op.drop_constraint("uq_api_keys_tenant_name", "api_keys", type_="unique")
    op.drop_constraint("uq_tenant_configs_tenant_id", "tenant_configs", type_="unique")

    # Drop tables
    op.drop_table("tenant_configs")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("tenants")

    # Drop functions
    conn.execute(sa.text("DROP FUNCTION IF EXISTS set_tenant_context;"))