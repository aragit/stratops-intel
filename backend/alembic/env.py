"""Alembic environment configuration for async SQLAlchemy."""

import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Get database URL from environment."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://stratops:stratops_dev_password@localhost:5432/stratops",
    )
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    return database_url


def do_run_migrations(connection: Connection) -> None:
    """Run migrations on the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=False,
        render_as_batch=False,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online_async() -> None:
    """Run migrations asynchronously using asyncpg driver."""
    connectable = create_async_engine(
        get_url(),
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        echo=False,
        connect_args={
            "server_settings": {
                "application_name": "alembic-migrator",
                "jit": "off",
            },
        },
    )

    async with connectable.connect() as connection:
        await connection.execute(
            text("SET app.current_tenant = '00000000-0000-0000-0000-000000000000'")
        )
        await connection.commit()
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_migrations_online_async())
