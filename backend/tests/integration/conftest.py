"""Shared fixtures for integration tests.

Uses testcontainers for PostgreSQL and Redis, and provides helper
fixtures for creating tenants, users, and API keys in a real database.

Creates a non-superuser role ('testuser') so that RLS is actually enforced.
Superusers bypass RLS even with FORCE ROW LEVEL SECURITY. Fixture setup/teardown
uses the admin (superuser) engine to bypass RLS; only test assertions use testuser.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import redis.asyncio as aioredis
import respx
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool


# Lazy import for aiobotocore to avoid OpenSSL conflicts
def _get_aiobotocore():
    try:
        import aiobotocore.session

        return aiobotocore.session, True
    except (ImportError, AttributeError) as e:
        if "GEN_EMAIL" in str(e):
            return None, False
        return None, False


AIOBOTOCORE_AVAILABLE = False

try:
    from testcontainers.community.postgres import PostgresContainer
except ImportError:
    from testcontainers.postgres import PostgresContainer

try:
    from testcontainers.community.redis import RedisContainer
except ImportError:
    from testcontainers.redis import RedisContainer

try:
    from testcontainers.neo4j import Neo4jContainer
except ImportError:
    Neo4jContainer = None

try:
    from testcontainers.minio import MinioContainer
except ImportError:
    MinioContainer = None

from backend.db.models import APIKey, Base, Tenant, User  # noqa: E402
from backend.db.neo4j_client import Neo4jClient  # noqa: E402
from backend.db.tenant_session import TenantSessionManager  # noqa: E402

logger = structlog.get_logger(__name__)


@pytest.fixture(scope="module")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Start a PostgreSQL testcontainer for integration tests."""
    with PostgresContainer("pgvector/pgvector:pg16") as postgres:
        yield postgres


@pytest.fixture(scope="module")
def redis_container() -> Generator[RedisContainer, None, None]:
    """Start a Redis testcontainer for integration tests."""
    with RedisContainer("redis:7-alpine") as redis_c:
        yield redis_c


def _to_asyncpg_url(database_url: str) -> str:
    """Convert a PostgreSQL URL to use the asyncpg driver."""
    if "+asyncpg" not in database_url:
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        database_url = database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    return database_url


@pytest.fixture(scope="module")
async def integration_engines(
    postgres_container: PostgresContainer,
) -> AsyncGenerator[tuple[AsyncEngine, AsyncEngine], None]:
    """Create admin and test async engines.

    The admin engine connects as the superuser for fixture setup/teardown
    (bypassing RLS). The test engine connects as a non-superuser so RLS
    is enforced during test assertions.
    """
    superuser_url = _to_asyncpg_url(postgres_container.get_connection_url())

    admin_engine = create_async_engine(
        superuser_url,
        echo=False,
        poolclass=NullPool,
        connect_args={"server_settings": {"application_name": "integration-admin"}},
    )

    async with admin_engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(
            text(
                "DO $$ BEGIN "
                "IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'testuser') THEN "
                "CREATE ROLE testuser LOGIN PASSWORD 'testpass'; "
                "END IF; "
                "END $$;"
            )
        )

    async with admin_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with admin_engine.begin() as conn:
        tables = ["tenant_configs", "api_keys", "users", "tenants"]
        for table in tables:
            await conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;"))
            await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;"))
            await conn.execute(text(f"GRANT ALL ON TABLE {table} TO testuser;"))
            await conn.execute(
                text("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO testuser;")
            )

        await conn.execute(
            text("""
            CREATE POLICY tenant_isolation ON tenants
            FOR ALL USING (id = current_setting('app.current_tenant')::UUID);
        """)
        )
        await conn.execute(
            text("""
            CREATE POLICY tenant_isolation ON users
            FOR ALL USING (tenant_id = current_setting('app.current_tenant')::UUID);
        """)
        )
        await conn.execute(
            text("""
            CREATE POLICY tenant_isolation ON api_keys
            FOR ALL USING (tenant_id = current_setting('app.current_tenant')::UUID);
        """)
        )
        await conn.execute(
            text("""
            CREATE POLICY tenant_isolation ON tenant_configs
            FOR ALL USING (tenant_id = current_setting('app.current_tenant')::UUID);
        """)
        )

    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(superuser_url)
    testuser_url = urlunparse(
        parsed._replace(netloc=f"testuser:testpass@{parsed.hostname}:{parsed.port}")
    )

    test_engine = create_async_engine(
        testuser_url,
        echo=False,
        poolclass=NullPool,
        connect_args={
            "server_settings": {
                "application_name": "integration-test",
            },
        },
    )

    yield admin_engine, test_engine
    await test_engine.dispose()
    await admin_engine.dispose()


@pytest.fixture(scope="module")
async def integration_engine(integration_engines: tuple[AsyncEngine, AsyncEngine]) -> AsyncEngine:
    """The admin engine for fixture setup/teardown (bypasses RLS)."""
    return integration_engines[0]


@pytest.fixture(scope="module")
async def test_engine(integration_engines: tuple[AsyncEngine, AsyncEngine]) -> AsyncEngine:
    """The test engine with RLS enforced (non-superuser)."""
    return integration_engines[1]


@pytest.fixture(scope="module")
async def integration_redis(
    redis_container: RedisContainer,
) -> AsyncGenerator[aioredis.Redis, None]:
    """Create an async Redis client pointing to the testcontainer."""
    redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"
    client = aioredis.from_url(redis_url, decode_responses=True)
    yield client
    try:
        await client.aclose()
    except RuntimeError:
        pass  # Event loop may be closed during module-scoped teardown


@pytest.fixture(scope="module")
def minio_container() -> Generator[MinioContainer, None, None]:
    """Start a MinIO testcontainer for integration tests."""
    if MinioContainer is None:
        pytest.skip("testcontainers-minio not installed")
    container = MinioContainer(
        image="minio/minio:latest",
        access_key="minioadmin",
        secret_key="minioadmin",
    )
    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="module")
async def s3_client(minio_container) -> AsyncGenerator[Any, None]:
    """Create an async S3 client pointing to the test MinIO container."""

    # Lazy import to avoid OpenSSL conflict at module load time
    def _get_aiobotocore_session():
        try:
            import aiobotocore.session

            return aiobotocore.session, True
        except (ImportError, AttributeError) as e:
            if "GEN_EMAIL" in str(e):
                pytest.skip("aiobotocore OpenSSL conflict in test environment (known issue)")
            raise

    aiobotocore_session, available = _get_aiobotocore_session()
    if not available:
        pytest.skip("aiobotocore not available")

    endpoint = (
        f"http://{minio_container.get_container_host_ip()}:{minio_container.get_exposed_port(9000)}"
    )
    session = aiobotocore_session.get_session()
    try:
        async with session.create_client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            region_name="us-east-1",
        ) as client:
            yield client
    except AttributeError as e:
        if "GEN_EMAIL" in str(e):
            pytest.skip("aiobotocore OpenSSL conflict in test environment (known issue)")
        raise


@pytest.fixture
async def test_tenant_a(integration_engine: AsyncEngine) -> AsyncGenerator[dict, None]:
    """Create tenant A for isolation tests with a unique UUID per test."""
    tenant_id = uuid4()
    session_factory = async_sessionmaker(
        bind=integration_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        t = Tenant(
            id=tenant_id,
            name=f"Tenant A {tenant_id.hex[:8]}",
            slug=f"tenant-a-{tenant_id.hex[:8]}",
            tier="pro",
        )
        session.add(t)
        await session.commit()
    yield {
        "id": tenant_id,
        "name": f"Tenant A {tenant_id.hex[:8]}",
        "slug": f"tenant-a-{tenant_id.hex[:8]}",
        "tier": "pro",
    }
    async with session_factory() as session:
        await session.execute(
            text("DELETE FROM api_keys WHERE tenant_id = :tid"), {"tid": str(tenant_id)}
        )
        await session.execute(
            text("DELETE FROM users WHERE tenant_id = :tid"), {"tid": str(tenant_id)}
        )
        await session.execute(
            text("DELETE FROM tenant_configs WHERE tenant_id = :tid"), {"tid": str(tenant_id)}
        )
        await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": str(tenant_id)})
        await session.commit()


@pytest.fixture
async def test_tenant_b(integration_engine: AsyncEngine) -> AsyncGenerator[dict, None]:
    """Create tenant B for isolation tests with a unique UUID per test."""
    tenant_id = uuid4()
    session_factory = async_sessionmaker(
        bind=integration_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        t = Tenant(
            id=tenant_id,
            name=f"Tenant B {tenant_id.hex[:8]}",
            slug=f"tenant-b-{tenant_id.hex[:8]}",
            tier="free",
        )
        session.add(t)
        await session.commit()
    yield {
        "id": tenant_id,
        "name": f"Tenant B {tenant_id.hex[:8]}",
        "slug": f"tenant-b-{tenant_id.hex[:8]}",
        "tier": "free",
    }
    async with session_factory() as session:
        await session.execute(
            text("DELETE FROM api_keys WHERE tenant_id = :tid"), {"tid": str(tenant_id)}
        )
        await session.execute(
            text("DELETE FROM users WHERE tenant_id = :tid"), {"tid": str(tenant_id)}
        )
        await session.execute(
            text("DELETE FROM tenant_configs WHERE tenant_id = :tid"), {"tid": str(tenant_id)}
        )
        await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": str(tenant_id)})
        await session.commit()


@pytest.fixture
async def test_user_a(
    integration_engine: AsyncEngine, test_tenant_a: dict
) -> AsyncGenerator[dict, None]:
    """Create a user in tenant A."""
    from api.auth import hash_password

    user_id = uuid4()
    session_factory = async_sessionmaker(
        bind=integration_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        u = User(
            id=user_id,
            tenant_id=test_tenant_a["id"],
            email=f"user_a_{user_id.hex[:8]}@test.com",
            hashed_password=hash_password("password_a"),
            role="admin",
            is_active=True,
        )
        session.add(u)
        await session.commit()
    yield {
        "id": user_id,
        "email": f"user_a_{user_id.hex[:8]}@test.com",
        "tenant_id": test_tenant_a["id"],
    }


@pytest.fixture
async def test_user_b(
    integration_engine: AsyncEngine, test_tenant_b: dict
) -> AsyncGenerator[dict, None]:
    """Create a user in tenant B."""
    from api.auth import hash_password

    user_id = uuid4()
    session_factory = async_sessionmaker(
        bind=integration_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        u = User(
            id=user_id,
            tenant_id=test_tenant_b["id"],
            email=f"user_b_{user_id.hex[:8]}@test.com",
            hashed_password=hash_password("password_b"),
            role="analyst",
            is_active=True,
        )
        session.add(u)
        await session.commit()
    yield {
        "id": user_id,
        "email": f"user_b_{user_id.hex[:8]}@test.com",
        "tenant_id": test_tenant_b["id"],
    }


@pytest.fixture
async def test_api_key_a(
    integration_engine: AsyncEngine, test_tenant_a: dict, test_user_a: dict
) -> AsyncGenerator[dict, None]:
    """Create an API key for tenant A."""
    raw_key = f"so_test_{uuid4().hex}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    session_factory = async_sessionmaker(
        bind=integration_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        k = APIKey(
            tenant_id=test_tenant_a["id"],
            user_id=test_user_a["id"],
            key_hash=key_hash,
            name=f"test-key-{uuid4().hex[:8]}",
            scopes={"read": True, "write": True},
            is_active=True,
        )
        session.add(k)
        await session.commit()
    yield {"raw_key": raw_key, "key_hash": key_hash, "tenant_id": test_tenant_a["id"]}


@pytest.fixture
async def integration_gateway(
    integration_engine: AsyncEngine,
    integration_redis: aioredis.Redis,
):
    """Provide a patched FastAPI app and httpx AsyncClient for integration tests.

    Patches the global session manager to use the testcontainer engine so
    that gateway endpoints (login, /health/tenant, etc.) work without
    requiring the full lifespan.
    """
    from unittest.mock import MagicMock, patch

    from httpx import ASGITransport, AsyncClient

    manager = TenantSessionManager.__new__(TenantSessionManager)
    manager._engine = integration_engine
    manager._session_maker = async_sessionmaker(
        bind=integration_engine, class_=AsyncSession, expire_on_commit=False
    )
    manager._is_connected = True
    manager._initialized = True
    manager._database_url = str(integration_engine.url)
    manager._pool_size = 5
    manager._max_overflow = 5

    mock_redis = MagicMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()

    with (
        patch("db.tenant_session.get_session_manager", return_value=manager),
        patch("db.dependencies.get_session_manager", return_value=manager),
        patch("api.gateway.get_session_manager", return_value=manager),
        patch("api.auth.get_session_manager", return_value=manager),
        patch("api.gateway._redis_pool", mock_redis),
        patch("api.gateway.aioredis", MagicMock(return_value=mock_redis)),
    ):
        from api.gateway import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# === Week 4 fixtures ===


@pytest.fixture(scope="module")
def neo4j_container():
    """Neo4j test container."""
    if Neo4jContainer is None:
        pytest.skip("testcontainers-neo4j not installed")
    container = Neo4jContainer("neo4j:5-community")
    container.start()
    yield container
    container.stop()


@pytest.fixture
def neo4j_client(neo4j_container: Neo4jContainer):
    """Neo4j client for test container."""
    return Neo4jClient(
        uri=neo4j_container.get_url(),
        user="neo4j",
        password=neo4j_container.password,
    )


@pytest.fixture
def test_tenant(neo4j_client: Neo4jClient) -> str:
    """Create a test tenant in Neo4j."""
    tenant_id = str(uuid4())
    asyncio.get_event_loop().run_until_complete(
        neo4j_client.run("CREATE (t:Tenant {id: $id, name: 'Test Tenant'})", {"id": tenant_id})
    )
    return tenant_id


@pytest.fixture
def sample_signal_text() -> str:
    """Sample signal text for testing."""
    return (
        "Apple Inc. reported record quarterly revenue of $94.8 billion, "
        "up 2% year over year. CEO Tim Cook said the company is seeing "
        "strong demand for iPhone 15 and services. CFO Luca Maestri "
        "guided for $81-83 billion revenue next quarter. "
        "Microsoft reported Azure growth of 29% in constant currency. "
        "Satya Nadella emphasized AI-driven growth across cloud and productivity."
    )


@pytest.fixture
def mock_extraction_service():
    """Mock BentoML extraction service responses."""
    with respx.mock() as mock:
        mock.post("http://bentoml-extraction:3000/v1/extract").mock(
            return_value=type(
                "obj",
                (object,),
                {
                    "status_code": 200,
                    "json": lambda: [
                        {
                            "result": {
                                "entities": [
                                    {"company_name": "Apple Inc.", "ticker": "AAPL"},
                                    {"name": "Tim Cook", "role": "CEO"},
                                    {"company_name": "Microsoft", "ticker": "MSFT"},
                                    {"name": "Satya Nadella", "role": "CEO"},
                                ]
                            }
                        }
                    ],
                },
            )
        )
        yield mock


@pytest.fixture
def mock_summarization_service():
    """Mock BentoML summarization service responses."""
    with respx.mock() as mock:
        mock.post("http://bentoml-summarization:3000/summarize").mock(
            return_value=type(
                "obj",
                (object,),
                {
                    "status_code": 200,
                    "json": lambda: [
                        {
                            "summaries": [
                                "Apple reported record revenue driven by iPhone and services."
                            ],
                            "model": "test",
                            "batch_size": 1,
                            "total_tokens": 50,
                        }
                    ],
                },
            )
        )
        yield mock


@pytest.fixture
def mock_narrative_service():
    """Mock BentoML narrative service responses."""
    with respx.mock() as mock:
        mock.post("http://bentoml-narrative:3000/generate").mock(
            return_value=type(
                "obj",
                (object,),
                {
                    "status_code": 200,
                    "json": lambda: {
                        "narrative": "# Executive Brief\n\nApple reported record revenue of $94.8B driven by iPhone and services.\n\nKey developments:\n- iPhone 15 demand strong\n- Services revenue growing\n- Guidance conservative\n\nRecommended actions:\n- Monitor iPhone demand in China\n- Invest in AI services",
                        "key_takeaways": [
                            "Apple reported record $94.8B revenue",
                            "iPhone and services driving growth",
                            "Conservative guidance for next quarter",
                        ],
                        "confidence": 0.85,
                        "model": "test-model",
                    },
                },
            )
        )
        yield mock


@pytest.fixture
def mock_bentoml_summarization():
    """Mock BentoML summarization service."""
    with respx.mock() as mock:
        mock.post("http://bentoml-summarization:3000/summarize").mock(
            return_value=type(
                "obj",
                (object,),
                {
                    "status_code": 200,
                    "json": lambda: [
                        {
                            "summaries": [
                                "Apple reported record revenue driven by iPhone and services."
                            ],
                            "model": "test",
                            "batch_size": 1,
                            "total_tokens": 50,
                        }
                    ],
                },
            )
        )
        yield mock


@pytest.fixture
def mock_bentoml_narrative():
    """Mock BentoML narrative service."""
    with respx.mock() as mock:
        mock.post("http://bentoml-narrative:3000/generate").mock(
            return_value=type(
                "obj",
                (object,),
                {
                    "status_code": 200,
                    "json": lambda: {
                        "narrative": "# Executive Brief\n\nApple reported record revenue of $94.8B driven by iPhone and services.\n\nKey developments:\n- iPhone 15 demand strong\n- Services revenue growing\n- Guidance conservative\n\nRecommended actions:\n- Monitor iPhone demand in China\n- Invest in AI services",
                        "key_takeaways": [
                            "Apple reported record $94.8B revenue",
                            "iPhone and services driving growth",
                            "Conservative guidance for next quarter",
                        ],
                        "confidence": 0.85,
                        "model": "test-model",
                    },
                },
            )
        )
        yield mock
