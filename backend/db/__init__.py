"""StratOps Intel Backend Database Package.

Exports the tenant-aware session manager, database initialisation helpers,
and FastAPI dependencies for database access.
"""

__version__ = "0.1.0"

from db.dependencies import get_admin_db, get_db, verify_api_key
from db.tenant_session import (
    TenantSessionManager,
    close_database,
    get_admin_session,
    get_database_url,
    get_session_manager,
    get_tenant_session,
    initialize_database,
)

__all__ = [
    "TenantSessionManager",
    "close_database",
    "get_admin_db",
    "get_admin_session",
    "get_database_url",
    "get_db",
    "get_session_manager",
    "get_tenant_session",
    "initialize_database",
    "verify_api_key",
]
