"""SQLAlchemy 2.0 declarative models for StratOps Intel.

This module defines the ORM models for the multi-tenant database schema.
All models use Mapped[] type annotations and SQLAlchemy 2.0 style.
"""

import enum
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class TenantTier(str, enum.Enum):
    """Tenant subscription tier enumeration."""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class UserRole(str, enum.Enum):
    """User role enumeration for RBAC."""

    OWNER = "owner"
    ADMIN = "admin"
    ANALYST = "analyst"
    MEMBER = "member"
    VIEWER = "viewer"


class Tenant(Base):
    """Tenant model for multi-tenancy.

    Represents a customer organization in the StratOps platform.
    Each tenant has its own isolated data via PostgreSQL RLS.

    Table: tenants
    """

    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    tier: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default=TenantTier.FREE.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    users: Mapped[list["User"]] = relationship(
        "User", back_populates="tenant", cascade="all, delete-orphan", lazy="selectin"
    )
    api_keys: Mapped[list["APIKey"]] = relationship(
        "APIKey", back_populates="tenant", cascade="all, delete-orphan", lazy="selectin"
    )
    tenant_configs: Mapped[list["TenantConfig"]] = relationship(
        "TenantConfig", back_populates="tenant", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_tenants_slug", "slug", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Tenant(id={self.id}, name={self.name!r}, slug={self.slug!r})>"


class User(Base):
    """User model with tenant association.

    Represents a user account within a tenant. Users are the primary
    actors in the system and are always scoped to a single tenant
    via RLS.

    Table: users
    """

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default=UserRole.MEMBER.value
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="users", lazy="selectin")
    api_keys: Mapped[list["APIKey"]] = relationship(
        "APIKey", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
        Index("ix_users_tenant_id", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email!r}, tenant_id={self.tenant_id})>"


class APIKey(Base):
    """API key model for programmatic access.

    Stores hashed API keys associated with a tenant and optionally
    a user. Keys are scoped per-tenant via RLS.

    Table: api_keys
    """

    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    scopes: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="api_keys", lazy="selectin")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="api_keys", lazy="selectin")

    __table_args__ = (
        UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
        UniqueConstraint("tenant_id", "name", name="uq_api_keys_tenant_name"),
        Index("ix_api_keys_tenant_id", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name!r})>"


class TenantConfig(Base):
    """Tenant-specific configuration model.

    Stores arbitrary JSON configuration for each tenant.

    Table: tenant_configs
    """

    __tablename__ = "tenant_configs"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="tenant_configs", lazy="selectin")

    __table_args__ = (
        UniqueConstraint("tenant_id", name="uq_tenant_configs_tenant_id"),
        Index("ix_tenant_configs_tenant_id", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"<TenantConfig(id={self.id}, tenant_id={self.tenant_id})>"


class Signal(Base):
    """Signal model for ingested structured data.

    Represents a normalized signal from any source adapter. Raw content
    is stored in MinIO/S3 (pointer-only), while structured metadata and
    fingerprints are stored in PostgreSQL with RLS enforcement.

    Table: signals
    """

    __tablename__ = "signals"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    content_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    structured_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    tenant: Mapped["Tenant"] = relationship("Tenant", lazy="selectin")

    __table_args__ = (
        UniqueConstraint("tenant_id", "fingerprint", name="uq_signals_tenant_fingerprint"),
        Index("ix_signals_tenant_id", "tenant_id"),
        Index("ix_signals_source_type", "source_type"),
        Index("ix_signals_collected_at", "collected_at"),
    )

    def __repr__(self) -> str:
        return f"<Signal(id={self.id}, source_type={self.source_type!r}, tenant_id={self.tenant_id})>"