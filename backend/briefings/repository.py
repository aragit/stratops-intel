"""Briefing repository with RLS enforcement.

Provides CRUD operations for briefings with tenant isolation.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.db.models import BriefingModel


class BriefingRepository:
    """Repository for briefing operations with RLS enforcement.

    All methods require a tenant-scoped session to enforce RLS.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with tenant-scoped session.

        Args:
            session: AsyncSession bound to a specific tenant via RLS context
        """
        self.session = session

    async def create(self, briefing: Any) -> Any:
        """Create a new briefing.

        Args:
            briefing: Briefing object with tenant_id, title, content_md_uri, etc.

        Returns:
            Created BriefingModel
        """
        model = BriefingModel(
            tenant_id=briefing.tenant_id,
            title=briefing.title,
            content_md_uri=briefing.content_md_uri,
            version=briefing.version,
            is_current=briefing.is_current,
            generated_by=briefing.metadata.get("generated_by") if briefing.metadata else None,
            metadata=briefing.metadata,
        )
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return model

    async def get_current(self, tenant_id: UUID, title: str) -> BriefingModel | None:
        """Get the current (is_current=True) version of a briefing by title.

        Args:
            tenant_id: Tenant ID (RLS context must match)
            title: Briefing title

        Returns:
            Current BriefingModel or None
        """
        stmt = (
            select(BriefingModel)
            .where(BriefingModel.title == title)
            .where(BriefingModel.is_current.is_(True))
            .options(selectinload(BriefingModel.tenant))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_version(self, tenant_id: UUID, title: str, version: int) -> BriefingModel | None:
        """Get a specific version of a briefing by title.

        Args:
            tenant_id: Tenant ID (RLS context must match)
            title: Briefing title
            version: Version number

        Returns:
            BriefingModel or None
        """
        stmt = (
            select(BriefingModel)
            .where(BriefingModel.title == title)
            .where(BriefingModel.version == version)
            .options(selectinload(BriefingModel.tenant))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: UUID,
        limit: int = 50,
        offset: int = 0,
        is_current_only: bool = False,
    ) -> list[Any]:
        """List briefings for a tenant with pagination.

        Args:
            tenant_id: Tenant ID (RLS context must match)
            limit: Maximum number of results
            offset: Pagination offset
            is_current_only: If True, only return current versions

        Returns:
            List of BriefingModel objects
        """
        stmt = select(BriefingModel).order_by(BriefingModel.created_at.desc())

        if is_current_only:
            stmt = stmt.where(BriefingModel.is_current.is_(True))

        stmt = stmt.limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_versions(self, tenant_id: UUID, title: str) -> list[Any]:
        """List all versions of a briefing by title.

        Args:
            tenant_id: Tenant ID (RLS context must match)
            title: Briefing title

        Returns:
            List of BriefingModel ordered by version DESC
        """
        stmt = (
            select(BriefingModel)
            .where(BriefingModel.title == title)
            .order_by(BriefingModel.version.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def mark_superseded(self, tenant_id: UUID, title: str, version: int) -> None:
        """Mark a specific version as superseded (is_current = False).

        Args:
            tenant_id: Tenant ID (RLS context must match)
            title: Briefing title
            version: Version to mark as superseded
        """
        stmt = (
            update(BriefingModel)
            .where(BriefingModel.title == title)
            .where(BriefingModel.version == version)
            .values(is_current=False, updated_at=func.now())
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def set_current_version(self, tenant_id: UUID, title: str, version: int) -> None:
        """Mark a specific version as current, superseding others.

        Args:
            tenant_id: Tenant ID (RLS context must match)
            title: Briefing title
            version: Version to set as current
        """
        # First, mark all other versions as not current
        await self.session.execute(
            update(BriefingModel)
            .where(BriefingModel.title == title)
            .values(is_current=False, updated_at=func.now())
        )

        # Then mark the target version as current
        await self.session.execute(
            update(BriefingModel)
            .where(BriefingModel.title == title)
            .where(BriefingModel.version == version)
            .values(is_current=True, updated_at=func.now())
        )
        await self.session.flush()
