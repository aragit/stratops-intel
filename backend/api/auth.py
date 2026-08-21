"""JWT authentication and API key validation for the FastAPI gateway.

Provides token creation, decoding, and FastAPI dependencies for
extracting the current user from either a JWT bearer token or an
``x-api-key`` header.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

import bcrypt
import jwt
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from backend.db.dependencies import verify_api_key
from backend.db.models import User
from backend.db.tenant_session import get_session_manager

logger = structlog.get_logger(__name__)

_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "stratops-dev-secret-change-in-production")
_ALGORITHM: str = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_EXPIRE_MINUTES", "15"))
_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7"))

oauth2_scheme = HTTPBearer(auto_error=False)


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a short-lived JWT access token.

    Args:
        data: Claims to encode into the token. Must include ``sub``.
        expires_delta: Custom expiration window. Defaults to
            :data:`_ACCESS_TOKEN_EXPIRE_MINUTES`.

    Returns:
        The encoded JWT string.
    """
    to_encode = dict(data)
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.now(UTC), "type": "access"})
    token = jwt.encode(to_encode, _SECRET_KEY, algorithm=_ALGORITHM)
    logger.debug("access_token_created", sub=data.get("sub"), expires=expire.isoformat())
    return token


def create_refresh_token(data: dict) -> str:
    """Create a long-lived JWT refresh token.

    Args:
        data: Claims to encode. Must include ``sub``.

    Returns:
        The encoded JWT string.
    """
    to_encode = dict(data)
    expire = datetime.now(UTC) + timedelta(days=_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "iat": datetime.now(UTC), "type": "refresh"})
    token = jwt.encode(to_encode, _SECRET_KEY, algorithm=_ALGORITHM)
    logger.debug("refresh_token_created", sub=data.get("sub"), expires=expire.isoformat())
    return token


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: The raw JWT string.

    Returns:
        The decoded claims dictionary.

    Raises:
        HTTPException: 401 if the token is expired or invalid.
    """
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("token_expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jwt.InvalidTokenError as exc:
        logger.warning("invalid_token", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        The bcrypt hash string.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash.

    Args:
        plain_password: The plaintext password.
        hashed_password: The stored bcrypt hash.

    Returns:
        ``True`` if the password matches.
    """
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(oauth2_scheme),
) -> User:
    """FastAPI dependency: extract and validate the current user from a JWT.

    Args:
        credentials: The bearer credentials extracted by FastAPI.

    Returns:
        The :class:`User` instance matching the ``sub`` claim.

    Raises:
        HTTPException: 401 if the token is missing, invalid, or the user
            is not found.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    user_id_str: str | None = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    try:
        user_id = UUID(user_id_str)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid subject claim",
        ) from None

    manager = get_session_manager()
    try:
        async with manager.admin_session() as session:
            stmt = select(User).where(User.id == user_id, User.is_active.is_(True))
            result = await session.execute(stmt)
            user: User | None = result.scalar_one_or_none()
    except Exception as exc:
        logger.error("user_lookup_error", user_id=user_id_str, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error during user lookup",
        ) from exc

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


async def get_current_user_with_api_key(request: Request) -> User:
    """FastAPI dependency: try JWT first, fall back to x-api-key header.

    Attempts to authenticate via ``Authorization: Bearer <token>``. If
    that fails, validates the ``x-api-key`` header via
    :func:`~db.dependencies.verify_api_key` and loads the user.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The authenticated :class:`User`.

    Raises:
        HTTPException: 401 if both methods fail.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        try:
            return await get_current_user(
                HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            )
        except HTTPException:
            pass

    api_key = request.headers.get("x-api-key")
    if api_key:
        try:
            tenant_id, scopes = await verify_api_key(request)
            manager = get_session_manager()
            async with manager.admin_session() as session:
                stmt = select(User).where(
                    User.tenant_id == tenant_id,
                    User.is_active.is_(True),
                )
                result = await session.execute(stmt)
                user: User | None = result.scalars().first()
            if user is not None:
                return user
        except HTTPException:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (Bearer token or x-api-key)",
        headers={"WWW-Authenticate": "Bearer"},
    )


class RoleChecker:
    """FastAPI dependency factory for role-based access control.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(RoleChecker(["admin", "owner"]))])
        async def admin_endpoint(user: User = Depends(get_current_user)):
            ...
    """

    def __init__(self, allowed_roles: list[str]) -> None:
        """Initialise the role checker.

        Args:
            allowed_roles: List of role strings that are permitted.
        """
        self._allowed_roles = allowed_roles

    async def __call__(self, user: User = Depends(get_current_user)) -> User:
        """Check the user's role against the allowed list.

        Args:
            user: The authenticated user.

        Returns:
            The user if authorised.

        Raises:
            HTTPException: 403 if the user's role is not in the allowed list.
        """
        if user.role not in self._allowed_roles:
            logger.warning(
                "role_check_failed",
                user_id=str(user.id),
                user_role=user.role,
                allowed_roles=self._allowed_roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not in allowed roles: {self._allowed_roles}",
            )
        return user
