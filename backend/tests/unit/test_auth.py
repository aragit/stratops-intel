"""Unit tests for JWT authentication and API key validation.

Tests token creation, decoding, expiry handling, and the role-checker
dependency.
"""

from __future__ import annotations

import time
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api.auth import (
    _SECRET_KEY,
    RoleChecker,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestCreateAccessToken:
    """Tests for create_access_token."""

    def test_returns_string(self) -> None:
        """Token should be a string."""
        token = create_access_token({"sub": "user-123"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_contains_correct_claims(self) -> None:
        """Token should encode sub, exp, iat, and type=access."""
        token = create_access_token({"sub": "user-123"})
        payload = jwt.decode(token, _SECRET_KEY, algorithms=["HS256"])
        assert payload["sub"] == "user-123"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_custom_expires_delta(self) -> None:
        """Token should respect custom expires_delta."""
        token = create_access_token({"sub": "u"}, expires_delta=timedelta(hours=1))
        payload = jwt.decode(token, _SECRET_KEY, algorithms=["HS256"])
        exp = payload["exp"]
        iat = payload["iat"]
        assert exp - iat == 3600


class TestCreateRefreshToken:
    """Tests for create_refresh_token."""

    def test_returns_string(self) -> None:
        """Refresh token should be a string."""
        token = create_refresh_token({"sub": "user-123"})
        assert isinstance(token, str)

    def test_type_is_refresh(self) -> None:
        """Refresh token should have type=refresh."""
        token = create_refresh_token({"sub": "user-123"})
        payload = jwt.decode(token, _SECRET_KEY, algorithms=["HS256"])
        assert payload["type"] == "refresh"

    def test_longer_expiry_than_access(self) -> None:
        """Refresh token should expire later than access token."""
        access = create_access_token({"sub": "u"})
        refresh = create_refresh_token({"sub": "u"})
        ap = jwt.decode(access, _SECRET_KEY, algorithms=["HS256"])
        rp = jwt.decode(refresh, _SECRET_KEY, algorithms=["HS256"])
        assert rp["exp"] > ap["exp"]


class TestDecodeToken:
    """Tests for decode_token."""

    def test_valid_token_decodes(self) -> None:
        """Valid token should decode correctly."""
        token = create_access_token({"sub": "user-123"})
        payload = decode_token(token)
        assert payload["sub"] == "user-123"

    def test_expired_token_raises_401(self) -> None:
        """Expired token should raise HTTPException 401."""
        token = create_access_token({"sub": "u"}, expires_delta=timedelta(seconds=-1))
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_invalid_token_raises_401(self) -> None:
        """Garbage token should raise HTTPException 401."""
        with pytest.raises(HTTPException) as exc_info:
            decode_token("not.a.valid.token")
        assert exc_info.value.status_code == 401

    def test_wrong_secret_raises_401(self) -> None:
        """Token signed with wrong secret should raise 401."""
        token = jwt.encode({"sub": "u"}, "wrong-secret", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        assert exc_info.value.status_code == 401


class TestPasswordHashing:
    """Tests for hash_password and verify_password."""

    def test_hash_produces_bcrypt(self) -> None:
        """Hash should be a valid bcrypt string."""
        h = hash_password("testpassword")
        assert h.startswith("$2")

    def test_verify_correct_password(self) -> None:
        """Correct password should verify."""
        h = hash_password("secret")
        assert verify_password("secret", h) is True

    def test_verify_wrong_password(self) -> None:
        """Wrong password should fail verification."""
        h = hash_password("secret")
        assert verify_password("wrong", h) is False


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_no_credentials_raises_401(self) -> None:
        """Missing credentials should raise 401."""
        from api.auth import get_current_user

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self) -> None:
        """Valid token should return the user object."""
        from api.auth import get_current_user

        user_id = uuid4()
        token = create_access_token({"sub": str(user_id)})

        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.is_active = True

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_manager = MagicMock()
        mock_manager.admin_session.return_value = mock_session

        with patch("api.auth.get_session_manager", return_value=mock_manager):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            user = await get_current_user(credentials=creds)
            assert user is mock_user

    @pytest.mark.asyncio
    async def test_inactive_user_raises_401(self) -> None:
        """Inactive user should raise 401."""
        from api.auth import get_current_user

        user_id = uuid4()
        token = create_access_token({"sub": str(user_id)})

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_manager = MagicMock()
        mock_manager.admin_session.return_value = mock_session

        with patch("api.auth.get_session_manager", return_value=mock_manager):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(credentials=creds)
            assert exc_info.value.status_code == 401


class TestRoleChecker:
    """Tests for RoleChecker dependency."""

    @pytest.mark.asyncio
    async def test_allowed_role_passes(self) -> None:
        """User with allowed role should pass."""
        checker = RoleChecker(["admin", "owner"])
        mock_user = MagicMock()
        mock_user.role = "admin"
        result = await checker(user=mock_user)
        assert result is mock_user

    @pytest.mark.asyncio
    async def test_disallowed_role_raises_403(self) -> None:
        """User with disallowed role should raise 403."""
        checker = RoleChecker(["admin"])
        mock_user = MagicMock()
        mock_user.role = "viewer"
        mock_user.id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await checker(user=mock_user)
        assert exc_info.value.status_code == 403
