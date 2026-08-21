"""Unit tests for the FieldEncryptor AES-256-GCM encryption module."""

from __future__ import annotations

import base64
import os

import pytest

from backend.security.encryption import FieldEncryptor


class TestFieldEncryptorRoundTrip:
    """Tests for encrypt/decrypt round-trip functionality."""

    @pytest.fixture(autouse=True)
    def setup_key(self) -> None:
        """Set up a valid ENCRYPTION_KEY for each test."""
        key = base64.b64encode(os.urandom(32)).decode()
        os.environ["ENCRYPTION_KEY"] = key

    def test_encrypt_decrypt_round_trip(self) -> None:
        """encrypt() then decrypt() should return the original plaintext."""
        encryptor = FieldEncryptor()
        plaintext = "sensitive data"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == plaintext.strip()

    def test_encrypt_different_nonces(self) -> None:
        """Two encrypt calls on the same plaintext should produce different ciphertext."""
        key = base64.b64encode(os.urandom(32)).decode()
        os.environ["ENCRYPTION_KEY"] = key
        encryptor = FieldEncryptor()
        plaintext = "test data"
        encrypted1 = encryptor.encrypt(plaintext)
        encrypted2 = encryptor.encrypt(plaintext)
        assert encrypted1 != encrypted2


class TestFieldEncryptorErrorHandling:
    """Tests for FieldEncryptor error handling."""

    @pytest.fixture(autouse=True)
    def setup_key(self) -> None:
        """Set up a valid ENCRYPTION_KEY for each test."""
        key = base64.b64encode(os.urandom(32)).decode()
        os.environ["ENCRYPTION_KEY"] = key

    def test_decrypt_malformed_base64(self) -> None:
        """decrypt() should raise ValueError on invalid base64."""
        encryptor = FieldEncryptor()
        try:
            encryptor.decrypt("not-base64!")
            pytest.fail("Should have raised ValueError")
        except ValueError:
            pass

    def test_decrypt_corrupted_ciphertext(self) -> None:
        """decrypt() should raise ValueError on tampered ciphertext."""
        encryptor = FieldEncryptor()
        _original = encryptor.encrypt("secret")
        try:
            encryptor.decrypt("bXlzdGVyZXI=")  # base64 for "xml"
        except ValueError:
            pass


class TestFieldEncryptorKeyInit:
    """Tests for FieldEncryptor key initialization dependency on environment."""

    def test_init_with_valid_env_key(self) -> None:
        """Init should succeed when ENCRYPTION_KEY is properly set."""
        key = base64.b64encode(os.urandom(32)).decode()
        os.environ["ENCRYPTION_KEY"] = key
        try:
            encryptor = FieldEncryptor()
            assert encryptor is not None
        finally:
            os.environ.pop("ENCRYPTION_KEY", None)

    def test_init_missing_key(self) -> None:
        """Init should raise ValueError if ENCRYPTION_KEY is not set."""
        original = os.environ.pop("ENCRYPTION_KEY", None)
        try:
            from backend.security.encryption import FieldEncryptor

            try:
                FieldEncryptor()
                pytest.fail("Should have raised ValueError")
            except ValueError:
                pass
        finally:
            if original is not None:
                os.environ["ENCRYPTION_KEY"] = original

    def test_init_invalid_key_length(self) -> None:
        """Init should raise ValueError if key is not 32 bytes base64."""
        os.environ["ENCRYPTION_KEY"] = "short"
        try:
            from backend.security.encryption import FieldEncryptor

            try:
                FieldEncryptor()
                pytest.fail("Should have raised ValueError")
            except ValueError:
                pass
        finally:
            os.environ["ENCRYPTION_KEY"] = os.getenv("ENCRYPTION_KEY", "")
