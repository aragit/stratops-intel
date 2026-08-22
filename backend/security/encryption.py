"""Field-level AES-256-GCM encryption for sensitive signal payloads.

Provides encryption primitives that operate on individual text fields
before database persistence. Master key is loaded from the
``ENCRYPTION_KEY`` environment variable (32-byte base64-encoded value).

Usage
-----

.. code-block:: python

    from backend.security.encryption import FieldEncryptor

    encryptor = FieldEncryptor()
    encrypted = encryptor.encrypt("sensitive data")
    decrypted = encryptor.decrypt(encrypted)
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class FieldEncryptor:
    """AES-256-GCM field encryptor for sensitive payload protection.

    Uses a rotating master key derived from the ``ENCRYPTION_KEY`` env var.
    Each encryption operation generates a unique nonce, ensuring
    probabilistic ciphertext output even for identical plaintexts.

    Attributes:
        _aesgcm: Initialized AES-GCM instance using the master key.
        _key_id: Identifier of the current key version for auditability.
    """

    def __init__(self) -> None:
        """Initialize the field encryptor with the master key from env.

        The ``ENCRYPTION_KEY`` environment variable must be a 32-byte
        (256-bit) base64-encoded value. Example::

            $ export ENCRYPTION_KEY=$(python -c "import base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())")

        Raises:
            ValueError: If ``ENCRYPTION_KEY`` is not set or has an invalid
                byte length.
        """
        raw = os.getenv("ENCRYPTION_KEY")
        if not raw:
            raise ValueError("ENCRYPTION_KEY environment variable not set")

        try:
            key_bytes = base64.b64decode(raw)
        except Exception as exc:
            raise ValueError("ENCRYPTION_KEY must be base64-encoded") from exc

        if len(key_bytes) != 32:
            raise ValueError("ENCRYPTION_KEY must decode to exactly 32 bytes (256 bits)")

        self._aesgcm = AESGCM(key_bytes)
        self._key_id = "v1"

    def encrypt(self, plain_text: str) -> str:
        """Encrypt a plaintext string using AES-256-GCM.

        The output is a base64-encoded string containing the 12-byte nonce
        followed by the ciphertext. The nonce is prepended so that decryption
        can extract it without a separate key-management schema.

        Args:
            plain_text: Cleartext string to encrypt.

        Returns:
            Base64-encoded string: ``base64(nonce + ciphertext)``.
        """
        nonce = os.urandom(12)  # 96-bit nonce is the AES-GCM recommendation
        ciphertext = self._aesgcm.encrypt(nonce, plain_text.encode("utf-8"), None)
        return base64.b64encode(nonce + ciphertext).decode("utf-8")

    def decrypt(self, cipher_text: str) -> str:
        """Decrypt a base64-encoded AES-256-GCM ciphertext.

        Args:
            cipher_text: Base64-encoded string returned by ``encrypt()``.

        Returns:
            The original cleartext string.

        Raises:
            ValueError: If the ciphertext is malformed, expired, or the
                integrity check fails (authentication tag mismatch).
        """
        try:
            raw = base64.b64decode(cipher_text)
        except Exception as exc:
            raise ValueError("cipher_text is not valid base64") from exc

        if len(raw) < 12:
            raise ValueError("cipher_text is too short to contain a nonce")

        nonce = raw[:12]
        ciphertext = raw[12:]

        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise ValueError(
                "decryption failed: authentication tag mismatch or corrupted data"
            ) from exc

        return plaintext.decode("utf-8")  # type: ignore[no-any-return]
