"""Re-export stub: SecretScanner now lives in application.security.

This file preserves backward compatibility for existing imports from
the skill_compression.privacy package (e.g. PrivacyPipeline).

The canonical location is application.security.secret_scanner.
"""

from application.security.secret_scanner import (  # noqa: F401
    ALLOWED_CLAIM_PATTERNS,
    SECRET_PATTERNS,
    SecretBlockedError,
    SecretMatch,
    SecretPattern,
    SecretScanner,
)

__all__ = [
    "ALLOWED_CLAIM_PATTERNS",
    "SECRET_PATTERNS",
    "SecretBlockedError",
    "SecretMatch",
    "SecretPattern",
    "SecretScanner",
]
