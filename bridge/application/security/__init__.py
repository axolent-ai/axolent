"""Security utilities for AXOLENT application layer.

Provides prompt-injection detection and secret scanning for user-supplied content.

Architecture note: env_scrubber lives in infrastructure/security/ because
it is used by infrastructure-layer modules (claude_process_pool, claude_cli).
Import it directly from infrastructure.security.env_scrubber if needed.
"""

from application.security.injection_detector import (
    InjectionDetector,
    InjectionMatch,
)
from application.security.input_normalizer import normalize_for_security_check
from application.security.prompt_delimiters import escape_prompt_delimited_text
from application.security.secret_scanner import (
    SecretBlockedError,
    SecretMatch,
    SecretScanner,
)

__all__ = [
    "InjectionDetector",
    "InjectionMatch",
    "SecretBlockedError",
    "SecretMatch",
    "SecretScanner",
    "escape_prompt_delimited_text",
    "normalize_for_security_check",
]
