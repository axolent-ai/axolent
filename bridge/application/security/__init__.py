"""Security utilities for AXOLENT application layer.

Provides prompt-injection detection for user-supplied content.

Architecture note: env_scrubber lives in infrastructure/security/ because
it is used by infrastructure-layer modules (claude_process_pool, claude_cli).
Import it directly from infrastructure.security.env_scrubber if needed.
"""

from application.security.injection_detector import (
    InjectionDetector,
    InjectionMatch,
)

__all__ = [
    "InjectionDetector",
    "InjectionMatch",
]
