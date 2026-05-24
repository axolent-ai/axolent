"""Infrastructure security utilities.

Contains subprocess environment scrubbing and other infrastructure-level
security primitives that cannot live in the application layer due to
hexagonal architecture constraints (infrastructure must not import application).
"""

from infrastructure.security.env_scrubber import (
    build_scrubbed_env,
    CLAUDE_SUBPROCESS_ENV_ALLOWLIST,
)

__all__ = [
    "build_scrubbed_env",
    "CLAUDE_SUBPROCESS_ENV_ALLOWLIST",
]
