"""Environment variable scrubbing for Claude CLI subprocesses.

GAP-11 mitigation: Claude CLI subprocesses should NOT inherit the full
process environment. Only variables required for correct operation are
passed through via an allowlist.

This prevents leakage of:
  - TELEGRAM_BOT_TOKEN
  - SENTRY_DSN
  - Database credentials
  - Any other secrets loaded into the parent process environment
"""

from __future__ import annotations

import os
import sys


# Allowlist of environment variables that Claude CLI needs to function.
# Rationale for each group:
#
# PATH/HOME/USERPROFILE: Required for process execution and CLI binary discovery.
# APPDATA/LOCALAPPDATA: Windows credential store and config paths.
# TEMP/TMP: Temporary file operations.
# ANTHROPIC_API_KEY: Claude CLI authentication (if API-key mode is used).
# CLAUDE_*: Any Claude CLI configuration variables.
# LANG/LC_*/TZ: Locale and timezone (correct text handling).
# AXOLENT_TIMEZONE: Custom timezone config used in prompts.
# NO_COLOR/FORCE_COLOR: Terminal output control (set by our spawn code).
# PYTHONUNBUFFERED: Buffering control (set by our spawn code).
# NODE_OPTIONS: Node.js runtime flags (set by our spawn code).
# SYSTEMROOT/WINDIR: Windows system directory (required for subprocess on Windows).
# COMSPEC: Windows command interpreter path (required for some child processes).
# SSL_CERT_FILE/SSL_CERT_DIR/REQUESTS_CA_BUNDLE: TLS certificate paths.
# PROGRAMFILES/PROGRAMFILES(X86): Standard Windows paths for binary discovery.
# CommonProgramFiles: Windows common files path.

CLAUDE_SUBPROCESS_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Process execution essentials
        "PATH",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        # Windows system paths (required for subprocess execution)
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "CommonProgramFiles",
        # Claude CLI authentication and configuration
        "ANTHROPIC_API_KEY",
        # Locale and timezone
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "AXOLENT_TIMEZONE",
        # Terminal/buffering control (set by our spawn code)
        "NO_COLOR",
        "FORCE_COLOR",
        "PYTHONUNBUFFERED",
        "NODE_OPTIONS",
        # TLS certificate paths (required for HTTPS to Anthropic)
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "CURL_CA_BUNDLE",
    }
)

# Prefix allowlist: any env var starting with these prefixes is allowed.
# This covers CLAUDE_CONFIG_DIR, CLAUDE_MODEL, etc. without listing each one.
_ALLOWED_PREFIXES: tuple[str, ...] = (
    "CLAUDE_",
    "ANTHROPIC_",
)


def build_scrubbed_env(
    source_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a scrubbed environment dict for Claude CLI subprocess.

    Only passes through variables that are on the allowlist or match
    an allowed prefix. All other variables (including secrets like
    TELEGRAM_BOT_TOKEN, SENTRY_DSN, etc.) are excluded.

    Args:
        source_env: Source environment dict. Defaults to os.environ.

    Returns:
        Filtered environment dictionary safe for subprocess use.
    """
    env = source_env if source_env is not None else dict(os.environ)
    scrubbed: dict[str, str] = {}

    for key, value in env.items():
        # Exact match
        if (
            key.upper() in CLAUDE_SUBPROCESS_ENV_ALLOWLIST
            or key in CLAUDE_SUBPROCESS_ENV_ALLOWLIST
        ):
            scrubbed[key] = value
            continue

        # Prefix match (case-insensitive for robustness)
        key_upper = key.upper()
        if any(key_upper.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
            scrubbed[key] = value
            continue

    # Ensure critical Windows paths are always present on Windows
    if sys.platform == "win32":
        if "SYSTEMROOT" not in scrubbed:
            system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
            scrubbed["SYSTEMROOT"] = system_root
        if "COMSPEC" not in scrubbed:
            comspec = os.environ.get("COMSPEC", r"C:\Windows\system32\cmd.exe")
            scrubbed["COMSPEC"] = comspec

    return scrubbed
