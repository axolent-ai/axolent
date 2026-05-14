"""Personality loader: reads config/*.md and builds PersonalityConfig.

I/O adapter that reads markdown files from the filesystem
and produces a PersonalityConfig object from the domain layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from domain.personality import PersonalityConfig
from infrastructure.encoding import open_utf8

log = logging.getLogger(__name__)

_CONFIG_DIR: Path = Path(__file__).resolve().parent.parent / "config"
_SYSTEM_PROMPT_PATH: Path = _CONFIG_DIR / "system_prompt.md"
_SYSTEM_PROMPT_EXAMPLE_PATH: Path = _CONFIG_DIR / "system_prompt.example.md"
_CONSTITUTION_PATH: Path = _CONFIG_DIR / "user_constitution.md"
_CONSTITUTION_EXAMPLE_PATH: Path = _CONFIG_DIR / "user_constitution.example.md"


def _load_config_file(path: Path, label: str, fallback_path: Path | None = None) -> str:
    """Load a config file as UTF-8 string with optional fallback.

    Tries to read ``path`` first (user override). If that file does not
    exist and ``fallback_path`` is given, the fallback file is loaded
    (generic example template from the repo).

    Args:
        path: Primary path to the config file (user override, possibly gitignored).
        label: Description of the file for logs (e.g. "system prompt").
        fallback_path: Optional fallback path (e.g. .example.md in the repo).

    Returns:
        File content as string, or empty string if neither primary
        nor fallback file was found.

    Raises:
        SystemExit: On critical errors (encoding, permissions, OS errors).
    """
    try:
        with open_utf8(path, "r") as f:
            content = f.read().strip()
        log.info("%s loaded: %d chars from %s", label, len(content), path)
        return content
    except FileNotFoundError:
        if fallback_path is not None:
            log.info(
                "%s not found: %s, trying fallback: %s",
                label,
                path,
                fallback_path,
            )
            return _load_config_file(fallback_path, f"{label} (example)")
        log.warning("%s not found: %s (fallback: empty)", label, path)
        return ""
    except (PermissionError, UnicodeDecodeError, OSError) as e:
        log.error("%s could not be read: %s: %s", label, path, e)
        raise SystemExit(
            f"{label} file {path} is corrupted. Bot start aborted. "
            f"Check file encoding (UTF-8) and permissions. Original error: {e}"
        ) from e


def load_system_prompt() -> str:
    """Load the system prompt from config/system_prompt.md.

    Fallback: config/system_prompt.example.md (generic template).
    The user can create system_prompt.md as a personal override
    (this file is in .gitignore and is not committed to the repo).

    Returns:
        File content as string, or empty string if not found.

    Raises:
        SystemExit: On critical file errors (encoding, permissions).
    """
    return _load_config_file(
        _SYSTEM_PROMPT_PATH, "System prompt", _SYSTEM_PROMPT_EXAMPLE_PATH
    )


def load_user_constitution() -> str:
    """Load the user constitution from config/user_constitution.md.

    Fallback: config/user_constitution.example.md (generic template).
    The user can create user_constitution.md as a personal override
    (this file is in .gitignore and is not committed to the repo).

    Returns:
        File content as string, or empty string if not found.

    Raises:
        SystemExit: On critical file errors (encoding, permissions).
    """
    return _load_config_file(
        _CONSTITUTION_PATH, "User constitution", _CONSTITUTION_EXAMPLE_PATH
    )


def build_combined_prompt() -> str:
    """Load both config files and combine them.

    Convenience wrapper: reads system prompt and constitution,
    builds PersonalityConfig, returns combined prompt.

    Returns:
        Combined prompt string for --append-system-prompt.
    """
    system = load_system_prompt()
    constitution = load_user_constitution()
    config = PersonalityConfig(system_prompt=system, user_constitution=constitution)
    return config.build_combined_prompt()
