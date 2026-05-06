"""Personality-Loader: Liest config/*.md und baut PersonalityConfig.

I/O-Adapter der Markdown-Dateien vom Dateisystem liest
und ein PersonalityConfig-Objekt aus dem Domain-Layer erzeugt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from domain.personality import PersonalityConfig
from infrastructure.encoding import open_utf8

log = logging.getLogger(__name__)

_CONFIG_DIR: Path = Path(__file__).resolve().parent.parent / "config"
_SYSTEM_PROMPT_PATH: Path = _CONFIG_DIR / "system_prompt.md"
_CONSTITUTION_PATH: Path = _CONFIG_DIR / "user_constitution.md"


def load_system_prompt() -> str:
    """Lädt den System-Prompt aus config/system_prompt.md.

    Returns:
        Inhalt der Datei als String, oder leerer String bei Fehler.
    """
    try:
        with open_utf8(_SYSTEM_PROMPT_PATH, "r") as f:
            content = f.read().strip()
        log.info(
            "System-Prompt geladen: %d Zeichen aus %s",
            len(content),
            _SYSTEM_PROMPT_PATH,
        )
        return content
    except FileNotFoundError:
        log.warning(
            "System-Prompt nicht gefunden: %s (Fallback: leer)", _SYSTEM_PROMPT_PATH
        )
        return ""
    except Exception as e:
        log.warning("Fehler beim Laden des System-Prompts: %s", e)
        return ""


def load_user_constitution() -> str:
    """Lädt die User-Constitution aus config/user_constitution.md.

    Returns:
        Inhalt der Datei als String, oder leerer String bei Fehler.
    """
    try:
        with open_utf8(_CONSTITUTION_PATH, "r") as f:
            content = f.read().strip()
        log.info(
            "User-Constitution geladen: %d Zeichen aus %s",
            len(content),
            _CONSTITUTION_PATH,
        )
        return content
    except FileNotFoundError:
        log.warning(
            "User-Constitution nicht gefunden: %s (Fallback: leer)", _CONSTITUTION_PATH
        )
        return ""
    except Exception as e:
        log.warning("Fehler beim Laden der User-Constitution: %s", e)
        return ""


def build_combined_prompt() -> str:
    """Lädt beide Config-Dateien und kombiniert sie.

    Convenience-Wrapper: Liest System-Prompt und Constitution,
    baut PersonalityConfig, gibt kombinierten Prompt zurück.

    Returns:
        Kombinierter Prompt-String für --append-system-prompt.
    """
    system = load_system_prompt()
    constitution = load_user_constitution()
    config = PersonalityConfig(system_prompt=system, user_constitution=constitution)
    return config.build_combined_prompt()
