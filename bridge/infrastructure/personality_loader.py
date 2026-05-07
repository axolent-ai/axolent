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


def _load_config_file(path: Path, label: str) -> str:
    """Lädt eine Config-Datei als UTF-8-String.

    Args:
        path: Pfad zur Config-Datei.
        label: Beschreibung der Datei für Logs (z.B. "System-Prompt").

    Returns:
        Inhalt der Datei als String, oder leerer String bei FileNotFoundError.

    Raises:
        SystemExit: Bei kritischen Fehlern (Encoding, Permissions, OS-Fehler).
    """
    try:
        with open_utf8(path, "r") as f:
            content = f.read().strip()
        log.info("%s geladen: %d Zeichen aus %s", label, len(content), path)
        return content
    except FileNotFoundError:
        log.warning("%s nicht gefunden: %s (Fallback: leer)", label, path)
        return ""
    except (PermissionError, UnicodeDecodeError, OSError) as e:
        log.error("%s konnte nicht gelesen werden: %s: %s", label, path, e)
        raise SystemExit(
            f"{label}-Datei {path} fehlerhaft. Bot-Start abgebrochen. "
            f"Pruefe Datei-Encoding (UTF-8) und -Rechte. Original-Fehler: {e}"
        ) from e


def load_system_prompt() -> str:
    """Lädt den System-Prompt aus config/system_prompt.md.

    Returns:
        Inhalt der Datei als String, oder leerer String wenn nicht vorhanden.

    Raises:
        SystemExit: Bei kritischen Datei-Fehlern (Encoding, Permissions).
    """
    return _load_config_file(_SYSTEM_PROMPT_PATH, "System-Prompt")


def load_user_constitution() -> str:
    """Lädt die User-Constitution aus config/user_constitution.md.

    Returns:
        Inhalt der Datei als String, oder leerer String wenn nicht vorhanden.

    Raises:
        SystemExit: Bei kritischen Datei-Fehlern (Encoding, Permissions).
    """
    return _load_config_file(_CONSTITUTION_PATH, "User-Constitution")


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
