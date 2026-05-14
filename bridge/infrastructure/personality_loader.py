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
_SYSTEM_PROMPT_EXAMPLE_PATH: Path = _CONFIG_DIR / "system_prompt.example.md"
_CONSTITUTION_PATH: Path = _CONFIG_DIR / "user_constitution.md"
_CONSTITUTION_EXAMPLE_PATH: Path = _CONFIG_DIR / "user_constitution.example.md"


def _load_config_file(path: Path, label: str, fallback_path: Path | None = None) -> str:
    """Lädt eine Config-Datei als UTF-8-String mit optionalem Fallback.

    Versucht zuerst ``path`` zu lesen (User-Override). Existiert diese
    Datei nicht und ``fallback_path`` ist angegeben, wird die Fallback-Datei
    geladen (generisches Example-Template aus dem Repo).

    Args:
        path: Primärer Pfad zur Config-Datei (User-Override, ggf. gitignored).
        label: Beschreibung der Datei für Logs (z.B. "System-Prompt").
        fallback_path: Optionaler Fallback-Pfad (z.B. .example.md im Repo).

    Returns:
        Inhalt der Datei als String, oder leerer String wenn weder Primär-
        noch Fallback-Datei gefunden wurde.

    Raises:
        SystemExit: Bei kritischen Fehlern (Encoding, Permissions, OS-Fehler).
    """
    try:
        with open_utf8(path, "r") as f:
            content = f.read().strip()
        log.info("%s geladen: %d Zeichen aus %s", label, len(content), path)
        return content
    except FileNotFoundError:
        if fallback_path is not None:
            log.info(
                "%s nicht gefunden: %s, versuche Fallback: %s",
                label,
                path,
                fallback_path,
            )
            return _load_config_file(fallback_path, f"{label} (example)")
        log.warning("%s nicht gefunden: %s (Fallback: leer)", label, path)
        return ""
    except (PermissionError, UnicodeDecodeError, OSError) as e:
        log.error("%s konnte nicht gelesen werden: %s: %s", label, path, e)
        raise SystemExit(
            f"{label}-Datei {path} fehlerhaft. Bot-Start abgebrochen. "
            f"Prüfe Datei-Encoding (UTF-8) und -Rechte. Original-Fehler: {e}"
        ) from e


def load_system_prompt() -> str:
    """Lädt den System-Prompt aus config/system_prompt.md.

    Fallback: config/system_prompt.example.md (generisches Template).
    Der User kann system_prompt.md als persönlichen Override anlegen
    (diese Datei ist in .gitignore und wird nicht ins Repo committed).

    Returns:
        Inhalt der Datei als String, oder leerer String wenn nicht vorhanden.

    Raises:
        SystemExit: Bei kritischen Datei-Fehlern (Encoding, Permissions).
    """
    return _load_config_file(
        _SYSTEM_PROMPT_PATH, "System-Prompt", _SYSTEM_PROMPT_EXAMPLE_PATH
    )


def load_user_constitution() -> str:
    """Lädt die User-Constitution aus config/user_constitution.md.

    Fallback: config/user_constitution.example.md (generisches Template).
    Der User kann user_constitution.md als persönlichen Override anlegen
    (diese Datei ist in .gitignore und wird nicht ins Repo committed).

    Returns:
        Inhalt der Datei als String, oder leerer String wenn nicht vorhanden.

    Raises:
        SystemExit: Bei kritischen Datei-Fehlern (Encoding, Permissions).
    """
    return _load_config_file(
        _CONSTITUTION_PATH, "User-Constitution", _CONSTITUTION_EXAMPLE_PATH
    )


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
