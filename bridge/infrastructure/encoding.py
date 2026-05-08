"""Encoding-Helper für Jarvis-LITE.

Standard-Pattern für alle File-, Subprocess- und JSON-Operationen.
Erzwingt UTF-8 explizit, damit Multi-User Multi-Language Robustheit
ab Tag 1 garantiert ist.

Niemals direkt open() oder subprocess.run() in Jarvis-LITE-Code,
sondern diese Helper benutzen.
"""

from __future__ import annotations

import json
import logging
import subprocess  # nosec B404 - bewusster zentraler Wrapper, alle Calls hier kontrolliert
from pathlib import Path
from typing import Any, TextIO

log = logging.getLogger(__name__)


def open_utf8(path: str | Path, mode: str = "r", **kwargs) -> TextIO:
    """Öffnet eine Datei mit UTF-8-Encoding und replace-Errors.

    Args:
        path: Datei-Pfad (str oder Path).
        mode: Datei-Modus (r, w, a, etc.).
        **kwargs: weitere open()-Argumente.

    Returns:
        Datei-Handle mit UTF-8 / errors=replace gesetzt.
    """
    return open(path, mode, encoding="utf-8", errors="replace", **kwargs)


def run_subprocess_utf8(
    cmd: list[str],
    timeout: float | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Ruft ein externes Programm auf mit garantiertem UTF-8-Encoding.

    Args:
        cmd: Befehl als Liste (z.B. ["claude", "-p", "..."]).
        timeout: Timeout in Sekunden, None für unendlich.
        **kwargs: weitere subprocess.run()-Argumente.

    Returns:
        subprocess.CompletedProcess mit stdout/stderr als str (UTF-8 dekodiert).
    """
    return subprocess.run(  # nosec B603 - shell=False, kein Injection-Risiko
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **kwargs,
    )


def write_json_utf8(data: Any, path: str | Path, indent: int = 2) -> None:
    """Schreibt ein Python-Objekt als JSON-Datei in UTF-8.

    Verwendet ensure_ascii=False damit Unicode-Zeichen direkt im JSON
    landen, nicht escaped (\\u00e4 etc.).

    Args:
        data: serialisierbares Python-Objekt.
        path: Ziel-Pfad.
        indent: JSON-Einrückung (default 2).
    """
    with open_utf8(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_json_utf8(path: str | Path) -> Any:
    """Liest eine JSON-Datei in UTF-8.

    Args:
        path: Quell-Pfad.

    Returns:
        Deserialisiertes Python-Objekt.
    """
    with open_utf8(path, "r") as f:
        return json.load(f)


def append_jsonl_utf8(entry: dict, path: str | Path) -> None:
    """Hängt einen Eintrag als JSON-Line an eine Datei an.

    Verwendet UTF-8 ohne BOM, ensure_ascii=False für Unicode-Lesbarkeit.

    Args:
        entry: zu schreibendes Dict.
        path: Ziel-Datei.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open_utf8(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
