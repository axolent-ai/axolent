"""Audit-Log-Writer mit Rotation.

Schreibt strukturierte Audit-Einträge als JSONL mit RotatingFileHandler.
Max 10 MB pro Datei, 5 Backups.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

AUDIT_LOG_PATH: Path = Path(__file__).resolve().parent.parent / "logs" / "audit.jsonl"
AUDIT_LOG_PATH.parent.mkdir(exist_ok=True)

# Dedizierter Logger für Audit-Einträge (nicht über root-logger)
_audit_logger = logging.getLogger("jarvis-bridge.audit")
_audit_logger.setLevel(logging.INFO)
_audit_handler = logging.handlers.RotatingFileHandler(
    AUDIT_LOG_PATH,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8",
)
_audit_handler.setFormatter(logging.Formatter("%(message)s"))
_audit_logger.addHandler(_audit_handler)
_audit_logger.propagate = False  # nicht über root-logger schicken


def write_audit_log(entry: dict[str, Any]) -> None:
    """Schreibt einen Audit-Log-Eintrag via RotatingFileHandler.

    Args:
        entry: Dictionary mit Audit-Daten (timestamp, user_id, etc.)
    """
    try:
        _audit_logger.info(json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        log.warning("Audit-Log Fehler: %s", e)
