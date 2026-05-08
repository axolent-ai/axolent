"""Audit-Service: Use-Case-Wrapper für strukturiertes Audit-Logging.

Stellt generische Funktionen bereit, die der Presentation-Layer nutzt,
um Command-, Callback- und Streaming-Aktionen im Audit-Log zu erfassen.
Der LLM-Pfad (chat_service) loggt weiterhin direkt via write_audit_log.

write_raw_audit() erlaubt dem Presentation-Layer, rohe Audit-Dicts zu
schreiben, ohne direkt auf infrastructure.audit_log zuzugreifen
(Layer-Contract: presentation darf nicht direkt aus infrastructure importieren).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from infrastructure.audit_log import write_audit_log

log = logging.getLogger(__name__)


def log_command_audit(
    *,
    action: str,
    user_id: int,
    chat_id: int,
    username: Optional[str] = None,
    entry_id: Optional[str] = None,
    success: bool = True,
    details: Optional[str] = None,
) -> None:
    """Schreibt einen Audit-Eintrag für Commands und Callbacks.

    Args:
        action: Bezeichnung der Aktion (z.B. "remember", "forget", "bm_del").
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        username: Telegram Username (optional).
        entry_id: Betroffene Entry/Bookmark-ID (optional, wenn anwendbar).
        success: Ob die Aktion erfolgreich war.
        details: Zusätzliche Info (optional).
    """
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "command",
        "action": action,
        "user_id": user_id,
        "chat_id": chat_id,
        "success": success,
    }
    if username is not None:
        entry["username"] = username
    if entry_id is not None:
        entry["entry_id"] = entry_id
    if details is not None:
        entry["details"] = details

    write_audit_log(entry)


def write_raw_audit(entry: dict[str, Any]) -> None:
    """Schreibt ein rohes Audit-Dict ins Audit-Log.

    Erlaubt dem Presentation-Layer, strukturierte Audit-Einträge zu
    schreiben, ohne direkt auf infrastructure.audit_log zuzugreifen.

    Args:
        entry: Dictionary mit Audit-Daten (timestamp, event_type, etc.).
    """
    write_audit_log(entry)
