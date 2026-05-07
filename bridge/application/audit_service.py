"""Audit-Service: Use-Case-Wrapper fuer strukturiertes Command-Audit-Logging.

Stellt eine generische Funktion bereit, die der Presentation-Layer nutzt,
um Command- und Callback-Aktionen im Audit-Log zu erfassen.
Der LLM-Pfad (chat_service) loggt weiterhin direkt via write_audit_log.
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
    """Schreibt einen Audit-Eintrag fuer Commands und Callbacks.

    Args:
        action: Bezeichnung der Aktion (z.B. "remember", "forget", "bm_del").
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        username: Telegram Username (optional).
        entry_id: Betroffene Entry/Bookmark-ID (optional, wenn anwendbar).
        success: Ob die Aktion erfolgreich war.
        details: Zusaetzliche Info (optional).
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
