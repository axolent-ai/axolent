"""Audit service: use-case wrapper for structured audit logging.

Provides generic functions that the presentation layer uses to capture
command, callback, and streaming actions in the audit log.
The LLM path (chat_service) continues to log directly via write_audit_log.

write_raw_audit() allows the presentation layer to write raw audit dicts
without importing infrastructure.audit_log directly
(layer contract: presentation must not import from infrastructure).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from infrastructure.audit_log import write_audit_log

log = logging.getLogger(__name__)

# Keys in task_meta that hold non-JSON-serializable objects (LanguageContext,
# StreamGuard, SkillMatch). Must be filtered before writing audit dicts.
# Shared between chat_service.save_streaming_result and handlers.py error paths.
AUDIT_NON_SERIALIZABLE_KEYS: frozenset[str] = frozenset(
    {"_skill_match", "_stream_guard", "_language_ctx"}
)


def filter_task_meta(task_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of task_meta with non-serializable keys removed.

    Safe for JSON serialization in audit log entries.
    """
    if not task_meta:
        return {}
    return {k: v for k, v in task_meta.items() if k not in AUDIT_NON_SERIALIZABLE_KEYS}


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
    """Write an audit entry for commands and callbacks.

    Args:
        action: Action identifier (e.g. "remember", "forget", "bm_del").
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.
        username: Telegram username (optional).
        entry_id: Affected entry/bookmark ID (optional, if applicable).
        success: Whether the action succeeded.
        details: Additional info (optional).
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
    """Write a raw audit dict to the audit log.

    Allows the presentation layer to write structured audit entries
    without importing infrastructure.audit_log directly.

    Args:
        entry: Dictionary with audit data (timestamp, event_type, etc.).
    """
    write_audit_log(entry)
