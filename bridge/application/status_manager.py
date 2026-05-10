"""Status-Manager: koordiniert Status-Updates waehrend Processing.

Zeigt dem User was Jarvis gerade macht, statt nur "..." als Placeholder.
Sprach-aware (DE + EN basierend auf Sticky-Language).
Rate-limited: max alle 0.5s ein Status-Update.

Phase 1: Interne Schritte (Memory, Prompt, Streaming)
Phase 2 (spaeter): Tool-Activity (Web-Suche, Datei-Lesen etc.)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Konfiguration
SHOW_STATUS_UPDATES: bool = True
STATUS_RATE_LIMIT_SECONDS: float = 0.5


# ---------------------------------------------------------------------------
# Status-Texte (sprach-aware)
# ---------------------------------------------------------------------------

_STATUS_TEXTS: dict[str, dict[str, str]] = {
    "memory_loading": {
        "de": "\U0001f9e0 Lade Notizen…",
        "en": "\U0001f9e0 Loading memory…",
    },
    "memory_loaded": {
        "de": "\U0001f9e0 Lade Notizen… ({n} gefunden)",
        "en": "\U0001f9e0 Loading memory… ({n} entries)",
    },
    "thinking": {
        "de": "\U0001f4ad Denke nach…",
        "en": "\U0001f4ad Thinking…",
    },
    "formatting": {
        "de": "✨ Formatiere…",
        "en": "✨ Formatting…",
    },
}


def get_status_text(key: str, lang: str = "de", **kwargs: Any) -> str:
    """Gibt den lokalisierten Status-Text zurueck.

    Args:
        key: Status-Schluessel (z.B. "memory_loading", "thinking").
        lang: Sprachcode ("de", "en", etc.). Fallback auf "de".
        **kwargs: Format-Parameter (z.B. n=3 fuer Memory-Count).

    Returns:
        Formatierter Status-Text.
    """
    texts = _STATUS_TEXTS.get(key, {})
    # Nur DE und EN unterstuetzt, Rest faellt auf DE zurueck
    template = texts.get(lang, texts.get("de", key))
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template


# ---------------------------------------------------------------------------
# StatusUpdate Protocol (fuer Presentation-Layer-Integration)
# ---------------------------------------------------------------------------


class StatusCallback(Protocol):
    """Protocol fuer Status-Update-Callbacks.

    Der Presentation-Layer implementiert dieses Protocol
    um Status-Updates als Telegram-Edits zu senden.
    """

    async def __call__(self, text: str) -> None:
        """Sendet ein Status-Update an den User.

        Args:
            text: Der Status-Text (mit Emoji).
        """
        ...


@dataclass
class StatusSession:
    """State einer laufenden Status-Session.

    Tracked wann das letzte Update gesendet wurde
    und ob Status-Updates aktiv sind.

    Attributes:
        callback: Async-Callable das den Status-Text an den User sendet.
        language: Aktive Sprache fuer diese Session.
        enabled: Ob Status-Updates aktiv sind.
        last_update_time: Zeitpunkt des letzten Status-Updates (monotonic).
        stream_started: True wenn der Token-Stream begonnen hat.
        _last_key: Letzter gesendeter Status-Key (fuer Phase-Change-Detection).
    """

    callback: StatusCallback
    language: str = "de"
    enabled: bool = field(default_factory=lambda: SHOW_STATUS_UPDATES)
    last_update_time: float = 0.0
    stream_started: bool = False
    _last_key: str = field(default="", repr=False)

    async def update(self, key: str, **kwargs: Any) -> None:
        """Sendet ein Status-Update (rate-limited, Phase-Change-Bypass).

        Rate-Limit wird uebersprungen wenn:
        - Es der allererste Aufruf ist (last_update_time == 0)
        - Der Status-Key sich aendert (neue Phase, z.B. memory_loading -> thinking)

        Args:
            key: Status-Schluessel (z.B. "memory_loading").
            **kwargs: Format-Parameter.
        """
        if not self.enabled or self.stream_started:
            return

        now = time.monotonic()
        is_phase_change = key != self._last_key

        # Rate-Limit nur anwenden wenn es KEIN Phase-Wechsel ist
        if (
            not is_phase_change
            and now - self.last_update_time < STATUS_RATE_LIMIT_SECONDS
        ):
            return

        text = get_status_text(key, self.language, **kwargs)
        try:
            await self.callback(text)
            self.last_update_time = now
            self._last_key = key
        except Exception as e:
            log.debug("Status-Update fehlgeschlagen: %s", e)

    def set_language(self, lang: str) -> None:
        """Aktualisiert die Sprache der Session.

        Wird aufgerufen sobald die tatsaechliche Sprache bestimmt ist
        (z.B. nach Sticky-Language-Lookup oder Sprach-Detection).
        Alle folgenden Status-Updates nutzen die neue Sprache.

        Args:
            lang: Sprachcode ("de", "en", etc.).
        """
        self.language = lang

    def mark_stream_started(self) -> None:
        """Markiert dass der Token-Stream begonnen hat.

        Ab hier werden keine Status-Updates mehr gesendet,
        der normale Streaming-Edit-Flow uebernimmt.
        """
        self.stream_started = True
