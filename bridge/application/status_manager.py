"""Status manager: coordinates status updates during processing.

Shows the user what Axolent is doing, instead of just "..." as placeholder.
Language-aware (DE + EN based on sticky language).
Rate-limited: max one status update every 0.5s.

Phase 1: Internal steps (memory, prompt, streaming)
Phase 2 (later): Tool activity (web search, file reading, etc.)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from i18n.domain.i18n import get_status_text  # noqa: F401 (re-export)

if TYPE_CHECKING:
    from application.execution.context import ExecutionContext

log = logging.getLogger(__name__)

# Configuration
SHOW_STATUS_UPDATES: bool = True
STATUS_RATE_LIMIT_SECONDS: float = 0.5
MIN_STATUS_DISPLAY_MS: int = 1100  # Minimum display duration per status update (ms)


# ---------------------------------------------------------------------------
# StatusUpdate Protocol (for presentation layer integration)
# ---------------------------------------------------------------------------


class StatusCallback(Protocol):
    """Protocol for status update callbacks.

    The presentation layer implements this protocol
    to send status updates as Telegram edits.
    """

    async def __call__(self, text: str) -> None:
        """Send a status update to the user.

        Args:
            text: The status text (with emoji).
        """
        ...


@dataclass
class StatusSession:
    """State of a running status session.

    Tracks when the last update was sent and whether status updates are active.

    Attributes:
        callback: Async callable that sends the status text to the user.
        language: Active language for this session.
        context: Optional ExecutionContext. If provided, language is
            derived from context.language.code (Phase 0 Commit 5).
        enabled: Whether status updates are active.
        last_update_time: Timestamp of last status update (monotonic).
        stream_started: True when the token stream has begun.
        _last_key: Last sent status key (for phase-change detection).
    """

    callback: StatusCallback
    language: str = "en"
    context: "ExecutionContext | None" = field(default=None, repr=False)
    enabled: bool = field(default_factory=lambda: SHOW_STATUS_UPDATES)
    last_update_time: float = 0.0
    stream_started: bool = False
    _last_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Derive language from context if provided."""
        if self.context is not None:
            self.language = self.context.language.code

    async def update(self, key: str, **kwargs: Any) -> None:
        """Send a status update (rate-limited, phase-change bypass).

        Enforces a minimum display duration (MIN_STATUS_DISPLAY_MS) between
        consecutive status updates so the user can read each status
        before it is replaced.

        Rate limit is skipped when:
        * It is the very first call (last_update_time == 0)
        * The status key changes (new phase, e.g. memory_loading -> thinking)

        Args:
            key: Status key (e.g. "memory_loading").
            **kwargs: Format parameters.
        """
        if not self.enabled or self.stream_started:
            return

        now = time.monotonic()
        is_phase_change = key != self._last_key

        # Apply rate limit only when it is NOT a phase change
        if (
            not is_phase_change
            and now - self.last_update_time < STATUS_RATE_LIMIT_SECONDS
        ):
            return

        # Minimum display duration: wait until previous status was visible long enough
        if self.last_update_time > 0:
            elapsed_ms = (now - self.last_update_time) * 1000
            remaining_ms = MIN_STATUS_DISPLAY_MS - elapsed_ms
            if remaining_ms > 0:
                await asyncio.sleep(remaining_ms / 1000)

        text = get_status_text(key, self.language, **kwargs)
        try:
            await self.callback(text)
            self.last_update_time = time.monotonic()
            self._last_key = key
        except Exception as e:
            log.debug("Status update failed: %s", e)

    def set_language(self, lang: str) -> None:
        """Update the session language.

        Called once the actual language is determined
        (e.g. after sticky-language lookup or language detection).
        All subsequent status updates use the new language.

        Args:
            lang: Language code ("de", "en", etc.).
        """
        self.language = lang

    def mark_stream_started(self) -> None:
        """Mark that the token stream has begun.

        From here on, no more status updates are sent;
        the normal streaming edit flow takes over.
        """
        self.stream_started = True
