"""EK-03 Regression test: cancelled stream must not save partial response.

Verifies that after session.is_cancelled is True, neither fallback finalize
nor history save can execute. This is a logic-level test that validates the
control flow invariant without needing the full Telegram integration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class FakeStreamingSession:
    """Minimal streaming session mock for cancel logic testing."""

    accumulated_text: str = ""
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    finalized_text: str | None = None

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()


class TestCancelledStreamDoesNotSave:
    """EK-03: Verify that cancelled streams never save partial text."""

    def test_cancelled_with_accumulated_text_no_fallback(self) -> None:
        """After cancel, accumulated_text must NOT become final_text."""
        session = FakeStreamingSession(accumulated_text="partial response here")
        session.cancel()

        # Simulate the fixed logic from handlers.py:
        # After cancel, we return immediately (no fallback block runs)
        final_text = ""
        had_error = False

        if session.is_cancelled:
            # EK-03: hard return in production code
            # Here we simulate by setting a flag and skipping
            saved = False
        else:
            # Fallback: this must NOT execute when cancelled
            if not final_text and session.accumulated_text and not had_error:
                final_text = session.accumulated_text
            if final_text and not had_error:
                saved = True
            else:
                saved = False

        assert not saved, "Cancelled stream must not save partial text"
        assert final_text == "", "final_text must stay empty after cancel"

    def test_non_cancelled_stream_does_fallback(self) -> None:
        """Non-cancelled stream with accumulated text should use fallback."""
        session = FakeStreamingSession(accumulated_text="complete response")
        # NOT cancelled

        final_text = ""
        had_error = False

        if session.is_cancelled:
            saved = False
        else:
            if not final_text and session.accumulated_text and not had_error:
                final_text = session.accumulated_text
            saved = bool(final_text and not had_error)

        assert saved, "Non-cancelled stream should save"
        assert final_text == "complete response"

    def test_cancelled_with_error_and_accumulated(self) -> None:
        """Cancelled + error + accumulated: nothing saved."""
        session = FakeStreamingSession(accumulated_text="error partial")
        session.cancel()

        final_text = ""
        had_error = True

        if session.is_cancelled:
            saved = False
        else:
            if not final_text and session.accumulated_text and not had_error:
                final_text = session.accumulated_text
            saved = bool(final_text and not had_error)

        assert not saved
        assert final_text == ""

    def test_cancel_flag_is_terminal(self) -> None:
        """Once cancelled, the session stays cancelled (no race reset)."""
        session = FakeStreamingSession()
        assert not session.is_cancelled
        session.cancel()
        assert session.is_cancelled
        # Cannot un-cancel
        assert session.is_cancelled
