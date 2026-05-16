"""Tests for the AST-based i18n hardcoded string scanner.

Verifies that the scanner correctly identifies hardcoded strings
in Telegram API calls while respecting whitelists and suppressions.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# Add scripts dir to path so we can import the scanner
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from i18n_scan import scan_file  # noqa: E402


class TestI18nScanAST:
    """Verifies the AST scanner detects hardcoded strings correctly."""

    def _write_and_scan(self, tmp_path: Path, content: str) -> list:
        """Helper: write content to a temp .py file and scan it."""
        f = tmp_path / "test_module.py"
        f.write_text(textwrap.dedent(content), encoding="utf-8")
        return scan_file(f)

    def test_detects_hardcoded_reply_text(self, tmp_path: Path) -> None:
        """reply_text with a string literal should be flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(update):
                await update.message.reply_text("Memory system not initialized.")
            """,
        )
        assert len(violations) == 1
        assert violations[0].method == "reply_text"
        assert "Memory system" in violations[0].string_value

    def test_detects_hardcoded_edit_message_text(self, tmp_path: Path) -> None:
        """edit_message_text with a string literal should be flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(query):
                await query.edit_message_text("Settings not available.")
            """,
        )
        assert len(violations) == 1
        assert violations[0].method == "edit_message_text"

    def test_detects_hardcoded_answer_text_kwarg(self, tmp_path: Path) -> None:
        """answer(text=...) with a string literal should be flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(query):
                await query.answer(text="Bookmark not found", show_alert=False)
            """,
        )
        assert len(violations) == 1
        assert violations[0].method == "answer"
        assert "Bookmark not found" in violations[0].string_value

    def test_detects_hardcoded_inline_keyboard_button(self, tmp_path: Path) -> None:
        """InlineKeyboardButton with hardcoded text should be flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            from telegram import InlineKeyboardButton
            btn = InlineKeyboardButton(text="Full text", callback_data="x")
            """,
        )
        assert len(violations) == 1
        assert violations[0].method == "InlineKeyboardButton"

    def test_detects_fstring_literal(self, tmp_path: Path) -> None:
        """f-string with hardcoded text parts should be flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(update, i):
                await update.message.reply_text(f"#{i} Full text")
            """,
        )
        assert len(violations) == 1
        assert "Full text" in violations[0].string_value

    def test_allows_t_call(self, tmp_path: Path) -> None:
        """Calls using t() are allowed (not flagged)."""
        violations = self._write_and_scan(
            tmp_path,
            """
            from i18n.domain.i18n import t
            async def handler(update, lang):
                await update.message.reply_text(t("some.key", lang))
            """,
        )
        assert len(violations) == 0

    def test_allows_variable_argument(self, tmp_path: Path) -> None:
        """Variable arguments are allowed (not string literals)."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(update, msg):
                await update.message.reply_text(msg)
            """,
        )
        assert len(violations) == 0

    def test_allows_symbol_only_strings(self, tmp_path: Path) -> None:
        """Pure symbol strings (markers, placeholders) are allowed."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(chat):
                await chat.send_message("...")
            """,
        )
        assert len(violations) == 0

    def test_allows_suppressed_line(self, tmp_path: Path) -> None:
        """Lines with # i18n: ok comment are suppressed."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(update):
                await update.message.reply_text("Debug info")  # i18n: ok
            """,
        )
        assert len(violations) == 0

    def test_mixed_file_only_flags_hardcoded(self, tmp_path: Path) -> None:
        """In a file with mixed calls, only hardcoded strings are flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            from i18n.domain.i18n import t

            async def handler(update, lang):
                # This is OK (uses t())
                await update.message.reply_text(t("reset.confirmation", lang))

                # This is OK (variable)
                msg = "computed"
                await update.message.reply_text(msg)

                # This should be flagged
                await update.message.reply_text("Rate limiter not initialized.")

                # This is OK (symbol only)
                await chat.send_message("...")

                # This should be flagged
                await query.edit_message_text("Unknown slot.")
            """,
        )
        assert len(violations) == 2
        methods = {v.method for v in violations}
        assert "reply_text" in methods
        assert "edit_message_text" in methods

    def test_does_not_flag_answer_without_text_kwarg(self, tmp_path: Path) -> None:
        """answer() without text= keyword is not flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            async def handler(query):
                await query.answer()
            """,
        )
        assert len(violations) == 0

    def test_inline_keyboard_button_first_arg(self, tmp_path: Path) -> None:
        """InlineKeyboardButton(string, ...) first positional arg is flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            from telegram import InlineKeyboardButton
            btn = InlineKeyboardButton("Click me", callback_data="x")
            """,
        )
        assert len(violations) == 1

    def test_inline_keyboard_button_variable_ok(self, tmp_path: Path) -> None:
        """InlineKeyboardButton(variable, ...) is not flagged."""
        violations = self._write_and_scan(
            tmp_path,
            """
            from telegram import InlineKeyboardButton
            label = t("some.key", "en")
            btn = InlineKeyboardButton(label, callback_data="x")
            """,
        )
        assert len(violations) == 0
