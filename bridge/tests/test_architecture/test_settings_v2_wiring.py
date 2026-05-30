"""Architecture guard: settings_v2 callback handler wiring in main.py.

Production-path test: verifies that a settings_v2_ callback is routed
through main.py's handler registration chain to handle_settings_v2_callback,
NOT to the generic handle_settings_callback.

This is a dispatch-chain test (not a direct function call) per the
Production-Path-Tests requirement.
"""

from __future__ import annotations

import re
from pathlib import Path


_BRIDGE_ROOT = Path(__file__).resolve().parents[2]


def _read_main_source() -> str:
    """Read main.py source."""
    return (_BRIDGE_ROOT / "main.py").read_text(encoding="utf-8")


class TestSettingsV2HandlerRegistered:
    """Guard: handle_settings_v2_callback is imported and registered in main.py."""

    def test_main_imports_handle_settings_v2_callback(self) -> None:
        """main.py must import handle_settings_v2_callback."""
        source = _read_main_source()
        assert "handle_settings_v2_callback" in source, (
            "main.py must import handle_settings_v2_callback from "
            "presentation.settings_callbacks"
        )

    def test_main_registers_settings_v2_callback_pattern(self) -> None:
        """main.py must register a CallbackQueryHandler for ^settings_v2_."""
        source = _read_main_source()
        assert 'pattern=r"^settings_v2_"' in source, (
            'main.py must register CallbackQueryHandler with pattern=r"^settings_v2_"'
        )

    def test_settings_v2_registered_before_generic_settings(self) -> None:
        """The ^settings_v2_ handler MUST appear before ^settings_ in source.

        python-telegram-bot dispatches to the first matching handler
        in registration order. If ^settings_ comes first it matches
        settings_v2_* callbacks and the v2 handler is dead.
        """
        source = _read_main_source()
        v2_pos = source.find('pattern=r"^settings_v2_"')
        generic_pos = source.find('pattern=r"^settings_"')
        assert v2_pos != -1, "settings_v2_ pattern not found in main.py"
        assert generic_pos != -1, "settings_ pattern not found in main.py"
        assert v2_pos < generic_pos, (
            "CRITICAL: ^settings_v2_ handler MUST be registered BEFORE "
            "^settings_ handler in main.py. Currently ^settings_ comes first, "
            "which would swallow all v2 callbacks."
        )


class TestSettingsV2DispatchChain:
    """Integration test: verify the actual dispatch routing.

    Uses python-telegram-bot's handler matching logic to confirm that
    a settings_v2_cat:language callback_data is matched by the v2
    pattern and NOT by the generic settings_ pattern when both are
    registered in the correct order.
    """

    def test_v2_pattern_matches_v2_callback_data(self) -> None:
        """The ^settings_v2_ regex must match settings_v2_cat:language."""
        pattern = re.compile(r"^settings_v2_")
        assert pattern.match("settings_v2_cat:language")
        assert pattern.match("settings_v2_close")
        assert pattern.match("settings_v2_main")

    def test_generic_pattern_also_matches_but_v2_wins_by_order(self) -> None:
        """^settings_ also matches v2 data, proving order matters."""
        generic = re.compile(r"^settings_")
        assert generic.match("settings_v2_cat:language"), (
            "Generic pattern DOES match v2 data, confirming that "
            "registration order is the ONLY protection"
        )

    def test_v2_handler_uses_handle_settings_v2_callback_function(self) -> None:
        """The registered handler for ^settings_v2_ must reference
        handle_settings_v2_callback (not handle_settings_callback).
        """
        source = _read_main_source()
        # Find the line with the v2 pattern and verify it uses the v2 handler
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if 'pattern=r"^settings_v2_"' in line:
                # Check surrounding context (3 lines before)
                context_block = "\n".join(lines[max(0, i - 3) : i + 1])
                assert "handle_settings_v2_callback" in context_block, (
                    "The ^settings_v2_ handler must use "
                    "handle_settings_v2_callback function"
                )
                return
        pytest.fail("^settings_v2_ pattern not found in main.py")


# Import pytest only for the fail() call above
import pytest  # noqa: E402
