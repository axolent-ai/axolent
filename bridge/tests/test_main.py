"""Tests for main.py: ALLOW_ALL_USERS safeguard (C-1).

Tests validate_allow_all_users() in all three scenarios:
1. ALLOW_ALL_USERS=true without DEV_MODE -> SystemExit
2. ALLOW_ALL_USERS=true with DEV_MODE -> warning only
3. ALLOW_ALL_USERS not set -> no output, normal behavior
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest


class TestValidateAllowAllUsers:
    """Tests for validate_allow_all_users safeguard."""

    def test_allow_all_without_dev_mode_exits(self) -> None:
        """ALLOW_ALL_USERS=true without AXOLENT_DEV_MODE=true blocks start."""
        with (
            patch("main.ALLOW_ALL_USERS", True),
            patch("main.AXOLENT_DEV_MODE", False),
        ):
            from main import validate_allow_all_users

            with pytest.raises(SystemExit) as exc_info:
                validate_allow_all_users()
            assert exc_info.value.code == 2

    def test_allow_all_with_dev_mode_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ALLOW_ALL_USERS=true with AXOLENT_DEV_MODE=true only warns."""
        with (
            patch("main.ALLOW_ALL_USERS", True),
            patch("main.AXOLENT_DEV_MODE", True),
            caplog.at_level(logging.WARNING, logger="axolent"),
        ):
            from main import validate_allow_all_users

            # Must NOT raise
            validate_allow_all_users()

            assert any(
                "ALLOW_ALL_USERS active in DEV_MODE" in r.message for r in caplog.records
            )

    def test_allow_all_disabled_no_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without ALLOW_ALL_USERS: no output, no SystemExit."""
        with (
            patch("main.ALLOW_ALL_USERS", False),
            patch("main.AXOLENT_DEV_MODE", False),
            caplog.at_level(logging.DEBUG, logger="axolent"),
        ):
            from main import validate_allow_all_users

            validate_allow_all_users()

            # No log output from validate_allow_all_users expected
            relevant = [
                r
                for r in caplog.records
                if "ALLOW_ALL_USERS" in r.message or "DEV_MODE" in r.message
            ]
            assert len(relevant) == 0

    def test_allow_all_exit_message_is_descriptive(self) -> None:
        """The exit message clearly explains what to do."""
        with (
            patch("main.ALLOW_ALL_USERS", True),
            patch("main.AXOLENT_DEV_MODE", False),
        ):
            import logging as _logging

            from main import validate_allow_all_users

            logger = _logging.getLogger("axolent")
            with (
                pytest.raises(SystemExit),
                patch.object(logger, "critical") as mock_critical,
            ):
                validate_allow_all_users()

            call_msg = mock_critical.call_args[0][0]
            assert "AXOLENT_DEV_MODE" in call_msg
            assert "ALLOW_ALL_USERS" in call_msg
