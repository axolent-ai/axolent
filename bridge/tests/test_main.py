"""Tests für main.py: ALLOW_ALL_USERS-Safeguard (C-1).

Testet validate_allow_all_users() in allen drei Szenarien:
1. ALLOW_ALL_USERS=true ohne DEV_MODE -> SystemExit
2. ALLOW_ALL_USERS=true mit DEV_MODE -> nur Warning
3. ALLOW_ALL_USERS nicht gesetzt -> kein Output, normales Verhalten
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest


class TestValidateAllowAllUsers:
    """Tests für validate_allow_all_users Safeguard."""

    def test_allow_all_without_dev_mode_exits(self) -> None:
        """ALLOW_ALL_USERS=true ohne AXOLENT_DEV_MODE=true blockiert Start."""
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
        """ALLOW_ALL_USERS=true mit AXOLENT_DEV_MODE=true warnt nur."""
        with (
            patch("main.ALLOW_ALL_USERS", True),
            patch("main.AXOLENT_DEV_MODE", True),
            caplog.at_level(logging.WARNING, logger="axolent"),
        ):
            from main import validate_allow_all_users

            # Darf NICHT raisen
            validate_allow_all_users()

            assert any(
                "ALLOW_ALL_USERS aktiv im DEV_MODE" in r.message for r in caplog.records
            )

    def test_allow_all_disabled_no_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Ohne ALLOW_ALL_USERS: kein Output, kein SystemExit."""
        with (
            patch("main.ALLOW_ALL_USERS", False),
            patch("main.AXOLENT_DEV_MODE", False),
            caplog.at_level(logging.DEBUG, logger="axolent"),
        ):
            from main import validate_allow_all_users

            validate_allow_all_users()

            # Kein Log-Output von validate_allow_all_users erwartet
            relevant = [
                r
                for r in caplog.records
                if "ALLOW_ALL_USERS" in r.message or "DEV_MODE" in r.message
            ]
            assert len(relevant) == 0

    def test_allow_all_exit_message_is_descriptive(self) -> None:
        """Die Exit-Meldung erklaert klar was zu tun ist."""
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
