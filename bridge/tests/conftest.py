"""Globale pytest-Fixtures fuer die Jarvis-LITE Bridge Test-Suite.

Stellt bereit:
    - tmp_data_dir: temporaeres Verzeichnis fuer Bookmark-JSONL und andere Testdaten
    - mock_claude_cli: Mock fuer den Claude-Subprozess (kein echter CLI-Aufruf)
    - sample_bookmark: ein Beispiel-Bookmark-Dict
    - event_loop: asyncio Event-Loop fuer async Tests
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, patch

import pytest

# Bridge-Root zum sys.path hinzufuegen damit Imports funktionieren
BRIDGE_ROOT = Path(__file__).resolve().parent.parent
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Erstellt ein temporaeres Verzeichnis fuer isolierte Testdaten.

    Nutzt pytests eingebautes tmp_path, liefert einen sauberen Ordner
    der nach dem Test automatisch aufgeraeumt wird.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_claude_cli() -> Generator[AsyncMock, None, None]:
    """Mockt den Claude-CLI-Subprozess.

    Gibt eine Dummy-Antwort zurueck ohne tatsaechlich claude aufzurufen.
    Return-Format: (exit_code=0, stdout="Mocked response", stderr="", duration=0.5)
    """
    mock_result = (0, "Das ist eine gemockte Claude-Antwort.", "", 0.5)

    with patch(
        "infrastructure.claude_cli.call_claude_async",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_fn:
        yield mock_fn


@pytest.fixture
def sample_bookmark() -> dict[str, Any]:
    """Liefert ein Beispiel-Bookmark-Dict fuer Tests."""
    return {
        "timestamp": "2026-05-06T12:00:00+00:00",
        "user_id": 12345,
        "username": "testuser",
        "message_id": 100,
        "chat_id": 67890,
        "content": "Das ist ein Test-Bookmark mit Umlauten: äöüß",
    }
