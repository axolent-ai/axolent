"""Globale pytest-Fixtures für die Axolent Bridge Test-Suite.

Stellt bereit:
    - tmp_data_dir: temporäres Verzeichnis für Bookmark-JSONL und andere Testdaten
    - sample_bookmark: ein Beispiel-Bookmark-Dict
    - event_loop: asyncio Event-Loop für async Tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Erstellt ein temporäres Verzeichnis für isolierte Testdaten.

    Nutzt pytests eingebautes tmp_path, liefert einen sauberen Ordner
    der nach dem Test automatisch aufgeräumt wird.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_bookmark() -> dict[str, Any]:
    """Liefert ein Beispiel-Bookmark-Dict für Tests."""
    return {
        "timestamp": "2026-05-06T12:00:00+00:00",
        "user_id": 12345,
        "username": "testuser",
        "message_id": 100,
        "chat_id": 67890,
        "content": "Das ist ein Test-Bookmark mit Umlauten: äöüß",
    }
