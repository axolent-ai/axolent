"""Global pytest fixtures for the Axolent Bridge test suite.

Provides:
    - tmp_data_dir: temporary directory for bookmark JSONL and other test data
    - sample_bookmark: a sample bookmark dict
    - event_loop: asyncio event loop for async tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for isolated test data.

    Uses pytest's built-in tmp_path, provides a clean directory
    that is automatically cleaned up after the test.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_bookmark() -> dict[str, Any]:
    """Provide a sample bookmark dict for tests."""
    return {
        "timestamp": "2026-05-06T12:00:00+00:00",
        "user_id": 12345,
        "username": "testuser",
        "message_id": 100,
        "chat_id": 67890,
        "content": "Das ist ein Test-Bookmark mit Umlauten: äöüß",
    }
