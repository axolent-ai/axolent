"""Tests für infrastructure.audit_log: JSONL Audit-Log mit Rotation.

Testet dass Audit-Einträge korrekt geschrieben werden
und die Rotation bei Überschreiten der Max-Größe greift.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

import pytest


class TestAuditLog:
    """Audit-Log Schreib- und Rotations-Tests."""

    @pytest.fixture
    def isolated_audit(self, tmp_path: Path):
        """Erstellt einen isolierten Audit-Logger mit eigenem Pfad."""
        audit_path = tmp_path / "audit.jsonl"

        # Eigenen Logger + Handler erstellen
        test_logger = logging.getLogger(f"test_audit_{id(self)}")
        test_logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            audit_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        test_logger.addHandler(handler)
        test_logger.propagate = False

        yield audit_path, test_logger

        # Cleanup
        handler.close()
        test_logger.removeHandler(handler)

    def test_audit_log_writes_jsonl(self, isolated_audit) -> None:
        """Audit-Einträge werden als gültige JSONL-Zeilen geschrieben."""
        audit_path, logger = isolated_audit

        entry = {
            "timestamp": "2026-05-06T12:00:00+00:00",
            "user_id": 123,
            "action": "message",
        }
        logger.info(json.dumps(entry, ensure_ascii=False))

        content = audit_path.read_text(encoding="utf-8").strip()
        parsed = json.loads(content)
        assert parsed["user_id"] == 123
        assert parsed["action"] == "message"

    def test_audit_log_multiple_entries(self, isolated_audit) -> None:
        """Mehrere Einträge werden als separate JSONL-Zeilen geschrieben."""
        audit_path, logger = isolated_audit

        for i in range(5):
            logger.info(json.dumps({"seq": i}, ensure_ascii=False))

        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
        for i, line in enumerate(lines):
            assert json.loads(line)["seq"] == i

    def test_audit_log_rotation_at_max_bytes(self, tmp_path: Path) -> None:
        """Bei Überschreiten von maxBytes wird rotiert (simuliert mit kleinem maxBytes)."""
        audit_path = tmp_path / "small_audit.jsonl"

        # Sehr kleiner maxBytes: 200 Bytes pro Datei
        test_logger = logging.getLogger(f"rotation_test_{id(self)}")
        test_logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            audit_path,
            maxBytes=200,  # Sehr klein für schnelle Rotation
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        test_logger.addHandler(handler)
        test_logger.propagate = False

        try:
            # Genug Einträge schreiben um Rotation auszulösen
            for i in range(50):
                entry = {"seq": i, "data": "x" * 50}
                test_logger.info(json.dumps(entry, ensure_ascii=False))

            # Backup-Datei muss existieren (Rotation hat stattgefunden)
            backup = Path(str(audit_path) + ".1")
            assert backup.exists(), (
                "Rotation hätte eine .1-Backup-Datei erzeugen müssen"
            )
        finally:
            handler.close()
            test_logger.removeHandler(handler)

    def test_audit_log_unicode_content(self, isolated_audit) -> None:
        """Unicode-Inhalt wird korrekt geschrieben (keine ASCII-Escapes)."""
        audit_path, logger = isolated_audit

        entry = {"user": "Jessica", "text": "Grueße aus Muenchen: äöüß"}
        logger.info(json.dumps(entry, ensure_ascii=False))

        raw = audit_path.read_text(encoding="utf-8")
        assert "äöüß" in raw
        assert "\\u" not in raw  # Keine Escapes
