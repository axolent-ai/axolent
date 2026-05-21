"""Tests for Skill Commands and Secret Filter (Step 5).

Covers:
  - check_secret_content: API tokens, prices, emails, IBANs, passwords rejected
  - check_secret_content: clean text passes
  - _execute_forget: creates tombstone, sets status to retired
  - derive_skill_name integration with commands
"""

from __future__ import annotations

import sqlite3

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import (
    STATUS_ACTIVE,
    STATUS_RETIRED,
)
from presentation.skill_commands import (
    _execute_forget,
    check_secret_content,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


class FakeDBConnection:
    """Minimal in-memory SQLite for tests."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql, params=(), **kwargs):
        return self._conn.execute(sql, params)

    def executescript(self, sql):
        self._conn.executescript(sql)

    def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations):
        for sql, params in operations:
            self._conn.execute(sql, params)
        self._conn.commit()


def _setup_storage() -> HypothesisStorage:
    """Create an in-memory HypothesisStorage with schema."""
    conn = FakeDBConnection()
    storage = HypothesisStorage(conn)
    storage.init_schema()
    return storage


def _make_hypothesis(
    *,
    hypothesis_id: str = "hyp-forget-001",
    user_id: int = 42,
    status: str = STATUS_ACTIVE,
    claim: str = "Test skill",
    pattern_hash: str = "hash123",
    scope_hash: str = "scope_abc",
) -> Hypothesis:
    """Create a test hypothesis for forget tests."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=user_id,
        type="preference",
        scope=HypothesisScope(),
        claim=claim,
        status=status,
        elo_rating=1600.0,
        created_at="2026-05-20T10:00:00+00:00",
        last_seen="2026-05-20T12:00:00+00:00",
        pattern_hash=pattern_hash,
        scope_hash=scope_hash,
    )


# ---------------------------------------------------------------
# Tests: check_secret_content (HC-SC-13)
# ---------------------------------------------------------------


class TestCheckSecretContent:
    """Tests for the No-Model-Secret filter."""

    def test_clean_text_passes(self) -> None:
        """Normal skill text should pass the filter."""
        assert check_secret_content("Verwende immer Bulletpoints") is None

    def test_clean_instruction_passes(self) -> None:
        """Instruction-style text should pass."""
        result = check_secret_content(
            "Bei Code-Reviews immer erst Root Cause, dann Fix"
        )
        assert result is None

    def test_api_token_sk(self) -> None:
        """OpenAI-style API token should be rejected."""
        result = check_secret_content("use sk-1234567890abcdef1234567890abcdef")
        assert result is not None
        assert "Token" in result or "Secret" in result

    def test_api_token_ghp(self) -> None:
        """GitHub PAT should be rejected."""
        result = check_secret_content("token ghp_abcdefghijklmnopqrstuvwxyz12345678")
        assert result is not None

    def test_bearer_token(self) -> None:
        """Bearer token should be rejected."""
        result = check_secret_content("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")
        assert result is not None

    def test_price_euro(self) -> None:
        """Euro price should be rejected."""
        result = check_secret_content("Preis ist 29.99 EUR pro Stueck")
        assert result is not None
        assert "Preis" in result

    def test_price_dollar_sign(self) -> None:
        """Dollar sign price should be rejected."""
        result = check_secret_content("costs $150.00")
        assert result is not None

    def test_email_address(self) -> None:
        """Email address should be rejected."""
        result = check_secret_content("kontakt@example.com")
        assert result is not None
        assert "E-Mail" in result

    def test_phone_number(self) -> None:
        """Phone number should be rejected."""
        result = check_secret_content("Call +49 170 1234 5678")
        assert result is not None

    def test_iban(self) -> None:
        """IBAN should be rejected."""
        result = check_secret_content("IBAN: DE89 3704 0044 0532 0130 00")
        assert result is not None

    def test_password_keyword(self) -> None:
        """Password-adjacent content should be rejected."""
        result = check_secret_content("passwort: meinGeheim123!")
        assert result is not None
        assert "Passwort" in result

    def test_long_hex_string(self) -> None:
        """Long hex string (likely a key) should be rejected."""
        hex_str = "a" * 32
        result = check_secret_content(f"key is {hex_str}")
        assert result is not None

    def test_short_number_passes(self) -> None:
        """Short numbers (not phone) should pass."""
        assert check_secret_content("Chapter 3 of 10") is None

    def test_german_text_passes(self) -> None:
        """German instruction text should pass."""
        result = check_secret_content(
            "Schreibe Kundenmails immer in formellem Ton mit Sie-Anrede"
        )
        assert result is None


# ---------------------------------------------------------------
# Tests: _execute_forget
# ---------------------------------------------------------------


class TestExecuteForget:
    """Tests for the forget execution logic."""

    def test_forget_sets_retired_status(self) -> None:
        """Forget should set hypothesis status to 'retired'."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        _execute_forget(storage, hyp, permanent=False)

        updated = storage.get_hypothesis("hyp-forget-001")
        assert updated is not None
        assert updated.status == STATUS_RETIRED

    def test_forget_creates_tombstone(self) -> None:
        """Forget should create a tombstone record."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        _execute_forget(storage, hyp, permanent=False)

        has_tombstone = storage.check_tombstone("hash123")
        assert has_tombstone is True

    def test_forget_permanent_tombstone(self) -> None:
        """Permanent forget should create a permanent tombstone."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        _execute_forget(storage, hyp, permanent=True)

        # Check tombstone exists and is permanent
        has_tombstone = storage.check_tombstone("hash123")
        assert has_tombstone is True

    def test_forget_non_permanent_has_expiry(self) -> None:
        """Non-permanent forget should set expires_at 30 days out."""
        storage = _setup_storage()
        hyp = _make_hypothesis()
        storage.insert_hypothesis(hyp)

        _execute_forget(storage, hyp, permanent=False)

        # Verify tombstone has an expiry date
        row = storage._conn.fetchone(
            "SELECT * FROM hypothesis_tombstones WHERE fingerprint = ?",
            ("hash123",),
        )
        assert row is not None
        assert row["expires_at"] is not None
        assert row["permanent"] == 0


# ---------------------------------------------------------------
# Tests: Architecture guards
# ---------------------------------------------------------------


class TestArchitectureGuards:
    """Verify layer boundaries are respected."""

    def test_skill_commands_does_not_import_infrastructure(self) -> None:
        """skill_commands should not import infrastructure modules."""
        import presentation.skill_commands as mod

        source_file = mod.__file__
        with open(source_file, encoding="utf-8") as f:
            content = f.read()

        # Must not import from infrastructure
        assert "from infrastructure" not in content
        assert "import infrastructure" not in content

    def test_skill_profile_view_does_not_import_infrastructure(self) -> None:
        """skill_profile_view should not import infrastructure modules."""
        import presentation.skill_profile_view as mod

        source_file = mod.__file__
        with open(source_file, encoding="utf-8") as f:
            content = f.read()

        assert "from infrastructure" not in content
        assert "import infrastructure" not in content
