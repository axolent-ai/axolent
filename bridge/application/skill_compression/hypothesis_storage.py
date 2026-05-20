"""Hypothesis storage schema for Skill-Compression (7 tables).

All tables reside in the encrypted SQLCipher database (via CryptoConnection
or SqliteConnection for tests).

Tables:
  1. hypotheses          - Core hypothesis records with Elo/FSRS/Bayesian state
  2. hypothesis_aliases  - Dynamic term pool per hypothesis
  3. hypothesis_evidence - Evidence ledger (support/contradict signals)
  4. hypothesis_versions - Version history with predecessor_context
  5. hypothesis_tombstones - Deleted hypotheses with TTL
  6. hypothesis_local_eval_set - Smoke-test example pairs
  7. pattern_difficulty   - Elo rating per fingerprint pattern

HC-SC-9: type column is TEXT, not ENUM (open for extension).
AG-SC-8: no CHECK constraint with fixed values on type columns.

All timestamps are ISO-8601 UTC strings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Schema DDL
# ──────────────────────────────────────────────────────────────

HYPOTHESIS_SCHEMA_SQL = """
-- Hypotheses: core pattern records
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id          TEXT PRIMARY KEY,
    user_id                INTEGER NOT NULL,
    type                   TEXT NOT NULL,
    scope_json             TEXT NOT NULL DEFAULT '{}',
    claim                  TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'candidate',
    version                INTEGER NOT NULL DEFAULT 1,
    elo_rating             REAL NOT NULL DEFAULT 1500.0,
    elo_games_played       INTEGER NOT NULL DEFAULT 0,
    bayes_confidence       REAL NOT NULL DEFAULT 0.5,
    support_count          INTEGER NOT NULL DEFAULT 0,
    contradict_count       INTEGER NOT NULL DEFAULT 0,
    last_contradiction_at  TEXT,
    fsrs_state_json        TEXT NOT NULL DEFAULT '{}',
    source_type            TEXT NOT NULL DEFAULT 'live_chat',
    decay_immune           INTEGER NOT NULL DEFAULT 0,
    created_at             TEXT NOT NULL,
    last_applied           TEXT,
    last_seen              TEXT NOT NULL,
    approval_state         TEXT NOT NULL DEFAULT 'pending',
    pattern_hash           TEXT,
    scope_hash             TEXT
);
CREATE INDEX IF NOT EXISTS idx_hypotheses_user_id
    ON hypotheses(user_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_user_status
    ON hypotheses(user_id, status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_pattern_hash
    ON hypotheses(pattern_hash);
CREATE INDEX IF NOT EXISTS idx_hypotheses_scope_hash
    ON hypotheses(scope_hash);
CREATE INDEX IF NOT EXISTS idx_hypotheses_fingerprint
    ON hypotheses(user_id, pattern_hash);

-- Hypothesis aliases: dynamic term pool
CREATE TABLE IF NOT EXISTS hypothesis_aliases (
    alias_id        TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL,
    alias_text      TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.5,
    evidence_count  INTEGER NOT NULL DEFAULT 0,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_aliases_hypothesis
    ON hypothesis_aliases(hypothesis_id);

-- Hypothesis evidence: structured proof ledger
CREATE TABLE IF NOT EXISTS hypothesis_evidence (
    evidence_id        TEXT PRIMARY KEY,
    hypothesis_id      TEXT NOT NULL,
    hypothesis_version INTEGER NOT NULL DEFAULT 1,
    episode_id         TEXT,
    request_id         TEXT,
    response_id        TEXT,
    signal_type        TEXT NOT NULL,
    signal_strength    REAL NOT NULL DEFAULT 1.0,
    created_at         TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_evidence_hypothesis
    ON hypothesis_evidence(hypothesis_id, hypothesis_version);

-- Hypothesis versions: version history with predecessor context
CREATE TABLE IF NOT EXISTS hypothesis_versions (
    version_id          TEXT PRIMARY KEY,
    hypothesis_id       TEXT NOT NULL,
    version             INTEGER NOT NULL,
    claim               TEXT NOT NULL,
    elo_rating_at_save  REAL NOT NULL DEFAULT 1500.0,
    change_reason       TEXT,
    predecessor_context TEXT,
    created_at          TEXT NOT NULL,
    deprecated_at       TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_versions_hypothesis
    ON hypothesis_versions(hypothesis_id, version);

-- Hypothesis tombstones: deleted hypotheses with TTL
CREATE TABLE IF NOT EXISTS hypothesis_tombstones (
    tombstone_id    TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL,
    fingerprint     TEXT NOT NULL,
    scope_hash      TEXT,
    deleted_at      TEXT NOT NULL,
    expires_at      TEXT,
    permanent       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tombstones_fingerprint
    ON hypothesis_tombstones(fingerprint);
CREATE INDEX IF NOT EXISTS idx_tombstones_scope
    ON hypothesis_tombstones(scope_hash);

-- Local evaluation set: smoke-test examples per hypothesis
CREATE TABLE IF NOT EXISTS hypothesis_local_eval_set (
    eval_id         TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL,
    example_input   TEXT NOT NULL,
    example_output  TEXT NOT NULL,
    was_correct     INTEGER,
    last_evaluated  TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_eval_hypothesis
    ON hypothesis_local_eval_set(hypothesis_id);

-- Pattern difficulty: Elo rating per fingerprint (for difficulty-aware Elo updates)
CREATE TABLE IF NOT EXISTS pattern_difficulty (
    fingerprint_hash    TEXT PRIMARY KEY,
    difficulty_rating   REAL NOT NULL DEFAULT 1500.0,
    games_played        INTEGER NOT NULL DEFAULT 0,
    last_updated        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_difficulty_fingerprint
    ON pattern_difficulty(fingerprint_hash);
"""


# ──────────────────────────────────────────────────────────────
# Connection protocol (for type safety without coupling to concrete class)
# ──────────────────────────────────────────────────────────────


class DBConnection(Protocol):
    """Minimal DB connection interface used by HypothesisStorage.

    Compatible with both SqliteConnection and CryptoConnection.
    """

    def execute(self, sql: str, params: tuple | dict = (), **kwargs) -> object: ...
    def executescript(self, sql: str) -> None: ...
    def fetchall(self, sql: str, params: tuple | dict = ()) -> list: ...
    def fetchone(self, sql: str, params: tuple | dict = ()) -> Optional[object]: ...
    def execute_in_transaction(self, operations: list[tuple[str, tuple]]) -> None: ...


# ──────────────────────────────────────────────────────────────
# Domain data classes
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HypothesisScope:
    """Scope definition for a hypothesis (where does this pattern apply?).

    Attributes:
        project: Project identifier (e.g. 'client_ads').
        client: Client identifier (e.g. 'honey-brand').
        context: Additional context tags.
    """

    project: str = ""
    client: str = ""
    context: tuple[str, ...] = ()

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "project": self.project,
                "client": self.client,
                "context": list(self.context),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> HypothesisScope:
        """Deserialize from JSON string."""
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
            return cls(
                project=data.get("project", ""),
                client=data.get("client", ""),
                context=tuple(data.get("context", [])),
            )
        except (json.JSONDecodeError, TypeError):
            return cls()


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Core hypothesis record (HC-SC-1: frozen=True, slots=True).

    Represents a learned pattern with full lifecycle state.
    Type is TEXT (HC-SC-9), not an enum, to allow future extension.

    Attributes:
        hypothesis_id: Unique identifier.
        user_id: Telegram user ID.
        type: Pattern type ('request', 'preference', 'negative', or future types).
        scope: Where this pattern applies.
        claim: Human-readable description of the pattern.
        status: Lifecycle stage (candidate/suggested/confirmed/active/needs_review/paused/archived).
        version: Current version number.
        elo_rating: Pattern confidence via Elo system (HC-SC-4).
        elo_games_played: Number of Elo matches.
        bayes_confidence: Bayesian confidence [0, 1].
        support_count: Number of supporting evidence items.
        contradict_count: Number of contradicting evidence items.
        last_contradiction_at: Timestamp of last contradiction.
        fsrs_state_json: FSRS v7 state for decay (HC-SC-5).
        source_type: Origin of the hypothesis.
        decay_immune: Whether this is user-created (HC-SC-6).
        evidence_ids: Tuple of associated evidence IDs.
        created_at: Creation timestamp.
        last_applied: Last time this hypothesis was applied.
        last_seen: Last time evidence was added.
        approval_state: Admin approval state.
        pattern_hash: Fingerprint hash for matching.
        scope_hash: Scope hash for collision detection.
    """

    hypothesis_id: str = ""
    user_id: int = 0
    type: str = "request"
    scope: HypothesisScope = field(default_factory=HypothesisScope)
    claim: str = ""
    status: str = "candidate"
    version: int = 1
    elo_rating: float = 1500.0
    elo_games_played: int = 0
    bayes_confidence: float = 0.5
    support_count: int = 0
    contradict_count: int = 0
    last_contradiction_at: Optional[str] = None
    fsrs_state_json: str = "{}"
    source_type: str = "live_chat"
    decay_immune: bool = False
    evidence_ids: tuple[str, ...] = ()
    created_at: str = ""
    last_applied: Optional[str] = None
    last_seen: str = ""
    approval_state: str = "pending"
    pattern_hash: Optional[str] = None
    scope_hash: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Storage class
# ──────────────────────────────────────────────────────────────


class HypothesisStorage:
    """CRUD operations for the hypothesis schema.

    Wraps the 7 tables and provides typed access methods.
    All write operations go through execute_in_transaction for atomicity.
    """

    def __init__(self, conn: DBConnection) -> None:
        self._conn = conn

    def init_schema(self) -> None:
        """Create all hypothesis tables (idempotent).

        Safe to call multiple times. Uses CREATE TABLE IF NOT EXISTS.
        """
        self._conn.executescript(HYPOTHESIS_SCHEMA_SQL)
        log.info("Hypothesis schema initialized (7 tables)")

    # ── Hypothesis CRUD ──────────────────────────────────────

    def insert_hypothesis(self, h: Hypothesis) -> None:
        """Insert a new hypothesis record.

        Args:
            h: Hypothesis to insert.
        """
        self._conn.execute(
            """INSERT INTO hypotheses (
                hypothesis_id, user_id, type, scope_json, claim, status,
                version, elo_rating, elo_games_played, bayes_confidence,
                support_count, contradict_count, last_contradiction_at,
                fsrs_state_json, source_type, decay_immune,
                created_at, last_applied, last_seen,
                approval_state, pattern_hash, scope_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                h.hypothesis_id,
                h.user_id,
                h.type,
                h.scope.to_json(),
                h.claim,
                h.status,
                h.version,
                h.elo_rating,
                h.elo_games_played,
                h.bayes_confidence,
                h.support_count,
                h.contradict_count,
                h.last_contradiction_at,
                h.fsrs_state_json,
                h.source_type,
                1 if h.decay_immune else 0,
                h.created_at,
                h.last_applied,
                h.last_seen,
                h.approval_state,
                h.pattern_hash,
                h.scope_hash,
            ),
        )

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve a hypothesis by ID.

        Args:
            hypothesis_id: The hypothesis ID.

        Returns:
            Hypothesis or None.
        """
        row = self._conn.fetchone(
            "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )
        if row is None:
            return None
        return self._row_to_hypothesis(row)

    def get_hypotheses_by_user(
        self,
        user_id: int,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[Hypothesis]:
        """Retrieve hypotheses for a user, optionally filtered by status.

        Args:
            user_id: Telegram user ID.
            status: Optional status filter.
            limit: Maximum results.

        Returns:
            List of Hypothesis objects.
        """
        if status is not None:
            rows = self._conn.fetchall(
                "SELECT * FROM hypotheses WHERE user_id = ? AND status = ? "
                "ORDER BY last_seen DESC LIMIT ?",
                (user_id, status, limit),
            )
        else:
            rows = self._conn.fetchall(
                "SELECT * FROM hypotheses WHERE user_id = ? "
                "ORDER BY last_seen DESC LIMIT ?",
                (user_id, limit),
            )
        return [self._row_to_hypothesis(r) for r in rows]

    def count_active_hypotheses(self, user_id: int) -> int:
        """Count active hypotheses for a user (for HC-SC-8: max 50 check).

        Args:
            user_id: Telegram user ID.

        Returns:
            Number of active hypotheses.
        """
        row = self._conn.fetchone(
            "SELECT count(*) as cnt FROM hypotheses "
            "WHERE user_id = ? AND status = 'active'",
            (user_id,),
        )
        return row["cnt"] if row else 0

    # ── Evidence CRUD ────────────────────────────────────────

    def insert_evidence(
        self,
        evidence_id: str,
        hypothesis_id: str,
        hypothesis_version: int,
        signal_type: str,
        signal_strength: float,
        created_at: str,
        *,
        episode_id: Optional[str] = None,
        request_id: Optional[str] = None,
        response_id: Optional[str] = None,
    ) -> None:
        """Insert a new evidence record into the ledger.

        Args:
            evidence_id: Unique evidence ID.
            hypothesis_id: Associated hypothesis.
            hypothesis_version: Version this evidence supports.
            signal_type: Type of signal (no_correction, correction, etc.).
            signal_strength: Signal strength [0, 1].
            created_at: ISO-8601 timestamp.
            episode_id: Optional conversation episode ID.
            request_id: Optional request ID.
            response_id: Optional response ID.
        """
        self._conn.execute(
            """INSERT INTO hypothesis_evidence (
                evidence_id, hypothesis_id, hypothesis_version,
                episode_id, request_id, response_id,
                signal_type, signal_strength, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence_id,
                hypothesis_id,
                hypothesis_version,
                episode_id,
                request_id,
                response_id,
                signal_type,
                signal_strength,
                created_at,
            ),
        )

    def get_evidence_for_hypothesis(
        self,
        hypothesis_id: str,
        version: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve evidence records for a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID.
            version: Optional version filter.

        Returns:
            List of evidence dicts.
        """
        if version is not None:
            rows = self._conn.fetchall(
                "SELECT * FROM hypothesis_evidence "
                "WHERE hypothesis_id = ? AND hypothesis_version = ? "
                "ORDER BY created_at DESC",
                (hypothesis_id, version),
            )
        else:
            rows = self._conn.fetchall(
                "SELECT * FROM hypothesis_evidence "
                "WHERE hypothesis_id = ? ORDER BY created_at DESC",
                (hypothesis_id,),
            )
        return [dict(r) for r in rows]

    # ── Tombstone CRUD ───────────────────────────────────────

    def insert_tombstone(
        self,
        tombstone_id: str,
        hypothesis_id: str,
        fingerprint: str,
        deleted_at: str,
        *,
        scope_hash: Optional[str] = None,
        expires_at: Optional[str] = None,
        permanent: bool = False,
    ) -> None:
        """Insert a tombstone record.

        Args:
            tombstone_id: Unique tombstone ID.
            hypothesis_id: The deleted hypothesis ID.
            fingerprint: Pattern fingerprint for matching.
            deleted_at: Deletion timestamp.
            scope_hash: Optional scope hash.
            expires_at: Expiration timestamp (None for permanent).
            permanent: Whether this is a permanent tombstone.
        """
        self._conn.execute(
            """INSERT INTO hypothesis_tombstones (
                tombstone_id, hypothesis_id, fingerprint, scope_hash,
                deleted_at, expires_at, permanent
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                tombstone_id,
                hypothesis_id,
                fingerprint,
                scope_hash,
                deleted_at,
                expires_at,
                1 if permanent else 0,
            ),
        )

    def check_tombstone(self, fingerprint: str) -> bool:
        """Check if a fingerprint has an active (non-expired) tombstone.

        Args:
            fingerprint: Pattern fingerprint hash.

        Returns:
            True if a blocking tombstone exists.
        """
        row = self._conn.fetchone(
            "SELECT 1 FROM hypothesis_tombstones "
            "WHERE fingerprint = ? AND (permanent = 1 OR expires_at > datetime('now'))",
            (fingerprint,),
        )
        return row is not None

    # ── Pattern Difficulty ───────────────────────────────────

    def upsert_pattern_difficulty(
        self,
        fingerprint_hash: str,
        difficulty_rating: float,
        games_played: int,
        last_updated: str,
    ) -> None:
        """Insert or update a pattern difficulty record.

        Args:
            fingerprint_hash: Fingerprint hash of the pattern.
            difficulty_rating: Current Elo difficulty rating.
            games_played: Number of games played.
            last_updated: ISO-8601 timestamp.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO pattern_difficulty (
                fingerprint_hash, difficulty_rating, games_played, last_updated
            ) VALUES (?, ?, ?, ?)""",
            (fingerprint_hash, difficulty_rating, games_played, last_updated),
        )

    def get_pattern_difficulty(self, fingerprint_hash: str) -> Optional[dict]:
        """Retrieve pattern difficulty by fingerprint.

        Args:
            fingerprint_hash: The fingerprint hash.

        Returns:
            Dict with difficulty data or None.
        """
        row = self._conn.fetchone(
            "SELECT * FROM pattern_difficulty WHERE fingerprint_hash = ?",
            (fingerprint_hash,),
        )
        return dict(row) if row else None

    # ── Version CRUD ─────────────────────────────────────────

    def insert_version(
        self,
        version_id: str,
        hypothesis_id: str,
        version: int,
        claim: str,
        elo_rating_at_save: float,
        created_at: str,
        *,
        change_reason: Optional[str] = None,
        predecessor_context: Optional[str] = None,
    ) -> None:
        """Insert a hypothesis version record.

        Args:
            version_id: Unique version ID.
            hypothesis_id: Associated hypothesis.
            version: Version number.
            claim: The claim text at this version.
            elo_rating_at_save: Elo rating when version was saved.
            created_at: ISO-8601 timestamp.
            change_reason: Reason for the version change.
            predecessor_context: Reference to prior version evidence.
        """
        self._conn.execute(
            """INSERT INTO hypothesis_versions (
                version_id, hypothesis_id, version, claim,
                elo_rating_at_save, change_reason, predecessor_context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                hypothesis_id,
                version,
                claim,
                elo_rating_at_save,
                change_reason,
                predecessor_context,
                created_at,
            ),
        )

    # ── Local Eval Set ───────────────────────────────────────

    def insert_eval_example(
        self,
        eval_id: str,
        hypothesis_id: str,
        example_input: str,
        example_output: str,
    ) -> None:
        """Insert a local evaluation example.

        Args:
            eval_id: Unique eval ID.
            hypothesis_id: Associated hypothesis.
            example_input: Example input text.
            example_output: Expected output text.
        """
        self._conn.execute(
            """INSERT INTO hypothesis_local_eval_set (
                eval_id, hypothesis_id, example_input, example_output
            ) VALUES (?, ?, ?, ?)""",
            (eval_id, hypothesis_id, example_input, example_output),
        )

    # ── Alias CRUD ───────────────────────────────────────────

    def insert_alias(
        self,
        alias_id: str,
        hypothesis_id: str,
        alias_text: str,
        first_seen: str,
        last_seen: str,
        *,
        confidence: float = 0.5,
        evidence_count: int = 0,
    ) -> None:
        """Insert a hypothesis alias.

        Args:
            alias_id: Unique alias ID.
            hypothesis_id: Associated hypothesis.
            alias_text: The alias text.
            first_seen: First seen timestamp.
            last_seen: Last seen timestamp.
            confidence: Alias confidence.
            evidence_count: Number of supporting evidence items.
        """
        self._conn.execute(
            """INSERT INTO hypothesis_aliases (
                alias_id, hypothesis_id, alias_text, confidence,
                evidence_count, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                alias_id,
                hypothesis_id,
                alias_text,
                confidence,
                evidence_count,
                first_seen,
                last_seen,
            ),
        )

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _row_to_hypothesis(row) -> Hypothesis:
        """Convert a DB row to a Hypothesis dataclass.

        Args:
            row: sqlite3.Row or dict-like object.

        Returns:
            Hypothesis instance.
        """
        return Hypothesis(
            hypothesis_id=row["hypothesis_id"],
            user_id=row["user_id"],
            type=row["type"],
            scope=HypothesisScope.from_json(row["scope_json"]),
            claim=row["claim"],
            status=row["status"],
            version=row["version"],
            elo_rating=row["elo_rating"],
            elo_games_played=row["elo_games_played"],
            bayes_confidence=row["bayes_confidence"],
            support_count=row["support_count"],
            contradict_count=row["contradict_count"],
            last_contradiction_at=row["last_contradiction_at"],
            fsrs_state_json=row["fsrs_state_json"],
            source_type=row["source_type"],
            decay_immune=bool(row["decay_immune"]),
            created_at=row["created_at"],
            last_applied=row["last_applied"],
            last_seen=row["last_seen"],
            approval_state=row["approval_state"],
            pattern_hash=row["pattern_hash"],
            scope_hash=row["scope_hash"],
        )
