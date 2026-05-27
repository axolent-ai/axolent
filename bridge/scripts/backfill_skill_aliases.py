"""One-time backfill: extract trigger aliases for existing skills.

Round-4 fix (2026-05-27): The pattern extraction was previously missing
the reversed form "wenn ich schreibe X" (only "wenn ich X schreibe"
worked). Skills stored before this fix may have no aliases in
hypothesis_aliases. This script re-runs extraction on all
confirmed/active hypotheses and inserts missing aliases.

Idempotent: safe to run multiple times (skips existing aliases).
Startup hook: runs once per schema version via marker in hypothesis_aliases.

Usage:
    cd bridge/
    .venv/Scripts/python.exe scripts/backfill_skill_aliases.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from uuid import uuid4

# Ensure bridge/ is on sys.path
_bridge_root = Path(__file__).resolve().parent.parent
if str(_bridge_root) not in sys.path:
    sys.path.insert(0, str(_bridge_root))

from datetime import datetime, timezone  # noqa: E402

from application.skill_compression.hypothesis_storage import HypothesisStorage  # noqa: E402
from application.skill_compression.skill_learning_service import (  # noqa: E402
    _extract_trigger_aliases,
)

log = logging.getLogger(__name__)

# Schema version marker: stored in a dedicated migration_markers table.
# Avoids FK constraint issues with hypothesis_aliases.
_BACKFILL_MARKER_KEY = "backfill_aliases_v1"


def _ensure_marker_table(storage: HypothesisStorage) -> None:
    """Create the migration_markers table if it does not exist."""
    storage._conn.execute(
        """CREATE TABLE IF NOT EXISTS migration_markers (
            marker_key TEXT PRIMARY KEY,
            completed_at TEXT NOT NULL
        )"""
    )


def _backfill_already_done(storage: HypothesisStorage) -> bool:
    """Check if the backfill marker exists."""
    _ensure_marker_table(storage)
    row = storage._conn.fetchone(
        "SELECT marker_key FROM migration_markers WHERE marker_key = ?",
        (_BACKFILL_MARKER_KEY,),
    )
    return row is not None


def _set_backfill_done(storage: HypothesisStorage) -> None:
    """Write the backfill marker to prevent re-runs."""
    _ensure_marker_table(storage)
    now_iso = datetime.now(timezone.utc).isoformat()
    storage._conn.execute(
        "INSERT OR IGNORE INTO migration_markers (marker_key, completed_at) "
        "VALUES (?, ?)",
        (_BACKFILL_MARKER_KEY, now_iso),
    )


def run(storage: HypothesisStorage) -> tuple[int, int]:
    """Run the backfill: extract and insert missing aliases.

    Args:
        storage: HypothesisStorage instance with active DB connection.

    Returns:
        Tuple of (aliases_added, hypotheses_processed).
    """
    if _backfill_already_done(storage):
        log.info("backfill_skill_aliases: already done (v1 marker found)")
        return 0, 0

    # Load all confirmed + active hypotheses across all users
    rows = storage._conn.fetchall(
        "SELECT hypothesis_id, claim FROM hypotheses "
        "WHERE status IN ('confirmed', 'active')",
    )

    aliases_added = 0
    hypotheses_processed = 0

    for row in rows:
        hyp_id = row["hypothesis_id"]
        claim = row["claim"]
        hypotheses_processed += 1

        # Extract aliases from claim text using the (now-fixed) extractor
        extracted = _extract_trigger_aliases(claim)
        if not extracted:
            continue

        # Get existing aliases for this hypothesis
        existing_rows = storage._conn.fetchall(
            "SELECT LOWER(alias_text) as alias_lower "
            "FROM hypothesis_aliases WHERE hypothesis_id = ?",
            (hyp_id,),
        )
        existing_set = {r["alias_lower"] for r in existing_rows}

        # Insert missing aliases
        now_iso = datetime.now(timezone.utc).isoformat()
        for alias_text in extracted:
            if alias_text.lower() in existing_set:
                continue
            alias_id = f"alias_{uuid4().hex[:12]}"
            storage.insert_alias(
                alias_id=alias_id,
                hypothesis_id=hyp_id,
                alias_text=alias_text,
                first_seen=now_iso,
                last_seen=now_iso,
                confidence=0.9,
                evidence_count=1,
            )
            existing_set.add(alias_text.lower())
            aliases_added += 1
            log.info(
                "backfill: added alias (len=%d) to hypothesis %s",
                len(alias_text),
                hyp_id,
            )

    # Mark as done
    _set_backfill_done(storage)
    log.info(
        "backfill_skill_aliases: %d aliases added across %d hypotheses",
        aliases_added,
        hypotheses_processed,
    )
    return aliases_added, hypotheses_processed


def main() -> None:
    """CLI entry point: run backfill against production DB."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from dotenv import load_dotenv

    load_dotenv()

    db_path = _bridge_root / "data" / "axolent.db"
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    from infrastructure.sqlite_storage import SqliteConnection

    conn = SqliteConnection(db_path)
    storage = HypothesisStorage(conn)

    added, processed = run(storage)
    print(f"{added} aliases added across {processed} hypotheses")


if __name__ == "__main__":
    main()
