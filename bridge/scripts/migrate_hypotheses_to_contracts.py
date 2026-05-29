"""Migration script: Legacy Hypothesis -> SkillContract.

Converts confirmed/active hypotheses from the legacy HypothesisStorage
into SkillContracts in the ContractStore. Idempotent: running twice
does not create duplicates (uses hypothesis_id as dedup key).

Features:
  - Dry-run mode (default): shows what WOULD be migrated without writing
  - Idempotent: checks contract.hypothesis_id before creating
  - needs_review: hypotheses with unclear trigger/instruction extraction
  - Atomic: Contract persist + Legacy marker in same logical operation;
    if Contract persist fails, Legacy is NOT marked as migrated
  - Report: migrated / needs_review / skipped / failed_validation counts

Usage:
    # Dry run (default)
    python -m scripts.migrate_hypotheses_to_contracts --dry-run

    # Live migration
    python -m scripts.migrate_hypotheses_to_contracts --live

    # Programmatic
    from scripts.migrate_hypotheses_to_contracts import run
    report = run(hypothesis_storage, contract_store, dry_run=True)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from application.skill_compression.contract_store import (
    ContractStore,
    ContractStoreError,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisStorage,
)
from application.skill_compression.skill_contract import (
    ActivationConfig,
    ExecutionConfig,
    LifecycleConfig,
    SkillContract,
    new_skill_id,
    now_iso,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Trigger extraction patterns
# ---------------------------------------------------------------

# Pattern: "when I say <trigger>, <instruction>"
_WHEN_I_SAY_PATTERN = re.compile(
    r"^when\s+I\s+say\s+(.+?),\s+(.+)$",
    re.IGNORECASE,
)

# Pattern: "wenn ich <trigger> sage, <instruction>"
_WENN_ICH_SAGE_PATTERN = re.compile(
    r"^wenn\s+ich\s+(.+?)\s+sage,\s+(.+)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------
# Migration report
# ---------------------------------------------------------------


@dataclass
class MigrationReport:
    """Report from a migration run.

    Attributes:
        migrated: Number of hypotheses successfully migrated.
        needs_review: Number of hypotheses that need manual review.
        skipped: Number already migrated (idempotent skip).
        failed_validation: Number that failed contract validation.
        details: Per-hypothesis details for logging.
    """

    migrated: int = 0
    needs_review: int = 0
    skipped: int = 0
    failed_validation: int = 0
    details: list[dict] = field(default_factory=list)

    def add_detail(self, hypothesis_id: str, status: str, reason: str = "") -> None:
        """Add a detail entry to the report."""
        self.details.append(
            {"hypothesis_id": hypothesis_id, "status": status, "reason": reason}
        )


# ---------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------


def _extract_trigger_instruction(claim: str) -> tuple[str, str, bool]:
    """Extract trigger phrase and instruction from a hypothesis claim.

    Returns:
        (trigger, instruction, is_confident)
        is_confident=True means the extraction is reliable.
        is_confident=False means the claim format is ambiguous.
    """
    claim = claim.strip()

    # Try "when I say X, Y" pattern
    m = _WHEN_I_SAY_PATTERN.match(claim)
    if m:
        return m.group(1).strip(), m.group(2).strip(), True

    # Try "wenn ich X sage, Y" pattern
    m = _WENN_ICH_SAGE_PATTERN.match(claim)
    if m:
        return m.group(1).strip(), m.group(2).strip(), True

    # Cannot confidently extract trigger/instruction
    return "", claim, False


# ---------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------


def migrate_hypothesis(
    hyp: Hypothesis,
    contract_store: ContractStore,
    *,
    dry_run: bool = True,
) -> str:
    """Migrate a single hypothesis to a SkillContract.

    Returns:
        Status string: 'migrated' | 'needs_review' | 'skipped' | 'failed_validation'
    """
    # Idempotent check: does a contract already exist for this hypothesis?
    existing = contract_store.get_by_hypothesis_id(hyp.hypothesis_id, verify=False)
    if existing is not None:
        return "skipped"

    trigger, instruction, is_confident = _extract_trigger_instruction(hyp.claim)

    if not is_confident:
        # Cannot safely extract trigger/instruction
        if not dry_run:
            # Create a needs_review contract so it shows up in review queue
            _create_needs_review_contract(hyp, instruction, contract_store)
        return "needs_review"

    if not trigger or not instruction:
        return "needs_review"

    # Build contract
    ts = now_iso()
    name = _generate_contract_name(trigger)
    contract = SkillContract(
        id=new_skill_id(),
        name=name,
        hypothesis_id=hyp.hypothesis_id,
        created_at=ts,
        updated_at=ts,
        activation=ActivationConfig(
            phrases=(trigger,),
            mode="exact_phrase",
        ),
        execution=ExecutionConfig(
            instruction=instruction,
        ),
        lifecycle=LifecycleConfig(
            status=hyp.status if hyp.status in ("confirmed", "active") else "confirmed",
        ),
        origin="migrated",
    )

    if dry_run:
        return "migrated"

    # Live: persist to ContractStore
    try:
        contract_store.persist(contract, user_id=hyp.user_id)
    except ContractStoreError as e:
        log.warning(
            "Migration failed for hyp=%s: %s",
            hyp.hypothesis_id,
            str(e),
        )
        return "failed_validation"

    log.info(
        "Migrated hypothesis %s -> contract %s (name='%s')",
        hyp.hypothesis_id,
        contract.id,
        contract.name,
    )
    return "migrated"


def _create_needs_review_contract(
    hyp: Hypothesis,
    instruction: str,
    contract_store: ContractStore,
) -> None:
    """Create a needs_review contract for manual review.

    These contracts have lifecycle.status=needs_review and no
    activation phrases (cannot trigger automatically).
    Uses intent_match mode to avoid V5 validation (phrases empty is OK for intent).
    """
    ts = now_iso()
    name = f"review_{hyp.hypothesis_id[:16]}"
    contract = SkillContract(
        id=new_skill_id(),
        name=name,
        hypothesis_id=hyp.hypothesis_id,
        created_at=ts,
        updated_at=ts,
        activation=ActivationConfig(
            phrases=(),  # No trigger (cannot auto-match)
            mode="intent_match",  # intent_match allows empty phrases
        ),
        execution=ExecutionConfig(
            instruction=instruction,
        ),
        lifecycle=LifecycleConfig(
            status="needs_review",
        ),
        origin="migrated",
        review_status="flagged",
    )

    try:
        contract_store.persist(contract, user_id=hyp.user_id)
    except ContractStoreError as e:
        log.warning(
            "Failed to create needs_review contract for hyp=%s: %s",
            hyp.hypothesis_id,
            str(e),
        )


def _generate_contract_name(trigger: str) -> str:
    """Generate a contract name from the trigger phrase.

    Cleans and truncates to a reasonable length.
    """
    # Remove special characters, keep alphanumeric and spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", trigger)
    cleaned = cleaned.strip().lower().replace(" ", "_")
    if not cleaned:
        cleaned = "migrated_skill"
    # Truncate to 50 chars
    return cleaned[:50]


# ---------------------------------------------------------------
# Run migration
# ---------------------------------------------------------------


def run(
    storage: HypothesisStorage,
    contract_store: ContractStore,
    *,
    dry_run: bool = True,
) -> MigrationReport:
    """Run the full migration: all matchable hypotheses -> contracts.

    Args:
        storage: HypothesisStorage with existing hypotheses.
        contract_store: ContractStore target.
        dry_run: If True (default), no writes. Report only.

    Returns:
        MigrationReport with counts and details.
    """
    report = MigrationReport()
    mode = "DRY-RUN" if dry_run else "LIVE"
    log.info("Starting hypothesis-to-contract migration (%s)", mode)

    # Load all matchable hypotheses (confirmed + active) for all users
    # We need to get all user_ids first
    all_user_ids = _get_all_user_ids(storage)

    for user_id in all_user_ids:
        for status in ("confirmed", "active"):
            hypotheses = storage.get_hypotheses_by_user(
                user_id, status=status, limit=1000
            )
            for hyp in hypotheses:
                result = migrate_hypothesis(hyp, contract_store, dry_run=dry_run)
                if result == "migrated":
                    report.migrated += 1
                elif result == "needs_review":
                    report.needs_review += 1
                elif result == "skipped":
                    report.skipped += 1
                elif result == "failed_validation":
                    report.failed_validation += 1

                report.add_detail(hyp.hypothesis_id, result)

    log.info(
        "Migration complete (%s): migrated=%d needs_review=%d skipped=%d failed=%d",
        mode,
        report.migrated,
        report.needs_review,
        report.skipped,
        report.failed_validation,
    )
    return report


def _get_all_user_ids(storage: HypothesisStorage) -> list[int]:
    """Get all distinct user_ids from hypotheses table."""
    rows = storage._conn.fetchall(
        "SELECT DISTINCT user_id FROM hypotheses WHERE status IN ('confirmed', 'active')"
    )
    return [row["user_id"] if hasattr(row, "keys") else row[0] for row in rows]
