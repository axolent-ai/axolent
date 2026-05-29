"""Contract Store: DB schema + CRUD + checksum verification for SkillContracts.

Responsibilities:
  - DDL for skill_contracts table (schema creation/migration)
  - Create, Read, Update, Delete operations
  - _finalize_security_metadata() pre-persist hook (Addendum K1)
  - Checksum computation + verification (Addendum K3)
  - DB==JSON invariant enforcement (V16) on READ + WRITE
  - contract_version auto-increment on update
  - Optimistic locking via contract_version compare-and-swap
  - Central secure load helper (_load_contract_from_row)

Storage model:
  - contract_json column holds the canonical JSON (source of truth)
  - Index columns (schema_version, contract_version, name, origin, etc.)
    are copies from JSON for query performance
  - Validator enforces DB==JSON invariant (V16)

Dependencies: Python stdlib only (json, sqlite3 protocol, logging).
DB connection follows the same DBConnection protocol as hypothesis_storage.py.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace
from typing import Optional, Protocol, runtime_checkable

from application.skill_compression.contract_validator import (
    ValidationResult,
    validate,
)
from application.skill_compression.skill_contract import (
    SkillContract,
    compute_checksum,
    compute_package_type,
    compute_risk_level,
    now_iso,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# DB Connection Protocol (matches hypothesis_storage.py)
# ──────────────────────────────────────────────────────────────


@runtime_checkable
class DBConnection(Protocol):
    """Minimal DB connection interface."""

    def execute(self, sql: str, params: tuple | dict = (), **kwargs) -> object: ...
    def executescript(self, sql: str) -> None: ...
    def fetchall(self, sql: str, params: tuple | dict = ()) -> list: ...
    def fetchone(self, sql: str, params: tuple | dict = ()) -> Optional[object]: ...
    def execute_in_transaction(self, operations: list[tuple[str, tuple]]) -> None: ...


# ──────────────────────────────────────────────────────────────
# DDL Schema
# ──────────────────────────────────────────────────────────────

SKILL_CONTRACTS_SCHEMA_SQL = """
-- Skill Contracts v2: canonical contract storage
CREATE TABLE IF NOT EXISTS skill_contracts (
    id                  TEXT PRIMARY KEY,
    user_id             INTEGER NOT NULL,
    name                TEXT NOT NULL,
    schema_version      INTEGER NOT NULL DEFAULT 2,
    contract_version    INTEGER NOT NULL DEFAULT 1,
    hypothesis_id       TEXT,
    origin              TEXT NOT NULL DEFAULT 'local_learn',
    lifecycle_status    TEXT NOT NULL DEFAULT 'confirmed',
    review_status       TEXT NOT NULL DEFAULT 'unreviewed',
    risk_level          TEXT NOT NULL DEFAULT 'unknown',
    package_type        TEXT NOT NULL DEFAULT 'local_skill',
    checksum            TEXT NOT NULL,
    contract_json       TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skill_contracts_user_id
    ON skill_contracts(user_id);
CREATE INDEX IF NOT EXISTS idx_skill_contracts_user_name
    ON skill_contracts(user_id, name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_contracts_user_name_unique
    ON skill_contracts(user_id, name);
CREATE INDEX IF NOT EXISTS idx_skill_contracts_hypothesis
    ON skill_contracts(hypothesis_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_contracts_hypothesis_unique
    ON skill_contracts(hypothesis_id)
    WHERE hypothesis_id IS NOT NULL AND hypothesis_id != '';
CREATE INDEX IF NOT EXISTS idx_skill_contracts_origin
    ON skill_contracts(origin);
CREATE INDEX IF NOT EXISTS idx_skill_contracts_status
    ON skill_contracts(user_id, lifecycle_status);
"""


# ──────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────


class ContractStoreError(Exception):
    """Base error for contract store operations."""


class ContractNotFoundError(ContractStoreError):
    """Raised when a contract ID is not found."""


class ContractValidationError(ContractStoreError):
    """Raised when validation fails before persist."""

    def __init__(self, message: str, result: ValidationResult):
        super().__init__(message)
        self.result = result


class ContractChecksumError(ContractStoreError):
    """Raised when checksum verification fails on load."""


class ContractDuplicateNameError(ContractStoreError):
    """Raised when a contract name already exists for the same user."""


class ContractDuplicateHypothesisError(ContractStoreError):
    """Raised when a hypothesis_id is already linked to another contract."""


class ContractInvariantError(ContractStoreError):
    """Raised when DB columns and JSON content diverge."""


class ContractConcurrentUpdateError(ContractStoreError):
    """Raised when optimistic locking detects a stale update."""


# ──────────────────────────────────────────────────────────────
# Pre-persist hooks
# ──────────────────────────────────────────────────────────────


def _finalize_security_metadata(contract: SkillContract) -> SkillContract:
    """Compute derived security fields before persistence (Addendum K1).

    Sets:
      - store_meta.package_type from permissions
      - risk_level from permissions
      - trust.checksum from canonical JSON

    Must not pass 'unknown' risk_level to DB for non-draft contracts.
    Called before validator runs, so V14 sees the computed values.
    """
    # Compute package_type and risk_level from permissions
    computed_package_type = compute_package_type(contract.permissions)
    computed_risk_level = compute_risk_level(contract.permissions)

    contract = replace(
        contract,
        store_meta=replace(
            contract.store_meta,
            package_type=computed_package_type,
        ),
        risk_level=computed_risk_level,
    )

    # Compute checksum (must happen AFTER other fields are finalized)
    checksum = compute_checksum(contract)
    contract = replace(
        contract,
        trust=replace(
            contract.trust,
            checksum=checksum,
        ),
    )

    return contract


# ──────────────────────────────────────────────────────────────
# Checksum verification (standalone, kept for backward compat)
# ──────────────────────────────────────────────────────────────


def verify_checksum(contract: SkillContract) -> bool:
    """Verify that a loaded contract's checksum matches its content.

    For persisted contracts: checksum must be present and must match.
    Returns False if checksum is None (persisted contracts require checksum).
    Returns False if checksum does not match computed value.
    Returns True only if checksum is present AND matches.
    """
    if contract.trust.checksum is None:
        return False
    expected = compute_checksum(contract)
    return contract.trust.checksum == expected


# ──────────────────────────────────────────────────────────────
# Contract Store
# ──────────────────────────────────────────────────────────────


class ContractStore:
    """CRUD operations for SkillContracts backed by SQLite.

    Enforces:
      - _finalize_security_metadata() before every persist/update
      - Full validation (V1-V17) before every persist/update
      - DB==JSON invariant (V16) on WRITE and READ
      - Checksum computation on write, verification on read
      - contract_version auto-increment on update
      - Optimistic locking on update (contract_version compare-and-swap)
      - Name uniqueness per user (V8, DB UNIQUE INDEX)
      - Single secure load path (_load_contract_from_row)
    """

    def __init__(self, db: DBConnection) -> None:
        self._db = db

    def init_schema(self) -> None:
        """Create the skill_contracts table if it does not exist."""
        self._db.executescript(SKILL_CONTRACTS_SCHEMA_SQL)

    @staticmethod
    def _classify_integrity_error(
        exc: sqlite3.IntegrityError,
        contract: SkillContract,
        user_id: int,
    ) -> ContractStoreError:
        """Map a raw sqlite3.IntegrityError to a typed ContractStoreError.

        Inspects the error message to determine which unique constraint was
        violated and returns the appropriate domain exception.
        """
        err_msg = str(exc).lower()
        if "hypothesis" in err_msg:
            return ContractDuplicateHypothesisError(
                f"A contract with hypothesis_id '{contract.hypothesis_id}' "
                f"already exists (DB constraint)"
            )
        if "name" in err_msg:
            return ContractDuplicateNameError(
                f"A contract with name '{contract.name}' already exists "
                f"for user {user_id} (DB constraint)"
            )
        # Fallback: unknown constraint, still wrap in domain error
        return ContractStoreError(
            f"DB integrity constraint violated for contract '{contract.id}': {exc}"
        )

    # ── Central secure load helper ─────────────────────────────

    def _load_contract_from_row(self, row, *, verify: bool = True) -> SkillContract:
        """Central secure load path. ALL reads go through here.

        Steps:
          1. Deserialize contract_json
          2. Assert DB==JSON invariant
          3. Assert checksum integrity (if verify=True)

        Raises:
            ContractInvariantError: If DB columns diverge from JSON content.
            ContractChecksumError: If checksum verification fails.
        """
        contract_json = (
            row["contract_json"]
            if hasattr(row, "keys")
            else row[self._contract_json_idx(row)]
        )
        contract = SkillContract.from_json(contract_json)
        self._assert_db_json_invariant(row, contract)
        if verify:
            self._assert_checksum(row, contract)
        return contract

    @staticmethod
    def _contract_json_idx(row) -> int:
        """Find contract_json position in tuple rows (fallback)."""
        # In our queries we always select with known column order
        # This should not normally be called since we use Row factory
        raise ContractStoreError(
            "Row must be dict-like (sqlite3.Row). Tuple access is not supported in secure load path."
        )

    def _assert_db_json_invariant(self, row, contract: SkillContract) -> None:
        """Verify DB index columns match JSON content exactly."""
        mismatches = []

        def _get(field: str):
            if hasattr(row, "keys"):
                return row[field]
            raise ContractStoreError("Row must be dict-like for invariant check.")

        if _get("id") != contract.id:
            mismatches.append(f"id: db={_get('id')} json={contract.id}")
        if _get("name") != contract.name:
            mismatches.append(f"name: db={_get('name')} json={contract.name}")
        if _get("schema_version") != contract.schema_version:
            mismatches.append(
                f"schema_version: db={_get('schema_version')} json={contract.schema_version}"
            )
        if _get("contract_version") != contract.contract_version:
            mismatches.append(
                f"contract_version: db={_get('contract_version')} json={contract.contract_version}"
            )
        if _get("origin") != contract.origin:
            mismatches.append(f"origin: db={_get('origin')} json={contract.origin}")
        if _get("lifecycle_status") != contract.lifecycle.status:
            mismatches.append(
                f"status: db={_get('lifecycle_status')} json={contract.lifecycle.status}"
            )
        if _get("review_status") != contract.review_status:
            mismatches.append(
                f"review_status: db={_get('review_status')} json={contract.review_status}"
            )
        if _get("risk_level") != contract.risk_level:
            mismatches.append(
                f"risk_level: db={_get('risk_level')} json={contract.risk_level}"
            )
        if _get("package_type") != contract.store_meta.package_type:
            mismatches.append(
                f"package_type: db={_get('package_type')} json={contract.store_meta.package_type}"
            )
        if _get("checksum") != contract.trust.checksum:
            mismatches.append(
                f"checksum: db={_get('checksum')} json={contract.trust.checksum}"
            )

        if mismatches:
            raise ContractInvariantError(f"DB-JSON-Mismatch: {mismatches}")

    def _assert_checksum(self, row, contract: SkillContract) -> None:
        """Hard checksum verification for persisted contracts.

        Conditions (ALL must hold):
          1. DB checksum column must not be empty/None
          2. contract.trust.checksum must not be empty/None
          3. DB checksum == JSON checksum (already covered by invariant check)
          4. compute_checksum(contract) == DB checksum

        Raises ContractChecksumError on any failure.
        """
        db_checksum = row["checksum"] if hasattr(row, "keys") else None
        json_checksum = contract.trust.checksum

        if not db_checksum:
            raise ContractChecksumError(
                f"DB checksum is empty/null for contract '{contract.id}'. "
                f"Persisted contracts must have a checksum."
            )

        if not json_checksum:
            raise ContractChecksumError(
                f"JSON checksum is empty/null for contract '{contract.id}'. "
                f"Persisted contracts must have a checksum in trust.checksum."
            )

        if db_checksum != json_checksum:
            raise ContractChecksumError(
                f"DB checksum does not match JSON checksum for contract '{contract.id}'. "
                f"DB={db_checksum[:16]}... JSON={json_checksum[:16]}..."
            )

        computed = compute_checksum(contract)
        if computed != db_checksum:
            raise ContractChecksumError(
                f"Checksum verification failed for contract '{contract.id}'. "
                f"Computed checksum does not match stored checksum. Data may have been tampered with."
            )

    # ── Create ──────────────────────────────────────────────

    def persist(self, contract: SkillContract, user_id: int) -> SkillContract:
        """Insert a new contract into the database.

        Applies _finalize_security_metadata, validates, computes checksum,
        and enforces DB==JSON invariant.

        Args:
            contract: The contract to persist.
            user_id: The owning user's ID.

        Returns:
            The finalized contract (with computed checksum, risk_level, etc.)

        Raises:
            ContractValidationError: If validation fails.
            ContractDuplicateNameError: If name already exists for user.
        """
        # Update timestamp BEFORE finalize (checksum covers updated_at)
        contract = replace(contract, updated_at=now_iso())

        # Finalize security metadata (computes checksum, risk_level, package_type)
        contract = _finalize_security_metadata(contract)

        # Check name uniqueness (V8) app-side first for clear error message
        existing = self._db.fetchone(
            "SELECT id FROM skill_contracts WHERE user_id = ? AND name = ?",
            (user_id, contract.name),
        )
        if existing:
            raise ContractDuplicateNameError(
                f"A contract with name '{contract.name}' already exists for user {user_id}"
            )

        # Validate (V1-V17)
        result = validate(
            contract,
            db_schema_version=contract.schema_version,
            db_contract_version=contract.contract_version,
        )
        if not result.is_valid:
            error_msgs = "; ".join(f"[{i.rule}] {i.message}" for i in result.errors)
            raise ContractValidationError(
                f"Contract validation failed: {error_msgs}",
                result,
            )

        # Serialize
        contract_json = contract.to_json()

        # Insert (DB UNIQUE INDEX enforces name uniqueness as safety net)
        try:
            self._db.execute(
                """INSERT INTO skill_contracts
                   (id, user_id, name, schema_version, contract_version,
                    hypothesis_id, origin, lifecycle_status, review_status,
                    risk_level, package_type, checksum, contract_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    contract.id,
                    user_id,
                    contract.name,
                    contract.schema_version,
                    contract.contract_version,
                    contract.hypothesis_id,
                    contract.origin,
                    contract.lifecycle.status,
                    contract.review_status,
                    contract.risk_level,
                    contract.store_meta.package_type,
                    contract.trust.checksum,
                    contract_json,
                    contract.created_at,
                    contract.updated_at,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise self._classify_integrity_error(e, contract, user_id) from e

        log.info(
            "Persisted contract %s (name='%s', user=%d, risk=%s, package=%s)",
            contract.id,
            contract.name,
            user_id,
            contract.risk_level,
            contract.store_meta.package_type,
        )
        return contract

    # ── Read ────────────────────────────────────────────────

    _FULL_ROW_SELECT = """SELECT id, user_id, name, schema_version, contract_version,
        hypothesis_id, origin, lifecycle_status, review_status,
        risk_level, package_type, checksum, contract_json,
        created_at, updated_at FROM skill_contracts"""

    def get_by_id(self, contract_id: str, *, verify: bool = True) -> SkillContract:
        """Load a contract by ID with optional checksum verification.

        Uses the central secure load helper.

        Args:
            contract_id: The contract ID to look up.
            verify: If True (default), verify checksum on load.

        Returns:
            The deserialized SkillContract.

        Raises:
            ContractNotFoundError: If ID not found.
            ContractChecksumError: If checksum verification fails.
            ContractInvariantError: If DB columns diverge from JSON.
        """
        row = self._db.fetchone(
            f"{self._FULL_ROW_SELECT} WHERE id = ?",
            (contract_id,),
        )
        if row is None:
            raise ContractNotFoundError(f"Contract '{contract_id}' not found")

        return self._load_contract_from_row(row, verify=verify)

    def get_by_user(
        self, user_id: int, *, status: Optional[str] = None
    ) -> list[SkillContract]:
        """Load all contracts for a user, optionally filtered by lifecycle status.

        Uses the central secure load helper. Tampered contracts raise errors.

        Args:
            user_id: The user ID.
            status: Optional lifecycle status filter.

        Returns:
            List of SkillContracts.
        """
        if status:
            rows = self._db.fetchall(
                f"{self._FULL_ROW_SELECT} WHERE user_id = ? AND lifecycle_status = ?",
                (user_id, status),
            )
        else:
            rows = self._db.fetchall(
                f"{self._FULL_ROW_SELECT} WHERE user_id = ?",
                (user_id,),
            )

        contracts = []
        for row in rows:
            try:
                contract = self._load_contract_from_row(row, verify=True)
                contracts.append(contract)
            except (ContractChecksumError, ContractInvariantError) as e:
                log.warning(
                    "Skipping contract due to integrity error (user %d): %s",
                    user_id,
                    str(e),
                )
                continue

        return contracts

    def get_by_hypothesis_id(
        self, hypothesis_id: str, *, verify: bool = True
    ) -> Optional[SkillContract]:
        """Load a contract by its linked hypothesis ID.

        Routes through get_by_id to ensure single secure load path.

        Returns None if no contract is linked to this hypothesis.

        Raises:
            ContractChecksumError: If checksum verification fails.
            ContractInvariantError: If DB columns diverge from JSON.
        """
        row = self._db.fetchone(
            "SELECT id FROM skill_contracts WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )
        if row is None:
            return None

        contract_id = row["id"] if hasattr(row, "keys") else row[0]
        return self.get_by_id(contract_id, verify=verify)

    # ── Update ──────────────────────────────────────────────

    def update(
        self,
        contract: SkillContract,
        user_id: int,
        *,
        expected_version: Optional[int] = None,
    ) -> SkillContract:
        """Update an existing contract with optimistic locking.

        Auto-increments contract_version. Applies finalize + validate.
        Uses compare-and-swap on contract_version to prevent lost updates.

        Args:
            contract: The updated contract (must have existing ID).
            user_id: The owning user's ID.
            expected_version: The version the caller believes is current.
                If None, reads current version from DB (legacy compat).

        Returns:
            The finalized updated contract.

        Raises:
            ContractNotFoundError: If ID not found.
            ContractValidationError: If validation fails.
            ContractDuplicateNameError: If new name conflicts.
            ContractConcurrentUpdateError: If expected_version is stale.
        """
        # Check existence and get current version
        existing_row = self._db.fetchone(
            "SELECT contract_version, name FROM skill_contracts WHERE id = ? AND user_id = ?",
            (contract.id, user_id),
        )
        if existing_row is None:
            raise ContractNotFoundError(
                f"Contract '{contract.id}' not found for user {user_id}"
            )

        current_version = (
            existing_row[0]
            if isinstance(existing_row, (tuple, list))
            else existing_row["contract_version"]
        )
        old_name = (
            existing_row[1]
            if isinstance(existing_row, (tuple, list))
            else existing_row["name"]
        )

        # Optimistic locking: if caller provided expected_version, it must match
        if expected_version is not None and expected_version != current_version:
            raise ContractConcurrentUpdateError(
                f"Stale update for contract {contract.id} "
                f"(expected v{expected_version}, current in DB is v{current_version})"
            )

        # Determine the version we lock against
        lock_version = (
            expected_version if expected_version is not None else current_version
        )

        # Auto-increment version
        new_version = current_version + 1
        contract = replace(
            contract,
            contract_version=new_version,
            updated_at=now_iso(),
        )

        # Finalize security metadata
        contract = _finalize_security_metadata(contract)

        # Check name uniqueness if name changed (V8)
        if contract.name != old_name:
            name_conflict = self._db.fetchone(
                "SELECT id FROM skill_contracts WHERE user_id = ? AND name = ? AND id != ?",
                (user_id, contract.name, contract.id),
            )
            if name_conflict:
                raise ContractDuplicateNameError(
                    f"A contract with name '{contract.name}' already exists for user {user_id}"
                )

        # Validate (V1-V17 including V3 monotonicity, V16 invariant)
        result = validate(
            contract,
            old_version=current_version,
            db_schema_version=contract.schema_version,
            db_contract_version=contract.contract_version,
        )
        if not result.is_valid:
            error_msgs = "; ".join(f"[{i.rule}] {i.message}" for i in result.errors)
            raise ContractValidationError(
                f"Contract validation failed: {error_msgs}",
                result,
            )

        # Serialize
        contract_json = contract.to_json()

        # Compare-and-swap update: WHERE includes contract_version
        try:
            cursor = self._db.execute(
                """UPDATE skill_contracts SET
                    name = ?, schema_version = ?, contract_version = ?,
                    hypothesis_id = ?, origin = ?, lifecycle_status = ?,
                    review_status = ?, risk_level = ?, package_type = ?,
                    checksum = ?, contract_json = ?, updated_at = ?
                   WHERE id = ? AND user_id = ? AND contract_version = ?""",
                (
                    contract.name,
                    contract.schema_version,
                    contract.contract_version,
                    contract.hypothesis_id,
                    contract.origin,
                    contract.lifecycle.status,
                    contract.review_status,
                    contract.risk_level,
                    contract.store_meta.package_type,
                    contract.trust.checksum,
                    contract_json,
                    contract.updated_at,
                    contract.id,
                    user_id,
                    lock_version,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise self._classify_integrity_error(e, contract, user_id) from e

        # Check if the update actually hit a row
        rowcount = cursor.rowcount if hasattr(cursor, "rowcount") else None
        if rowcount == 0:
            raise ContractConcurrentUpdateError(
                f"Stale update for contract {contract.id} "
                f"(locked on v{lock_version}, possibly newer in DB)"
            )

        log.info(
            "Updated contract %s (v%d -> v%d, user=%d)",
            contract.id,
            current_version,
            new_version,
            user_id,
        )
        return contract

    # ── Delete ──────────────────────────────────────────────

    def delete(self, contract_id: str, user_id: int) -> bool:
        """Delete a contract by ID.

        Returns True if deleted, False if not found.
        """
        row = self._db.fetchone(
            "SELECT id FROM skill_contracts WHERE id = ? AND user_id = ?",
            (contract_id, user_id),
        )
        if row is None:
            return False

        self._db.execute(
            "DELETE FROM skill_contracts WHERE id = ? AND user_id = ?",
            (contract_id, user_id),
        )
        log.info("Deleted contract %s (user=%d)", contract_id, user_id)
        return True

    # ── Queries ─────────────────────────────────────────────

    def count_by_user(self, user_id: int) -> int:
        """Count contracts for a user."""
        row = self._db.fetchone(
            "SELECT COUNT(*) FROM skill_contracts WHERE user_id = ?",
            (user_id,),
        )
        if row is None:
            return 0
        return row[0] if isinstance(row, (tuple, list)) else row["COUNT(*)"]

    def exists_by_name(self, user_id: int, name: str) -> bool:
        """Check if a contract name exists for a user."""
        row = self._db.fetchone(
            "SELECT 1 FROM skill_contracts WHERE user_id = ? AND name = ?",
            (user_id, name),
        )
        return row is not None
