"""Draft Store: ephemeral in-memory store for skill drafts.

Manages the lifecycle of skill drafts during the /learn flow:
  - Create draft with TTL (15 minutes default)
  - Get with ownership + TTL check
  - Update with optimistic locking (etag)
  - Delete (cancel)
  - Cleanup expired drafts (called on every create to prevent leaks)

Key: (user_id, chat_id, draft_id)
Thread-safe via asyncio.Lock.

Dependencies: Python stdlib only (hashlib, json, uuid, datetime, asyncio).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import uuid4

from application.skill_compression.skill_contract import SkillContract

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────


class DraftStoreError(Exception):
    """Base error for draft store operations."""


class DraftNotFoundError(DraftStoreError):
    """Raised when a draft does not exist or has expired."""


class DraftExpiredError(DraftStoreError):
    """Raised when a draft's TTL has been exceeded."""


class DraftEtagMismatchError(DraftStoreError):
    """Raised when the provided etag does not match the current draft."""


class DraftOwnershipError(DraftStoreError):
    """Raised when a user attempts to access another user's draft."""


# ──────────────────────────────────────────────────────────────
# Draft dataclass
# ──────────────────────────────────────────────────────────────


def _compute_etag(contract: SkillContract) -> str:
    """Compute an etag (SHA-256) from the draft's contract JSON.

    Used for optimistic locking: the caller must provide the current
    etag to prove they are working on the latest version.
    """
    raw = contract.to_json(canonical=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SkillDraft:
    """A skill draft in the edit/preview flow.

    Attributes:
        draft_id: Unique draft identifier.
        user_id: Telegram user ID of the creator.
        chat_id: Chat context where the draft was created.
        contract: Current draft state (SkillContract).
        etag: SHA-256 of the current contract JSON (optimistic locking).
        created_at: When the draft was created (UTC).
        expires_at: When the draft expires (created_at + TTL).
        status: Current status (pending, editing, needs_input).
        edit_count: Number of edits applied so far.
    """

    draft_id: str
    user_id: int
    chat_id: int
    contract: SkillContract
    etag: str
    created_at: datetime
    expires_at: datetime
    status: str = "pending"  # pending | editing | needs_input
    edit_count: int = 0


# ──────────────────────────────────────────────────────────────
# Draft Store
# ──────────────────────────────────────────────────────────────

# Default TTL: 15 minutes
DEFAULT_TTL_SECONDS = 900


class DraftStore:
    """Ephemeral in-memory store for skill drafts.

    Key: (user_id, chat_id, draft_id).
    Each (user_id, chat_id) pair can have at most one active draft
    (creating a new one replaces the old one).

    Thread-safe via asyncio.Lock.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        # Key: (user_id, chat_id, draft_id) -> SkillDraft
        self._drafts: dict[tuple[int, int, str], SkillDraft] = {}
        # Quick lookup: (user_id, chat_id) -> draft_id (one active draft per chat)
        self._active: dict[tuple[int, int], str] = {}
        # O(1) ownership index: (chat_id, draft_id) -> user_id
        # Prevents cross-user scan on get() miss.
        self._draft_owners: dict[tuple[int, str], int] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        """Raw number of stored drafts (may include expired until cleanup).

        This property does NOT acquire the lock and does NOT mutate state.
        For an accurate count that cleans up expired drafts first, use
        the async count_active() method instead.
        """
        return len(self._drafts)

    async def count_active(self) -> int:
        """Accurate count of non-expired drafts (async, lock-safe).

        Acquires the lock, runs cleanup of expired drafts, then returns
        the count. Prefer this over active_count when precision matters.
        """
        async with self._lock:
            self._cleanup_expired_locked()
            return len(self._drafts)

    async def create(
        self,
        user_id: int,
        chat_id: int,
        contract: SkillContract,
    ) -> SkillDraft:
        """Create a new draft. Replaces any existing draft for the same chat.

        Calls cleanup_expired() first to prevent zombie-draft memory leaks.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            contract: Initial draft contract.

        Returns:
            The created SkillDraft.
        """
        async with self._lock:
            # Cleanup expired drafts first (prevents memory leaks)
            self._cleanup_expired_locked()

            # Remove existing draft for this (user_id, chat_id) if any
            old_draft_id = self._active.get((user_id, chat_id))
            if old_draft_id is not None:
                self._drafts.pop((user_id, chat_id, old_draft_id), None)
                self._draft_owners.pop((chat_id, old_draft_id), None)

            # Create new draft
            draft_id = uuid4().hex
            now = datetime.now(timezone.utc)
            expires = now + timedelta(seconds=self._ttl_seconds)
            etag = _compute_etag(contract)

            draft = SkillDraft(
                draft_id=draft_id,
                user_id=user_id,
                chat_id=chat_id,
                contract=contract,
                etag=etag,
                created_at=now,
                expires_at=expires,
                status="pending",
                edit_count=0,
            )

            self._drafts[(user_id, chat_id, draft_id)] = draft
            self._active[(user_id, chat_id)] = draft_id
            self._draft_owners[(chat_id, draft_id)] = user_id
            return draft

    async def get(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
    ) -> Optional[SkillDraft]:
        """Get a draft by key. Returns None if not found or expired.

        Checks ownership: only the creating user may access their draft.

        Args:
            user_id: Requesting user's ID.
            chat_id: Chat context.
            draft_id: Draft identifier.

        Returns:
            SkillDraft or None if not found/expired.

        Raises:
            DraftOwnershipError: If user_id does not match the draft creator.
        """
        async with self._lock:
            key = (user_id, chat_id, draft_id)
            draft = self._drafts.get(key)

            if draft is None:
                # O(1) ownership check via secondary index instead of
                # scanning all drafts (prevents cross-user probe / DoS).
                owner_key = (chat_id, draft_id)
                if owner_key in self._draft_owners:
                    owner_uid = self._draft_owners[owner_key]
                    if owner_uid != user_id:
                        raise DraftOwnershipError("Draft ownership check failed.")
                return None

            # Check ownership
            if draft.user_id != user_id:
                raise DraftOwnershipError("Draft ownership check failed.")

            # Check expiry
            if datetime.now(timezone.utc) > draft.expires_at:
                # Expired: remove and return None
                self._drafts.pop(key, None)
                self._active.pop((user_id, chat_id), None)
                self._draft_owners.pop((chat_id, draft_id), None)
                return None

            return draft

    async def update(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
        new_contract: SkillContract,
        expected_etag: str,
        new_status: Optional[str] = None,
    ) -> SkillDraft:
        """Update a draft with optimistic locking (etag check).

        Args:
            user_id: Requesting user's ID.
            chat_id: Chat context.
            draft_id: Draft identifier.
            new_contract: Updated contract.
            expected_etag: The etag the caller expects (must match current).
            new_status: Optional new status (defaults to "editing").

        Returns:
            Updated SkillDraft with new etag.

        Raises:
            DraftNotFoundError: If draft does not exist.
            DraftExpiredError: If TTL has been exceeded.
            DraftEtagMismatchError: If expected_etag does not match.
            DraftOwnershipError: If user_id does not match.
        """
        async with self._lock:
            key = (user_id, chat_id, draft_id)
            draft = self._drafts.get(key)

            if draft is None:
                # O(1) ownership check via secondary index
                owner_key = (chat_id, draft_id)
                if owner_key in self._draft_owners:
                    owner_uid = self._draft_owners[owner_key]
                    if owner_uid != user_id:
                        raise DraftOwnershipError("Draft ownership check failed.")
                raise DraftNotFoundError(f"Draft '{draft_id}' not found.")

            # Check ownership
            if draft.user_id != user_id:
                raise DraftOwnershipError("Draft ownership check failed.")

            # Check expiry
            if datetime.now(timezone.utc) > draft.expires_at:
                self._drafts.pop(key, None)
                self._active.pop((user_id, chat_id), None)
                self._draft_owners.pop((chat_id, draft_id), None)
                raise DraftExpiredError(
                    f"Draft '{draft_id}' has expired. Please start with /learn again."
                )

            # Check etag (optimistic locking)
            if draft.etag != expected_etag:
                raise DraftEtagMismatchError(
                    f"Draft '{draft_id}' has been modified concurrently. "
                    f"Expected etag '{expected_etag[:16]}...', "
                    f"current is '{draft.etag[:16]}...'."
                )

            # Update
            new_etag = _compute_etag(new_contract)
            updated = SkillDraft(
                draft_id=draft.draft_id,
                user_id=draft.user_id,
                chat_id=draft.chat_id,
                contract=new_contract,
                etag=new_etag,
                created_at=draft.created_at,
                expires_at=draft.expires_at,
                status=new_status or "editing",
                edit_count=draft.edit_count + 1,
            )

            self._drafts[key] = updated
            return updated

    async def delete(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
    ) -> bool:
        """Delete a draft (cancel flow). Returns True if deleted.

        Checks ownership: only the creating user may delete their draft.

        Args:
            user_id: Requesting user's ID.
            chat_id: Chat context.
            draft_id: Draft identifier.

        Returns:
            True if the draft was found and deleted, False if not found.

        Raises:
            DraftOwnershipError: If user_id does not match.
        """
        async with self._lock:
            key = (user_id, chat_id, draft_id)
            draft = self._drafts.get(key)

            if draft is None:
                # O(1) ownership check via secondary index
                owner_key = (chat_id, draft_id)
                if owner_key in self._draft_owners:
                    owner_uid = self._draft_owners[owner_key]
                    if owner_uid != user_id:
                        raise DraftOwnershipError("Draft ownership check failed.")
                return False

            if draft.user_id != user_id:
                raise DraftOwnershipError("Draft ownership check failed.")

            self._drafts.pop(key, None)
            self._active.pop((user_id, chat_id), None)
            self._draft_owners.pop((chat_id, draft_id), None)
            return True

    async def cleanup_expired(self) -> int:
        """Remove all expired drafts. Returns count of removed drafts.

        Should be called periodically or at every create() to prevent leaks.
        """
        async with self._lock:
            return self._cleanup_expired_locked()

    def _cleanup_expired_locked(self) -> int:
        """Internal cleanup (must be called while holding the lock)."""
        now = datetime.now(timezone.utc)
        expired_keys = [
            key for key, draft in self._drafts.items() if now > draft.expires_at
        ]
        for key in expired_keys:
            self._drafts.pop(key)
            user_id, chat_id, draft_id = key
            if self._active.get((user_id, chat_id)) == draft_id:
                self._active.pop((user_id, chat_id), None)
            self._draft_owners.pop((chat_id, draft_id), None)
        return len(expired_keys)

    async def get_active_for_chat(
        self,
        user_id: int,
        chat_id: int,
    ) -> Optional[SkillDraft]:
        """Get the active draft for a (user_id, chat_id) pair, if any.

        Convenience method for the /learn flow which has one draft per chat.

        Returns:
            SkillDraft or None if no active draft (or expired).
        """
        async with self._lock:
            draft_id = self._active.get((user_id, chat_id))
            if draft_id is None:
                return None
            draft = self._drafts.get((user_id, chat_id, draft_id))
            if draft is None:
                self._active.pop((user_id, chat_id), None)
                return None
            # Check expiry
            if datetime.now(timezone.utc) > draft.expires_at:
                self._drafts.pop((user_id, chat_id, draft_id), None)
                self._active.pop((user_id, chat_id), None)
                self._draft_owners.pop((chat_id, draft_id), None)
                return None
            return draft
