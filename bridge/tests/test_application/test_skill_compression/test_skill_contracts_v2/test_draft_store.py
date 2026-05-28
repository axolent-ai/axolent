"""T6 Tests: DraftStore (4-path: Happy + Malicious + Rejection + Privacy).

Coverage:
  D1: Foreign user cannot save (ownership check)
  D2: Expired callback -> clean error, no crash
  D3: Edit re-validates (tested via contract change + etag tracking)
  D4: Edit trigger empty (tested via contract replacement)
  D5: Cancel removes completely

  U10: Draft TTL expiry
  U11: Etag mismatch
  U12: Ownership check (foreign user)

  Plus: cleanup_expired removes only expired drafts,
        create replaces existing draft for same chat,
        get_active_for_chat convenience method.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from application.skill_compression.draft_store import (
    DEFAULT_TTL_SECONDS,
    DraftEtagMismatchError,
    DraftExpiredError,
    DraftNotFoundError,
    DraftOwnershipError,
    DraftStore,
    _compute_etag,
)
from application.skill_compression.skill_contract import (
    ActivationConfig,
    SkillContract,
    create_minimal_contract,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

USER_A = 1001
USER_B = 2002
CHAT_1 = 5001
CHAT_2 = 5002


@pytest.fixture
def store() -> DraftStore:
    """A fresh DraftStore with short TTL for testing."""
    return DraftStore(ttl_seconds=DEFAULT_TTL_SECONDS)


@pytest.fixture
def short_ttl_store() -> DraftStore:
    """DraftStore with 1-second TTL for expiry tests."""
    return DraftStore(ttl_seconds=1)


@pytest.fixture
def contract_a() -> SkillContract:
    return create_minimal_contract(
        name="draft-test-a",
        phrases=("hello",),
        instruction="say hello back",
    )


@pytest.fixture
def contract_b() -> SkillContract:
    return create_minimal_contract(
        name="draft-test-b",
        phrases=("goodbye",),
        instruction="say goodbye back",
    )


# ──────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────


class TestDraftStoreHappy:
    """Standard create/get/update/delete operations."""

    @pytest.mark.asyncio
    async def test_create_and_get(self, store, contract_a):
        """Create a draft and retrieve it."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        assert draft.user_id == USER_A
        assert draft.chat_id == CHAT_1
        assert draft.contract.name == contract_a.name
        assert draft.etag
        assert draft.status == "pending"
        assert draft.edit_count == 0

        # Get it back
        retrieved = await store.get(USER_A, CHAT_1, draft.draft_id)
        assert retrieved is not None
        assert retrieved.draft_id == draft.draft_id
        assert retrieved.etag == draft.etag

    @pytest.mark.asyncio
    async def test_update_changes_etag(self, store, contract_a, contract_b):
        """Update changes the etag (optimistic locking)."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        old_etag = draft.etag

        updated = await store.update(
            USER_A,
            CHAT_1,
            draft.draft_id,
            new_contract=contract_b,
            expected_etag=old_etag,
        )
        assert updated.etag != old_etag
        assert updated.edit_count == 1
        assert updated.status == "editing"

    @pytest.mark.asyncio
    async def test_cancel_removes_draft(self, store, contract_a):
        """D5: Cancel removes draft completely."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        deleted = await store.delete(USER_A, CHAT_1, draft.draft_id)
        assert deleted is True

        # Should not be retrievable
        retrieved = await store.get(USER_A, CHAT_1, draft.draft_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self, store):
        """Deleting a nonexistent draft returns False, no error."""
        result = await store.delete(USER_A, CHAT_1, "nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_create_replaces_existing_for_same_chat(
        self, store, contract_a, contract_b
    ):
        """Creating a new draft replaces the old one for same (user, chat)."""
        draft1 = await store.create(USER_A, CHAT_1, contract_a)
        draft2 = await store.create(USER_A, CHAT_1, contract_b)

        assert draft1.draft_id != draft2.draft_id
        assert store.active_count == 1

        # Old draft not retrievable
        old = await store.get(USER_A, CHAT_1, draft1.draft_id)
        assert old is None

        # New draft exists
        new = await store.get(USER_A, CHAT_1, draft2.draft_id)
        assert new is not None
        assert new.contract.name == contract_b.name

    @pytest.mark.asyncio
    async def test_different_chats_different_drafts(
        self, store, contract_a, contract_b
    ):
        """Different chats can have independent drafts."""
        draft1 = await store.create(USER_A, CHAT_1, contract_a)
        draft2 = await store.create(USER_A, CHAT_2, contract_b)

        assert store.active_count == 2

        r1 = await store.get(USER_A, CHAT_1, draft1.draft_id)
        r2 = await store.get(USER_A, CHAT_2, draft2.draft_id)
        assert r1 is not None
        assert r2 is not None
        assert r1.contract.name != r2.contract.name

    @pytest.mark.asyncio
    async def test_get_active_for_chat(self, store, contract_a):
        """get_active_for_chat returns the most recent draft."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        active = await store.get_active_for_chat(USER_A, CHAT_1)
        assert active is not None
        assert active.draft_id == draft.draft_id

    @pytest.mark.asyncio
    async def test_get_active_for_chat_none(self, store):
        """get_active_for_chat returns None when no draft exists."""
        active = await store.get_active_for_chat(USER_A, CHAT_1)
        assert active is None

    @pytest.mark.asyncio
    async def test_update_with_custom_status(self, store, contract_a, contract_b):
        """Update can set a custom status."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        updated = await store.update(
            USER_A,
            CHAT_1,
            draft.draft_id,
            new_contract=contract_b,
            expected_etag=draft.etag,
            new_status="needs_input",
        )
        assert updated.status == "needs_input"


# ──────────────────────────────────────────────────────────────
# Malicious path: ownership violations
# ──────────────────────────────────────────────────────────────


class TestDraftStoreMalicious:
    """D1: Foreign user cannot save/edit/access another user's draft."""

    @pytest.mark.asyncio
    async def test_foreign_user_cannot_get(self, store, contract_a):
        """User B cannot get User A's draft."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        with pytest.raises(DraftOwnershipError, match="ownership check failed"):
            await store.get(USER_B, CHAT_1, draft.draft_id)

    @pytest.mark.asyncio
    async def test_foreign_user_cannot_update(self, store, contract_a, contract_b):
        """User B cannot update User A's draft."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        with pytest.raises(DraftOwnershipError, match="ownership check failed"):
            await store.update(
                USER_B,
                CHAT_1,
                draft.draft_id,
                new_contract=contract_b,
                expected_etag=draft.etag,
            )

    @pytest.mark.asyncio
    async def test_foreign_user_cannot_delete(self, store, contract_a):
        """User B cannot delete User A's draft."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        with pytest.raises(DraftOwnershipError, match="ownership check failed"):
            await store.delete(USER_B, CHAT_1, draft.draft_id)


# ──────────────────────────────────────────────────────────────
# Rejection path: TTL, etag, not found
# ──────────────────────────────────────────────────────────────


class TestDraftStoreRejection:
    """D2: Expired/stale/missing drafts produce clean errors."""

    @pytest.mark.asyncio
    async def test_expired_draft_returns_none_on_get(self, short_ttl_store, contract_a):
        """U10: Draft after TTL returns None on get (no crash)."""
        draft = await short_ttl_store.create(USER_A, CHAT_1, contract_a)
        # Wait for TTL to expire
        await asyncio.sleep(1.5)
        result = await short_ttl_store.get(USER_A, CHAT_1, draft.draft_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_draft_raises_on_update(
        self, short_ttl_store, contract_a, contract_b
    ):
        """D2: Expired draft on update raises DraftExpiredError with clean message."""
        draft = await short_ttl_store.create(USER_A, CHAT_1, contract_a)
        await asyncio.sleep(1.5)
        with pytest.raises(DraftExpiredError, match="expired"):
            await short_ttl_store.update(
                USER_A,
                CHAT_1,
                draft.draft_id,
                new_contract=contract_b,
                expected_etag=draft.etag,
            )

    @pytest.mark.asyncio
    async def test_stale_etag_rejected(self, store, contract_a, contract_b):
        """U11: Stale etag on update raises DraftEtagMismatchError."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        old_etag = draft.etag

        # First update succeeds and changes etag
        await store.update(
            USER_A,
            CHAT_1,
            draft.draft_id,
            new_contract=contract_b,
            expected_etag=old_etag,
        )

        # Second update with old etag fails
        with pytest.raises(DraftEtagMismatchError, match="modified concurrently"):
            await store.update(
                USER_A,
                CHAT_1,
                draft.draft_id,
                new_contract=contract_a,
                expected_etag=old_etag,
            )

    @pytest.mark.asyncio
    async def test_update_nonexistent_draft_raises(self, store, contract_a):
        """Update on nonexistent draft raises DraftNotFoundError."""
        with pytest.raises(DraftNotFoundError, match="not found"):
            await store.update(
                USER_A,
                CHAT_1,
                "nonexistent-id",
                new_contract=contract_a,
                expected_etag="fake-etag",
            )

    @pytest.mark.asyncio
    async def test_double_save_second_not_found(self, store, contract_a):
        """Double-click scenario: after delete, second get returns None."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        await store.delete(USER_A, CHAT_1, draft.draft_id)
        result = await store.get(USER_A, CHAT_1, draft.draft_id)
        assert result is None


# ──────────────────────────────────────────────────────────────
# Cleanup tests
# ──────────────────────────────────────────────────────────────


class TestDraftStoreCleanup:
    """cleanup_expired removes only expired drafts."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_only(self, contract_a, contract_b):
        """Expired drafts removed, non-expired survive."""
        short_store = DraftStore(ttl_seconds=1)
        long_store = DraftStore(ttl_seconds=3600)

        # Create draft in short-TTL store (will expire)
        await short_store.create(USER_A, CHAT_1, contract_a)
        # Create draft in long-TTL store (must survive)
        long_draft = await long_store.create(USER_A, CHAT_1, contract_b)

        await asyncio.sleep(1.5)

        # After sleep, the short-TTL draft is expired
        removed_short = await short_store.cleanup_expired()
        assert removed_short == 1
        assert short_store.active_count == 0

        # Long-TTL draft must survive cleanup
        removed_long = await long_store.cleanup_expired()
        assert removed_long == 0
        assert long_store.active_count == 1

        # Verify long-TTL draft is still retrievable
        surviving = await long_store.get(USER_A, CHAT_1, long_draft.draft_id)
        assert surviving is not None
        assert surviving.contract.name == contract_b.name

    @pytest.mark.asyncio
    async def test_cleanup_on_create(self, contract_a, contract_b):
        """cleanup_expired is called on every create() to prevent leaks."""
        store = DraftStore(ttl_seconds=1)
        await store.create(USER_A, CHAT_1, contract_a)
        await asyncio.sleep(1.5)

        # Creating a new draft should cleanup the expired one
        await store.create(USER_A, CHAT_2, contract_b)
        # Only the new draft should exist
        assert store.active_count == 1

    @pytest.mark.asyncio
    async def test_get_active_for_chat_expired_returns_none(self, contract_a):
        """Expired draft via get_active_for_chat returns None."""
        store = DraftStore(ttl_seconds=1)
        await store.create(USER_A, CHAT_1, contract_a)
        await asyncio.sleep(1.5)
        result = await store.get_active_for_chat(USER_A, CHAT_1)
        assert result is None


# ──────────────────────────────────────────────────────────────
# Privacy path: no sensitive data leaked
# ──────────────────────────────────────────────────────────────


class TestDraftStorePrivacy:
    """Error messages must not leak sensitive contract content."""

    @pytest.mark.asyncio
    async def test_ownership_error_does_not_leak_instruction(self, store):
        """Ownership error should not contain the draft's instruction."""
        secret_contract = create_minimal_contract(
            name="secret-draft",
            phrases=("secret-trigger",),
            instruction="my-api-key-sk-123456",
        )
        draft = await store.create(USER_A, CHAT_1, secret_contract)
        with pytest.raises(DraftOwnershipError) as exc_info:
            await store.get(USER_B, CHAT_1, draft.draft_id)
        assert "my-api-key-sk-123456" not in str(exc_info.value)
        assert "secret-trigger" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ownership_error_does_not_leak_user_ids(self, store):
        """Ownership error must not contain any user IDs."""
        contract = create_minimal_contract(
            name="test", phrases=("test",), instruction="test"
        )
        draft = await store.create(USER_A, CHAT_1, contract)
        with pytest.raises(DraftOwnershipError) as exc_info:
            await store.get(USER_B, CHAT_1, draft.draft_id)
        error_msg = str(exc_info.value)
        assert str(USER_A) not in error_msg
        assert str(USER_B) not in error_msg

    @pytest.mark.asyncio
    async def test_expired_error_does_not_leak_instruction(self, contract_a):
        """Expired error should not contain contract content."""
        store = DraftStore(ttl_seconds=1)
        secret_contract = create_minimal_contract(
            name="secret-draft",
            phrases=("trigger",),
            instruction="top-secret-instruction",
        )
        draft = await store.create(USER_A, CHAT_1, secret_contract)
        await asyncio.sleep(1.5)
        with pytest.raises(DraftExpiredError) as exc_info:
            await store.update(
                USER_A,
                CHAT_1,
                draft.draft_id,
                new_contract=contract_a,
                expected_etag=draft.etag,
            )
        assert "top-secret-instruction" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_etag_mismatch_does_not_leak_full_etag(
        self, store, contract_a, contract_b
    ):
        """Etag mismatch error only shows truncated etag, not full hash."""
        draft = await store.create(USER_A, CHAT_1, contract_a)
        await store.update(
            USER_A,
            CHAT_1,
            draft.draft_id,
            new_contract=contract_b,
            expected_etag=draft.etag,
        )
        with pytest.raises(DraftEtagMismatchError) as exc_info:
            await store.update(
                USER_A,
                CHAT_1,
                draft.draft_id,
                new_contract=contract_a,
                expected_etag=draft.etag,
            )
        # Full 64-char hex should not appear in error message
        error_msg = str(exc_info.value)
        assert draft.etag not in error_msg  # Only first 16 chars shown


# ──────────────────────────────────────────────────────────────
# Codex v1 tests: trigger phrase editing edge cases
# ──────────────────────────────────────────────────────────────


class TestDraftStoreTriggerEditing:
    """Codex v1 required: edited trigger phrase validation."""

    @pytest.mark.asyncio
    async def test_edit_to_empty_trigger_tracked_via_etag(self, store, contract_a):
        """D4: Editing trigger to empty is trackable (contract change = new etag).

        The DraftStore tracks that the contract changed (new etag).
        Actual trigger validation is done by ContractValidator, not DraftStore.
        This test verifies the DraftStore correctly tracks the edit.
        """
        draft = await store.create(USER_A, CHAT_1, contract_a)

        # Edit to empty trigger (will be caught by validator at save time)
        empty_trigger_contract = replace(
            contract_a,
            activation=ActivationConfig(phrases=()),
        )
        updated = await store.update(
            USER_A,
            CHAT_1,
            draft.draft_id,
            new_contract=empty_trigger_contract,
            expected_etag=draft.etag,
            new_status="needs_input",
        )
        assert updated.status == "needs_input"
        assert updated.contract.activation.phrases == ()
        assert updated.etag != draft.etag

    @pytest.mark.asyncio
    async def test_etag_is_deterministic(self, contract_a):
        """Etag computation is deterministic for the same contract."""
        etag1 = _compute_etag(contract_a)
        etag2 = _compute_etag(contract_a)
        assert etag1 == etag2
        assert len(etag1) == 64  # SHA-256 hex
