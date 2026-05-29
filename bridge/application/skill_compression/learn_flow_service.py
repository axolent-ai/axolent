"""Learn Flow Service: orchestrates /learn Preview/Confirm/Edit/Cancel + --quick.

Application-layer service. All business logic for the /learn flow lives here.
Presentation handlers (skill_commands.py) stay thin: parse args, call this
service, render results.

State Machine (Codex-approved):
  /learn
    -> build contract (lifecycle=draft)
    -> safety precheck
    -> if needs_input: pending_input(trigger)
    -> if quick: safety final -> persist contract + legacy ATOMAR
    -> else: draft preview
         -> save: safety final -> persist contract + legacy ATOMAR
         -> edit: pending_edit(trigger|instruction)
         -> cancel: delete draft
  pending_edit / pending_input
    -> next message (real follow-up handler)
    -> validate edit
    -> safety precheck
    -> update draft etag
    -> render preview again

Dual-write strategy (transition until Etappe 4):
  When a contract is saved, both a SkillContract (ContractStore) AND a
  legacy Hypothesis (SkillLearningService) are created. This ensures
  the existing Matcher (which is not yet contract-aware) can still
  trigger the skill. The legacy write is removed in Etappe 4 when
  the Matcher becomes contract-aware.

One Safety Gate rule:
  _validate_contract_safety() is the SINGLE entry point for all safety
  checks before persist. All persist paths (quick, save_draft, edit-save,
  needs_input-complete) MUST call it. ContractStore.persist() validates
  only structure/schema, never policy/privacy.

Dependencies:
  - ContractBuilder (build drafts from free-text)
  - DraftStore (ephemeral preview/edit drafts)
  - ContractStore (persistent contract storage)
  - SkillLearningService (legacy dual-write for matcher compatibility)
  - PrivacyPipeline / SecretScanner (safety checks)
  - PendingEditStore (in-memory edit/input state for follow-up handler)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from application.skill_compression.contract_builder import (
    BuildResult,
    ContractBuilder,
)
from application.skill_compression.contract_store import (
    ContractDuplicateNameError,
    ContractStore,
    ContractStoreError,
)
from application.skill_compression.draft_store import (
    DraftEtagMismatchError,
    DraftExpiredError,
    DraftNotFoundError,
    DraftOwnershipError,
    DraftStore,
    SkillDraft,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.privacy.secret_scanner import SecretScanner
from application.skill_compression.skill_contract import LifecycleConfig, SkillContract
from application.skill_compression.skill_learning_service import SkillLearningService

log = logging.getLogger(__name__)

# Shared scanner for secret checks on edits
_secret_scanner = SecretScanner()


# ---------------------------------------------------------------
# Pending Edit/Input State Store
# ---------------------------------------------------------------

# Default TTL for pending edit/input states: 10 minutes
PENDING_STATE_TTL_SECONDS = 600


@dataclass(frozen=True, slots=True)
class PendingState:
    """State for a user awaiting follow-up input (edit or needs_input).

    Attributes:
        user_id: Telegram user ID (ownership).
        chat_id: Chat context.
        draft_id: Draft being edited.
        etag: Current etag of the draft.
        action: 'edit_trigger' | 'edit_instruction' | 'needs_input'
        created_at: When the pending state was set.
        expires_at: When the pending state expires.
    """

    user_id: int
    chat_id: int
    draft_id: str
    etag: str
    action: str  # edit_trigger | edit_instruction | needs_input
    created_at: datetime
    expires_at: datetime


class PendingEditStore:
    """In-memory store for pending edit/input states.

    Key: (user_id, chat_id). One pending state per user+chat.
    Thread-safe via asyncio.Lock.
    """

    def __init__(self, ttl_seconds: int = PENDING_STATE_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._states: dict[tuple[int, int], PendingState] = {}
        self._lock = asyncio.Lock()

    async def set_pending(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
        etag: str,
        action: str,
    ) -> PendingState:
        """Set a pending edit/input state for a user+chat.

        Replaces any existing pending state for the same user+chat.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            draft_id: Draft being edited.
            etag: Current etag of the draft.
            action: Type of pending action.

        Returns:
            The created PendingState.
        """
        async with self._lock:
            self._cleanup_expired_locked()
            now = datetime.now(timezone.utc)
            state = PendingState(
                user_id=user_id,
                chat_id=chat_id,
                draft_id=draft_id,
                etag=etag,
                action=action,
                created_at=now,
                expires_at=now + timedelta(seconds=self._ttl_seconds),
            )
            self._states[(user_id, chat_id)] = state
            return state

    async def get_pending(self, user_id: int, chat_id: int) -> Optional[PendingState]:
        """Get the pending state for a user+chat, if any.

        Returns None if no state or expired.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.

        Returns:
            PendingState or None.
        """
        async with self._lock:
            state = self._states.get((user_id, chat_id))
            if state is None:
                return None
            if state.user_id != user_id:
                return None
            if datetime.now(timezone.utc) > state.expires_at:
                self._states.pop((user_id, chat_id), None)
                return None
            return state

    async def clear_pending(self, user_id: int, chat_id: int) -> bool:
        """Clear a pending state. Returns True if cleared.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.

        Returns:
            True if a state was removed, False if none existed.
        """
        async with self._lock:
            state = self._states.get((user_id, chat_id))
            if state is None:
                return False
            if state.user_id != user_id:
                return False
            self._states.pop((user_id, chat_id), None)
            return True

    def _cleanup_expired_locked(self) -> int:
        """Remove expired states (must hold lock)."""
        now = datetime.now(timezone.utc)
        expired = [key for key, state in self._states.items() if now > state.expires_at]
        for key in expired:
            self._states.pop(key)
        return len(expired)


# ---------------------------------------------------------------
# Result types
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LearnFlowResult:
    """Outcome of start_learn().

    Attributes:
        status: 'preview' | 'needs_input' | 'saved' | 'rejected'
        draft: The created draft (for preview/needs_input).
        build_result: Raw BuildResult from ContractBuilder.
        rejection_reason: Why rejected (for rejected/needs_input).
        saved_contract_name: Name of saved contract (for quick mode).
    """

    status: str
    draft: Optional[SkillDraft] = None
    build_result: Optional[BuildResult] = None
    rejection_reason: str = ""
    saved_contract_name: str = ""


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Outcome of save_draft()."""

    success: bool
    contract_name: str = ""
    error: str = ""
    error_type: str = ""  # expired | not_found | stale | ownership | validation


@dataclass(frozen=True, slots=True)
class EditResult:
    """Outcome of edit_trigger() / edit_instruction()."""

    success: bool
    draft: Optional[SkillDraft] = None
    error: str = ""
    error_type: str = ""  # expired | not_found | stale | ownership | rejected


# ---------------------------------------------------------------
# Service
# ---------------------------------------------------------------


class LearnFlowService:
    """Orchestrates the /learn flow: build -> preview -> save/edit/cancel.

    Business logic lives here. Handlers stay thin.

    State Machine:
      /learn -> build -> safety precheck -> preview/quick/needs_input
      preview -> save/edit/cancel
      edit -> pending_edit -> follow-up -> update -> preview
      needs_input -> pending_input -> follow-up -> update -> preview
    """

    def __init__(
        self,
        contract_builder: ContractBuilder,
        draft_store: DraftStore,
        contract_store: ContractStore,
        privacy_pipeline: PrivacyPipeline,
        skill_learning_service: SkillLearningService,
        pending_edit_store: Optional["PendingEditStore"] = None,
    ) -> None:
        self._builder = contract_builder
        self._draft_store = draft_store
        self._contract_store = contract_store
        self._privacy = privacy_pipeline
        self._legacy_service = skill_learning_service
        self._pending_store = pending_edit_store or PendingEditStore()

    @property
    def pending_edit_store(self) -> "PendingEditStore":
        """Access the pending edit store (for handler registration)."""
        return self._pending_store

    # ---------------------------------------------------------------
    # ONE SAFETY GATE: central safety validation before any persist
    # ---------------------------------------------------------------

    def _validate_contract_safety(
        self,
        contract: SkillContract,
        user_id: int,
        source: str,
    ) -> Optional[str]:
        """Central safety gate: runs full PrivacyPipeline on canonical claim form.

        This is THE SINGLE safety check before any persist. All paths
        (quick, save_draft, edit-save, needs_input-complete) MUST call this.

        Checks SecretScanner + HealthcareFilter + NudgeFilter on the
        canonical skill claim form: "when I say <trigger>, <instruction>".

        Args:
            contract: The SkillContract to validate.
            user_id: User who owns the contract.
            source: Context string for logging (e.g. 'quick', 'save', 'edit').

        Returns:
            None if safe, or rejection reason string if blocked.
        """
        from application.skill_compression.hypothesis_storage import (
            Hypothesis,
            HypothesisScope,
        )

        # Build canonical claim: "when I say <trigger>, <instruction>"
        trigger = ""
        if contract.activation.phrases:
            trigger = contract.activation.phrases[0]
        instruction = contract.execution.instruction

        if trigger:
            canonical_claim = f"when I say {trigger}, {instruction}"
        else:
            canonical_claim = instruction

        # Build temporary hypothesis for PrivacyPipeline
        now_iso = datetime.now(timezone.utc).isoformat()
        temp_hyp = Hypothesis(
            hypothesis_id=f"hyp_{uuid4().hex[:16]}",
            user_id=user_id,
            type="preference",
            scope=HypothesisScope(),
            claim=canonical_claim,
            status="confirmed",
            version=1,
            elo_rating=1500.0,
            elo_games_played=0,
            bayes_confidence=0.5,
            support_count=1,
            contradict_count=0,
            source_type="learn_command",
            decay_immune=True,
            created_at=now_iso,
            last_applied=None,
            last_seen=now_iso,
        )

        # Run full privacy pipeline (Healthcare + Secret + Nudge)
        rejection = self._privacy.check(temp_hyp)
        if rejection is not None:
            log.info(
                "Safety gate blocked (%s): user=%d source=%s reason_len=%d",
                source,
                user_id,
                rejection.source.value,
                len(rejection.reason),
            )
            return rejection.reason

        return None

    async def start_learn(
        self,
        user_id: int,
        chat_id: int,
        text: str,
        *,
        quick: bool = False,
    ) -> LearnFlowResult:
        """Start a /learn flow: build contract draft, optionally persist directly.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            text: Raw text after /learn (without the command prefix).
            quick: If True, skip preview and persist immediately.

        Returns:
            LearnFlowResult with status and draft/contract details.
        """
        # Build contract from free-text
        build_result = self._builder.build(text)

        # needs_input: cannot extract trigger or trigger rejected
        if build_result.status == "needs_input":
            # HIGH-1 fix: run safety BEFORE creating draft.
            # Secrets/healthcare/nudge must be rejected immediately,
            # never stored even temporarily in DraftStore.
            safety_rejection = self._validate_contract_safety(
                build_result.contract, user_id, "needs_input_pre_draft"
            )
            if safety_rejection is not None:
                return LearnFlowResult(
                    status="rejected",
                    build_result=build_result,
                    rejection_reason=safety_rejection,
                )

            # Safety passed: create draft so the user can provide the trigger
            draft = await self._draft_store.create(
                user_id=user_id,
                chat_id=chat_id,
                contract=build_result.contract,
            )
            # Set pending_input state so follow-up handler can pick up
            await self._pending_store.set_pending(
                user_id=user_id,
                chat_id=chat_id,
                draft_id=draft.draft_id,
                etag=draft.etag,
                action="needs_input",
            )
            return LearnFlowResult(
                status="needs_input",
                draft=draft,
                build_result=build_result,
                rejection_reason=build_result.needs_input_reason,
            )

        # Safety precheck (One Safety Gate) before any further processing
        safety_rejection = self._validate_contract_safety(
            build_result.contract, user_id, "start_learn"
        )
        if safety_rejection is not None:
            return LearnFlowResult(
                status="rejected",
                build_result=build_result,
                rejection_reason=safety_rejection,
            )

        # Quick mode: skip preview, persist directly
        if quick:
            return await self._persist_contract(
                user_id=user_id,
                contract=build_result.contract,
                text=text,
                build_result=build_result,
            )

        # Normal mode: create draft for preview
        draft = await self._draft_store.create(
            user_id=user_id,
            chat_id=chat_id,
            contract=build_result.contract,
        )

        return LearnFlowResult(
            status="preview",
            draft=draft,
            build_result=build_result,
        )

    async def save_draft(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
        etag: str,
    ) -> SaveResult:
        """Save a draft: safety check, transition draft->confirmed, persist atomically.

        One Safety Gate: full PrivacyPipeline runs before any persist.
        Atomic dual-write: if legacy fails, contract is NOT persisted.
        Idempotent: if draft already deleted (double-click), returns
        already_saved error.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            draft_id: Draft identifier.
            etag: Expected etag for optimistic locking.

        Returns:
            SaveResult with success flag and contract name.
        """
        try:
            draft = await self._draft_store.get(user_id, chat_id, draft_id)
        except DraftOwnershipError:
            return SaveResult(success=False, error="ownership", error_type="ownership")

        if draft is None:
            return SaveResult(
                success=False, error="already_saved", error_type="not_found"
            )

        # Etag check
        if draft.etag != etag:
            return SaveResult(success=False, error="stale", error_type="stale")

        # ONE SAFETY GATE: full privacy check BEFORE any persist
        safety_rejection = self._validate_contract_safety(
            draft.contract, user_id, "save_draft"
        )
        if safety_rejection is not None:
            return SaveResult(
                success=False,
                error=safety_rejection,
                error_type="rejected",
            )

        # Transition lifecycle: draft -> confirmed before persist
        contract = replace(
            draft.contract,
            lifecycle=LifecycleConfig(
                status="confirmed",
                editable=draft.contract.lifecycle.editable,
                decay=draft.contract.lifecycle.decay,
                last_schema_migration=draft.contract.lifecycle.last_schema_migration,
            ),
        )

        # BLOCKER-1 fix: Atomic dual-write with Contract Preflight + Rollback.
        # Strategy:
        #   1. Contract Preflight: validate name-uniqueness and schema BEFORE legacy
        #   2. Legacy write
        #   3. Contract persist
        #   4. If contract persist fails AFTER legacy: tombstone the legacy hypothesis
        #
        # This prevents split-brain in BOTH directions:
        #   Legacy-fail -> no Contract (step 2 fails, step 3 never runs)
        #   Contract-fail-after-Legacy -> Legacy tombstoned (step 4)

        trigger = ""
        if contract.activation.phrases:
            trigger = contract.activation.phrases[0]
        instruction = contract.execution.instruction
        if trigger:
            legacy_text = f"when I say {trigger}, {instruction}"
        else:
            legacy_text = instruction

        # Step 1: Contract Preflight (catch foreseeable failures before legacy write)
        preflight_error = self._contract_preflight(contract, user_id)
        if preflight_error is not None:
            return SaveResult(
                success=False,
                error=preflight_error,
                error_type="validation",
            )

        # Step 2: Legacy write
        legacy_result = self._legacy_service.learn(
            claim_text=legacy_text,
            user_id=user_id,
            source="learn_command",
        )

        if not legacy_result.success:
            log.warning(
                "Dual-write legacy blocked: user=%d reason_len=%d",
                user_id,
                len(legacy_result.rejection_reason),
            )
            return SaveResult(
                success=False,
                error=legacy_result.rejection_reason or "Legacy dual-write failed",
                error_type="rejected",
            )

        # Step 3: Persist to ContractStore (legacy succeeded)
        try:
            saved = self._contract_store.persist(contract, user_id=user_id)
        except ContractStoreError as e:
            # Step 4: Contract persist failed AFTER legacy success -> ROLLBACK legacy
            log.error(
                "Contract persist failed after legacy success, rolling back: "
                "user=%d hyp=%s error=%s",
                user_id,
                legacy_result.hypothesis_id,
                str(e),
            )
            self._rollback_legacy(legacy_result.hypothesis_id)
            error_type = "validation"
            if isinstance(e, ContractDuplicateNameError):
                error_type = "validation"
            return SaveResult(
                success=False,
                error=str(e),
                error_type=error_type,
            )

        # Delete draft + clear any pending edit state
        await self._draft_store.delete(user_id, chat_id, draft_id)
        await self._pending_store.clear_pending(user_id, chat_id)

        log.info(
            "Learn flow saved: contract=%s user=%d",
            saved.id,
            user_id,
        )
        return SaveResult(success=True, contract_name=saved.name)

    async def cancel_draft(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
    ) -> bool:
        """Cancel a draft: delete from DraftStore.

        Returns True if deleted, False if not found.
        """
        try:
            deleted = await self._draft_store.delete(user_id, chat_id, draft_id)
        except DraftOwnershipError:
            return False
        return deleted

    async def edit_trigger(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
        etag: str,
        new_trigger: str,
    ) -> EditResult:
        """Edit the trigger phrase of a draft.

        Revalidates trigger, runs safety precheck, updates draft with new etag.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            draft_id: Draft identifier.
            etag: Expected etag.
            new_trigger: New trigger phrase.

        Returns:
            EditResult with success flag and updated draft.
        """
        try:
            draft = await self._draft_store.get(user_id, chat_id, draft_id)
        except DraftOwnershipError:
            return EditResult(success=False, error="ownership", error_type="ownership")

        if draft is None:
            return EditResult(success=False, error="not_found", error_type="not_found")

        # Apply trigger edit (validates trigger format/stopwords)
        updated_contract, err = self._builder.apply_trigger_edit(
            draft.contract, new_trigger
        )
        if err is not None:
            return EditResult(success=False, error=err, error_type="rejected")

        # Safety precheck on the updated contract (One Safety Gate)
        safety_rejection = self._validate_contract_safety(
            updated_contract, user_id, "edit_trigger"
        )
        if safety_rejection is not None:
            return EditResult(
                success=False, error=safety_rejection, error_type="rejected"
            )

        # Update draft in store
        try:
            updated_draft = await self._draft_store.update(
                user_id, chat_id, draft_id, updated_contract, etag
            )
        except DraftExpiredError:
            return EditResult(success=False, error="expired", error_type="expired")
        except DraftEtagMismatchError:
            return EditResult(success=False, error="stale", error_type="stale")
        except DraftNotFoundError:
            return EditResult(success=False, error="not_found", error_type="not_found")

        return EditResult(success=True, draft=updated_draft)

    async def edit_instruction(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
        etag: str,
        new_instruction: str,
    ) -> EditResult:
        """Edit the instruction of a draft.

        Runs safety precheck on new instruction, revalidates, updates draft.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            draft_id: Draft identifier.
            etag: Expected etag.
            new_instruction: New instruction text.

        Returns:
            EditResult with success flag and updated draft.
        """
        try:
            draft = await self._draft_store.get(user_id, chat_id, draft_id)
        except DraftOwnershipError:
            return EditResult(success=False, error="ownership", error_type="ownership")

        if draft is None:
            return EditResult(success=False, error="not_found", error_type="not_found")

        # Apply instruction edit
        updated_contract, err = self._builder.apply_instruction_edit(
            draft.contract, new_instruction
        )
        if err is not None:
            return EditResult(success=False, error=err, error_type="rejected")

        # Safety precheck on updated contract (One Safety Gate)
        safety_rejection = self._validate_contract_safety(
            updated_contract, user_id, "edit_instruction"
        )
        if safety_rejection is not None:
            return EditResult(
                success=False, error=safety_rejection, error_type="rejected"
            )

        # Update draft in store
        try:
            updated_draft = await self._draft_store.update(
                user_id, chat_id, draft_id, updated_contract, etag
            )
        except DraftExpiredError:
            return EditResult(success=False, error="expired", error_type="expired")
        except DraftEtagMismatchError:
            return EditResult(success=False, error="stale", error_type="stale")
        except DraftNotFoundError:
            return EditResult(success=False, error="not_found", error_type="not_found")

        return EditResult(success=True, draft=updated_draft)

    async def set_pending_edit(
        self,
        user_id: int,
        chat_id: int,
        draft_id: str,
        etag: str,
        action: str,
    ) -> PendingState:
        """Set a pending edit/input state for follow-up handler.

        Called by the presentation layer when user clicks Edit or
        when needs_input requires follow-up.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            draft_id: Draft being edited.
            etag: Current draft etag.
            action: 'edit_trigger' | 'edit_instruction' | 'needs_input'

        Returns:
            The created PendingState.
        """
        return await self._pending_store.set_pending(
            user_id=user_id,
            chat_id=chat_id,
            draft_id=draft_id,
            etag=etag,
            action=action,
        )

    async def get_pending_state(
        self, user_id: int, chat_id: int
    ) -> Optional[PendingState]:
        """Get the current pending edit/input state for a user+chat.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.

        Returns:
            PendingState or None.
        """
        return await self._pending_store.get_pending(user_id, chat_id)

    async def clear_pending_state(self, user_id: int, chat_id: int) -> bool:
        """Clear the pending state for a user+chat.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.

        Returns:
            True if cleared.
        """
        return await self._pending_store.clear_pending(user_id, chat_id)

    async def handle_follow_up(
        self,
        user_id: int,
        chat_id: int,
        text: str,
    ) -> Optional["EditResult"]:
        """Handle a follow-up message for pending edit/input.

        This is the core method called by the follow-up message handler.
        It resolves the pending state and calls the appropriate edit method.

        Args:
            user_id: Telegram user ID.
            chat_id: Chat context.
            text: The user's follow-up text.

        Returns:
            EditResult if a pending state was found and processed,
            None if no pending state (message should be handled normally).
        """
        state = await self._pending_store.get_pending(user_id, chat_id)
        if state is None:
            return None

        # MEDIUM-1 fix: Do NOT clear pending state before the edit succeeds.
        # Only clear on success or terminal errors (expired/not_found/ownership).
        # On rejected/stale: keep or refresh pending so user can retry.

        # Dispatch based on action
        if state.action == "edit_trigger":
            result = await self.edit_trigger(
                user_id=user_id,
                chat_id=chat_id,
                draft_id=state.draft_id,
                etag=state.etag,
                new_trigger=text.strip(),
            )
        elif state.action == "edit_instruction":
            result = await self.edit_instruction(
                user_id=user_id,
                chat_id=chat_id,
                draft_id=state.draft_id,
                etag=state.etag,
                new_instruction=text.strip(),
            )
        elif state.action == "needs_input":
            # User is providing the missing trigger
            result = await self.edit_trigger(
                user_id=user_id,
                chat_id=chat_id,
                draft_id=state.draft_id,
                etag=state.etag,
                new_trigger=text.strip(),
            )
        else:
            log.warning("Unknown pending action: %s user=%d", state.action, user_id)
            await self._pending_store.clear_pending(user_id, chat_id)
            return None

        # Post-dispatch: manage pending state based on result
        if result.success:
            # Edit succeeded: clear pending, flow continues to preview
            await self._pending_store.clear_pending(user_id, chat_id)
        elif result.error_type in ("expired", "not_found", "ownership"):
            # Terminal errors: clear pending (draft is gone/unreachable)
            await self._pending_store.clear_pending(user_id, chat_id)
        else:
            # Retryable errors (rejected, stale): keep pending state active.
            # If stale: refresh etag from the draft if possible.
            if result.error_type == "stale" and result.draft is not None:
                await self._pending_store.set_pending(
                    user_id=user_id,
                    chat_id=chat_id,
                    draft_id=state.draft_id,
                    etag=result.draft.etag,
                    action=state.action,
                )
            # For 'rejected': pending stays as-is, user can try again

        return result

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    async def _persist_contract(
        self,
        user_id: int,
        contract: SkillContract,
        text: str,
        build_result: BuildResult,
    ) -> LearnFlowResult:
        """Persist a contract (used by quick mode).

        One Safety Gate already ran in start_learn() before this is called.
        Transitions lifecycle draft->confirmed, persists to ContractStore,
        atomic dual-writes to legacy HypothesisStorage.
        """
        # Transition lifecycle: draft -> confirmed
        confirmed_contract = replace(
            contract,
            lifecycle=LifecycleConfig(
                status="confirmed",
                editable=contract.lifecycle.editable,
                decay=contract.lifecycle.decay,
                last_schema_migration=contract.lifecycle.last_schema_migration,
            ),
        )

        # BLOCKER-1 fix: Atomic dual-write with Contract Preflight + Rollback (quick path).
        # Same strategy as save_draft: preflight -> legacy -> contract -> rollback on fail.
        trigger = ""
        if confirmed_contract.activation.phrases:
            trigger = confirmed_contract.activation.phrases[0]
        instruction = confirmed_contract.execution.instruction
        if trigger:
            legacy_text = f"when I say {trigger}, {instruction}"
        else:
            legacy_text = text

        # Step 1: Contract Preflight
        preflight_error = self._contract_preflight(confirmed_contract, user_id)
        if preflight_error is not None:
            return LearnFlowResult(
                status="rejected",
                build_result=build_result,
                rejection_reason=preflight_error,
            )

        # Step 2: Legacy write
        legacy_result = self._legacy_service.learn(
            claim_text=legacy_text,
            user_id=user_id,
            source="learn_command",
        )

        if not legacy_result.success:
            log.warning(
                "Dual-write legacy blocked (quick): user=%d reason_len=%d",
                user_id,
                len(legacy_result.rejection_reason),
            )
            return LearnFlowResult(
                status="rejected",
                build_result=build_result,
                rejection_reason=legacy_result.rejection_reason
                or "Legacy dual-write failed",
            )

        # Step 3: Persist to ContractStore (legacy succeeded)
        try:
            saved = self._contract_store.persist(confirmed_contract, user_id=user_id)
        except ContractStoreError as e:
            # Step 4: Contract persist failed -> ROLLBACK legacy
            log.error(
                "Contract persist failed after legacy success (quick), rolling back: "
                "user=%d hyp=%s error=%s",
                user_id,
                legacy_result.hypothesis_id,
                str(e),
            )
            self._rollback_legacy(legacy_result.hypothesis_id)
            return LearnFlowResult(
                status="rejected",
                build_result=build_result,
                rejection_reason=str(e),
            )

        log.info(
            "Learn flow quick-saved: contract=%s user=%d",
            saved.id,
            user_id,
        )
        return LearnFlowResult(
            status="saved",
            build_result=build_result,
            saved_contract_name=saved.name,
        )

    def _contract_preflight(
        self,
        contract: SkillContract,
        user_id: int,
    ) -> Optional[str]:
        """Pre-validate contract before legacy write to catch foreseeable errors.

        Checks name-uniqueness and runs ContractStore validation logic
        WITHOUT actually writing to DB. This catches ContractDuplicateNameError,
        ContractValidationError, etc. BEFORE the legacy hypothesis is created.

        Returns:
            None if preflight passes, or error message string if it would fail.
        """
        # Check name uniqueness (the most common failure cause)
        if contract.name and self._contract_store.exists_by_name(
            user_id, contract.name
        ):
            return f"A contract with name '{contract.name}' already exists for user {user_id}"

        # Validate contract schema (V1-V17) without persisting
        from application.skill_compression.contract_store import (
            _finalize_security_metadata,
        )
        from application.skill_compression.contract_validator import validate

        try:
            finalized = _finalize_security_metadata(contract)
        except Exception as e:
            return f"Contract finalization failed: {e}"

        result = validate(
            finalized,
            db_schema_version=finalized.schema_version,
            db_contract_version=finalized.contract_version,
        )
        if not result.is_valid:
            error_msgs = "; ".join(f"[{i.rule}] {i.message}" for i in result.errors)
            return f"Contract validation failed: {error_msgs}"

        return None

    def _rollback_legacy(self, hypothesis_id: str) -> None:
        """Tombstone a legacy hypothesis after contract persist failure.

        Sets the hypothesis status to 'rejected' so the Matcher ignores it.
        This is the rollback mechanism for the dual-write atomicity guarantee.

        If the hypothesis_id is empty or the tombstone fails, we log but do not
        raise (to avoid masking the original ContractStoreError).
        """
        if not hypothesis_id:
            log.warning("Cannot rollback legacy: empty hypothesis_id")
            return

        try:
            self._legacy_service._storage.update_hypothesis_status(
                hypothesis_id, "rejected"
            )
            log.info(
                "Rolled back legacy hypothesis: hyp=%s (set to rejected)",
                hypothesis_id,
            )
        except Exception as e:
            log.error(
                "Failed to rollback legacy hypothesis %s: %s",
                hypothesis_id,
                str(e),
            )
