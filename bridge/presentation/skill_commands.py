"""Skill Chat-Shortcuts: /skills, /skill, /skillforget, /learn, /explain, /import.

Layer 6 (UI): Telegram command handlers for skill management.
Integrates with SkillMatcher (Layer 5) and HypothesisStorage.

Commands:
  /skills         - Show top 10 active skills
  /skill X        - Show details for skill X (by ID or name fragment)
  /skillforget X  - Delete skill (30-day tombstone). Named /skillforget
                    because /forget is already used by Memory.
  /learn          - Save last bot interaction as permanent skill
  /explain X      - Explain a skill decision (8 question types)
  /import PATH    - Import conversations from a folder (dry-run first)

HC-SC-7 [BLOCKER]: Tombstones 30 days default, "nie wieder" as permanent.
HC-SC-6 [BLOCKER]: /learn creates decay-immune skills.
HC-SC-13 [BLOCKER]: No-Model-Secret Rule for /learn (allowlist filter).
HC-SC-16 [BLOCKER]: Import strictly opt-in, dry-run first, never periodic.
HC-SC-18 [BLOCKER]: 8 Explainer question types via /explain.
HC-IMPORT-1 [BLOCKER]: All imported hypotheses start as 'suggested'.

Architecture guard: Presentation layer uses only Application services.
No direct infra-layer or raw domain-layer access (except domain types).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationHandlerStop, ContextTypes

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import (
    STATUS_CONFIRMED,
    STATUS_PAUSED,
    STATUS_RETIRED,
)
from application.skill_compression.privacy.secret_scanner import SecretScanner
from application.skill_compression.skill_explainer import (
    ExplainerQuestionType,
    SkillExplainer,
)
from domain.language import DEFAULT_LANGUAGE
from i18n.domain.i18n import t
from typeguard import typechecked

from presentation.decorators import lcp_aware, require_private_chat, require_whitelist
from presentation.skill_profile_view import (
    PROFILE_VISIBLE_STATUSES,
    build_profile_list_keyboard,
    build_skill_actions_keyboard,
    derive_skill_name,
    render_profile,
    render_skill_detail_text,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Language resolution helper
# ---------------------------------------------------------------


async def _resolve_lang(
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    chat_id: int,
) -> str:
    """Resolve user language from chat_service.

    Falls back to DEFAULT_LANGUAGE if chat_service is unavailable.

    Args:
        context: Telegram handler context.
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.

    Returns:
        ISO 639-1 language code.
    """
    chat_service = context.application.bot_data.get("chat_service")
    if chat_service is not None and hasattr(chat_service, "get_chat_language"):
        raw = await chat_service.get_chat_language(user_id, chat_id)
        return raw or DEFAULT_LANGUAGE
    return DEFAULT_LANGUAGE


# ---------------------------------------------------------------
# Secret filter for /learn (HC-SC-13)
# Delegates to the centralized SecretScanner (Step 8).
# ---------------------------------------------------------------

# Shared scanner instance (Step 8 consolidation).
_secret_scanner = SecretScanner()


def check_secret_content(text: str) -> Optional[str]:
    """Check if text contains secret/sensitive content (HC-SC-13).

    No-Model-Secret Rule: Skills must not store API tokens, prices,
    passwords, private identifiers, or raw data.

    Delegates to SecretScanner (Step 8) for multi-layered detection.

    Args:
        text: Text to check.

    Returns:
        Description of the detected secret type, or None if clean.
    """
    matches = _secret_scanner.scan(text)
    if matches:
        return matches[0].description_de
    return None


# ---------------------------------------------------------------
# Helper: get storage from bot_data
# ---------------------------------------------------------------


def _get_hypothesis_storage(
    context: ContextTypes.DEFAULT_TYPE,
) -> Optional[HypothesisStorage]:
    """Get HypothesisStorage from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        HypothesisStorage or None if not initialized.
    """
    return context.application.bot_data.get("hypothesis_storage")


def _get_skill_learning_service(context: ContextTypes.DEFAULT_TYPE):
    """Get SkillLearningService from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        SkillLearningService or None if not initialized.
    """
    return context.application.bot_data.get("skill_learning_service")


def _get_skill_explainer(
    context: ContextTypes.DEFAULT_TYPE,
) -> Optional[SkillExplainer]:
    """Get SkillExplainer from bot_data.

    Falls back to creating one from storage if not pre-initialized.

    Args:
        context: Telegram handler context.

    Returns:
        SkillExplainer or None.
    """
    explainer = context.application.bot_data.get("skill_explainer")
    if explainer is not None:
        return explainer

    # Fallback: create from storage
    storage = _get_hypothesis_storage(context)
    if storage is None:
        return None
    return SkillExplainer(storage)


# ---------------------------------------------------------------
# Question type mapping for /explain
# ---------------------------------------------------------------

_QUESTION_TYPE_ALIASES: dict[str, ExplainerQuestionType] = {
    "1": ExplainerQuestionType.WHAT_RECOGNIZED,
    "was": ExplainerQuestionType.WHAT_RECOGNIZED,
    "what": ExplainerQuestionType.WHAT_RECOGNIZED,
    "erkannt": ExplainerQuestionType.WHAT_RECOGNIZED,
    "2": ExplainerQuestionType.WHY_NOT_SKILL,
    "warum-nicht": ExplainerQuestionType.WHY_NOT_SKILL,
    "why-not": ExplainerQuestionType.WHY_NOT_SKILL,
    "3": ExplainerQuestionType.WHY_PROMOTED,
    "warum": ExplainerQuestionType.WHY_PROMOTED,
    "why": ExplainerQuestionType.WHY_PROMOTED,
    "promotet": ExplainerQuestionType.WHY_PROMOTED,
    "4": ExplainerQuestionType.WHEN_DRIFT,
    "drift": ExplainerQuestionType.WHEN_DRIFT,
    "wann": ExplainerQuestionType.WHEN_DRIFT,
    "5": ExplainerQuestionType.WHAT_NEEDED,
    "needed": ExplainerQuestionType.WHAT_NEEDED,
    "6": ExplainerQuestionType.LESSONS_LEARNED,
    "lessons": ExplainerQuestionType.LESSONS_LEARNED,
    "gelernt": ExplainerQuestionType.LESSONS_LEARNED,
    "7": ExplainerQuestionType.SCOPE_BOUNDARIES,
    "scope": ExplainerQuestionType.SCOPE_BOUNDARIES,
    "grenzen": ExplainerQuestionType.SCOPE_BOUNDARIES,
    "8": ExplainerQuestionType.COUNTER_EVIDENCE,
    "gegen": ExplainerQuestionType.COUNTER_EVIDENCE,
    "counter": ExplainerQuestionType.COUNTER_EVIDENCE,
    "gegenbelege": ExplainerQuestionType.COUNTER_EVIDENCE,
}


# ---------------------------------------------------------------
# Command: /skills
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_skills_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /skills command: show top 10 active skills.

    Shows a compact list with inline buttons for details.

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(t("skill.system_not_initialized", lang))
        return

    # Load all visible hypotheses for this user
    hypotheses: list[Hypothesis] = []
    for status in PROFILE_VISIBLE_STATUSES:
        hypotheses.extend(
            storage.get_hypotheses_by_user(user_id, status=status, limit=50)
        )

    # Render profile text
    profile_text = render_profile(hypotheses, lang=lang)

    # Build inline keyboard
    keyboard = build_profile_list_keyboard(hypotheses)

    await update.message.reply_text(
        text=profile_text,
        reply_markup=keyboard,
    )


# ---------------------------------------------------------------
# Command: /skill X
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
@lcp_aware
async def handle_skill_detail_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /skill X command: show details for a specific skill.

    X can be a hypothesis_id or a name fragment (fuzzy match).
    If X is ambiguous: shows a list for selection.

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(t("skill.system_not_initialized", lang))
        return

    # Parse argument
    args = context.args
    if not args:
        await update.message.reply_text(t("skill.detail_usage", lang))
        return

    query = " ".join(args).strip()

    # Try exact ID match first
    hyp = storage.get_hypothesis(query)
    if hyp is not None and hyp.user_id == user_id:
        version_history = storage.get_version_history(query)
        detail_text = render_skill_detail_text(hyp, version_history, lang=lang)
        keyboard = build_skill_actions_keyboard(hyp, lang=lang)
        await update.message.reply_text(
            text=detail_text,
            reply_markup=keyboard,
        )
        return

    # Fuzzy name match: search all visible hypotheses
    all_hyps: list[Hypothesis] = []
    for status in PROFILE_VISIBLE_STATUSES:
        all_hyps.extend(
            storage.get_hypotheses_by_user(user_id, status=status, limit=100)
        )

    query_lower = query.lower()
    matches = [
        h
        for h in all_hyps
        if query_lower in derive_skill_name(h).lower() or query_lower in h.claim.lower()
    ]

    if not matches:
        await update.message.reply_text(t("skill.not_found_query", lang, query=query))
        return

    if len(matches) == 1:
        hyp = matches[0]
        version_history = storage.get_version_history(hyp.hypothesis_id)
        detail_text = render_skill_detail_text(hyp, version_history, lang=lang)
        keyboard = build_skill_actions_keyboard(hyp, lang=lang)
        await update.message.reply_text(
            text=detail_text,
            reply_markup=keyboard,
        )
        return

    # Ambiguous: show selection list
    buttons: list[list[InlineKeyboardButton]] = []
    for h in matches[:10]:
        name = derive_skill_name(h)
        buttons.append(
            [
                InlineKeyboardButton(
                    text=name,
                    callback_data=f"skill_detail:{h.hypothesis_id}",
                )
            ]
        )

    await update.message.reply_text(
        text=t("skill.ambiguous_query", lang, query=query),
        reply_markup=InlineKeyboardMarkup(buttons),
    )


# ---------------------------------------------------------------
# Command: /forget X
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
@lcp_aware
async def handle_skill_forget_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /forget X command: delete a skill with tombstone.

    HC-SC-7: 30-day tombstone default. --permanent for "nie wieder".

    If X is not provided: shows list of skills to forget (IC-CMD-1).

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(t("skill.system_not_initialized", lang))
        return

    args = context.args
    if not args:
        # IC-CMD-1: show list of forgettable skills
        all_hyps: list[Hypothesis] = []
        for status in PROFILE_VISIBLE_STATUSES:
            all_hyps.extend(
                storage.get_hypotheses_by_user(user_id, status=status, limit=50)
            )

        if not all_hyps:
            await update.message.reply_text(t("skill.none_available", lang))
            return

        buttons: list[list[InlineKeyboardButton]] = []
        for h in all_hyps[:10]:
            name = derive_skill_name(h)
            buttons.append(
                [
                    InlineKeyboardButton(
                        text=t("skill.forget_button", lang, name=name),
                        callback_data=f"skill_forget:{h.hypothesis_id}",
                    )
                ]
            )

        await update.message.reply_text(
            text=t("skill.forget_prompt", lang),
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return

    # Parse arguments
    raw_args = " ".join(args).strip()
    permanent = "--permanent" in raw_args
    query = raw_args.replace("--permanent", "").strip()

    # Find the skill
    hyp = storage.get_hypothesis(query)
    if hyp is None or hyp.user_id != user_id:
        # Try fuzzy match
        all_hyps = []
        for status in PROFILE_VISIBLE_STATUSES:
            all_hyps.extend(
                storage.get_hypotheses_by_user(user_id, status=status, limit=100)
            )
        query_lower = query.lower()
        matches = [
            h
            for h in all_hyps
            if query_lower in derive_skill_name(h).lower()
            or query_lower in h.claim.lower()
        ]

        if not matches:
            await update.message.reply_text(
                t("skill.not_found_query_short", lang, query=query)
            )
            return

        if len(matches) > 1:
            buttons = []
            suffix = ":perm" if permanent else ""
            for h in matches[:10]:
                name = derive_skill_name(h)
                buttons.append(
                    [
                        InlineKeyboardButton(
                            text=name,
                            callback_data=f"skill_forget{suffix}:{h.hypothesis_id}",
                        )
                    ]
                )
            await update.message.reply_text(
                text=t("skill.ambiguous_query", lang, query=query),
                reply_markup=InlineKeyboardMarkup(buttons),
            )
            return

        hyp = matches[0]

    # Execute forget
    _execute_forget(storage, hyp, permanent)

    name = derive_skill_name(hyp)
    if permanent:
        await update.message.reply_text(t("skill.forgotten_permanent", lang, name=name))
    else:
        await update.message.reply_text(t("skill.forgotten_tombstone", lang, name=name))


def _execute_forget(
    storage: HypothesisStorage,
    hypothesis: Hypothesis,
    permanent: bool,
) -> None:
    """Execute the forget operation on a hypothesis.

    Sets status to 'retired' and creates a tombstone record.

    Args:
        storage: Hypothesis storage.
        hypothesis: The hypothesis to forget.
        permanent: Whether to create a permanent tombstone.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Set status to retired
    storage.update_hypothesis_status(hypothesis.hypothesis_id, STATUS_RETIRED)

    # Calculate expiration
    expires_at: Optional[str] = None
    if not permanent:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

    # Create tombstone
    tombstone_id = f"tomb_{uuid4().hex[:16]}"
    storage.insert_tombstone(
        tombstone_id=tombstone_id,
        hypothesis_id=hypothesis.hypothesis_id,
        fingerprint=hypothesis.pattern_hash or "",
        deleted_at=now_iso,
        scope_hash=hypothesis.scope_hash,
        expires_at=expires_at,
        permanent=permanent,
    )

    log.info(
        "Skill forgotten: hyp=%s permanent=%s tombstone=%s",
        hypothesis.hypothesis_id,
        permanent,
        tombstone_id,
    )


# ---------------------------------------------------------------
# Command: /learn
# ---------------------------------------------------------------


def _get_learn_flow_service(context: ContextTypes.DEFAULT_TYPE):
    """Get LearnFlowService from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        LearnFlowService or None if not initialized.
    """
    return context.application.bot_data.get("learn_flow_service")


def _render_preview_text(draft, lang: str) -> str:
    """Render a draft preview message using i18n keys.

    Args:
        draft: SkillDraft with contract data.
        lang: Language code.

    Returns:
        Formatted preview text string.
    """
    contract = draft.contract
    trigger = ""
    if contract.activation.phrases:
        trigger = contract.activation.phrases[0]
    lines = [
        t("skill.learn_preview_header", lang),
        t("skill.learn_preview_name", lang, name=contract.name),
        t("skill.learn_preview_trigger", lang, trigger=trigger),
        t(
            "skill.learn_preview_action",
            lang,
            instruction=contract.execution.instruction,
        ),
        t("skill.learn_preview_permissions", lang),
    ]
    return "\n".join(lines)


def _build_learn_buttons(draft_id: str, etag: str, lang: str) -> InlineKeyboardMarkup:
    """Build Save/Edit/Cancel buttons for learn preview.

    Args:
        draft_id: Draft identifier.
        etag: Current etag for optimistic locking.
        lang: Language code.

    Returns:
        InlineKeyboardMarkup with 3 buttons.
    """
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text=t("skill.learn_btn_save", lang),
                    callback_data=f"skill_learn:save:{draft_id}:{etag}",
                ),
                InlineKeyboardButton(
                    text=t("skill.learn_btn_edit", lang),
                    callback_data=f"skill_learn:edit:{draft_id}:{etag}",
                ),
                InlineKeyboardButton(
                    text=t("skill.learn_btn_cancel", lang),
                    callback_data=f"skill_learn:cancel:{draft_id}",
                ),
            ]
        ]
    )


@require_whitelist
@require_private_chat
@lcp_aware
@typechecked
async def handle_learn_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /learn command: create a skill via contract-based flow.

    Supports --quick flag for immediate persist (skip preview).
    Normal mode: builds draft, shows preview with Save/Edit/Cancel buttons.

    HC-SC-13 [BLOCKER]: Checks for secrets before storing.

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(t("skill.system_not_initialized", lang))
        return

    # Get the content to learn: either from args or from reply
    args = context.args
    reply = update.message.reply_to_message

    if args:
        skill_text = " ".join(args).strip()
    elif reply and reply.text:
        skill_text = reply.text.strip()
    else:
        await update.message.reply_text(t("skill.learn_usage", lang))
        return

    if not skill_text:
        await update.message.reply_text(t("skill.learn_empty", lang))
        return

    # Parse --quick flag
    quick = False
    if "--quick" in skill_text:
        quick = True
        skill_text = skill_text.replace("--quick", "").strip()
        if not skill_text:
            await update.message.reply_text(t("skill.learn_empty", lang))
            return

    # HC-SC-8: Check max active skills
    active_count = storage.count_active_hypotheses(user_id)
    confirmed_count = len(
        storage.get_hypotheses_by_user(user_id, status=STATUS_CONFIRMED, limit=51)
    )
    total_skills = active_count + confirmed_count
    if total_skills >= 50:
        await update.message.reply_text(t("skill.learn_max_reached", lang))
        return

    # Use LearnFlowService (new contract-based flow)
    learn_flow = _get_learn_flow_service(context)
    if learn_flow is None:
        # Fallback: legacy path if service not wired
        learning_service = _get_skill_learning_service(context)
        if learning_service is None:
            secret_type = check_secret_content(skill_text)
            if secret_type is not None:
                await update.message.reply_text(
                    t("skill.learn_secret_blocked", lang, secret_type=secret_type)
                )
                log.warning(  # nosemgrep: python-logger-credential-disclosure
                    "Secret filter blocked /learn: user=%d type=%s",
                    user_id,
                    secret_type,
                )
                return
            await update.message.reply_text(t("skill.system_not_initialized", lang))
            return

        result = learning_service.learn(
            claim_text=skill_text,
            user_id=user_id,
            source="learn_command",
        )

        if not result.success:
            await update.message.reply_text(
                t("skill.learn_privacy_blocked", lang, reason=result.rejection_reason)
            )
            return

        hyp = storage.get_hypothesis(result.hypothesis_id)
        name = derive_skill_name(hyp) if hyp else skill_text[:40]
        await update.message.reply_text(t("skill.learn_saved", lang, name=name))
        return

    # Contract-based learn flow
    flow_result = await learn_flow.start_learn(
        user_id=user_id,
        chat_id=chat_id,
        text=skill_text,
        quick=quick,
    )

    if flow_result.status == "rejected":
        await update.message.reply_text(
            t("skill.learn_quick_rejected", lang, reason=flow_result.rejection_reason)
        )
        log.info(
            "Learn flow rejected: user=%d reason_len=%d",
            user_id,
            len(flow_result.rejection_reason),
        )
        return

    if flow_result.status == "needs_input":
        # Ask the user for the missing trigger
        if flow_result.rejection_reason == "trigger_rejected":
            await update.message.reply_text(
                t("skill.learn_trigger_rejected", lang, reason="reserved word")
            )
        else:
            await update.message.reply_text(t("skill.learn_needs_trigger", lang))
        return

    if flow_result.status == "saved":
        # Quick mode: already persisted
        await update.message.reply_text(
            t("skill.learn_quick_saved", lang, name=flow_result.saved_contract_name)
        )
        log.info(
            "Skill quick-saved via /learn: user=%d name_len=%d",
            user_id,
            len(flow_result.saved_contract_name),
        )
        return

    if flow_result.status == "preview" and flow_result.draft is not None:
        # Show preview with buttons
        preview_text = _render_preview_text(flow_result.draft, lang)
        keyboard = _build_learn_buttons(
            flow_result.draft.draft_id, flow_result.draft.etag, lang
        )
        await update.message.reply_text(
            text=preview_text,
            reply_markup=keyboard,
        )
        log.info(
            "Learn preview shown: user=%d draft=%s",
            user_id,
            flow_result.draft.draft_id,
        )
        return


# ---------------------------------------------------------------
# Learn flow callback handler (skill_learn:*)
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_learn_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle learn flow inline keyboard callbacks.

    Callback patterns:
      skill_learn:save:<draft_id>:<etag>
      skill_learn:edit:<draft_id>:<etag>
      skill_learn:cancel:<draft_id>

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data
    user = query.from_user
    if not user:
        return
    user_id = user.id
    chat_id = query.message.chat_id if query.message else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    learn_flow = _get_learn_flow_service(context)
    if learn_flow is None:
        await query.answer(
            text=t("skill.system_not_initialized_short", lang), show_alert=True
        )
        return

    # Parse callback data
    parts = data.split(":")
    if len(parts) < 3:
        await query.answer()
        return

    action = parts[1]  # save | edit | cancel

    if action == "save" and len(parts) >= 4:
        draft_id = parts[2]
        etag = parts[3]
        await _handle_learn_save(
            query, learn_flow, user_id, chat_id, draft_id, etag, lang
        )

    elif action == "edit" and len(parts) >= 4:
        draft_id = parts[2]
        etag = parts[3]
        await _handle_learn_edit_prompt(
            query, learn_flow, user_id, chat_id, draft_id, etag, lang
        )

    elif action == "edit_trigger" and len(parts) >= 4:
        draft_id = parts[2]
        etag = parts[3]
        await _handle_learn_set_pending_edit(
            query,
            learn_flow,
            user_id,
            chat_id,
            draft_id,
            etag,
            action="edit_trigger",
            lang=lang,
        )

    elif action == "edit_instruction" and len(parts) >= 4:
        draft_id = parts[2]
        etag = parts[3]
        await _handle_learn_set_pending_edit(
            query,
            learn_flow,
            user_id,
            chat_id,
            draft_id,
            etag,
            action="edit_instruction",
            lang=lang,
        )

    elif action == "cancel" and len(parts) >= 3:
        draft_id = parts[2]
        await _handle_learn_cancel(query, learn_flow, user_id, chat_id, draft_id, lang)

    else:
        await query.answer()


async def _handle_learn_save(
    query, learn_flow, user_id: int, chat_id: int, draft_id: str, etag: str, lang: str
) -> None:
    """Save a learn draft via callback."""
    await query.answer()

    result = await learn_flow.save_draft(user_id, chat_id, draft_id, etag)

    if result.success:
        await query.edit_message_text(
            t("skill.learn_saved", lang, name=result.contract_name)
        )
        return

    # Error mapping to i18n keys
    if result.error_type == "not_found":
        await query.edit_message_text(t("skill.learn_draft_already_saved", lang))
    elif result.error_type == "stale":
        await query.edit_message_text(t("skill.learn_draft_stale", lang))
    elif result.error_type == "ownership":
        await query.edit_message_text(t("skill.learn_draft_not_yours", lang))
    elif result.error_type == "rejected":
        await query.edit_message_text(
            t("skill.learn_quick_rejected", lang, reason=result.error)
        )
    elif result.error_type == "validation":
        await query.edit_message_text(
            t("skill.learn_validation_failed", lang, reason=result.error)
        )
    else:
        await query.edit_message_text(
            t("skill.learn_validation_failed", lang, reason=result.error)
        )


async def _handle_learn_edit_prompt(
    query, learn_flow, user_id: int, chat_id: int, draft_id: str, etag: str, lang: str
) -> None:
    """Show edit choice (trigger or instruction) and set pending state.

    Sets a pending edit state so the follow-up message handler can
    intercept the user's next message and apply the edit.
    """
    await query.answer()

    # Show choice buttons: edit trigger or edit instruction
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text=t("skill.learn_edit_trigger_btn", lang),
                    callback_data=f"skill_learn:edit_trigger:{draft_id}:{etag}",
                ),
                InlineKeyboardButton(
                    text=t("skill.learn_edit_instruction_btn", lang),
                    callback_data=f"skill_learn:edit_instruction:{draft_id}:{etag}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=t("skill.learn_btn_cancel", lang),
                    callback_data=f"skill_learn:cancel:{draft_id}",
                ),
            ],
        ]
    )
    await query.edit_message_text(
        text=t("skill.learn_edit_choice", lang),
        reply_markup=keyboard,
    )


async def _handle_learn_set_pending_edit(
    query,
    learn_flow,
    user_id: int,
    chat_id: int,
    draft_id: str,
    etag: str,
    action: str,
    lang: str,
) -> None:
    """Set pending edit state and prompt user for new value.

    After this, the follow-up message handler will intercept the user's
    next text message and apply the edit.
    """
    await query.answer()

    # Set pending state in LearnFlowService
    await learn_flow.set_pending_edit(
        user_id=user_id,
        chat_id=chat_id,
        draft_id=draft_id,
        etag=etag,
        action=action,
    )

    # Prompt user for the new value
    if action == "edit_trigger":
        prompt_key = "skill.learn_edit_trigger_prompt"
    else:
        prompt_key = "skill.learn_edit_instruction_prompt"

    await query.edit_message_text(t(prompt_key, lang))


async def _handle_learn_cancel(
    query, learn_flow, user_id: int, chat_id: int, draft_id: str, lang: str
) -> None:
    """Cancel a learn draft via callback."""
    await query.answer()
    await learn_flow.cancel_draft(user_id, chat_id, draft_id)
    # Also clear any pending edit state
    await learn_flow.clear_pending_state(user_id, chat_id)
    await query.edit_message_text(t("skill.learn_cancelled", lang))


# ---------------------------------------------------------------
# Follow-up Message Handler for Edit/Needs-Input
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_learn_followup_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle follow-up messages for pending edit/needs_input states.

    This handler is registered in main.py in group 0 (higher priority).
    It checks if the user has a pending edit/input state. If yes, it
    processes the message as an edit and raises ApplicationHandlerStop
    to prevent group 1 (handle_message) from processing the same message.
    If no pending state, it returns normally so the next group runs.

    Args:
        update: Telegram update.
        context: Telegram handler context.

    Raises:
        ApplicationHandlerStop: When the message was consumed by the edit flow.
    """
    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else 0
    text = update.message.text if update.message else ""

    if not text or not text.strip():
        return

    learn_flow = _get_learn_flow_service(context)
    if learn_flow is None:
        return

    # Check for pending state
    pending = await learn_flow.get_pending_state(user_id, chat_id)
    if pending is None:
        return  # No pending state: let handle_message in group 1 process it

    # We have a pending state: process the follow-up
    lang = await _resolve_lang(context, user_id, chat_id)

    edit_result = await learn_flow.handle_follow_up(
        user_id=user_id,
        chat_id=chat_id,
        text=text,
    )

    if edit_result is None:
        # Should not happen (we checked pending above), but be safe
        return

    if not edit_result.success:
        # Edit failed: inform user, they can try again
        await update.message.reply_text(
            t("skill.learn_edit_failed", lang, reason=edit_result.error)
        )
        raise ApplicationHandlerStop

    # Edit succeeded: show new preview with updated draft
    if edit_result.draft is not None:
        preview_text = _render_preview_text(edit_result.draft, lang)
        keyboard = _build_learn_buttons(
            edit_result.draft.draft_id, edit_result.draft.etag, lang
        )
        await update.message.reply_text(
            text=preview_text,
            reply_markup=keyboard,
        )
        log.info(
            "Learn edit applied: user=%d draft=%s action=%s",
            user_id,
            edit_result.draft.draft_id,
            pending.action,
        )
    else:
        await update.message.reply_text(
            t("skill.learn_edit_failed", lang, reason="Draft not found after edit")
        )

    # Consumed: stop further handler groups from processing this message
    raise ApplicationHandlerStop


# ---------------------------------------------------------------
# Command: /explain X
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
@lcp_aware
async def handle_explain_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /explain X command: explain a skill decision.

    HC-SC-18 [BLOCKER]: 8 question types available.

    Usage:
      /explain <skill_name>                - Default: what was recognized
      /explain <skill_name> <question>     - Specific question type

    Question types (by number or keyword):
      1/was     - Was hat das Pattern erkannt?
      2/warum-nicht - Warum nicht zum Skill?
      3/warum   - Warum promotet?
      4/drift   - Wann Drift erkannt?
      5/nötig  - Was wäre nötig?
      6/lessons - Welche Lessons?
      7/scope   - Wo gilt es NICHT?
      8/gegen   - Welche Gegenbelege?

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(t("skill.system_not_initialized", lang))
        return

    explainer = _get_skill_explainer(context)
    if explainer is None:
        await update.message.reply_text(t("skill.system_not_initialized", lang))
        return

    args = context.args
    if not args:
        # Show help with all question types
        types_list = explainer.list_question_types()
        lines = [
            t("skill.explain_usage", lang),
            "",
            "Fragetypen:",  # noqa: en-only
        ]
        for i, (_, desc) in enumerate(types_list, 1):
            lines.append(f"  {i}. {desc}")
        lines.append("")
        lines.append(t("skill.explain_example", lang))
        await update.message.reply_text("\n".join(lines))
        return

    # Parse: last arg might be a question type number/keyword
    raw_args = list(args)
    question_type = ExplainerQuestionType.WHAT_RECOGNIZED  # default

    # Check if last argument is a question type
    last_arg_lower = raw_args[-1].lower()
    if last_arg_lower in _QUESTION_TYPE_ALIASES:
        question_type = _QUESTION_TYPE_ALIASES[last_arg_lower]
        raw_args = raw_args[:-1]

    if not raw_args:
        await update.message.reply_text(t("skill.explain_name_required", lang))
        return

    query = " ".join(raw_args).strip()

    # Find the hypothesis
    hyp = storage.get_hypothesis(query)
    if hyp is not None and hyp.user_id == user_id:
        response = explainer.explain(hyp.hypothesis_id, question_type)
        await update.message.reply_text(
            f"{response.title}\n{'=' * len(response.title)}\n\n{response.explanation}"
        )
        return

    # Fuzzy match
    all_hyps: list[Hypothesis] = []
    for status in PROFILE_VISIBLE_STATUSES:
        all_hyps.extend(
            storage.get_hypotheses_by_user(user_id, status=status, limit=100)
        )

    query_lower = query.lower()
    matches = [
        h
        for h in all_hyps
        if query_lower in derive_skill_name(h).lower() or query_lower in h.claim.lower()
    ]

    if not matches:
        await update.message.reply_text(t("skill.not_found_query", lang, query=query))
        return

    if len(matches) == 1:
        response = explainer.explain(matches[0].hypothesis_id, question_type)
        await update.message.reply_text(
            f"{response.title}\n{'=' * len(response.title)}\n\n{response.explanation}"
        )
        return

    # Ambiguous: show selection with explain buttons
    buttons: list[list[InlineKeyboardButton]] = []
    for h in matches[:10]:
        name = derive_skill_name(h)
        buttons.append(
            [
                InlineKeyboardButton(
                    text=name,
                    callback_data=f"skill_explain:{question_type.value}:{h.hypothesis_id}",
                )
            ]
        )

    await update.message.reply_text(
        text=t("skill.ambiguous_query", lang, query=query),
        reply_markup=InlineKeyboardMarkup(buttons),
    )


# ---------------------------------------------------------------
# Callback handlers for inline buttons
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
async def handle_skill_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle skill-related inline keyboard callbacks.

    Callback patterns:
      skill_detail:<hyp_id>  - Show skill details
      skill_pause:<hyp_id>   - Pause a skill
      skill_resume:<hyp_id>  - Resume a paused skill
      skill_forget:<hyp_id>  - Forget a skill (30-day)
      skill_forget:perm:<hyp_id> - Forget permanently
      skill_versions:<hyp_id> - Show version history
      skill_undo:<hyp_id>    - Undo last skill application
      skill_explain:<hyp_id> - Show explanation (Schritt 6 stub)

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data
    user = query.from_user
    if not user:
        return
    user_id = user.id
    chat_id = query.message.chat_id if query.message else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await query.answer(
            text=t("skill.system_not_initialized_short", lang),
            show_alert=True,
        )
        return

    # Parse callback data
    # RISK-3: Ask-Before-Apply confirmation (skill_confirm:yes/no/never:<id>)
    if data.startswith("skill_confirm:"):
        # Delegate to the dedicated confirm handler (shares this callback space)
        await _handle_skill_confirm_inline(query, context, user_id, chat_id, data, lang)
        return

    if data.startswith("skill_detail:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_detail_callback(query, storage, user_id, hyp_id, lang)

    elif data.startswith("skill_pause:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_pause_callback(query, storage, user_id, hyp_id, lang)

    elif data.startswith("skill_resume:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_resume_callback(query, storage, user_id, hyp_id, lang)

    elif data.startswith("skill_forget:perm:"):
        hyp_id = data.split(":", 2)[2]
        await _handle_forget_callback(
            query, storage, user_id, hyp_id, permanent=True, lang=lang
        )

    elif data.startswith("skill_forget:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_forget_callback(
            query, storage, user_id, hyp_id, permanent=False, lang=lang
        )

    elif data.startswith("skill_versions:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_versions_callback(query, storage, user_id, hyp_id, lang)

    elif data.startswith("skill_undo:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_undo_callback(query, storage, user_id, hyp_id, lang)

    elif data.startswith("skill_explain:"):
        # Explainer callback: skill_explain:<question_type>:<hyp_id>
        parts = data.split(":", 2)
        if len(parts) == 3:
            q_type_str = parts[1]
            hyp_id = parts[2]
            await _handle_explain_callback(
                query, context, user_id, hyp_id, q_type_str, lang
            )
        elif len(parts) == 2:
            hyp_id = parts[1]
            await _handle_explain_callback(
                query, context, user_id, hyp_id, "what_recognized", lang
            )

    else:
        await query.answer()


async def _handle_detail_callback(query, storage, user_id, hyp_id, lang):
    """Show skill detail via callback."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    version_history = storage.get_version_history(hyp_id)
    detail_text = render_skill_detail_text(hyp, version_history, lang=lang)
    keyboard = build_skill_actions_keyboard(hyp, lang=lang)
    await query.edit_message_text(text=detail_text, reply_markup=keyboard)


async def _handle_pause_callback(query, storage, user_id, hyp_id, lang):
    """Pause a skill."""
    from application.skill_compression.hypothesis_storage import (
        InvalidStatusTransition,
    )

    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    try:
        storage.transition_hypothesis_status(hyp_id, STATUS_PAUSED)
    except InvalidStatusTransition:
        storage.update_hypothesis_status(hyp_id, STATUS_PAUSED)
    name = derive_skill_name(hyp)
    await query.edit_message_text(t("skill.paused", lang, name=name))


async def _handle_resume_callback(query, storage, user_id, hyp_id, lang):
    """Resume a paused skill."""
    from application.skill_compression.hypothesis_storage import (
        InvalidStatusTransition,
    )

    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    # Resume to confirmed (safe default, user can re-earn active).
    # paused -> active is also valid; paused -> confirmed uses force
    # because transition matrix has paused -> active but not paused -> confirmed.
    try:
        storage.transition_hypothesis_status(hyp_id, STATUS_CONFIRMED, force=True)
    except InvalidStatusTransition:
        storage.update_hypothesis_status(hyp_id, STATUS_CONFIRMED)
    name = derive_skill_name(hyp)
    await query.edit_message_text(t("skill.resumed", lang, name=name))


async def _handle_forget_callback(query, storage, user_id, hyp_id, permanent, lang):
    """Forget a skill via callback."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    _execute_forget(storage, hyp, permanent)
    name = derive_skill_name(hyp)

    if permanent:
        text = t("skill.forgotten_permanent", lang, name=name)
    else:
        text = t("skill.forgotten_tombstone", lang, name=name)

    await query.edit_message_text(text)


async def _handle_versions_callback(query, storage, user_id, hyp_id, lang):
    """Show version history via callback."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    version_history = storage.get_version_history(hyp_id)
    if not version_history:
        await query.edit_message_text(t("skill.no_version_history", lang))
        return

    lines = [f"Versionen von '{derive_skill_name(hyp)}':", ""]
    lines.append(f"Aktuell: v{hyp.version} - {hyp.claim}")
    lines.append("")

    for vh in version_history:
        v_num = vh.get("version", "?")
        v_claim = vh.get("claim", "")
        v_reason = vh.get("change_reason", "")
        entry = f"v{v_num}: {v_claim}"
        if v_reason:
            entry += f"\n  Grund: {v_reason}"
        lines.append(entry)

    await query.edit_message_text("\n".join(lines))


async def _handle_undo_callback(query, storage, user_id, hyp_id, lang):
    """Undo last skill application (logs contradiction evidence)."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    # Log contradiction evidence
    now_iso = datetime.now(timezone.utc).isoformat()
    evidence_id = f"ev_{uuid4().hex[:16]}"
    storage.insert_evidence(
        evidence_id=evidence_id,
        hypothesis_id=hyp_id,
        hypothesis_version=hyp.version,
        signal_type="correction",
        signal_strength=1.0,
        created_at=now_iso,
    )

    # Update contradiction count
    storage.update_hypothesis_support(
        hypothesis_id=hyp_id,
        increment_contradict=1,
        last_contradiction_at=now_iso,
    )

    name = derive_skill_name(hyp)
    await query.edit_message_text(t("skill.undo_done", lang, name=name))
    log.info("Skill undo: hyp=%s user=%d", hyp_id, user_id)


async def _handle_explain_callback(query, context, user_id, hyp_id, q_type_str, lang):
    """Show skill explanation via callback."""
    await query.answer()

    explainer = _get_skill_explainer(context)
    if explainer is None:
        await query.edit_message_text(t("skill.system_not_initialized_short", lang))
        return

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await query.edit_message_text(t("skill.system_not_initialized_short", lang))
        return

    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    # Resolve question type
    try:
        question_type = ExplainerQuestionType(q_type_str)
    except ValueError:
        question_type = ExplainerQuestionType.WHAT_RECOGNIZED

    response = explainer.explain(hyp.hypothesis_id, question_type)
    text = f"{response.title}\n{'=' * min(len(response.title), 40)}\n\n{response.explanation}"

    # Truncate if too long for Telegram (4096 char limit)
    if len(text) > 4000:
        text = text[:3997] + "..."

    await query.edit_message_text(text)


# ---------------------------------------------------------------
# Command: /import (Step 7 - Conversation Import)
# ---------------------------------------------------------------

# Import state keys for context.user_data
_IMPORT_PENDING_KEY = "import_pending_folder"


def _get_import_orchestrator(
    context: ContextTypes.DEFAULT_TYPE,
):
    """Get ImportOrchestrator from bot_data.

    Args:
        context: Telegram handler context.

    Returns:
        ImportOrchestrator or None if not initialized.
    """
    return context.application.bot_data.get("import_orchestrator")


@require_whitelist
@require_private_chat
@lcp_aware
async def handle_import_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /import command: import conversations from a folder.

    HC-SC-16: Strictly opt-in, dry-run first, progress display.
    HC-IMPORT-1: All imported hypotheses start as 'suggested'.

    Workflow:
      1. User: /import ~/Documents/ChatGPT-Export/
      2. Bot: shows dry-run preview
      3. User: confirms (via callback button)
      4. Bot: shows progress and result

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    from pathlib import Path as _Path  # noqa: E402 -- deferred to avoid circular

    from application.skill_compression.conversation_import.orchestrator import (  # noqa: E402
        ImportOrchestrator,
    )

    user = update.effective_user
    if not user:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    orchestrator = _get_import_orchestrator(context)
    storage = _get_hypothesis_storage(context)

    # Fallback: create orchestrator from storage if not pre-initialized
    if orchestrator is None and storage is not None:
        orchestrator = ImportOrchestrator(storage)
        orchestrator.init_schema()
        context.application.bot_data["import_orchestrator"] = orchestrator

    if orchestrator is None:
        await update.message.reply_text(t("skill.system_not_initialized", lang))
        return

    args = context.args
    if not args:
        await update.message.reply_text(t("skill.import_usage", lang))
        return

    folder_str = " ".join(args).strip()
    folder_path = _Path(folder_str).expanduser().resolve()

    if not folder_path.exists():
        await update.message.reply_text(
            t("skill.import_folder_not_found", lang, path=folder_path)
        )
        return

    if not folder_path.is_dir():
        await update.message.reply_text(
            t("skill.import_not_a_folder", lang, path=folder_path)
        )
        return

    # Execute dry-run (HC-SC-16)
    try:
        dry_run = orchestrator.dry_run(folder_path)
    except Exception as exc:
        await update.message.reply_text(t("skill.import_scan_error", lang, error=exc))
        return

    if not dry_run.files:
        await update.message.reply_text(t("skill.import_no_files", lang))
        return

    # Build dry-run preview message
    lines = [
        t("skill.import_preview_header", lang),
        "",
    ]
    for fp in dry_run.files[:15]:  # Limit display to 15 files
        size_kb = fp.size_bytes / 1024
        conv_label = t(
            "skill.import_conversations_label", lang, count=fp.conversation_count
        )
        lines.append(f"  {fp.path} ({size_kb:.0f} KB, {conv_label}, {fp.source_type})")
    if len(dry_run.files) > 15:
        lines.append(
            f"  {t('skill.import_more_files', lang, count=len(dry_run.files) - 15)}"
        )

    lines.extend(
        [
            "",
            t(
                "skill.import_total",
                lang,
                conversations=dry_run.total_conversations,
                files=len(dry_run.files),
            ),
            "",
            t("skill.import_privacy_note", lang),
            t(
                "skill.import_duration",
                lang,
                seconds=dry_run.estimated_duration_seconds,
            ),
        ]
    )

    preview_text = "\n".join(lines)

    # Truncate for Telegram
    if len(preview_text) > 3800:
        preview_text = preview_text[:3797] + "..."

    # Store pending import in user_data for callback confirmation
    context.user_data[_IMPORT_PENDING_KEY] = str(folder_path)

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text=t("skill.import_start_btn", lang),
                    callback_data="import_confirm",
                ),
                InlineKeyboardButton(
                    text=t("skill.import_cancel_btn", lang),
                    callback_data="import_cancel",
                ),
            ]
        ]
    )

    await update.message.reply_text(
        text=preview_text,
        reply_markup=keyboard,
    )


@require_whitelist
@require_private_chat
async def handle_import_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle import confirmation/cancellation callbacks.

    Callback patterns:
      import_confirm - Start the actual import
      import_cancel  - Cancel the pending import
      import_delete:<import_id> - Delete imported source (HC-IMPORT-3)

    Args:
        update: Telegram update.
        context: Telegram handler context.
    """
    from pathlib import Path as _Path  # noqa: E402

    query = update.callback_query
    if not query or not query.data:
        return

    data = query.data
    user = query.from_user
    if not user:
        return

    user_id = user.id
    chat_id = query.message.chat_id if query.message else 0
    lang = await _resolve_lang(context, user_id, chat_id)

    if data == "import_cancel":
        await query.answer()
        context.user_data.pop(_IMPORT_PENDING_KEY, None)
        await query.edit_message_text(t("skill.import_cancelled", lang))
        return

    if data == "import_confirm":
        await query.answer()

        folder_str = context.user_data.pop(_IMPORT_PENDING_KEY, None)
        if not folder_str:
            await query.edit_message_text(t("skill.import_no_pending", lang))
            return

        orchestrator = _get_import_orchestrator(context)
        if orchestrator is None:
            await query.edit_message_text(t("skill.system_not_initialized_short", lang))
            return

        folder_path = _Path(folder_str)

        # Progress update via message edit
        progress_msg = await query.edit_message_text(
            t("skill.import_progress", lang, count=0)
        )

        last_update_count = [0]  # Mutable for closure

        def on_progress(done, total, hyps):
            # Only update every 5 files to avoid rate limiting
            if done - last_update_count[0] >= 5 or done == total:
                last_update_count[0] = done

        # Run import
        try:
            result = orchestrator.import_folder(
                folder_path,
                user_id=user.id,
                on_progress=on_progress,
            )
        except Exception as exc:
            await progress_msg.edit_text(t("skill.import_failed", lang, error=exc))
            log.error("Import failed: %s", exc, exc_info=True)
            return

        # Show result
        result_lines = [
            t("skill.import_done_header", lang, duration=result.duration_seconds),
            "",
            f"  {t('skill.import_files_processed', lang, count=result.files_processed)}",
            f"  {t('skill.import_conversations_parsed', lang, count=result.conversations_parsed)}",
            f"  {t('skill.import_hypotheses_created', lang, count=result.hypotheses_created)}",
            f"  {t('skill.import_hypotheses_skipped', lang, count=result.hypotheses_skipped)}",
        ]

        if result.errors:
            result_lines.append(
                f"  {t('skill.import_errors', lang, count=len(result.errors))}"
            )

        result_lines.extend(
            [
                "",
                t("skill.import_suggested_note", lang),
            ]
        )

        # Add delete button (HC-IMPORT-3)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text=t("skill.import_undo_btn", lang),
                        callback_data=f"import_delete:{result.import_id}",
                    )
                ]
            ]
        )

        await progress_msg.edit_text(
            text="\n".join(result_lines),
            reply_markup=keyboard,
        )
        return

    if data.startswith("import_delete:"):
        await query.answer()
        import_id = data.split(":", 1)[1]

        orchestrator = _get_import_orchestrator(context)
        if orchestrator is None:
            await query.edit_message_text(t("skill.system_not_initialized_short", lang))
            return

        deleted = orchestrator.delete_from_source(import_id)
        await query.edit_message_text(t("skill.import_undone", lang, count=deleted))


# ---------------------------------------------------------------
# RISK-3: Ask-Before-Apply confirmation flow
# ---------------------------------------------------------------

# Timeout for pending skill confirmations (seconds)
SKILL_CONFIRM_TIMEOUT_SECONDS: float = 300.0


def get_pending_skill_confirmations(
    context: ContextTypes.DEFAULT_TYPE,
) -> dict:
    """Get or create the pending skill confirmation store in bot_data.

    Structure: {(user_id, chat_id, hyp_id): {
        "skill_match": SkillMatch,
        "original_text": str,
        "original_update_data": dict,
        "timestamp": float,
        "envelope": RequestEnvelope,
    }}

    Keys are composite tuples ``(user_id, chat_id, hypothesis_id)`` so that
    multiple pending confirmations (different chats or different hypotheses)
    can coexist without overwriting each other (C3-SC-01 / Phase-1a fix).

    Args:
        context: Telegram handler context.

    Returns:
        Dict of pending confirmations keyed by (user_id, chat_id, hyp_id).
    """
    store = context.application.bot_data.get("_pending_skill_confirmations")
    if store is None:
        store = {}
        context.application.bot_data["_pending_skill_confirmations"] = store
    return store


def build_skill_confirm_keyboard(
    hypothesis_id: str, lang: str = "en"
) -> InlineKeyboardMarkup:
    """Build the inline keyboard for skill confirmation.

    Buttons: [Yes] [No] [Never again]

    Args:
        hypothesis_id: ID of the hypothesis to confirm.
        lang: Language code for button labels.

    Returns:
        InlineKeyboardMarkup with 3 buttons.
    """
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text=t("skill.confirm_apply_yes", lang),
                    callback_data=f"skill_confirm:yes:{hypothesis_id}",
                ),
                InlineKeyboardButton(
                    text=t("skill.confirm_apply_no", lang),
                    callback_data=f"skill_confirm:no:{hypothesis_id}",
                ),
                InlineKeyboardButton(
                    text=t("skill.confirm_apply_never", lang),
                    callback_data=f"skill_confirm:never:{hypothesis_id}",
                ),
            ]
        ]
    )


async def _handle_skill_confirm_inline(
    query, context, user_id: int, chat_id: int, data: str, lang: str
) -> None:
    """Internal handler for skill_confirm: callbacks (RISK-3: Ask-Before-Apply).

    Called from handle_skill_callback when pattern matches skill_confirm:*.

    After the user clicks a button, this handler:
      - yes: writes "user_confirmed" evidence
      - no: writes "user_declined_once" evidence
      - never: transitions hypothesis to "paused", writes "user_declined_permanent"

    Args:
        query: Callback query object.
        context: Telegram handler context.
        user_id: User ID.
        chat_id: Chat ID.
        data: Callback data string (skill_confirm:<action>:<hyp_id>).
        lang: Language code.
    """
    import time

    # Parse: skill_confirm:<action>:<hypothesis_id>
    parts = data.split(":", 2)
    if len(parts) != 3:
        await query.answer()
        return

    action = parts[1]  # "yes", "no", "never"
    hyp_id = parts[2]

    # C3-SC-01: Use composite key (user_id, chat_id, hyp_id) and get() first.
    # Only pop() after successful validation + action to avoid consuming a
    # pending confirmation that belongs to a different button press.
    pending_store = get_pending_skill_confirmations(context)
    store_key = (user_id, chat_id, hyp_id)
    pending = pending_store.get(store_key)

    if pending is None:
        # Expired or already handled
        await query.answer(text=t("skill.confirm_expired", lang), show_alert=True)
        return

    # R2-SC-03: Ownership validation before processing callback
    skill_match = pending.get("skill_match")
    if skill_match is not None:
        pending_hyp = skill_match.hypothesis
        if hyp_id != pending_hyp.hypothesis_id or pending_hyp.user_id != user_id:
            # Mismatch: do NOT pop, the real pending stays for its own button
            await query.answer(text=t("skill.confirm_expired", lang), show_alert=True)
            return

    # Timeout check: pop on timeout (pending is stale, no value keeping it)
    elapsed = time.time() - pending.get("timestamp", 0)
    if elapsed > SKILL_CONFIRM_TIMEOUT_SECONDS:
        pending_store.pop(store_key, None)
        await query.answer(text=t("skill.confirm_expired", lang), show_alert=True)
        return

    await query.answer()

    # Get services
    chat_service = context.application.bot_data.get("chat_service")
    storage = _get_hypothesis_storage(context)

    if chat_service is None or skill_match is None:
        await query.edit_message_text(t("skill.system_not_initialized_short", lang))
        return

    if action == "yes":
        # Write user_confirmed evidence
        chat_service._write_skill_evidence(
            skill_match, signal_type="user_confirmed", signal_strength=0.5
        )

        # Round-5: Promote hypothesis from 'confirmed' to 'active' so future
        # triggers auto-apply without asking again (user already said yes).
        if storage is not None:
            try:
                storage.transition_hypothesis_status(hyp_id, "active")
                log.info(
                    "Skill promoted to active after user confirmation: hyp=%s",
                    hyp_id,
                )
            except Exception:
                # Fallback: force update if transition matrix rejects
                try:
                    storage.update_hypothesis_status(hyp_id, "active")
                    log.info("Skill force-promoted to active: hyp=%s", hyp_id)
                except Exception:
                    log.debug(
                        "Failed to promote hypothesis %s to active",
                        hyp_id,
                        exc_info=True,
                    )

        # Edit confirmation message to indicate skill is being applied
        await query.edit_message_text(t("skill.confirm_applied", lang))

        # Round-5 CRITICAL: Re-process the original user message with the
        # skill now active. Previously, the flow stopped here and the user
        # never got a skill-powered response. The original text and envelope
        # are stored in the pending confirmation data.
        original_text = pending.get("original_text", "")
        original_envelope = pending.get("envelope")
        if original_text and original_envelope is not None:
            from presentation.handlers import reprocess_after_skill_confirmation

            try:
                await reprocess_after_skill_confirmation(
                    context=context,
                    chat_id=chat_id,
                    user_id=user_id,
                    username=query.from_user.username if query.from_user else None,
                    text=original_text,
                    envelope=original_envelope,
                )
                # Round-5b: Write skill_executed evidence after successful start
                chat_service._write_skill_evidence(
                    skill_match, signal_type="skill_executed", signal_strength=0.5
                )
            except Exception as exc:
                # Round-5b: Write skill_execution_failed evidence on failure
                chat_service._write_skill_evidence(
                    skill_match,
                    signal_type="skill_execution_failed",
                    signal_strength=0.0,
                )
                log.error(
                    "Skill execution failed after confirmation: hyp=%s error=%s",
                    hyp_id,
                    exc,
                    exc_info=True,
                )

        log.info(
            "Skill confirmed by user: hyp=%s user=%d, re-processing message",
            hyp_id,
            user_id,
        )

    elif action == "no":
        # Write user_declined_once evidence
        chat_service._write_skill_evidence(
            skill_match, signal_type="user_declined_once", signal_strength=-0.2
        )
        await query.edit_message_text(t("skill.confirm_declined", lang))
        log.info(
            "Skill declined (once) by user: hyp=%s user=%d",
            hyp_id,
            user_id,
        )

    elif action == "never":
        # Write user_declined_permanent evidence
        chat_service._write_skill_evidence(
            skill_match, signal_type="user_declined_permanent", signal_strength=-0.8
        )
        # Transition hypothesis to paused
        if storage is not None:
            try:
                storage.transition_hypothesis_status(hyp_id, "paused")
            except Exception:
                # If transition fails (e.g. already paused), log but continue
                log.debug(
                    "Failed to pause hypothesis %s after permanent decline",
                    hyp_id,
                    exc_info=True,
                )
        await query.edit_message_text(t("skill.confirm_never", lang))
        log.info(
            "Skill declined permanently by user: hyp=%s user=%d",
            hyp_id,
            user_id,
        )

    # C3-SC-01: Pop pending AFTER successful action so a stale button click
    # on a different confirmation cannot accidentally consume the live one.
    pending_store.pop(store_key, None)

    # Round-5 UX flow:
    # - "yes": Original message IS re-processed with the skill active.
    #   The hypothesis is promoted to 'active' and the streaming pipeline
    #   runs with the skill instruction block in the prompt. The user
    #   sees the skill-powered response immediately after clicking "Ja".
    # - "no": Evidence recorded, next message proceeds normally.
    # - "never": Hypothesis paused, will not trigger again.


# Public alias for the handler (used in tests)
handle_skill_confirm_callback = _handle_skill_confirm_inline
