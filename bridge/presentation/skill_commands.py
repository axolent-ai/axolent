"""Skill Chat-Shortcuts: /skills, /skill, /forget, /learn, /explain, /import.

Layer 6 (UI): Telegram command handlers for skill management.
Integrates with SkillMatcher (Layer 5) and HypothesisStorage.

Commands:
  /skills      - Show top 10 active skills
  /skill X     - Show details for skill X (by ID or name fragment)
  /forget X    - Delete skill (30-day tombstone)
  /learn       - Save last bot interaction as permanent skill
  /explain X   - Explain a skill decision (8 question types)
  /import PATH - Import conversations from a folder (dry-run first)

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
from telegram.ext import ContextTypes

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
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
from presentation.decorators import require_private_chat, require_whitelist
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


@require_whitelist
@require_private_chat
async def handle_learn_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /learn command: create a skill from last interaction.

    Creates a new hypothesis with:
      - status = confirmed (immediately usable)
      - decay_immune = True (HC-SC-6)
      - source_type = "learn_command"

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

    # HC-SC-13: No-Model-Secret check
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

    # HC-SC-8: Check max active skills
    active_count = storage.count_active_hypotheses(user_id)
    confirmed_count = len(
        storage.get_hypotheses_by_user(user_id, status=STATUS_CONFIRMED, limit=51)
    )
    total_skills = active_count + confirmed_count
    if total_skills >= 50:
        await update.message.reply_text(t("skill.learn_max_reached", lang))
        return

    # Create the hypothesis
    now_iso = datetime.now(timezone.utc).isoformat()
    hyp_id = f"hyp_{uuid4().hex[:16]}"

    hypothesis = Hypothesis(
        hypothesis_id=hyp_id,
        user_id=user_id,
        type="preference",
        scope=HypothesisScope(),
        claim=skill_text,
        status=STATUS_CONFIRMED,
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
        approval_state="approved",
    )

    storage.insert_hypothesis(hypothesis)

    name = derive_skill_name(hypothesis)
    await update.message.reply_text(t("skill.learn_saved", lang, name=name))

    log.info(
        "Skill learned via /learn: hyp=%s user=%d claim='%s'",
        hyp_id,
        user_id,
        skill_text[:50],
    )


# ---------------------------------------------------------------
# Command: /explain X
# ---------------------------------------------------------------


@require_whitelist
@require_private_chat
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
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    storage.update_hypothesis_status(hyp_id, STATUS_PAUSED)
    name = derive_skill_name(hyp)
    await query.edit_message_text(t("skill.paused", lang, name=name))


async def _handle_resume_callback(query, storage, user_id, hyp_id, lang):
    """Resume a paused skill."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text(t("skill.not_found", lang))
        return

    # Resume to confirmed (safe default, user can re-earn active)
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
