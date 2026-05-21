"""Skill Chat-Shortcuts: /skills, /skill, /forget, /learn, /explain commands.

Layer 6 (UI): Telegram command handlers for skill management.
Integrates with SkillMatcher (Layer 5) and HypothesisStorage.

Commands:
  /skills      - Show top 10 active skills
  /skill X     - Show details for skill X (by ID or name fragment)
  /forget X    - Delete skill (30-day tombstone)
  /learn       - Save last bot interaction as permanent skill
  /explain X   - Explain a skill decision (8 question types)

HC-SC-7 [BLOCKER]: Tombstones 30 days default, "nie wieder" as permanent.
HC-SC-6 [BLOCKER]: /learn creates decay-immune skills.
HC-SC-13 [BLOCKER]: No-Model-Secret Rule for /learn (allowlist filter).
HC-SC-18 [BLOCKER]: 8 Explainer question types via /explain.

Architecture guard: Presentation layer uses only Application services.
No direct infra-layer or raw domain-layer access (except domain types).
"""

from __future__ import annotations

import logging
import re
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
from application.skill_compression.skill_explainer import (
    ExplainerQuestionType,
    SkillExplainer,
)
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
# Secret filter for /learn (HC-SC-13)
# ---------------------------------------------------------------

# Allowlist: only these field patterns are acceptable in a /learn skill.
# Anything matching SECRET_PATTERNS is rejected.
SECRET_PATTERNS: list[re.Pattern] = [
    # API tokens (sk-, ghp_, gho_, xox, bearer, etc.)
    re.compile(
        r"(?:sk-|ghp_|gho_|xox[bpas]-|bearer\s+|token[:\s=]+)\S{8,}",
        re.IGNORECASE,
    ),
    # Currency amounts with digits (prices)
    re.compile(
        r"(?:[$€£¥])\s*\d+[.,]?\d*|\d+[.,]?\d*\s*(?:EUR|USD|GBP|CHF)",
        re.IGNORECASE,
    ),
    # Email addresses
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # Phone numbers (international format)
    re.compile(r"\+?\d[\d\s\-()]{8,}\d"),
    # IBANs
    re.compile(r"[A-Z]{2}\d{2}\s?[\dA-Z]{4,}"),
    # Password-adjacent content
    re.compile(
        r"(?:passwor[td]|kennwort|password|pwd|secret)[:\s=]+\S+",
        re.IGNORECASE,
    ),
    # Long hex/base64 strings (likely tokens/keys)
    re.compile(r"[a-fA-F0-9]{32,}"),
    re.compile(r"[A-Za-z0-9+/=]{40,}"),
]


def check_secret_content(text: str) -> Optional[str]:
    """Check if text contains secret/sensitive content (HC-SC-13).

    No-Model-Secret Rule: Skills must not store API tokens, prices,
    passwords, private identifiers, or raw data.

    Args:
        text: Text to check.

    Returns:
        Description of the detected secret type, or None if clean.
    """
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            # Map pattern to user-friendly description
            pattern_str = pattern.pattern
            if "sk-" in pattern_str or "token" in pattern_str:
                return "API-Token oder Secret"
            if "$" in pattern_str or "EUR" in pattern_str:
                return "Preisangabe"
            if "@" in pattern_str:
                return "E-Mail-Adresse"
            if r"\+" in pattern_str or "phone" in pattern_str.lower():
                return "Telefonnummer"
            if "IBAN" in pattern_str.upper() or r"[A-Z]{2}\d{2}" in pattern_str:
                return "IBAN/Kontonummer"
            if "passwor" in pattern_str:
                return "Passwort"
            if "hex" in pattern_str.lower() or "32," in pattern_str:
                return "Langer Token/Key"
            return "Sensible Daten"
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
    "nötig": ExplainerQuestionType.WHAT_NEEDED,
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

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(
            "Skill-System noch nicht initialisiert."  # i18n: ok
        )
        return

    # Load all visible hypotheses for this user
    hypotheses: list[Hypothesis] = []
    for status in PROFILE_VISIBLE_STATUSES:
        hypotheses.extend(
            storage.get_hypotheses_by_user(user_id, status=status, limit=50)
        )

    # Render profile text
    profile_text = render_profile(hypotheses)

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

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(
            "Skill-System noch nicht initialisiert."  # i18n: ok
        )
        return

    # Parse argument
    args = context.args
    if not args:
        await update.message.reply_text(  # i18n: ok
            "Bitte einen Skill-Namen oder ID angeben.\nBeispiel: /skill Drehkonzepte"
        )
        return

    query = " ".join(args).strip()

    # Try exact ID match first
    hyp = storage.get_hypothesis(query)
    if hyp is not None and hyp.user_id == user_id:
        version_history = storage.get_version_history(query)
        detail_text = render_skill_detail_text(hyp, version_history)
        keyboard = build_skill_actions_keyboard(hyp)
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
        await update.message.reply_text(  # i18n: ok
            f"Kein Skill gefunden für '{query}'.\nNutze /skills für eine Übersicht."
        )
        return

    if len(matches) == 1:
        hyp = matches[0]
        version_history = storage.get_version_history(hyp.hypothesis_id)
        detail_text = render_skill_detail_text(hyp, version_history)
        keyboard = build_skill_actions_keyboard(hyp)
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

    await update.message.reply_text(  # i18n: ok
        text=f"Mehrere Skills gefunden für '{query}':",
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

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(
            "Skill-System noch nicht initialisiert."  # i18n: ok
        )
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
            await update.message.reply_text("Keine Skills vorhanden.")  # i18n: ok
            return

        buttons: list[list[InlineKeyboardButton]] = []
        for h in all_hyps[:10]:
            name = derive_skill_name(h)
            buttons.append(
                [
                    InlineKeyboardButton(
                        text=f"Vergessen: {name}",  # i18n: ok
                        callback_data=f"skill_forget:{h.hypothesis_id}",
                    )
                ]
            )

        await update.message.reply_text(  # i18n: ok
            text="Welchen Skill möchtest du vergessen?",
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
                f"Kein Skill gefunden für '{query}'."  # i18n: ok
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
            await update.message.reply_text(  # i18n: ok
                text=f"Mehrere Skills gefunden für '{query}':",
                reply_markup=InlineKeyboardMarkup(buttons),
            )
            return

        hyp = matches[0]

    # Execute forget
    _execute_forget(storage, hyp, permanent)

    name = derive_skill_name(hyp)
    if permanent:
        await update.message.reply_text(  # i18n: ok
            f"Skill '{name}' permanent vergessen.\n"
            "Dieses Muster wird nie wieder gelernt."
        )
    else:
        await update.message.reply_text(  # i18n: ok
            f"Skill '{name}' vergessen (30-Tage Tombstone).\n"
            "Nach 30 Tagen kann das Muster erneut gelernt werden."
        )


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

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(
            "Skill-System noch nicht initialisiert."  # i18n: ok
        )
        return

    # Get the content to learn: either from args or from reply
    args = context.args
    reply = update.message.reply_to_message

    if args:
        skill_text = " ".join(args).strip()
    elif reply and reply.text:
        skill_text = reply.text.strip()
    else:
        await update.message.reply_text(  # i18n: ok
            "Bitte beschreibe den Skill oder antworte auf eine Bot-Nachricht.\n"
            "Beispiel: /learn Verwende immer Bulletpoints in Zusammenfassungen"
        )
        return

    if not skill_text:
        await update.message.reply_text(  # i18n: ok
            "Leerer Skill-Text. Bitte beschreibe was ich mir merken soll."
        )
        return

    # HC-SC-13: No-Model-Secret check
    secret_type = check_secret_content(skill_text)
    if secret_type is not None:
        await update.message.reply_text(  # i18n: ok
            f"Dieser Skill kann nicht gespeichert werden: "
            f"enthält möglicherweise {secret_type}.\n"
            "Skills dürfen keine Passwörter, API-Keys, "
            "Preise oder persönliche Daten enthalten."
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
        await update.message.reply_text(  # i18n: ok
            "Du hast bereits 50 aktive Skills.\n"
            "Nutze /forget um Platz zu schaffen, oder /skills für eine Übersicht."
        )
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
    await update.message.reply_text(  # i18n: ok
        f"Skill gespeichert: '{name}'\n"
        "Status: Bestätigt (wird bei passender Anfrage angewendet).\n"
        "Dieser Skill ist decay-immun und bleibt bis du ihn vergisst."
    )

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
      5/nötig   - Was wäre nötig?
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

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await update.message.reply_text(
            "Skill-System noch nicht initialisiert."  # i18n: ok
        )
        return

    explainer = _get_skill_explainer(context)
    if explainer is None:
        await update.message.reply_text(
            "Skill-System noch nicht initialisiert."  # i18n: ok
        )
        return

    args = context.args
    if not args:
        # Show help with all question types
        types_list = explainer.list_question_types()
        lines = [
            "Nutze /explain <Skill> [Frage] um einen Skill zu erklären.",  # i18n: ok
            "",
            "Fragetypen:",
        ]
        for i, (_, desc) in enumerate(types_list, 1):
            lines.append(f"  {i}. {desc}")
        lines.append("")
        lines.append("Beispiel: /explain Drehkonzepte 3")  # i18n: ok
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
        await update.message.reply_text(
            "Bitte einen Skill-Namen angeben.\n"  # i18n: ok
            "Beispiel: /explain Drehkonzepte"
        )
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
        await update.message.reply_text(
            f"Kein Skill gefunden für '{query}'.\n"  # i18n: ok
            "Nutze /skills für eine Übersicht."
        )
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
        text=f"Mehrere Skills gefunden für '{query}':",  # i18n: ok
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

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await query.answer(
            text="Skill-System nicht initialisiert.",  # i18n: ok
            show_alert=True,
        )
        return

    # Parse callback data
    if data.startswith("skill_detail:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_detail_callback(query, storage, user_id, hyp_id)

    elif data.startswith("skill_pause:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_pause_callback(query, storage, user_id, hyp_id)

    elif data.startswith("skill_resume:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_resume_callback(query, storage, user_id, hyp_id)

    elif data.startswith("skill_forget:perm:"):
        hyp_id = data.split(":", 2)[2]
        await _handle_forget_callback(query, storage, user_id, hyp_id, permanent=True)

    elif data.startswith("skill_forget:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_forget_callback(query, storage, user_id, hyp_id, permanent=False)

    elif data.startswith("skill_versions:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_versions_callback(query, storage, user_id, hyp_id)

    elif data.startswith("skill_undo:"):
        hyp_id = data.split(":", 1)[1]
        await _handle_undo_callback(query, storage, user_id, hyp_id)

    elif data.startswith("skill_explain:"):
        # Explainer callback: skill_explain:<question_type>:<hyp_id>
        parts = data.split(":", 2)
        if len(parts) == 3:
            q_type_str = parts[1]
            hyp_id = parts[2]
            await _handle_explain_callback(query, context, user_id, hyp_id, q_type_str)
        elif len(parts) == 2:
            hyp_id = parts[1]
            await _handle_explain_callback(
                query, context, user_id, hyp_id, "what_recognized"
            )

    else:
        await query.answer()


async def _handle_detail_callback(query, storage, user_id, hyp_id):
    """Show skill detail via callback."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text("Skill nicht gefunden.")  # i18n: ok
        return

    version_history = storage.get_version_history(hyp_id)
    detail_text = render_skill_detail_text(hyp, version_history)
    keyboard = build_skill_actions_keyboard(hyp)
    await query.edit_message_text(text=detail_text, reply_markup=keyboard)


async def _handle_pause_callback(query, storage, user_id, hyp_id):
    """Pause a skill."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text("Skill nicht gefunden.")  # i18n: ok
        return

    storage.update_hypothesis_status(hyp_id, STATUS_PAUSED)
    name = derive_skill_name(hyp)
    await query.edit_message_text(  # i18n: ok
        f"Skill '{name}' pausiert.\n"
        "Wird nicht mehr angewendet, bleibt aber gespeichert."
    )


async def _handle_resume_callback(query, storage, user_id, hyp_id):
    """Resume a paused skill."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text("Skill nicht gefunden.")  # i18n: ok
        return

    # Resume to confirmed (safe default, user can re-earn active)
    storage.update_hypothesis_status(hyp_id, STATUS_CONFIRMED)
    name = derive_skill_name(hyp)
    await query.edit_message_text(  # i18n: ok
        f"Skill '{name}' wieder aktiv (bestätigt).\n"
        "Wird bei passender Anfrage mit Rückfrage angewendet."
    )


async def _handle_forget_callback(query, storage, user_id, hyp_id, permanent):
    """Forget a skill via callback."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text("Skill nicht gefunden.")  # i18n: ok
        return

    _execute_forget(storage, hyp, permanent)
    name = derive_skill_name(hyp)

    if permanent:
        text = f"Skill '{name}' permanent vergessen."
    else:
        text = f"Skill '{name}' vergessen (30-Tage Tombstone)."

    await query.edit_message_text(text)


async def _handle_versions_callback(query, storage, user_id, hyp_id):
    """Show version history via callback."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text("Skill nicht gefunden.")  # i18n: ok
        return

    version_history = storage.get_version_history(hyp_id)
    if not version_history:
        await query.edit_message_text("Keine Versionshistorie vorhanden.")  # i18n: ok
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


async def _handle_undo_callback(query, storage, user_id, hyp_id):
    """Undo last skill application (logs contradiction evidence)."""
    await query.answer()
    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text("Skill nicht gefunden.")  # i18n: ok
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
    await query.edit_message_text(  # i18n: ok
        f"Skill-Anwendung '{name}' rückgängig gemacht.\nWiderspruch wurde notiert."
    )
    log.info("Skill undo: hyp=%s user=%d", hyp_id, user_id)


async def _handle_explain_callback(query, context, user_id, hyp_id, q_type_str):
    """Show skill explanation via callback."""
    await query.answer()

    explainer = _get_skill_explainer(context)
    if explainer is None:
        await query.edit_message_text(
            "Skill-System nicht initialisiert."  # i18n: ok
        )
        return

    storage = _get_hypothesis_storage(context)
    if storage is None:
        await query.edit_message_text(
            "Skill-System nicht initialisiert."  # i18n: ok
        )
        return

    hyp = storage.get_hypothesis(hyp_id)
    if hyp is None or hyp.user_id != user_id:
        await query.edit_message_text("Skill nicht gefunden.")  # i18n: ok
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
