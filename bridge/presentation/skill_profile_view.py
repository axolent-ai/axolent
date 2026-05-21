"""Skill Profile View: renders user skills as compact Markdown for Telegram.

Layer 6 (UI): Notiz-style bullet-point format (HC-UI-1).
Telegram max message length = 4096 chars; views stay within this limit.

The profile view is the user-facing representation of learned skills.
It uses neutral, informative language (no emotional/"I miss you" wording).

Architecture guard: This module imports ONLY from application.skill_compression
(HypothesisStorage, Hypothesis) and presentation utils. It does NOT import
any LCP, infrastructure, or lower-layer modules directly.

No external dependencies beyond python-telegram-bot (for InlineKeyboardMarkup).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
)
from application.skill_compression.pattern_judge import (
    STATUS_ACTIVE,
    STATUS_CONFIRMED,
    STATUS_PAUSED,
)
from application.skill_compression.skill_formatting import (
    derive_skill_name,
    format_skill_indicator,  # noqa: F401 (re-export for backward compat)
)
from i18n.domain.i18n import t

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

# Max skills shown in /skills overview (IC-UI-2)
MAX_PROFILE_SKILLS: int = 10

# Telegram message length limit
TELEGRAM_MAX_CHARS: int = 4096

# Statuses that are visible in the profile
PROFILE_VISIBLE_STATUSES: frozenset[str] = frozenset(
    {STATUS_CONFIRMED, STATUS_ACTIVE, STATUS_PAUSED}
)


# ---------------------------------------------------------------
# Skill name derivation: re-exported from application.skill_compression.skill_formatting
# ---------------------------------------------------------------


def _format_date(iso_str: Optional[str]) -> str:
    """Format ISO date string to compact DD.MM. format.

    Args:
        iso_str: ISO-8601 timestamp string.

    Returns:
        Formatted date string or empty string.
    """
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d.%m.")
    except (ValueError, TypeError):
        return ""


def _escape_markdown(text: str) -> str:
    """Escape special Markdown characters for Telegram.

    Args:
        text: Raw text.

    Returns:
        Escaped text safe for Telegram Markdown.
    """
    # Telegram MarkdownV2 special chars
    special = r"_*[]()~`>#+-=|{}.!"
    escaped = []
    for ch in text:
        if ch in special:
            escaped.append(f"\\{ch}")
        else:
            escaped.append(ch)
    return "".join(escaped)


# ---------------------------------------------------------------
# Profile rendering
# ---------------------------------------------------------------


def render_skill_line(hypothesis: Hypothesis, lang: str = "de") -> str:
    """Render a single skill as a bullet-point line.

    Format (HC-UI-1, Notiz-style):
        * Drehkonzepte (v2): 45s Brand Awareness
          [Status: aktiv]

    Args:
        hypothesis: The hypothesis to render.
        lang: Language code for i18n.

    Returns:
        Formatted bullet-point string (1-2 lines).
    """
    name = derive_skill_name(hypothesis)
    version_tag = f" (v{hypothesis.version})" if hypothesis.version > 1 else ""

    # Status indicator
    status_map = {
        STATUS_ACTIVE: t("skill.status_active", lang),
        STATUS_CONFIRMED: t("skill.status_confirmed", lang),
        STATUS_PAUSED: t("skill.status_paused", lang),
    }
    status_label = status_map.get(hypothesis.status, hypothesis.status)

    # Build the line
    line = f"* {name}{version_tag}"
    if hypothesis.status == STATUS_PAUSED:
        line += f"  [{status_label}]"

    return line


def render_skill_detail_text(
    hypothesis: Hypothesis,
    version_history: list[dict],
    lang: str = "de",
) -> str:
    """Render detailed view of a single skill with version history.

    Shows: name, claim, status, version info, evidence counts.

    Args:
        hypothesis: The hypothesis to display.
        version_history: Archived versions from storage.
        lang: Language code for i18n.

    Returns:
        Formatted detail string.
    """
    name = derive_skill_name(hypothesis)
    lines: list[str] = []

    # Header
    lines.append(f"Skill: {name}")
    lines.append("─" * min(len(f"Skill: {name}"), 30))
    lines.append("")

    # Current version
    version_tag = f"v{hypothesis.version}" if hypothesis.version > 1 else "v1"
    lines.append(t("skill.detail_version", lang, version=version_tag))
    lines.append(t("skill.detail_description", lang, claim=hypothesis.claim))

    # Status
    status_map = {
        STATUS_ACTIVE: t("skill.detail_status_active", lang),
        STATUS_CONFIRMED: t("skill.detail_status_confirmed", lang),
        STATUS_PAUSED: t("skill.detail_status_paused", lang),
    }
    lines.append(f"Status: {status_map.get(hypothesis.status, hypothesis.status)}")

    # Evidence
    lines.append(
        t(
            "skill.detail_evidence",
            lang,
            support=hypothesis.support_count,
            contradict=hypothesis.contradict_count,
        )
    )

    # Type
    type_map = {
        "preference": t("skill.type_preference", lang),
        "negative": t("skill.type_negative", lang),
        "request": t("skill.type_request", lang),
    }
    type_label = type_map.get(hypothesis.type, hypothesis.type)
    lines.append(f"Typ: {type_label}")

    # Scope
    scope_parts: list[str] = []
    if hypothesis.scope.project:
        scope_parts.append(f"Projekt: {hypothesis.scope.project}")
    if hypothesis.scope.client:
        scope_parts.append(f"Kunde: {hypothesis.scope.client}")
    if scope_parts:
        lines.append(f"Geltungsbereich: {', '.join(scope_parts)}")
    else:
        lines.append(f"Geltungsbereich: {t('skill.scope_global', lang)}")

    # Source
    source_map = {
        "live_chat": t("skill.source_live_chat", lang),
        "learn_command": t("skill.source_learn_command", lang),
        "manual": t("skill.source_manual", lang),
        "import": t("skill.source_import", lang),
    }
    lines.append(
        f"Quelle: {source_map.get(hypothesis.source_type, hypothesis.source_type)}"
    )

    # Dates
    created = _format_date(hypothesis.created_at)
    last_applied = _format_date(hypothesis.last_applied)
    if created:
        lines.append(f"Erstellt: {created}")
    if last_applied:
        lines.append(f"Zuletzt angewendet: {last_applied}")

    # Decay immunity
    if hypothesis.decay_immune:
        lines.append(t("skill.decay_immune", lang))

    # Version history
    if version_history:
        lines.append("")
        lines.append("Versionshistorie:")
        for vh in version_history:
            v_num = vh.get("version", "?")
            v_claim = vh.get("claim", "")
            v_reason = vh.get("change_reason", "")
            v_date = _format_date(vh.get("created_at"))
            entry = f"  v{v_num}: {v_claim}"
            if v_reason:
                entry += f" (Grund: {v_reason})"
            if v_date:
                entry += f" [{v_date}]"
            lines.append(entry)

    return "\n".join(lines)


def render_profile(
    hypotheses: list[Hypothesis],
    max_skills: int = MAX_PROFILE_SKILLS,
    lang: str = "de",
) -> str:
    """Render the complete skill profile as compact Markdown.

    Shows top N active skills sorted by last_applied DESC.
    Fits within Telegram's 4096 char limit.

    Args:
        hypotheses: All user hypotheses (pre-filtered by caller).
        max_skills: Maximum skills to show.
        lang: Language code for i18n.

    Returns:
        Formatted profile string.
    """
    profile_header_text = t("skill.profile_header", lang)
    header = f"{profile_header_text}\n───────────"

    # Filter to profile-visible statuses
    visible = [h for h in hypotheses if h.status in PROFILE_VISIBLE_STATUSES]

    if not visible:
        return f"{header}\n\n{t('skill.profile_empty', lang)}"

    # Sort by last_applied DESC (None = bottom)
    visible.sort(
        key=lambda h: h.last_applied or "0000-00-00",
        reverse=True,
    )

    # Limit to max
    shown = visible[:max_skills]

    lines: list[str] = [header, ""]

    for h in shown:
        lines.append(render_skill_line(h, lang=lang))

    # Footer
    total = len(visible)
    if total > max_skills:
        lines.append("")
        lines.append(t("skill.profile_more", lang, count=total - max_skills))

    profile_text = "\n".join(lines)

    # Truncate if exceeding Telegram limit (safety net)
    if len(profile_text) > TELEGRAM_MAX_CHARS - 100:
        profile_text = profile_text[: TELEGRAM_MAX_CHARS - 150] + "\n\n[...]"

    return profile_text


# ---------------------------------------------------------------
# Inline keyboard builders
# ---------------------------------------------------------------


def build_skill_actions_keyboard(
    hypothesis: Hypothesis,
    lang: str = "de",
) -> InlineKeyboardMarkup:
    """Build inline keyboard with skill action buttons.

    Buttons: [bearbeiten] [pausieren/fortsetzen] [vergessen] [Versionen?]

    Args:
        hypothesis: The hypothesis to build actions for.
        lang: Language code for i18n.

    Returns:
        InlineKeyboardMarkup with action buttons.
    """
    hyp_id = hypothesis.hypothesis_id
    buttons: list[list[InlineKeyboardButton]] = []

    # Row 1: edit + pause/resume
    row1: list[InlineKeyboardButton] = []

    if hypothesis.status == STATUS_PAUSED:
        row1.append(
            InlineKeyboardButton(
                text=t("skill.btn_resume", lang),
                callback_data=f"skill_resume:{hyp_id}",
            )
        )
    else:
        row1.append(
            InlineKeyboardButton(
                text=t("skill.btn_pause", lang),
                callback_data=f"skill_pause:{hyp_id}",
            )
        )

    row1.append(
        InlineKeyboardButton(
            text=t("skill.btn_forget", lang),
            callback_data=f"skill_forget:{hyp_id}",
        )
    )
    buttons.append(row1)

    # Row 2: versions (only if version > 1)
    if hypothesis.version > 1:
        buttons.append(
            [
                InlineKeyboardButton(
                    text=t("skill.btn_versions", lang),
                    callback_data=f"skill_versions:{hyp_id}",
                )
            ]
        )

    return InlineKeyboardMarkup(buttons)


def build_profile_list_keyboard(
    hypotheses: list[Hypothesis],
    max_skills: int = MAX_PROFILE_SKILLS,
) -> InlineKeyboardMarkup:
    """Build inline keyboard for skill list (/skills command).

    Each skill gets a "Details" button.

    Args:
        hypotheses: Visible hypotheses, pre-sorted.
        max_skills: Maximum to show.

    Returns:
        InlineKeyboardMarkup with detail buttons.
    """
    visible = [h for h in hypotheses if h.status in PROFILE_VISIBLE_STATUSES]
    visible.sort(key=lambda h: h.last_applied or "0000-00-00", reverse=True)
    shown = visible[:max_skills]

    buttons: list[list[InlineKeyboardButton]] = []
    for h in shown:
        name = derive_skill_name(h)
        version_tag = f" (v{h.version})" if h.version > 1 else ""
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{name}{version_tag}",
                    callback_data=f"skill_detail:{h.hypothesis_id}",
                )
            ]
        )

    return InlineKeyboardMarkup(buttons)


# ---------------------------------------------------------------
# Skill application indicator (HC-UI-2)
# ---------------------------------------------------------------


# format_skill_indicator: re-exported from application.skill_compression.skill_formatting


def build_indicator_keyboard(
    hypothesis_id: str,
    lang: str = "de",
) -> InlineKeyboardMarkup:
    """Build inline keyboard for skill indicator buttons.

    Buttons: [/undo] [Warum?]

    Args:
        hypothesis_id: The applied hypothesis ID.
        lang: Language code for i18n.

    Returns:
        InlineKeyboardMarkup with undo and explain buttons.
    """
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text=t("skill.btn_undo", lang),
                    callback_data=f"skill_undo:{hypothesis_id}",
                ),
                InlineKeyboardButton(
                    text=t("skill.btn_why", lang),
                    callback_data=f"skill_explain:{hypothesis_id}",
                ),
            ]
        ]
    )
