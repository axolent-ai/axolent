"""Tests for Skill Profile View (Step 5, Layer 6 UI).

Covers:
  - render_profile with 0/1/5/10/50 skills
  - render_skill_detail_text with version history
  - derive_skill_name with various claim lengths
  - format_skill_indicator appends correctly
  - Inline keyboard builders produce correct structure
  - Telegram max length respected
"""

from __future__ import annotations

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.pattern_judge import (
    STATUS_ACTIVE,
    STATUS_PAUSED,
)
from i18n.domain.i18n import t
from presentation.skill_profile_view import (
    TELEGRAM_MAX_CHARS,
    build_indicator_keyboard,
    build_profile_list_keyboard,
    build_skill_actions_keyboard,
    derive_skill_name,
    format_skill_indicator,
    render_profile,
    render_skill_detail_text,
    render_skill_line,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_hypothesis(
    *,
    hypothesis_id: str = "hyp-001",
    user_id: int = 42,
    status: str = STATUS_ACTIVE,
    claim: str = "User prefers bullet points",
    h_type: str = "preference",
    version: int = 1,
    last_applied: str = "2026-05-20T12:00:00+00:00",
    support_count: int = 5,
    contradict_count: int = 0,
    source_type: str = "live_chat",
    decay_immune: bool = False,
    project: str = "",
    client: str = "",
) -> Hypothesis:
    """Create a test hypothesis."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        user_id=user_id,
        type=h_type,
        scope=HypothesisScope(project=project, client=client),
        claim=claim,
        status=status,
        version=version,
        elo_rating=1600.0,
        support_count=support_count,
        contradict_count=contradict_count,
        source_type=source_type,
        decay_immune=decay_immune,
        created_at="2026-05-20T10:00:00+00:00",
        last_applied=last_applied,
        last_seen="2026-05-20T12:00:00+00:00",
    )


# ---------------------------------------------------------------
# Tests: derive_skill_name
# ---------------------------------------------------------------


class TestDeriveSkillName:
    """Tests for skill name derivation (IC-UI-3)."""

    def test_short_claim(self) -> None:
        """Short claim used as-is."""
        hyp = _make_hypothesis(claim="Bullet Points")
        assert derive_skill_name(hyp) == "Bullet Points"

    def test_claim_with_period(self) -> None:
        """Claim with period: use text before period."""
        hyp = _make_hypothesis(claim="Bullet Points. Always in Markdown.")
        assert derive_skill_name(hyp) == "Bullet Points"

    def test_long_claim_truncated(self) -> None:
        """Long claim should be truncated at word boundary."""
        hyp = _make_hypothesis(
            claim="This is a very long claim that exceeds forty characters limit"
        )
        name = derive_skill_name(hyp)
        assert len(name) <= 43  # 40 + "..."
        assert name.endswith("...")

    def test_empty_claim(self) -> None:
        """Empty claim uses type fallback."""
        hyp = _make_hypothesis(claim="")
        assert "unnamed" in derive_skill_name(hyp)

    def test_claim_exactly_40_chars(self) -> None:
        """Claim at exactly 40 chars should not be truncated."""
        claim = "A" * 40
        hyp = _make_hypothesis(claim=claim)
        assert derive_skill_name(hyp) == claim


# ---------------------------------------------------------------
# Tests: render_profile
# ---------------------------------------------------------------


class TestRenderProfile:
    """Tests for profile rendering."""

    def test_empty_skills(self) -> None:
        """Profile with 0 skills shows empty message."""
        result = render_profile([])
        profile_header = t("skill.profile_header", "de")
        assert profile_header in result
        assert t("skill.profile_empty", "de").split("\n")[0] in result

    def test_single_skill(self) -> None:
        """Profile with 1 skill renders correctly."""
        hyp = _make_hypothesis()
        result = render_profile([hyp])
        profile_header = t("skill.profile_header", "de")
        assert profile_header in result
        assert "User prefers bullet points" in result

    def test_five_skills(self) -> None:
        """Profile with 5 skills shows all."""
        hyps = [
            _make_hypothesis(
                hypothesis_id=f"hyp-{i:03d}",
                claim=f"Skill number {i}",
            )
            for i in range(5)
        ]
        result = render_profile(hyps)
        for i in range(5):
            assert f"Skill number {i}" in result

    def test_ten_skills_is_max_default(self) -> None:
        """Default max is 10 skills."""
        hyps = [
            _make_hypothesis(
                hypothesis_id=f"hyp-{i:03d}",
                claim=f"Skill {i}",
                last_applied=f"2026-05-{20 - i:02d}T12:00:00+00:00",
            )
            for i in range(15)
        ]
        result = render_profile(hyps)
        assert "und 5 weitere" in result

    def test_fifty_skills_shows_overflow(self) -> None:
        """50 skills shows overflow notice."""
        hyps = [
            _make_hypothesis(
                hypothesis_id=f"hyp-{i:03d}",
                claim=f"Skill {i}",
            )
            for i in range(50)
        ]
        result = render_profile(hyps)
        assert "und 40 weitere" in result

    def test_candidate_not_shown(self) -> None:
        """Candidate status should not appear in profile."""
        hyp = _make_hypothesis(status="candidate")
        result = render_profile([hyp])
        assert "Noch keine Skills" in result

    def test_paused_shown(self) -> None:
        """Paused skills should appear in profile."""
        hyp = _make_hypothesis(status=STATUS_PAUSED)
        result = render_profile([hyp])
        assert "pausiert" in result

    def test_sorted_by_last_applied(self) -> None:
        """Skills should be sorted by last_applied DESC."""
        hyp_old = _make_hypothesis(
            hypothesis_id="hyp-old",
            claim="Old skill",
            last_applied="2026-01-01T12:00:00+00:00",
        )
        hyp_new = _make_hypothesis(
            hypothesis_id="hyp-new",
            claim="New skill",
            last_applied="2026-05-20T12:00:00+00:00",
        )
        result = render_profile([hyp_old, hyp_new])
        # New should appear before old
        pos_new = result.index("New skill")
        pos_old = result.index("Old skill")
        assert pos_new < pos_old

    def test_telegram_char_limit(self) -> None:
        """Profile must not exceed Telegram max chars."""
        hyps = [
            _make_hypothesis(
                hypothesis_id=f"hyp-{i:03d}",
                claim=f"Skill {i} with a somewhat longer description text",
            )
            for i in range(100)
        ]
        result = render_profile(hyps, max_skills=100)
        assert len(result) <= TELEGRAM_MAX_CHARS


# ---------------------------------------------------------------
# Tests: render_skill_detail_text
# ---------------------------------------------------------------


class TestRenderSkillDetailText:
    """Tests for detailed skill view."""

    def test_basic_detail(self) -> None:
        """Basic detail view shows essential info."""
        hyp = _make_hypothesis()
        detail = render_skill_detail_text(hyp, [])
        assert "User prefers bullet points" in detail
        assert "Präferenz" in detail
        assert "Belege" in detail

    def test_detail_with_version_history(self) -> None:
        """Detail with version history shows version entries."""
        hyp = _make_hypothesis(version=2)
        history = [
            {
                "version": 1,
                "claim": "Old claim",
                "change_reason": "User corrected",
                "elo_rating_at_save": 1700.0,
                "created_at": "2026-05-01T10:00:00+00:00",
            }
        ]
        detail = render_skill_detail_text(hyp, history)
        assert "v1" in detail
        assert "Old claim" in detail
        assert "User corrected" in detail

    def test_detail_shows_scope(self) -> None:
        """Detail shows scope when set."""
        hyp = _make_hypothesis(project="ads", client="acme")
        detail = render_skill_detail_text(hyp, [])
        assert "Projekt: ads" in detail
        assert "Kunde: acme" in detail

    def test_detail_global_scope(self) -> None:
        """Detail shows 'Global' for empty scope."""
        hyp = _make_hypothesis()
        detail = render_skill_detail_text(hyp, [])
        assert "Global" in detail

    def test_detail_decay_immune(self) -> None:
        """Detail shows decay immunity."""
        hyp = _make_hypothesis(decay_immune=True, source_type="learn_command")
        detail = render_skill_detail_text(hyp, [])
        assert "Immun" in detail


# ---------------------------------------------------------------
# Tests: render_skill_line
# ---------------------------------------------------------------


class TestRenderSkillLine:
    """Tests for single skill line rendering."""

    def test_basic_line(self) -> None:
        """Basic line format."""
        hyp = _make_hypothesis()
        line = render_skill_line(hyp)
        assert line.startswith("*")
        assert "User prefers bullet points" in line

    def test_version_tag(self) -> None:
        """Version > 1 shows (vN) tag."""
        hyp = _make_hypothesis(version=3)
        line = render_skill_line(hyp)
        assert "(v3)" in line

    def test_no_version_tag_for_v1(self) -> None:
        """Version 1 has no version tag."""
        hyp = _make_hypothesis(version=1)
        line = render_skill_line(hyp)
        assert "(v1)" not in line

    def test_paused_status_shown(self) -> None:
        """Paused status is shown inline."""
        hyp = _make_hypothesis(status=STATUS_PAUSED)
        line = render_skill_line(hyp)
        assert "pausiert" in line


# ---------------------------------------------------------------
# Tests: format_skill_indicator (HC-UI-2)
# ---------------------------------------------------------------


class TestFormatSkillIndicator:
    """Tests for skill application indicator."""

    def test_indicator_appended(self) -> None:
        """Indicator must be appended after response."""
        hyp = _make_hypothesis(claim="Drehkonzepte")
        result = format_skill_indicator(hyp, "Bot response text")
        assert "Bot response text" in result
        assert "Skill" in result
        assert "angewendet" in result

    def test_indicator_contains_separator(self) -> None:
        """Indicator must have a visual separator."""
        hyp = _make_hypothesis()
        result = format_skill_indicator(hyp, "Response")
        assert "─" in result  # Unicode separator

    def test_indicator_with_version(self) -> None:
        """Indicator shows version for v>1."""
        hyp = _make_hypothesis(version=2, claim="Drehkonzepte")
        result = format_skill_indicator(hyp, "Response")
        assert "v2" in result

    def test_indicator_no_version_for_v1(self) -> None:
        """Indicator does not show version for v1."""
        hyp = _make_hypothesis(version=1, claim="Drehkonzepte")
        result = format_skill_indicator(hyp, "Response")
        assert " v1" not in result


# ---------------------------------------------------------------
# Tests: Inline keyboard builders
# ---------------------------------------------------------------


class TestInlineKeyboards:
    """Tests for keyboard builders."""

    def test_skill_actions_keyboard_has_buttons(self) -> None:
        """Action keyboard has pause and forget buttons."""
        hyp = _make_hypothesis()
        kb = build_skill_actions_keyboard(hyp)
        # Flatten buttons
        all_text = [btn.text for row in kb.inline_keyboard for btn in row]
        assert "pausieren" in all_text
        assert "vergessen" in all_text

    def test_versions_button_only_when_v_gt_1(self) -> None:
        """Versions button only appears for version > 1."""
        hyp_v1 = _make_hypothesis(version=1)
        kb_v1 = build_skill_actions_keyboard(hyp_v1)
        all_text_v1 = [btn.text for row in kb_v1.inline_keyboard for btn in row]
        assert "Versionen" not in all_text_v1

        hyp_v2 = _make_hypothesis(version=2)
        kb_v2 = build_skill_actions_keyboard(hyp_v2)
        all_text_v2 = [btn.text for row in kb_v2.inline_keyboard for btn in row]
        assert "Versionen" in all_text_v2

    def test_paused_skill_shows_resume(self) -> None:
        """Paused skill shows 'fortsetzen' instead of 'pausieren'."""
        hyp = _make_hypothesis(status=STATUS_PAUSED)
        kb = build_skill_actions_keyboard(hyp)
        all_text = [btn.text for row in kb.inline_keyboard for btn in row]
        assert "fortsetzen" in all_text
        assert "pausieren" not in all_text

    def test_profile_list_keyboard(self) -> None:
        """Profile list keyboard has one button per skill."""
        hyps = [
            _make_hypothesis(
                hypothesis_id=f"hyp-{i}",
                claim=f"Skill {i}",
            )
            for i in range(5)
        ]
        kb = build_profile_list_keyboard(hyps)
        assert len(kb.inline_keyboard) == 5

    def test_indicator_keyboard_has_undo_and_explain(self) -> None:
        """Indicator keyboard has /undo and Warum? buttons."""
        kb = build_indicator_keyboard("hyp-001")
        all_text = [btn.text for row in kb.inline_keyboard for btn in row]
        assert "/undo" in all_text
        assert "Warum?" in all_text
