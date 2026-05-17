"""Tests for application.style_adaption_service.

Tests the style profile observation, update logic, and prompt block generation.
Part of P3 (Contextual Silence with Style Adaptation).
"""

from application.style_adaption_service import (
    StyleAdaptionService,
    StyleProfile,
    _MIN_MESSAGES_FOR_PROFILE,
)


class TestStyleProfile:
    """StyleProfile data structure and prompt block generation."""

    def test_immature_profile_returns_empty_block(self) -> None:
        """Profile with fewer than MIN messages produces no prompt block."""
        profile = StyleProfile(observed_messages=2)
        assert profile.to_prompt_block() == ""

    def test_mature_profile_with_emojis(self) -> None:
        """Mature profile with high emoji usage produces guidance."""
        profile = StyleProfile(
            emoji_frequency=0.5,
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "emojis" in block.lower()
        assert "mirror" in block.lower()

    def test_mature_profile_no_emojis(self) -> None:
        """Mature profile with low emoji usage produces emoji-free guidance."""
        profile = StyleProfile(
            emoji_frequency=0.01,
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "emoji-free" in block.lower()

    def test_formal_profile(self) -> None:
        """Profile with Sie-formality produces formal guidance."""
        profile = StyleProfile(
            formality="sie",
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "Sie" in block

    def test_informal_profile(self) -> None:
        """Profile with Du-formality produces informal guidance."""
        profile = StyleProfile(
            formality="du",
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "Du" in block

    def test_terse_tonality(self) -> None:
        """Terse tonality produces concise style guidance."""
        profile = StyleProfile(
            tonality="terse",
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "terse" in block.lower() or "short" in block.lower()

    def test_warm_tonality(self) -> None:
        """Warm tonality produces elaborate style guidance."""
        profile = StyleProfile(
            tonality="warm",
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "warm" in block.lower()

    def test_mobile_device_signal(self) -> None:
        """Mobile device signal produces compact formatting guidance."""
        profile = StyleProfile(
            device_signal="mobile",
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "mobile" in block.lower()

    def test_code_switching_profile(self) -> None:
        """Code-switching detection produces natural acceptance guidance."""
        profile = StyleProfile(
            uses_code_switching=True,
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert "code-switching" in block.lower() or "mix" in block.lower()

    def test_neutral_profile_minimal_block(self) -> None:
        """Profile with all neutral/unknown values produces empty block."""
        profile = StyleProfile(
            emoji_frequency=0.15,  # between thresholds
            formality="unknown",
            tonality="neutral",
            device_signal="unknown",
            uses_code_switching=False,
            observed_messages=_MIN_MESSAGES_FOR_PROFILE,
        )
        block = profile.to_prompt_block()
        assert block == ""


class TestStyleAdaptionService:
    """StyleAdaptionService observation and profile building."""

    def test_observe_builds_profile(self) -> None:
        """Observing messages creates a profile for the user."""
        service = StyleAdaptionService()
        service.observe(123, "Hallo, wie geht es dir?")
        profile = service.get_profile(123)
        assert profile is not None
        assert profile.observed_messages == 1

    def test_empty_message_ignored(self) -> None:
        """Empty messages are not observed."""
        service = StyleAdaptionService()
        service.observe(123, "")
        service.observe(123, "   ")
        profile = service.get_profile(123)
        assert profile is None

    def test_emoji_detection(self) -> None:
        """Messages with emojis increase emoji_frequency."""
        service = StyleAdaptionService()
        for _ in range(6):
            service.observe(123, "Super cool! \U0001f60d\U0001f389")
        profile = service.get_profile(123)
        assert profile is not None
        assert profile.emoji_frequency > 0.5

    def test_no_emoji_detection(self) -> None:
        """Messages without emojis keep emoji_frequency low."""
        service = StyleAdaptionService()
        for _ in range(6):
            service.observe(123, "Das ist ein normaler Satz ohne Emojis.")
        profile = service.get_profile(123)
        assert profile is not None
        assert profile.emoji_frequency < 0.1

    def test_formality_du_detection(self) -> None:
        """Messages with Du-form detected correctly."""
        service = StyleAdaptionService()
        messages = [
            "Kannst du mir helfen?",
            "Was meinst du dazu?",
            "Hast du das schon probiert?",
            "Du bist der Beste!",
            "Sag mir doch was du denkst.",
            "Wie findest du das?",
        ]
        for msg in messages:
            service.observe(456, msg)
        profile = service.get_profile(456)
        assert profile is not None
        assert profile.formality == "du"

    def test_terse_tonality_detection(self) -> None:
        """Short, direct messages lead to terse tonality."""
        service = StyleAdaptionService()
        messages = ["Ja", "Nein", "Ok danke", "Mach das", "Passt", "Gut"]
        for msg in messages:
            service.observe(789, msg)
        profile = service.get_profile(789)
        assert profile is not None
        assert profile.tonality == "terse"

    def test_warm_tonality_detection(self) -> None:
        """Long, elaborate messages lead to warm tonality."""
        service = StyleAdaptionService()
        long_msg = (
            "Also ich habe mir das mal angesehen und finde es wirklich super "
            "interessant wie das alles zusammenhaengt. Besonders der Teil mit "
            "der Integration hat mir gut gefallen und ich wuerde gerne mehr "
            "darueber erfahren wenn du Zeit hast."
        )
        for _ in range(6):
            service.observe(101, long_msg)
        profile = service.get_profile(101)
        assert profile is not None
        assert profile.tonality == "warm"

    def test_code_switching_detection(self) -> None:
        """Messages mixing German and English detected as code-switching."""
        service = StyleAdaptionService()
        messages = [
            "Das ist really cool btw",
            "Ich habe just das Setup gemacht",
            "Sure, mach das like whatever",
            "Nice, actually ganz gut",
            "Okay cool, maybe morgen",
            "Sorry, habe das vergessen lol",
        ]
        for msg in messages:
            service.observe(202, msg)
        profile = service.get_profile(202)
        assert profile is not None
        assert profile.uses_code_switching is True

    def test_get_prompt_block_immature(self) -> None:
        """Prompt block for immature profiles returns empty (no style data)."""
        service = StyleAdaptionService()
        service.observe(303, "Hallo")
        block = service.get_prompt_block(303)
        # Anti-repetition moved to PromptComposer (i18n, all 20 languages)
        assert block == ""
        assert "[USER STYLE PROFILE]" not in block

    def test_get_prompt_block_mature(self) -> None:
        """Prompt block is non-empty for mature profiles with signals."""
        service = StyleAdaptionService()
        for _ in range(_MIN_MESSAGES_FOR_PROFILE + 1):
            service.observe(404, "Super! \U0001f60d Wie geht es dir?")
        block = service.get_prompt_block(404)
        assert "[USER STYLE PROFILE]" in block

    def test_unknown_user_returns_empty(self) -> None:
        """Unknown user_id returns empty block (anti-repetition via PromptComposer)."""
        service = StyleAdaptionService()
        block = service.get_prompt_block(999)
        # Anti-repetition is now handled by PromptComposer, not StyleAdaptionService
        assert block == ""
        assert service.get_profile(999) is None

    def test_mobile_detection(self) -> None:
        """Short messages with high emoji density suggest mobile."""
        service = StyleAdaptionService()
        for _ in range(10):
            service.observe(505, "\U0001f60d\U0001f389 ok")
        profile = service.get_profile(505)
        assert profile is not None
        assert profile.device_signal == "mobile"


class TestAntiRepetition:
    """Tests for the anti-repetition feature (NEU-03 / Item 11).

    Note: Anti-repetition prompt injection moved to PromptComposer (i18n).
    StyleAdaptionService retains only the post-response check
    (check_repetition_warning) and the filler word list (REPETITION_FILLERS).
    """

    def test_anti_repetition_not_in_style_block(self) -> None:
        """Anti-repetition is no longer part of get_prompt_block (moved to Composer)."""
        service = StyleAdaptionService()
        block = service.get_prompt_block(999, lang="de")
        # Anti-repetition is now exclusively via PromptComposer
        assert "[ANTI-REPETITION]" not in block

    def test_filler_list_still_available(self) -> None:
        """Filler list is retained for check_repetition_warning."""
        service = StyleAdaptionService()
        assert "Gerne" in service.REPETITION_FILLERS["de"]
        assert "Sure" in service.REPETITION_FILLERS["en"]

    def test_check_repetition_warning_detects_gerne(self) -> None:
        """check_repetition_warning flags repeated 'Gerne'."""
        service = StyleAdaptionService()
        response = (
            "Gerne helfe ich dir dabei! Hier sind die Informationen. "
            "Gerne erklaere ich das naeher."
        )
        warning = service.check_repetition_warning(response, lang="de")
        assert warning is not None
        assert "Gerne" in warning

    def test_check_repetition_warning_no_issue(self) -> None:
        """check_repetition_warning returns None when no repetition."""
        service = StyleAdaptionService()
        response = "Hier ist die Antwort auf deine Frage. Die Loesung ist einfach."
        warning = service.check_repetition_warning(response, lang="de")
        assert warning is None

    def test_check_repetition_warning_english(self) -> None:
        """English repetition detection works."""
        service = StyleAdaptionService()
        response = "Sure, I can help! Sure, here's what you need to know."
        warning = service.check_repetition_warning(response, lang="en")
        assert warning is not None
        assert "Sure" in warning

    def test_prompt_block_mature_only_style(self) -> None:
        """Mature profile prompt block includes only style, not anti-repetition."""
        service = StyleAdaptionService()
        for _ in range(_MIN_MESSAGES_FOR_PROFILE + 1):
            service.observe(606, "Super toll! \U0001f60d Danke dir!")
        block = service.get_prompt_block(606, lang="de")
        # Only style profile, anti-repetition handled by PromptComposer
        assert "[USER STYLE PROFILE]" in block
        assert "[ANTI-REPETITION]" not in block
