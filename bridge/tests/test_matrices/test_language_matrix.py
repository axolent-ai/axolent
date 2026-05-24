"""Language matrix tests: cross-cutting language behavior across all 10 languages.

Production-path tests verifying:
  - Language detection works for representative text in each language
  - Sticky language persists after short messages
  - LanguageResolver accepts all supported languages
  - System prompt includes language directive for all languages
  - i18n translations exist for all languages
"""

from __future__ import annotations


import pytest

from application.language.resolver import LanguageResolver
from application.prompt_composer import PromptComposer
from domain.language import detect_language, detect_language_with_confidence
from domain.personality import build_effective_prompt
from i18n.domain.i18n import is_supported, t
from infrastructure.conversation_storage import (
    get_language,
    set_language,
)

from .conftest import LANGUAGE_MARKER_TEXTS, LANGUAGES


pytestmark = pytest.mark.matrix


@pytest.mark.parametrize("lang", LANGUAGES)
class TestLanguageDetection:
    """Language detection produces correct results for marker texts."""

    def test_detect_language_for_marker_text(self, lang: str) -> None:
        """Each language has marker text that detect_language() returns correctly."""
        text = LANGUAGE_MARKER_TEXTS[lang]
        detected = detect_language(text)
        assert detected == lang, (
            f"detect_language() for {lang} returned '{detected}' "
            f"for text: '{text[:40]}...'"
        )

    def test_detect_with_confidence_above_threshold(self, lang: str) -> None:
        """Each language marker text produces confidence > 0.0."""
        text = LANGUAGE_MARKER_TEXTS[lang]
        detected, confidence = detect_language_with_confidence(text)
        assert detected == lang, (
            f"detect_language_with_confidence() for {lang} returned '{detected}'"
        )
        # Most should have high confidence; at minimum > 0
        assert confidence > 0.0, (
            f"Confidence for {lang} was 0.0 (text: '{text[:40]}...')"
        )


@pytest.mark.parametrize("lang", LANGUAGES)
class TestStickyLanguage:
    """Sticky language behavior is identical across all 10 languages."""

    async def test_sticky_language_persists_after_short_message(
        self, lang: str
    ) -> None:
        """Sticky-language persists when user sends a very short message."""
        user_id, chat_id = 42, 100
        await set_language(user_id, chat_id, lang)

        resolver = LanguageResolver(default_lang="de")
        # "ok" is too short for confident detection -> sticky should hold
        ctx = await resolver.resolve(user_id=user_id, chat_id=chat_id, text="ok")

        assert ctx.code == lang, (
            f"Sticky language {lang} did not persist: resolved to {ctx.code}"
        )
        assert ctx.source == "sticky"

    async def test_sticky_language_can_be_set_and_retrieved(self, lang: str) -> None:
        """set_language / get_language roundtrips for all supported codes."""
        user_id, chat_id = 99, 200
        await set_language(user_id, chat_id, lang)
        stored = await get_language(user_id, chat_id)
        assert stored == lang


@pytest.mark.parametrize("lang", LANGUAGES)
class TestLanguageResolver:
    """LanguageResolver accepts all supported languages."""

    async def test_resolver_returns_supported_language(
        self, lang: str, language_resolver: LanguageResolver
    ) -> None:
        """LanguageResolver.resolve() returns valid code for all languages."""
        user_id, chat_id = 77, 300
        await set_language(user_id, chat_id, lang)
        ctx = await language_resolver.resolve(
            user_id=user_id, chat_id=chat_id, text="hello"
        )
        assert ctx.code == lang

    def test_from_code_factory_accepts_all_languages(self, lang: str) -> None:
        """LanguageResolver.from_code() wraps any supported code."""
        ctx = LanguageResolver.from_code(lang)
        assert ctx.code == lang
        assert ctx.confidence == 1.0


@pytest.mark.parametrize("lang", LANGUAGES)
class TestSystemPromptLanguage:
    """System prompt includes language directive for all languages."""

    def test_system_prompt_includes_language_directive(self, lang: str) -> None:
        """build_effective_prompt injects LANGUAGE LOCK for every language."""
        result = build_effective_prompt("Base prompt.", lang)
        assert "IMPORTANT: Respond only in the language" in result, (
            f"Language directive missing for '{lang}'"
        )
        assert f"'{lang}'" in result, (
            f"Language code '{lang}' not found in system prompt"
        )

    def test_system_prompt_includes_no_switch_instruction(self, lang: str) -> None:
        """build_effective_prompt includes no-switch instruction for all languages."""
        result = build_effective_prompt("Base prompt.", lang)
        assert "Do not switch languages mid-response" in result

    def test_prompt_composer_language_block(self, lang: str) -> None:
        """PromptComposer with language block injects code for all languages."""
        composer = PromptComposer()
        ctx = LanguageResolver.from_code(lang)
        result = composer.compose(
            base_prompt="You are helpful.",
            ctx=ctx,
            purpose="chat",
            blocks=["language"],
        )
        assert f"'{lang}'" in result


@pytest.mark.parametrize("lang", LANGUAGES)
class TestI18nSupport:
    """i18n system supports all 10 matrix languages."""

    def test_language_is_supported(self, lang: str) -> None:
        """is_supported() returns True for all 10 languages."""
        assert is_supported(lang), f"Language '{lang}' not in i18n supported set"

    def test_reset_confirmation_key_exists(self, lang: str) -> None:
        """The 'reset.confirmation' key returns non-placeholder text."""
        result = t("reset.confirmation", lang)
        # A missing key returns [key.name] format
        assert not result.startswith("["), (
            f"i18n key 'reset.confirmation' missing for '{lang}': got '{result}'"
        )
        assert len(result) > 5, (
            f"i18n translation suspiciously short for '{lang}': '{result}'"
        )

    def test_help_body_key_exists(self, lang: str) -> None:
        """The 'help.body' key returns non-placeholder text for all languages."""
        result = t("help.body", lang)
        assert not result.startswith("["), (
            f"i18n key 'help.body' missing for '{lang}': got '{result}'"
        )

    def test_remember_saved_key_exists(self, lang: str) -> None:
        """The 'remember.saved' key returns valid text for all languages."""
        result = t("remember.saved", lang)
        assert not result.startswith("["), (
            f"i18n key 'remember.saved' missing for '{lang}': got '{result}'"
        )
