"""Tests for application.language_resolver: LanguageResolver + LanguageContext.

Tests the single-entry-point language resolution logic:
- No sticky, no text -> default
- No sticky, detected with confidence -> detected
- Sticky, no switch (low confidence or same lang) -> sticky
- Sticky, high confidence, different lang -> smart switch
- Override always wins
"""

from __future__ import annotations

import pytest

from application.language_resolver import LanguageContext, LanguageResolver
from infrastructure.conversation_storage import _reset_all_for_tests


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Reset conversation storage before each test."""
    _reset_all_for_tests()


class TestLanguageResolver:
    """Tests for LanguageResolver.resolve()."""

    async def test_resolver_no_sticky_no_text_uses_default(self) -> None:
        """Empty text with no sticky language falls back to default."""
        resolver = LanguageResolver(default_lang="de")
        ctx = await resolver.resolve(user_id=1, chat_id=10, text="")

        assert ctx.code == "de"
        assert ctx.source == "default"
        assert ctx.confidence == 0.0
        assert ctx.switched_from is None
        assert ctx.request_id  # non-empty

    async def test_resolver_no_sticky_detected_with_confidence(self) -> None:
        """Clear English text with no sticky -> detected as English."""
        resolver = LanguageResolver(default_lang="de")
        ctx = await resolver.resolve(
            user_id=1, chat_id=10, text="What is the weather like today?"
        )

        assert ctx.code == "en"
        assert ctx.source == "detected"
        assert ctx.confidence > 0.0
        assert ctx.switched_from is None

    async def test_resolver_no_sticky_german_text(self) -> None:
        """German text with no sticky -> detected as German."""
        resolver = LanguageResolver(default_lang="de")
        ctx = await resolver.resolve(
            user_id=1, chat_id=10, text="Wie wird das Wetter heute?"
        )

        assert ctx.code == "de"
        assert ctx.source == "detected"
        assert ctx.confidence > 0.0

    async def test_resolver_sticky_no_switch(self) -> None:
        """With sticky language and low-confidence detection: keep sticky."""
        from infrastructure.conversation_storage import set_language

        await set_language(1, 10, "it")

        resolver = LanguageResolver(default_lang="de")
        # Very short text: low confidence detection
        ctx = await resolver.resolve(user_id=1, chat_id=10, text="ok")

        assert ctx.code == "it"
        assert ctx.source == "sticky"
        assert ctx.confidence == 1.0
        assert ctx.switched_from is None

    async def test_resolver_sticky_high_confidence_switch(self) -> None:
        """With sticky=de but clear English text: smart switch to en."""
        from infrastructure.conversation_storage import set_language

        await set_language(1, 10, "de")

        resolver = LanguageResolver(default_lang="de")
        ctx = await resolver.resolve(
            user_id=1,
            chat_id=10,
            text="Can you please help me with this problem? I would really appreciate your assistance.",
        )

        assert ctx.code == "en"
        assert ctx.source == "detected"
        assert ctx.confidence > 0.7
        assert ctx.switched_from == "de"
        assert ctx.was_smart_switched is True

    async def test_resolver_sticky_low_confidence_no_switch(self) -> None:
        """With sticky=en and ambiguous text: no switch."""
        from infrastructure.conversation_storage import set_language

        await set_language(1, 10, "en")

        resolver = LanguageResolver(default_lang="de")
        # "Hallo" alone might not be high-confidence enough to switch
        ctx = await resolver.resolve(user_id=1, chat_id=10, text="Hallo")

        # Should keep sticky 'en' since confidence for de from "Hallo" alone
        # is likely below 0.7
        assert ctx.code == "en"
        assert ctx.source == "sticky"
        assert ctx.switched_from is None

    async def test_resolver_override_wins(self) -> None:
        """Override always takes priority over sticky and detection."""
        from infrastructure.conversation_storage import set_language

        await set_language(1, 10, "de")

        resolver = LanguageResolver(default_lang="de")
        ctx = await resolver.resolve(
            user_id=1,
            chat_id=10,
            text="This is English text",
            override="fr",
        )

        assert ctx.code == "fr"
        assert ctx.source == "override"
        assert ctx.confidence == 1.0
        assert ctx.switched_from is None

    async def test_resolver_request_id_unique(self) -> None:
        """Each resolve() call generates a unique request_id."""
        resolver = LanguageResolver(default_lang="de")
        ctx1 = await resolver.resolve(user_id=1, chat_id=10, text="Hallo")
        ctx2 = await resolver.resolve(user_id=1, chat_id=10, text="World")

        assert ctx1.request_id != ctx2.request_id


class TestLanguageContext:
    """Tests for LanguageContext dataclass."""

    def test_effective_lang_returns_code(self) -> None:
        """effective_lang() is just a convenience for ctx.code."""
        ctx = LanguageContext(
            code="it",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="abc123",
        )
        assert ctx.effective_lang() == "it"

    def test_was_smart_switched_true(self) -> None:
        """was_smart_switched is True when switched_from is set."""
        ctx = LanguageContext(
            code="en",
            source="detected",
            confidence=0.9,
            switched_from="de",
            request_id="abc123",
        )
        assert ctx.was_smart_switched is True

    def test_was_smart_switched_false(self) -> None:
        """was_smart_switched is False when switched_from is None."""
        ctx = LanguageContext(
            code="de",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="abc123",
        )
        assert ctx.was_smart_switched is False

    def test_frozen_dataclass(self) -> None:
        """LanguageContext is immutable (frozen)."""
        ctx = LanguageContext(
            code="de",
            source="default",
            confidence=0.0,
            switched_from=None,
            request_id="abc123",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.code = "en"  # type: ignore[misc]

    def test_from_code_factory(self) -> None:
        """from_code creates a valid LanguageContext from a plain string."""
        ctx = LanguageResolver.from_code("fr")
        assert ctx.code == "fr"
        assert ctx.source == "override"
        assert ctx.confidence == 1.0
        assert ctx.request_id  # non-empty
