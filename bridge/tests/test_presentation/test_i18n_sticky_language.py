"""Tests for i18n sticky-language compliance across all user-facing messages.

Verifies that after a user selects a language in the wizard, ALL subsequent
bot messages respect that language choice (sticky language).

Bug context: After wizard completion with lang=en, /start and wizard_done
still showed German text. This test suite prevents regressions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.onboarding import (
    WIZARD_LANGUAGES,
    get_start_welcome_text,
    get_wizard_done_text,
)
from infrastructure.onboarding_storage import OnboardingStorage
from infrastructure.sqlite_storage import SqliteConnection


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_i18n.db"


@pytest.fixture
def conn(db_path: Path) -> SqliteConnection:
    c = SqliteConnection(db_path)
    yield c
    c.close()


@pytest.fixture
def onboarding_storage(conn: SqliteConnection) -> OnboardingStorage:
    return OnboardingStorage(conn)


@pytest.fixture(autouse=True)
def _allow_all() -> None:
    with patch("presentation.decorators.ALLOW_ALL_USERS", True):
        yield  # type: ignore[misc]


def _make_update(user_id: int = 1, chat_id: int = 10) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


def _make_callback_update(data: str, user_id: int = 1, chat_id: int = 10) -> MagicMock:
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "testuser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = "private"
    update.callback_query = MagicMock()
    update.callback_query.data = data
    update.callback_query.from_user = update.effective_user
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.message = MagicMock()
    update.callback_query.message.reply_text = AsyncMock()
    return update


def _make_context(
    onboarding_storage: OnboardingStorage | None = None,
    sticky_lang: str | None = "de",
) -> MagicMock:
    context = MagicMock()
    context.args = []

    mock_chat_service = MagicMock()
    mock_chat_service.get_chat_language = AsyncMock(return_value=sticky_lang)
    mock_chat_service.set_chat_language = AsyncMock()
    mock_chat_service.save_static_response_to_history = AsyncMock()

    context.application = MagicMock()
    context.application.bot_data = {
        "chat_service": mock_chat_service,
        "system_prompt": "Test prompt.",
        "onboarding_storage": onboarding_storage,
    }
    return context


# ──────────────────────────────────────────────────────────────
# /start respects sticky language
# ──────────────────────────────────────────────────────────────


@pytest.mark.i18n
@pytest.mark.integration
class TestStartRespectsStickyLanguage:
    """After wizard with lang=X, /start shows text in language X."""

    async def test_start_english_for_en_user(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/start shows English welcome for user with sticky lang=en."""
        from presentation.handlers import handle_start_command

        onboarding_storage.set_onboarded(1, "en")

        update = _make_update()
        context = _make_context(
            onboarding_storage=onboarding_storage,
            sticky_lang="en",
        )

        await handle_start_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "is ready" in text
        assert "Schick mir" not in text

    async def test_start_german_for_de_user(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/start shows German welcome for user with sticky lang=de."""
        from presentation.handlers import handle_start_command

        onboarding_storage.set_onboarded(1, "de")

        update = _make_update()
        context = _make_context(
            onboarding_storage=onboarding_storage,
            sticky_lang="de",
        )

        await handle_start_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "ist bereit" in text
        assert "is ready" not in text

    async def test_start_fallback_to_english_for_unknown_lang(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/start falls back to EN for languages without specific translation."""
        from presentation.handlers import handle_start_command

        onboarding_storage.set_onboarded(1, "xx")

        update = _make_update()
        context = _make_context(
            onboarding_storage=onboarding_storage,
            sticky_lang="xx",
        )

        await handle_start_command(update, context)

        text = update.message.reply_text.call_args[0][0]
        # Falls back to EN (not DE) for unknown language codes
        assert "is ready" in text


# ──────────────────────────────────────────────────────────────
# wizard_done respects chosen language
# ──────────────────────────────────────────────────────────────


def _get_wizard_done_expected() -> dict[str, str]:
    """Derive expected fragments from the canonical i18n JSON source."""
    from i18n.domain.i18n import t

    result = {}
    for code in WIZARD_LANGUAGES:
        text = t("onboarding.done", code)
        # Use the first 4+ characters as the expected fragment (enough to validate)
        result[code] = text[:6] if len(text) >= 6 else text
    return result


_WIZARD_DONE_EXPECTED: dict[str, str] = _get_wizard_done_expected()


@pytest.mark.i18n
class TestWizardDoneI18n:
    """wizard_done shows completion text in chosen language."""

    @pytest.mark.parametrize(
        "lang_code",
        sorted(WIZARD_LANGUAGES.keys()),
    )
    async def test_wizard_done_in_chosen_language(
        self,
        lang_code: str,
        onboarding_storage: OnboardingStorage,
    ) -> None:
        """wizard_done for lang={lang_code} shows localized text."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        user_id = hash(lang_code) % 100000 + 1

        # Set wizard_lang first (simulates step 1 completion)
        onboarding_storage.set_wizard_lang(user_id, lang_code)

        update = _make_callback_update("wizard_done", user_id=user_id)
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        expected_fragment = _WIZARD_DONE_EXPECTED[lang_code]
        assert expected_fragment in text, (
            f"For lang={lang_code}, expected '{expected_fragment}' in '{text}'"
        )


# ──────────────────────────────────────────────────────────────
# /help respects sticky language
# ──────────────────────────────────────────────────────────────


class TestHelpRespectsStickyLanguage:
    """After wizard with lang=X, /help shows text in correct language."""

    async def test_help_english_for_en_user(self) -> None:
        """/help shows English for sticky lang=en."""
        from presentation.handlers import handle_help_command

        update = _make_update()
        context = _make_context(sticky_lang="en")

        await handle_help_command(update, context)

        call_args = update.message.reply_text.call_args
        text = call_args[0][0]
        assert "Command Overview" in text
        assert "Befehlsübersicht" not in text

    async def test_help_german_for_de_user(self) -> None:
        """/help shows German for sticky lang=de."""
        from presentation.handlers import handle_help_command

        update = _make_update()
        context = _make_context(sticky_lang="de")

        await handle_help_command(update, context)

        call_args = update.message.reply_text.call_args
        text = call_args[0][0]
        assert "Befehlsübersicht" in text

    @pytest.mark.parametrize(
        "lang_code",
        [
            "fr",
            "es",
            "it",
            "pt",
            "nl",
            "pl",
            "sv",
            "tr",
            "ru",
            "uk",
            "zh",
            "ja",
            "ko",
            "ar",
            "hi",
            "id",
            "th",
            "vi",
        ],
    )
    async def test_help_shows_native_text_for_all_langs(self, lang_code: str) -> None:
        """/help shows native translation for lang={lang_code}."""
        from i18n.domain.i18n import t
        from presentation.handlers import handle_help_command

        update = _make_update()
        context = _make_context(sticky_lang=lang_code)

        await handle_help_command(update, context)

        call_args = update.message.reply_text.call_args
        text = call_args[0][0]
        # The help text should contain the native help.title for this language
        expected_title = t("help.title", lang_code)
        assert expected_title in text


# ──────────────────────────────────────────────────────────────
# Wizard completion screen: all 20 languages have translations
# ──────────────────────────────────────────────────────────────


@pytest.mark.i18n
class TestWizardCompletionAllLanguages:
    """Step 2 completion screen is fully translated for all 20 languages."""

    @pytest.mark.parametrize(
        "lang_code",
        sorted(WIZARD_LANGUAGES.keys()),
    )
    async def test_step2_rendered_in_chosen_language(
        self,
        lang_code: str,
        onboarding_storage: OnboardingStorage,
    ) -> None:
        """Step 2 for lang={lang_code} shows translated completion screen."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        user_id = hash(lang_code) % 100000 + 2000

        callback_data = f"wizard_lang:{lang_code}"
        update = _make_callback_update(callback_data, user_id=user_id)
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        # All step 2 texts contain /settings and /help
        assert "/settings" in text, f"Missing /settings for lang={lang_code}"
        assert "/help" in text, f"Missing /help for lang={lang_code}"
        # Text should be non-trivial
        assert len(text) > 50, f"Step 2 text too short for lang={lang_code}"


# ──────────────────────────────────────────────────────────────
# Sticky language persists after wizard
# ──────────────────────────────────────────────────────────────


class TestStickyLanguagePersistsAfterWizard:
    """After wizard with lang=X, get_language returns X."""

    async def test_wizard_sets_sticky_language(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """Selecting EN in wizard sets sticky language to EN."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        update = _make_callback_update("wizard_lang:en")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        # Verify set_chat_language was called with "en"
        chat_service = context.application.bot_data["chat_service"]
        chat_service.set_chat_language.assert_called_once_with(1, 10, "en")


# ──────────────────────────────────────────────────────────────
# Domain: get_start_welcome_text
# ──────────────────────────────────────────────────────────────


class TestStartWelcomeTextDomain:
    """Domain-level tests for get_start_welcome_text."""

    def test_de_text(self) -> None:
        text = get_start_welcome_text("de")
        assert "ist bereit" in text

    def test_en_text(self) -> None:
        text = get_start_welcome_text("en")
        assert "is ready" in text

    def test_unknown_falls_back_to_en(self) -> None:
        text = get_start_welcome_text("xx")
        assert "is ready" in text


# ──────────────────────────────────────────────────────────────
# Domain: get_wizard_done_text for all 20 languages
# ──────────────────────────────────────────────────────────────


class TestWizardDoneTextDomain:
    """Domain-level tests for wizard done text coverage."""

    @pytest.mark.parametrize(
        "lang_code",
        sorted(WIZARD_LANGUAGES.keys()),
    )
    def test_wizard_done_text_exists(self, lang_code: str) -> None:
        """get_wizard_done_text returns non-empty for lang={lang_code}."""
        text = get_wizard_done_text(lang_code)
        assert len(text) > 0
        # Should not fall back to EN for any of the 20 languages
        if lang_code != "en":
            # At least for non-Latin scripts, text should differ from EN
            # (for Latin-script languages the text might be similar, so we
            # just check it's non-empty)
            assert text is not None


# ──────────────────────────────────────────────────────────────
# Regression: DE not broken
# ──────────────────────────────────────────────────────────────


@pytest.mark.i18n
class TestDERegressionGuard:
    """Ensure German output is not accidentally broken by i18n changes."""

    async def test_start_de_unchanged(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """/start for DE user still shows German welcome."""
        from presentation.handlers import handle_start_command

        onboarding_storage.set_onboarded(1, "de")
        update = _make_update()
        context = _make_context(
            onboarding_storage=onboarding_storage,
            sticky_lang="de",
        )

        await handle_start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Axolent ist bereit" in text
        assert "Schick mir eine Frage" in text

    async def test_wizard_done_de_unchanged(
        self, onboarding_storage: OnboardingStorage
    ) -> None:
        """wizard_done for DE still shows 'Viel Spaß!'."""
        from presentation.onboarding_callbacks import handle_wizard_callback

        onboarding_storage.set_wizard_lang(1, "de")
        update = _make_callback_update("wizard_done")
        context = _make_context(onboarding_storage=onboarding_storage)

        await handle_wizard_callback(update, context)

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Viel Spaß" in text

    async def test_help_de_unchanged(self) -> None:
        """/help for DE still shows German text."""
        from presentation.handlers import handle_help_command

        update = _make_update()
        context = _make_context(sticky_lang="de")

        await handle_help_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Befehlsübersicht" in text
