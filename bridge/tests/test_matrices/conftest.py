"""Parametrization matrices for cross-cutting AXOLENT behavior tests.

Three dimensions used systematically:
  LANGUAGES = standard set of 10 supported languages
  COMMANDS_WITH_ARGS = standard set of 8 slash commands expecting arguments
  COMMANDS_NO_ARGS = standard set of 6 slash commands without arguments
  CHANNELS = standard set of 4 message channel types

Models dimension intentionally excluded from broad matrices because:
  - Most AXOLENT logic is model-agnostic
  - Model-specific tests live in test_application/test_routing/
  - Cartesian explosion (3 models * 10 langs * 8 cmds * 4 channels = 960 tests) is
    not worth the runtime if logic is provably model-independent

Use single-dimension and 2D matrices where possible. Reserve 3D for actual
cross-cutting concerns (e.g. test_language_x_command_x_channel for sticky check).
"""

from __future__ import annotations


import pytest

from application.language.resolver import LanguageResolver
from infrastructure.conversation_storage import _reset_all_for_tests


# ---------------------------------------------------------------------------
# Dimension 1: Languages (the 10 core wizard languages)
# ---------------------------------------------------------------------------

LANGUAGES: list[str] = ["de", "en", "nl", "sv", "fr", "es", "it", "pt", "pl", "tr"]

# ---------------------------------------------------------------------------
# Dimension 2: Commands
# ---------------------------------------------------------------------------

COMMANDS_WITH_ARGS: list[tuple[str, str]] = [
    ("/remember", "test memory content"),
    ("/learn", "test pattern to learn"),
    ("/forget", "skill_id_123"),
    ("/explain", "skill_id_456"),
    ("/memory", ""),
    ("/skills", ""),
    ("/skill", "skill_id_789"),
    ("/usage", ""),
]

COMMANDS_NO_ARGS: list[str] = [
    "/reset",
    "/stop",
    "/help",
    "/start",
    "/settings",
    "/onboarding",
]

# ---------------------------------------------------------------------------
# Dimension 3: Channel types
# ---------------------------------------------------------------------------

CHANNELS: list[str] = ["normal", "reply", "long_message", "streaming"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def language_corpus() -> dict[str, str]:
    """Returns dict {lang: representative_text} for all 10 LANGUAGES.

    Each text is long enough (>6 words) to produce reliable detection.
    """
    return {
        "de": "Erklaere mir die Quantenphysik in einfachen Worten bitte",
        "en": "Explain quantum physics to me in simple terms please",
        "nl": "Leg me kwantumfysica uit in eenvoudige woorden alsjeblieft",
        "sv": "Beraetta foer mig om kvantfysik paa enkel svenska tack",
        "fr": "Explique-moi la physique quantique en mots simples svp",
        "es": "Explicame la fisica cuantica en palabras simples por favor",
        "it": "Spiegami la fisica quantistica con parole semplici per favore",
        "pt": "Explica-me a fisica quantica com palavras simples por favor",
        "pl": "Wyjasnij mi fizyke kwantowa w prostych slowach prosze",
        "tr": "Bana kuantum fizigini basit kelimelerle anlat lutfen",
    }


# Representative sentences that produce HIGH confidence in detect_language.
# These use actual characters (not ASCII approximations) for reliable detection.
LANGUAGE_MARKER_TEXTS: dict[str, str] = {
    "de": "Wie wird das Wetter heute? Ich hoffe es wird ein sonniger Tag.",
    "en": "What is the weather like today? I hope it will be a sunny day.",
    "nl": "Hoe is het weer vandaag? Ik hoop dat het een zonnige dag wordt.",
    "sv": "Hur blir vädret idag? Jag hoppas att det blir en solig dag.",
    "fr": "Quel temps fait-il aujourd'hui? J'espère qu'il fera beau.",
    "es": "Como esta el tiempo hoy? Espero que sea un dia soleado.",
    "it": "Come sara il tempo oggi? Spero che sara una giornata soleggiata.",
    "pt": "Como esta o tempo hoje? Espero que seja um dia ensolarado.",
    "pl": "Jaka jest dzisiaj pogoda? Mam nadzieje ze bedzie sloneczny dzien.",
    "tr": "Buguen hava nasil? Umarim guneşli bir guen olur.",
}


@pytest.fixture
def language_resolver() -> LanguageResolver:
    """Provide a fresh LanguageResolver with default settings."""
    return LanguageResolver(default_lang="de")


@pytest.fixture(autouse=True)
def _reset_conversation_state() -> None:
    """Reset conversation storage before each test to prevent cross-contamination."""
    _reset_all_for_tests()
