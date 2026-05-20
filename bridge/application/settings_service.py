"""Settings service: manages user preferences for the /settings v2 menu.

Architecture:
    - SettingsService is the application-layer facade for all 6 settings categories.
    - Storage backend: SqliteSettingsStorage (single user_settings table).
    - Returns UserSettings dataclass (immutable) for each read.
    - Backwards-compat: get_settings() returns defaults from existing sources
      (user_profiles, model_service sticky) when no explicit settings row exists.

Integration points (callers must update their lookups):
    1. chat_service.process_user_message_streaming() -- model via get_settings()
    2. debate_orchestrator.debate()                  -- debate_providers via get_settings()
    3. rate_limiter.check_and_consume()              -- profile via get_settings()
    4. language_resolver                             -- language via get_settings()
    5. TimeContext resolver                          -- timezone via get_settings()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from infrastructure.sqlite_storage import SqliteSettingsStorage

log = logging.getLogger(__name__)

# Valid rate-limit profile names (must stay in sync with rate_limiter.PROFILES)
VALID_RATE_LIMIT_PROFILES: frozenset[str] = frozenset(
    {"light", "normal", "power", "unlimited"}
)

# Personality feature identifiers (UI order)
PERSONALITY_FLAGS: tuple[str, ...] = (
    "personality_p1",
    "personality_p2",
    "personality_p3",
    "personality_p4",
    "personality_p5",
    "personality_p6",
)

# Default personality state: P1-P3 ON, P4 OFF, P5-P6 ON
_PERSONALITY_DEFAULTS: dict[str, bool] = {
    "personality_p1": True,
    "personality_p2": True,
    "personality_p3": True,
    "personality_p4": False,
    "personality_p5": True,
    "personality_p6": True,
}

# Known debate providers (active + planned)
# active=True means it can be toggled ON; active=False means it is planned (tap = info toast)
DEBATE_PROVIDERS: list[dict[str, object]] = [
    {"id": "claude", "label": "Claude", "active": True},
    {"id": "llama", "label": "Llama (lokal)", "active": True},
    {"id": "gpt4o", "label": "GPT-4o (geplant)", "active": False},
    {"id": "mistral", "label": "Mistral", "active": False},
    {"id": "gemini", "label": "Gemini 3.5", "active": False},
    {"id": "groq_llama", "label": "Groq Llama", "active": False},
]

# Default active debate providers (active ones that should be ON by default)
DEFAULT_DEBATE_PROVIDERS: tuple[str, ...] = ("claude",)

# Top-20 timezones to show in the quick-select list
COMMON_TIMEZONES: tuple[str, ...] = (
    "UTC",
    "Europe/Vienna",
    "Europe/Berlin",
    "Europe/Zurich",
    "Europe/London",
    "Europe/Paris",
    "Europe/Rome",
    "Europe/Warsaw",
    "Europe/Stockholm",
    "Europe/Istanbul",
    "Europe/Moscow",
    "Europe/Kiev",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Sao_Paulo",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Asia/Kolkata",
)


@dataclass(frozen=True, slots=True)
class UserSettings:
    """Immutable snapshot of all user settings for one user.

    None values mean "not explicitly set" -> caller should use the system default
    from the respective legacy source.
    """

    user_id: int
    language: Optional[str]  # None = use sticky/auto
    model: Optional[str]  # None = use system default from model_service
    debate_providers: tuple[str, ...]
    rate_limit_profile: Literal["light", "normal", "power", "unlimited"]
    personality_p1_proactive: bool
    personality_p2_less_ai_talk: bool
    personality_p3_style_adaption: bool
    personality_p4_confidence_signal: bool
    personality_p5_time_awareness: bool
    personality_p6_show_weakness: bool
    timezone: str  # IANA string, default "UTC"


def _row_to_settings(user_id: int, row: Optional[dict]) -> UserSettings:
    """Convert a raw DB row (or None) to a UserSettings instance.

    None row -> all defaults applied.

    Args:
        user_id: Telegram user ID.
        row: Dict from SqliteSettingsStorage.get_settings_row() or None.

    Returns:
        UserSettings with either explicit values or defaults.
    """
    if row is None:
        return UserSettings(
            user_id=user_id,
            language=None,
            model=None,
            debate_providers=DEFAULT_DEBATE_PROVIDERS,
            rate_limit_profile="normal",
            personality_p1_proactive=_PERSONALITY_DEFAULTS["personality_p1"],
            personality_p2_less_ai_talk=_PERSONALITY_DEFAULTS["personality_p2"],
            personality_p3_style_adaption=_PERSONALITY_DEFAULTS["personality_p3"],
            personality_p4_confidence_signal=_PERSONALITY_DEFAULTS["personality_p4"],
            personality_p5_time_awareness=_PERSONALITY_DEFAULTS["personality_p5"],
            personality_p6_show_weakness=_PERSONALITY_DEFAULTS["personality_p6"],
            timezone="UTC",
        )

    providers_raw: str = row.get("debate_providers") or ""
    providers: tuple[str, ...] = tuple(
        p.strip() for p in providers_raw.split(",") if p.strip()
    )
    if not providers:
        providers = DEFAULT_DEBATE_PROVIDERS

    profile_raw: str = row.get("rate_limit_profile") or "normal"
    profile: Literal["light", "normal", "power", "unlimited"] = (
        profile_raw  # type: ignore[assignment]
        if profile_raw in VALID_RATE_LIMIT_PROFILES
        else "normal"
    )

    return UserSettings(
        user_id=user_id,
        language=row.get("language") or None,
        model=row.get("model") or None,
        debate_providers=providers,
        rate_limit_profile=profile,
        personality_p1_proactive=bool(row.get("personality_p1", 1)),
        personality_p2_less_ai_talk=bool(row.get("personality_p2", 1)),
        personality_p3_style_adaption=bool(row.get("personality_p3", 1)),
        personality_p4_confidence_signal=bool(row.get("personality_p4", 0)),
        personality_p5_time_awareness=bool(row.get("personality_p5", 1)),
        personality_p6_show_weakness=bool(row.get("personality_p6", 1)),
        timezone=row.get("timezone") or "UTC",
    )


class SettingsService:
    """Application-layer facade for user settings v2.

    All mutations go through this service; storage is delegated to
    SqliteSettingsStorage. The service validates inputs and logs changes.
    """

    def __init__(self, storage: "SqliteSettingsStorage") -> None:
        self._storage = storage

    async def get_settings(self, user_id: int) -> UserSettings:
        """Read all settings for a user.

        Returns defaults (UserSettings with None model/language) when
        the user has no explicit row yet.

        Args:
            user_id: Telegram user ID.

        Returns:
            UserSettings snapshot.
        """
        row = self._storage.get_settings_row(user_id)
        return _row_to_settings(user_id, row)

    async def set_language(self, user_id: int, lang: Optional[str]) -> None:
        """Set the language preference.

        Args:
            user_id: Telegram user ID.
            lang: ISO 639-1 language code or None to clear.
        """
        self._storage.set_language(user_id, lang)
        log.info("Settings: user_id=%d language -> %s", user_id, lang)

    async def set_model(self, user_id: int, model: Optional[str]) -> None:
        """Set the model preference.

        Args:
            user_id: Telegram user ID.
            model: Full model ID or None to clear.
        """
        self._storage.set_model(user_id, model)
        log.info("Settings: user_id=%d model -> %s", user_id, model)

    async def toggle_debate_provider(
        self, user_id: int, provider: str
    ) -> tuple[str, ...]:
        """Toggle a debate provider on or off.

        Only providers with active=True in DEBATE_PROVIDERS may be toggled.
        Inactive (planned) providers raise ValueError.

        Args:
            user_id: Telegram user ID.
            provider: Provider ID to toggle.

        Returns:
            Updated tuple of active provider IDs.

        Raises:
            ValueError: If provider is not known or is not active.
        """
        known = {p["id"] for p in DEBATE_PROVIDERS}
        if provider not in known:
            raise ValueError(f"Unknown debate provider: '{provider}'")
        active_set = {p["id"] for p in DEBATE_PROVIDERS if p["active"]}
        if provider not in active_set:
            raise ValueError(
                f"Provider '{provider}' is planned (not yet active). Cannot toggle."
            )
        updated = self._storage.toggle_debate_provider(user_id, provider)
        log.info(
            "Settings: user_id=%d debate_providers -> %s (toggled: %s)",
            user_id,
            updated,
            provider,
        )
        return tuple(updated)

    async def set_rate_limit(self, user_id: int, profile: str) -> bool:
        """Set the rate limit profile.

        Args:
            user_id: Telegram user ID.
            profile: Profile name (light, normal, power, unlimited).

        Returns:
            True if valid profile was set, False if unknown.
        """
        if profile not in VALID_RATE_LIMIT_PROFILES:
            log.warning(
                "Settings: invalid rate_limit_profile '%s' for user_id=%d",
                profile,
                user_id,
            )
            return False
        self._storage.set_rate_limit_profile(user_id, profile)
        log.info("Settings: user_id=%d rate_limit_profile -> %s", user_id, profile)
        return True

    async def toggle_personality(self, user_id: int, feature: str, on: bool) -> None:
        """Toggle a personality feature flag.

        Args:
            user_id: Telegram user ID.
            feature: Feature name (personality_p1 .. personality_p6).
            on: True = enable, False = disable.

        Raises:
            ValueError: If feature name is not valid.
        """
        if feature not in PERSONALITY_FLAGS:
            valid = ", ".join(PERSONALITY_FLAGS)
            raise ValueError(
                f"Unknown personality feature: '{feature}'. Valid: {valid}"
            )
        self._storage.set_personality_flag(user_id, feature, on)
        log.info(
            "Settings: user_id=%d %s -> %s", user_id, feature, "ON" if on else "OFF"
        )

    async def set_timezone(self, user_id: int, tz: str) -> None:
        """Set the timezone.

        Does NOT validate against the full IANA database here
        (validation happens in the presentation layer via zoneinfo).

        Args:
            user_id: Telegram user ID.
            tz: IANA timezone string (e.g. 'Europe/Vienna').
        """
        self._storage.set_timezone(user_id, tz)
        log.info("Settings: user_id=%d timezone -> %s", user_id, tz)
