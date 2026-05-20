"""Tests for ModelAdherenceProfile and get_profile()."""

import pytest

from application.language.model_profiles import (
    ModelAdherenceProfile,
    get_profile,
    list_profiles,
)


class TestGetProfile:
    """Tests for get_profile() lookup logic."""

    def test_exact_match_opus(self) -> None:
        """Exact model ID match returns correct profile."""
        profile = get_profile("claude-opus-4-7")
        assert profile.model_id == "claude-opus-4-7"
        assert profile.enforcement_level == "normal"
        assert profile.verify_required is False
        assert profile.repair_enabled is False
        assert profile.stream_guard_enabled is False

    def test_exact_match_haiku(self) -> None:
        """Haiku has strict_with_verify enforcement."""
        profile = get_profile("claude-haiku-4-5")
        assert profile.enforcement_level == "strict_with_verify"
        assert profile.verify_required is True
        assert profile.repair_enabled is True
        assert profile.stream_guard_enabled is True

    def test_prefix_match_llama(self) -> None:
        """Llama-3.1-8b should match 'llama' prefix."""
        profile = get_profile("llama-3.1-8b")
        assert profile.model_id == "llama"
        assert profile.enforcement_level == "strict_with_verify"

    def test_prefix_match_gemini(self) -> None:
        """gemini-2.0-flash should match 'gemini' prefix."""
        profile = get_profile("gemini-2.0-flash")
        assert profile.model_id == "gemini"
        assert profile.enforcement_level == "strict"

    def test_unknown_model_returns_default(self) -> None:
        """Unknown model ID falls back to default profile."""
        profile = get_profile("some-unknown-model-xyz")
        assert profile.model_id == "default"
        assert profile.enforcement_level == "strict"
        assert profile.verify_required is True

    def test_none_model_returns_default(self) -> None:
        """None model_id returns default profile."""
        profile = get_profile(None)
        assert profile.model_id == "default"

    def test_empty_string_returns_default(self) -> None:
        """Empty string model_id returns default profile."""
        profile = get_profile("")
        assert profile.model_id == "default"

    def test_profile_is_frozen(self) -> None:
        """ModelAdherenceProfile is immutable."""
        profile = get_profile("claude-opus-4-7")
        with pytest.raises(AttributeError):
            profile.enforcement_level = "strict"  # type: ignore[misc]


class TestListProfiles:
    """Tests for list_profiles()."""

    def test_list_returns_all_profiles(self) -> None:
        """list_profiles() returns a non-empty list."""
        profiles = list_profiles()
        assert len(profiles) >= 8  # At least our known profiles
        assert all(isinstance(p, ModelAdherenceProfile) for p in profiles)

    def test_default_in_list(self) -> None:
        """Default profile is in the list."""
        profiles = list_profiles()
        ids = [p.model_id for p in profiles]
        assert "default" in ids
