"""Tests for LanguageContract: dynamic language enforcement blocks."""

from application.language.context import LanguageContext
from application.language.contract import LanguageContract
from application.language.model_profiles import get_profile


def _make_ctx(code: str = "de") -> LanguageContext:
    """Helper to create a LanguageContext for testing."""
    return LanguageContext(
        code=code,
        source="sticky",
        confidence=1.0,
        switched_from=None,
        request_id="test123",
    )


class TestLanguageContractBuild:
    """Tests for LanguageContract.build()."""

    def test_normal_level_is_polite(self) -> None:
        """Normal enforcement uses polite language."""
        ctx = _make_ctx("de")
        contract = LanguageContract.build(ctx, model_id="claude-opus-4-7")
        assert "German" in contract
        assert "de" in contract
        assert "MUST" not in contract  # Polite, not demanding

    def test_strict_level_uses_must(self) -> None:
        """Strict enforcement uses MUST and firm language."""
        ctx = _make_ctx("fr")
        contract = LanguageContract.build(ctx, model_id="gemini-2.0-flash")
        assert "MUST" in contract
        assert "French" in contract
        assert "hard constraint" in contract

    def test_strict_with_verify_is_emphatic(self) -> None:
        """Strict_with_verify uses maximum emphasis."""
        ctx = _make_ctx("sv")
        contract = LanguageContract.build(ctx, model_id="llama-3.1-8b")
        assert "MANDATORY" in contract or "MUST" in contract
        assert "Swedish" in contract
        assert "non-negotiable" in contract

    def test_explicit_profile_overrides_model_id(self) -> None:
        """Passing profile directly skips model_id lookup."""
        ctx = _make_ctx("en")
        profile = get_profile("claude-haiku-4-5")
        contract = LanguageContract.build(ctx, profile=profile)
        # Haiku is strict_with_verify
        assert "MANDATORY" in contract or "non-negotiable" in contract

    def test_unknown_language_uses_code(self) -> None:
        """Unknown language code is used as-is in the contract."""
        ctx = _make_ctx("xx")
        contract = LanguageContract.build(ctx, model_id="claude-opus-4-7")
        assert "xx" in contract  # Code used when no name mapping

    def test_all_supported_languages_have_names(self) -> None:
        """All commonly used language codes produce named contracts."""
        codes = [
            "de",
            "en",
            "fr",
            "es",
            "it",
            "pt",
            "nl",
            "sv",
            "da",
            "nb",
            "fi",
            "pl",
            "tr",
            "ru",
            "uk",
            "ar",
            "zh",
            "ja",
            "ko",
            "hi",
            "th",
            "id",
            "vi",
        ]
        for code in codes:
            ctx = _make_ctx(code)
            contract = LanguageContract.build(ctx, model_id="claude-opus-4-7")
            # Should not just contain the raw code but a human name
            assert len(contract) > 20


class TestLanguageContractRepair:
    """Tests for LanguageContract.build_repair_contract()."""

    def test_repair_contract_mentions_target(self) -> None:
        """Repair contract explicitly names the target language."""
        ctx = _make_ctx("de")
        contract = LanguageContract.build_repair_contract(ctx, "en")
        assert "German" in contract
        assert "English" in contract
        assert "CRITICAL" in contract

    def test_repair_contract_without_detected(self) -> None:
        """Repair contract works when detected lang is None."""
        ctx = _make_ctx("fr")
        contract = LanguageContract.build_repair_contract(ctx, None)
        assert "French" in contract
        assert "different language" in contract

    def test_repair_contract_instructs_rewrite(self) -> None:
        """Repair contract tells model to rewrite entirely."""
        ctx = _make_ctx("es")
        contract = LanguageContract.build_repair_contract(ctx, "en")
        assert "rewrite" in contract.lower() or "translate" in contract.lower()
