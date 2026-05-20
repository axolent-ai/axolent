"""Dynamic Language Contract: builds natural-language enforcement blocks.

Instead of bracket-markers like [RESPOND IN: de], this module
generates human-readable, natural-language contract blocks that
are inserted into the system prompt. The intensity of the contract
varies by ModelAdherenceProfile enforcement level.

The contract is THE authoritative language instruction in the prompt.
No other component may inject language instructions independently.

Phase 2 migration (Step 4/4):
    HC-R7: _LANGUAGE_NAMES dict replaced by LanguageRegistry lookups.
    All language metadata is now centralized in the Registry (single
    source of truth). No more dual-maintenance of language names.
"""

from __future__ import annotations

import logging

from application.language.context import LanguageContext
from application.language.model_profiles import (
    ModelAdherenceProfile,
    get_profile,
)
from application.language.registry import InMemoryLanguageRegistry

log = logging.getLogger(__name__)

# Singleton registry instance for language name lookups (HC-R7).
# The registry is read-only and thread-safe, so a module-level
# instance is fine.
_registry = InMemoryLanguageRegistry()


def _get_language_name(code: str) -> str:
    """Get human-readable language name for a code via Registry (HC-R7).

    Args:
        code: ISO-639-1 language code.

    Returns:
        English name of the language, or the code itself if unknown.
    """
    entry = _registry.get_or_none(code)
    if entry is not None:
        return entry.name
    return code


class LanguageContract:
    """Builds dynamic language enforcement contracts for system prompts.

    The contract intensity scales with the model's enforcement level:
    - normal: polite instruction
    - strict: firm instruction with explicit prohibition
    - strict_with_verify: emphatic instruction with consequences stated
    """

    @staticmethod
    def build(
        ctx: LanguageContext,
        model_id: str | None = None,
        profile: ModelAdherenceProfile | None = None,
    ) -> str:
        """Build a language contract block for the system prompt.

        Args:
            ctx: Resolved LanguageContext for this request.
            model_id: Model ID (used to look up profile if profile not given).
            profile: Pre-resolved profile (takes precedence over model_id).

        Returns:
            Natural-language contract string to inject into system prompt.
        """
        if profile is None:
            profile = get_profile(model_id)

        lang_code = ctx.code
        lang_name = _get_language_name(lang_code)
        level = profile.enforcement_level

        if level == "normal":
            return _build_normal_contract(lang_code, lang_name)
        elif level == "strict":
            return _build_strict_contract(lang_code, lang_name)
        else:  # strict_with_verify
            return _build_strict_with_verify_contract(lang_code, lang_name)

    @staticmethod
    def build_repair_contract(
        ctx: LanguageContext,
        original_detected_lang: str | None = None,
    ) -> str:
        """Build a reinforced contract for repair re-queries.

        Used by RepairService when the first response was in the wrong
        language. This is maximally explicit.

        Args:
            ctx: Target LanguageContext.
            original_detected_lang: What language was detected in the
                failed response (for context in the repair instruction).

        Returns:
            Repair contract string for the re-query system prompt.
        """
        lang_name = _get_language_name(ctx.code)
        detected_name = (
            _get_language_name(original_detected_lang)
            if original_detected_lang
            else "a different language"
        )

        return (
            f"CRITICAL LANGUAGE CORRECTION: Your previous response was in "
            f"{detected_name}, but the user expects {lang_name} ({ctx.code}). "
            f"You MUST rewrite your entire response in {lang_name}. "
            f"Preserve the meaning, tone, and formatting exactly, "
            f"but translate everything to {lang_name}. "
            f"Do not acknowledge this correction in your response. "
            f"Simply provide the answer in {lang_name}."
        )


def _build_normal_contract(lang_code: str, lang_name: str) -> str:
    """Normal enforcement: polite but clear instruction.

    For models that reliably follow instructions (Claude Opus/Sonnet).
    """
    return (
        f"Respond only in {lang_name} ({lang_code}). "
        f"This overrides any default language behavior. "
        f"Do not switch languages mid-response."
    )


def _build_strict_contract(lang_code: str, lang_name: str) -> str:
    """Strict enforcement: firm instruction with explicit prohibition.

    For models that occasionally drift (Gemini, Mistral).
    """
    return (
        f"LANGUAGE REQUIREMENT: You MUST respond entirely in {lang_name} "
        f"({lang_code}). This is a hard constraint, not a suggestion. "
        f"Do not use any other language in your response, not even for "
        f"greetings, transitions, or filler words. "
        f"If the user writes in a different language, still respond in "
        f"{lang_name}. The response language is determined by the system, "
        f"not by the user's input language."
    )


def _build_strict_with_verify_contract(lang_code: str, lang_name: str) -> str:
    """Maximum enforcement: emphatic with stated consequences.

    For models that frequently drift (local Llama, Haiku).
    """
    return (
        f"MANDATORY LANGUAGE CONTRACT: Your ENTIRE response MUST be in "
        f"{lang_name} ({lang_code}). This is non-negotiable. "
        f"Every single word, sentence, and paragraph must be in {lang_name}. "
        f"Do not mix languages. Do not default to English. "
        f"Do not switch languages even if the topic involves foreign terms. "
        f"Technical terms and proper nouns may remain in their original form, "
        f"but all explanatory text must be in {lang_name}. "
        f"Your response will be automatically verified for language compliance."
    )
