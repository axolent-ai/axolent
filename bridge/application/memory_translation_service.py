"""Memory translation service: translates memory entries on-the-fly for /memory display.

Design:
    - Original entries in DB are NEVER modified
    - Translation happens only at display time (/memory command)
    - Caching: translated text is cached per (entry_id, target_lang)
    - Language detection: uses domain.language to detect source language
    - Fallback: if translation fails, return original text
    - Provider: uses ProviderRouter with a lightweight model (haiku)

Since T26 (2026-05-16): Option B implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from domain.language import detect_language

if TYPE_CHECKING:
    from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)

# In-memory cache: (entry_id, target_lang) -> translated_text
_translation_cache: dict[tuple[str, str], str] = {}

# Maximum cache size to prevent unbounded growth
_MAX_CACHE_SIZE = 500

# Translation prompt template (minimal, fast, no hallucination)
_TRANSLATION_PROMPT = (
    "Translate the following note from {source_lang} to {target_lang}. "
    "Keep meaning, keep tone, no additions, no explanations, no commentary. "
    "Return ONLY the translated text, nothing else.\n\n"
    "{text}"
)

_TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise translator. Translate exactly what is given. "
    "Do not add any information. Do not explain. Do not comment. "
    "Output ONLY the translation."
)


def clear_cache() -> None:
    """Clear the entire translation cache. Used in tests and /reset."""
    _translation_cache.clear()


def cache_size() -> int:
    """Return current cache size. Exposed for monitoring/tests."""
    return len(_translation_cache)


def _evict_oldest_if_needed() -> None:
    """Evict oldest entries if cache exceeds max size.

    Simple FIFO: remove first entries added (dict preserves insertion order).
    """
    while len(_translation_cache) > _MAX_CACHE_SIZE:
        oldest_key = next(iter(_translation_cache))
        del _translation_cache[oldest_key]


async def translate_entry(
    entry_id: str,
    content: str,
    target_lang: str,
    provider_router: "ProviderRouter",
    model: Optional[str] = None,
) -> str:
    """Translate a single memory entry to the target language.

    Args:
        entry_id: Unique ID of the memory entry (for caching).
        content: Original text content.
        target_lang: ISO 639-1 target language code.
        provider_router: Router for LLM calls.
        model: Optional model override (default: haiku for speed).

    Returns:
        Translated text, or original if translation fails or is unnecessary.
    """
    # Detect source language
    source_lang = detect_language(content)

    # Same language: no translation needed
    if source_lang == target_lang:
        return content

    # Check cache
    cache_key = (entry_id, target_lang)
    cached = _translation_cache.get(cache_key)
    if cached is not None:
        return cached

    # Translate via LLM
    try:
        prompt = _TRANSLATION_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=content,
        )

        result = await provider_router.route(
            prompt=prompt,
            system_prompt=_TRANSLATION_SYSTEM_PROMPT,
            model=model or "claude-haiku-4-5-20251001",
            timeout_seconds=15,
        )

        if result.error or not result.text.strip():
            log.warning(
                "Translation failed for entry %s (%s->%s): %s",
                entry_id,
                source_lang,
                target_lang,
                result.error or "empty response",
            )
            return content  # Fallback: return original

        translated = result.text.strip()

        # Cache the result
        _translation_cache[cache_key] = translated
        _evict_oldest_if_needed()

        log.debug(
            "Translated entry %s: %s->%s (%d->%d chars)",
            entry_id,
            source_lang,
            target_lang,
            len(content),
            len(translated),
        )
        return translated

    except Exception as e:
        log.warning(
            "Translation exception for entry %s (%s->%s): %s",
            entry_id,
            source_lang,
            target_lang,
            e,
        )
        return content  # Fallback: return original


async def translate_entries(
    entries: list[dict[str, Any]],
    target_lang: str,
    provider_router: "ProviderRouter",
    model: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Translate a list of memory entries for display.

    Non-destructive: returns new dicts with translated 'content' field.
    Original entries are NOT modified.

    Args:
        entries: List of memory entry dicts (must have 'id' and 'content').
        target_lang: Target language code.
        provider_router: Router for LLM calls.
        model: Optional model override.

    Returns:
        List of entry dicts with translated content.
        Original order preserved.
    """
    if not entries:
        return entries

    translated_entries: list[dict[str, Any]] = []
    for entry in entries:
        entry_id = entry.get("id", "unknown")
        content = entry.get("content", "")

        if not content:
            translated_entries.append(entry)
            continue

        translated_text = await translate_entry(
            entry_id=entry_id,
            content=content,
            target_lang=target_lang,
            provider_router=provider_router,
            model=model,
        )

        # Create a new dict (do not mutate original)
        translated_entry = {**entry, "content": translated_text}
        translated_entries.append(translated_entry)

    return translated_entries
