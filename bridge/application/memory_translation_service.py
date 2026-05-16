"""Memory translation service: translates memory entries on-the-fly for /memory display.

Design:
    - Original entries in DB are NEVER modified
    - Translation happens only at display time (/memory command)
    - Caching: translated text is cached per (entry_id, target_lang)
    - Language detection: uses domain.language to detect source language
    - Fallback: if translation fails, return original text
    - Provider: uses ProviderRouter with a lightweight model (haiku)
    - Performance: batch translation (single LLM call for multiple entries)
      with parallel chunks for large lists

Since T26 (2026-05-16): Option B implementation.
Performance fix (2026-05-16): Batch + Parallel + Cache.
"""

from __future__ import annotations

import asyncio
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

# Max entries per single LLM batch call
BATCH_SIZE = 10

# Break marker for batch translation parsing
_BREAK_MARKER = "---NOTE-BREAK---"

# Translation prompt template (minimal, fast, no hallucination) - single entry
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
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
) -> str:
    """Translate a single memory entry to the target language.

    Args:
        entry_id: Unique ID of the memory entry (for caching).
        content: Original text content.
        target_lang: ISO 639-1 target language code.
        provider_router: Router for LLM calls.
        model: Optional model override (default: haiku for speed).
        user_id: Telegram user ID (required by claude_persistent provider).
        chat_id: Telegram chat ID (required by claude_persistent provider).

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

    log.info(
        "translate_entry: source_lang=%s, target_lang=%s, content_preview=%s",
        source_lang,
        target_lang,
        content[:30],
    )

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
            user_id=user_id,
            chat_id=chat_id,
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


async def _batch_translate(
    entries_with_source: list[tuple[dict[str, Any], str]],
    target_lang: str,
    provider_router: "ProviderRouter",
    model: Optional[str] = None,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
) -> dict[str, str]:
    """Translate a batch of entries in a single LLM call.

    Args:
        entries_with_source: List of (entry_dict, source_lang) tuples.
        target_lang: Target language code.
        provider_router: Router for LLM calls.
        model: Optional model override.
        user_id: Telegram user ID.
        chat_id: Telegram chat ID.

    Returns:
        dict mapping entry_id -> translated_text.
        On failure, returns original content for each entry.
    """
    if not entries_with_source:
        return {}

    # Single entry: use simpler prompt format
    if len(entries_with_source) == 1:
        entry, source_lang = entries_with_source[0]
        entry_id = entry.get("id", "unknown")
        content = entry.get("content", "")
        translated = await translate_entry(
            entry_id=entry_id,
            content=content,
            target_lang=target_lang,
            provider_router=provider_router,
            model=model,
            user_id=user_id,
            chat_id=chat_id,
        )
        return {entry_id: translated}

    notes_text = _BREAK_MARKER.join(
        [entry.get("content", "") for entry, _ in entries_with_source]
    )

    prompt = (
        f"Translate the following {len(entries_with_source)} notes to {target_lang}. "
        f"Return the translations in the EXACT SAME ORDER, separated by '{_BREAK_MARKER}'. "
        f"Keep meaning, keep tone, no additions, no commentary. "
        f"Keep proper nouns (brand names, product names like 'Flat White', 'Python', "
        f"personal names) in their original form. "
        f"Return ONLY the translations, separated by the break marker.\n\n"
        f"{notes_text}"
    )

    try:
        result = await provider_router.route(
            prompt=prompt,
            system_prompt=_TRANSLATION_SYSTEM_PROMPT,
            model=model or "claude-haiku-4-5-20251001",
            timeout_seconds=30,
            user_id=user_id,
            chat_id=chat_id,
        )

        if result.error or not result.text.strip():
            log.warning("Batch translation failed: %s", result.error or "empty")
            return {
                entry.get("id", "unknown"): entry.get("content", "")
                for entry, _ in entries_with_source
            }

        translations = result.text.strip().split(_BREAK_MARKER)

        if len(translations) != len(entries_with_source):
            log.warning(
                "Batch translation parse error: expected %d, got %d",
                len(entries_with_source),
                len(translations),
            )
            return {
                entry.get("id", "unknown"): entry.get("content", "")
                for entry, _ in entries_with_source
            }

        return {
            entry.get("id", "unknown"): translation.strip()
            for (entry, _), translation in zip(entries_with_source, translations)
        }

    except Exception as e:
        log.warning("Batch translation exception: %s", e)
        return {
            entry.get("id", "unknown"): entry.get("content", "")
            for entry, _ in entries_with_source
        }


async def translate_entries(
    entries: list[dict[str, Any]],
    target_lang: str,
    provider_router: "ProviderRouter",
    model: Optional[str] = None,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Translate a list of memory entries for display.

    Non-destructive: returns new dicts with translated 'content' field.
    Original entries are NOT modified.

    Performance: uses batch translation (single LLM call per batch of entries)
    with parallel execution for large lists (>BATCH_SIZE).

    Args:
        entries: List of memory entry dicts (must have 'id' and 'content').
        target_lang: Target language code.
        provider_router: Router for LLM calls.
        model: Optional model override.
        user_id: Telegram user ID (required by claude_persistent provider).
        chat_id: Telegram chat ID (required by claude_persistent provider).

    Returns:
        List of entry dicts with translated content.
        Original order preserved.
    """
    if not entries:
        return entries

    # Phase 1: Separate cache hits, same-language, and entries needing translation
    cache_hits: dict[str, str] = {}
    to_translate: list[tuple[dict[str, Any], str]] = []
    passthrough_ids: set[str] = set()  # entries with empty content

    for entry in entries:
        entry_id = entry.get("id", "unknown")
        content = entry.get("content", "")

        if not content:
            passthrough_ids.add(entry_id)
            continue

        # Check cache first
        cache_key = (entry_id, target_lang)
        cached = _translation_cache.get(cache_key)
        if cached is not None:
            cache_hits[entry_id] = cached
            continue

        # Check same-language short-circuit
        source_lang = detect_language(content)
        if source_lang == target_lang:
            cache_hits[entry_id] = content
            continue

        to_translate.append((entry, source_lang))

    # Phase 2: Batch-translate cache misses
    translated: dict[str, str] = {}

    if to_translate:
        if len(to_translate) <= BATCH_SIZE:
            translated = await _batch_translate(
                to_translate, target_lang, provider_router, model, user_id, chat_id
            )
        else:
            # Split into chunks and run in parallel
            chunks = [
                to_translate[i : i + BATCH_SIZE]
                for i in range(0, len(to_translate), BATCH_SIZE)
            ]
            results = await asyncio.gather(
                *[
                    _batch_translate(
                        chunk, target_lang, provider_router, model, user_id, chat_id
                    )
                    for chunk in chunks
                ]
            )
            for r in results:
                translated.update(r)

    # Phase 3: Cache newly translated entries
    for entry_id, text in translated.items():
        _translation_cache[(entry_id, target_lang)] = text
        _evict_oldest_if_needed()

    # Phase 4: Reassemble result in original order
    result: list[dict[str, Any]] = []
    for entry in entries:
        entry_id = entry.get("id", "unknown")
        content = entry.get("content", "")

        if entry_id in passthrough_ids:
            result.append(entry)
        elif entry_id in cache_hits:
            result.append({**entry, "content": cache_hits[entry_id]})
        elif entry_id in translated:
            result.append({**entry, "content": translated[entry_id]})
        else:
            result.append({**entry, "content": content})  # Fallback

    return result
