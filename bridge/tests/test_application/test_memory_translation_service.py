"""Tests for application.memory_translation_service (T26).

Covers:
    - Same-language short-circuit (no LLM call)
    - Successful translation with cache
    - Cache hit on repeated call
    - Provider failure fallback to original
    - Empty entries
    - translate_entries batch
    - Cache eviction at max size
    - Batch translation (single LLM call for multiple entries)
    - Parallel batches for large lists
    - Partial cache hits
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.memory_translation_service import (
    BATCH_SIZE,
    _MAX_CACHE_SIZE,
    _translation_cache,
    cache_size,
    clear_cache,
    translate_entries,
    translate_entry,
)
from infrastructure.providers.base import ProviderResponse


@pytest.fixture(autouse=True)
def _clear_translation_cache() -> None:
    """Clear the translation cache before each test."""
    clear_cache()


def _make_router(response_text: str = "translated", error: str = "") -> MagicMock:
    """Create a mock ProviderRouter that returns a fixed response."""
    router = MagicMock()
    router.route = AsyncMock(
        return_value=ProviderResponse(
            text=response_text,
            duration_seconds=0.5,
            provider_name="claude",
            error=error,
        )
    )
    return router


def _make_batch_router(translations: list[str], error: str = "") -> MagicMock:
    """Create a mock router that returns batch translations joined by the marker from the prompt.

    The UUID-based marker is generated at runtime, so this mock extracts it from
    the prompt and uses it to join the translations in the response.
    """
    import re

    async def _side_effect(**kwargs):
        prompt = kwargs.get("prompt", "")
        # Extract the dynamic marker from the prompt: "separated by '---NOTE-BREAK-<hex>---'"
        m = re.search(r"separated by '(---NOTE-BREAK-[a-f0-9]+---)'", prompt)
        if m:
            marker = m.group(1)
            text = marker.join(translations)
        else:
            # Single-entry fallback (no marker in prompt)
            text = translations[0] if translations else ""
        return ProviderResponse(
            text=text,
            duration_seconds=0.5,
            provider_name="claude",
            error=error,
        )

    router = MagicMock()
    router.route = AsyncMock(side_effect=_side_effect)
    return router


class TestTranslateEntry:
    """Tests for translate_entry()."""

    @pytest.mark.asyncio
    async def test_same_language_returns_original(self) -> None:
        """If source and target language match, no LLM call is made."""
        router = _make_router()
        result = await translate_entry(
            entry_id="ep_test1",
            content="Ich mag Delfine",
            target_lang="de",
            provider_router=router,
        )
        assert result == "Ich mag Delfine"
        router.route.assert_not_called()

    @pytest.mark.asyncio
    async def test_translation_calls_provider(self) -> None:
        """Translation makes an LLM call when languages differ."""
        router = _make_router(response_text="I like dolphins")
        result = await translate_entry(
            entry_id="ep_test2",
            content="Ich mag Delfine",
            target_lang="en",
            provider_router=router,
        )
        assert result == "I like dolphins"
        router.route.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_provider_call(self) -> None:
        """Second call with same entry+lang uses cache, no LLM call."""
        router = _make_router(response_text="I like dolphins")

        # First call: populates cache
        result1 = await translate_entry(
            entry_id="ep_test3",
            content="Ich mag Delfine",
            target_lang="en",
            provider_router=router,
        )
        assert result1 == "I like dolphins"
        assert router.route.call_count == 1

        # Second call: cache hit
        result2 = await translate_entry(
            entry_id="ep_test3",
            content="Ich mag Delfine",
            target_lang="en",
            provider_router=router,
        )
        assert result2 == "I like dolphins"
        assert router.route.call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_provider_error_returns_original(self) -> None:
        """When provider returns an error, original text is returned."""
        router = _make_router(response_text="", error="rate_limited")
        result = await translate_entry(
            entry_id="ep_test4",
            content="Ich mag Delfine",
            target_lang="en",
            provider_router=router,
        )
        assert result == "Ich mag Delfine"

    @pytest.mark.asyncio
    async def test_provider_exception_returns_original(self) -> None:
        """When provider raises an exception, original text is returned."""
        router = MagicMock()
        router.route = AsyncMock(side_effect=RuntimeError("connection lost"))
        result = await translate_entry(
            entry_id="ep_test5",
            content="Ich mag Delfine",
            target_lang="en",
            provider_router=router,
        )
        assert result == "Ich mag Delfine"

    @pytest.mark.asyncio
    async def test_different_target_langs_separate_cache_entries(self) -> None:
        """Same entry translated to different languages creates separate cache entries."""
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            lang_map = {1: "I like dolphins", 2: "J'aime les dauphins"}
            return ProviderResponse(
                text=lang_map.get(call_count, "translated"),
                duration_seconds=0.1,
                provider_name="claude",
            )

        router = MagicMock()
        router.route = AsyncMock(side_effect=_side_effect)

        en = await translate_entry(
            entry_id="ep_test6",
            content="Ich mag Delfine",
            target_lang="en",
            provider_router=router,
        )
        fr = await translate_entry(
            entry_id="ep_test6",
            content="Ich mag Delfine",
            target_lang="fr",
            provider_router=router,
        )
        assert en == "I like dolphins"
        assert fr == "J'aime les dauphins"
        assert cache_size() == 2


class TestTranslateEntries:
    """Tests for translate_entries() batch function."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self) -> None:
        """Empty input returns empty output."""
        router = _make_router()
        result = await translate_entries([], "en", router)
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_translates_all_entries(self) -> None:
        """All entries in a batch are translated."""
        router = _make_batch_router(["I like cats", "I like dogs"])
        entries = [
            {"id": "ep_a", "content": "Ich mag Katzen"},
            {"id": "ep_b", "content": "Ich mag Hunde"},
        ]
        result = await translate_entries(entries, "en", router)
        assert len(result) == 2
        assert result[0]["content"] == "I like cats"
        assert result[1]["content"] == "I like dogs"

    @pytest.mark.asyncio
    async def test_batch_preserves_non_content_fields(self) -> None:
        """Non-content fields (id, timestamp, etc.) are preserved."""
        router = _make_router(response_text="translated")
        entries = [
            {
                "id": "ep_c",
                "content": "Ich mag Fische",
                "timestamp": "2026-05-16T10:00:00Z",
                "importance": 7,
            }
        ]
        result = await translate_entries(entries, "en", router)
        assert result[0]["id"] == "ep_c"
        assert result[0]["timestamp"] == "2026-05-16T10:00:00Z"
        assert result[0]["importance"] == 7

    @pytest.mark.asyncio
    async def test_batch_does_not_mutate_originals(self) -> None:
        """Original entry dicts are not modified."""
        router = _make_router(response_text="translated")
        original = {"id": "ep_d", "content": "Das ist ein deutscher Text"}
        entries = [original]
        result = await translate_entries(entries, "en", router)
        assert original["content"] == "Das ist ein deutscher Text"
        assert result[0]["content"] == "translated"

    @pytest.mark.asyncio
    async def test_entry_with_empty_content_passes_through(self) -> None:
        """Entries with empty content are returned as-is."""
        router = _make_router()
        entries = [{"id": "ep_e", "content": ""}]
        result = await translate_entries(entries, "en", router)
        assert result[0]["content"] == ""
        router.route.assert_not_called()


class TestBatchTranslation:
    """Tests for batch translation performance optimization."""

    @pytest.mark.asyncio
    async def test_batch_translates_multiple_entries_in_one_call(self) -> None:
        """Multiple entries (<=BATCH_SIZE) result in exactly one LLM call."""
        router = _make_batch_router(
            ["I like cats", "I like dogs", "I like fish", "I like birds"]
        )

        entries = [
            {"id": "ep_b1", "content": "Ich mag Katzen"},
            {"id": "ep_b2", "content": "Ich mag Hunde"},
            {"id": "ep_b3", "content": "Ich mag Fische"},
            {"id": "ep_b4", "content": "Ich mag Voegel"},
        ]

        result = await translate_entries(entries, "en", router)

        # Only 1 LLM call for all 4 entries
        assert router.route.call_count == 1
        assert len(result) == 4
        assert result[0]["content"] == "I like cats"
        assert result[1]["content"] == "I like dogs"
        assert result[2]["content"] == "I like fish"
        assert result[3]["content"] == "I like birds"

    @pytest.mark.asyncio
    async def test_batch_parses_break_marker_correctly(self) -> None:
        """Break marker parsing correctly splits and trims translations."""
        # Include extra whitespace around translations
        router = _make_batch_router([" First translation \n", "  Second translation  "])

        entries = [
            {"id": "ep_p1", "content": "Das ist der erste deutsche Satz"},
            {"id": "ep_p2", "content": "Das ist der zweite deutsche Satz"},
        ]

        result = await translate_entries(entries, "en", router)

        assert result[0]["content"] == "First translation"
        assert result[1]["content"] == "Second translation"

    @pytest.mark.asyncio
    async def test_batch_falls_back_on_parse_error(self) -> None:
        """When LLM returns wrong number of segments, originals are returned."""
        # Return only 2 translations for 3 entries (mismatch triggers fallback)
        router = _make_batch_router(["Translation 1", "Translation 2"])

        entries = [
            {"id": "ep_f1", "content": "Erster deutscher Satz"},
            {"id": "ep_f2", "content": "Zweiter deutscher Satz"},
            {"id": "ep_f3", "content": "Dritter deutscher Satz"},
        ]

        result = await translate_entries(entries, "en", router)

        # Fallback: return originals
        assert result[0]["content"] == "Erster deutscher Satz"
        assert result[1]["content"] == "Zweiter deutscher Satz"
        assert result[2]["content"] == "Dritter deutscher Satz"

    @pytest.mark.asyncio
    async def test_parallel_batches_for_large_lists(self) -> None:
        """Lists larger than BATCH_SIZE are split into parallel batches."""
        import re

        num_entries = 25
        call_count = 0

        async def _batch_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            prompt_text = kwargs.get("prompt", "")
            # Extract dynamic marker from prompt
            marker_match = re.search(
                r"separated by '(---NOTE-BREAK-[a-f0-9]+---)'", prompt_text
            )
            marker = marker_match.group(1) if marker_match else "---FALLBACK---"
            # Parse how many notes are in this batch
            m = re.search(r"Translate the following (\d+) notes", prompt_text)
            num_in_batch = int(m.group(1)) if m else 1
            translations = [f"Translated note {i}" for i in range(num_in_batch)]
            return ProviderResponse(
                text=marker.join(translations),
                duration_seconds=0.5,
                provider_name="claude",
            )

        router = MagicMock()
        router.route = AsyncMock(side_effect=_batch_side_effect)

        entries = [
            {
                "id": f"ep_large_{i}",
                "content": f"Das ist mein deutscher Satz Nummer {i}",
            }
            for i in range(num_entries)
        ]

        result = await translate_entries(entries, "en", router)

        # 25 entries / BATCH_SIZE(10) = 3 batches
        expected_batches = (num_entries + BATCH_SIZE - 1) // BATCH_SIZE
        assert call_count == expected_batches
        assert len(result) == num_entries
        # All entries should be translated (not original)
        for entry in result:
            assert entry["content"].startswith("Translated note")

    @pytest.mark.asyncio
    async def test_cache_hits_skip_batch_call(self) -> None:
        """When all entries are cached, no LLM call is made."""
        router = _make_router()

        entries = [
            {"id": "ep_ch1", "content": "Ich mag Katzen"},
            {"id": "ep_ch2", "content": "Ich mag Hunde"},
            {"id": "ep_ch3", "content": "Ich mag Fische"},
        ]

        # Pre-populate cache (3-tuple: entry_id, target_lang, user_id)
        _translation_cache[("ep_ch1", "en", 0)] = "I like cats"
        _translation_cache[("ep_ch2", "en", 0)] = "I like dogs"
        _translation_cache[("ep_ch3", "en", 0)] = "I like fish"

        result = await translate_entries(entries, "en", router)

        # No LLM call needed
        router.route.assert_not_called()
        assert result[0]["content"] == "I like cats"
        assert result[1]["content"] == "I like dogs"
        assert result[2]["content"] == "I like fish"

    @pytest.mark.asyncio
    async def test_partial_cache_hits_only_translate_remainder(self) -> None:
        """When some entries are cached, only uncached ones go to LLM."""
        # 3 entries cached (3-tuple: entry_id, target_lang, user_id)
        _translation_cache[("ep_pc1", "en", 0)] = "Cached translation 1"
        _translation_cache[("ep_pc2", "en", 0)] = "Cached translation 2"
        _translation_cache[("ep_pc3", "en", 0)] = "Cached translation 3"

        # Router that dynamically builds response with the UUID marker from prompt
        router = _make_batch_router(["Fresh translation 4", "Fresh translation 5"])

        entries = [
            {"id": "ep_pc1", "content": "Ich habe einen deutschen Text hier"},
            {"id": "ep_pc2", "content": "Ich habe noch einen deutschen Text hier"},
            {"id": "ep_pc3", "content": "Das ist der dritte deutsche Satz"},
            {"id": "ep_pc4", "content": "Das ist mein vierter deutscher Satz"},
            {"id": "ep_pc5", "content": "Das ist mein fuenfter deutscher Satz"},
        ]

        result = await translate_entries(entries, "en", router)

        # Only 1 LLM call for the 2 uncached entries
        assert router.route.call_count == 1
        assert result[0]["content"] == "Cached translation 1"
        assert result[1]["content"] == "Cached translation 2"
        assert result[2]["content"] == "Cached translation 3"
        assert result[3]["content"] == "Fresh translation 4"
        assert result[4]["content"] == "Fresh translation 5"

        # Newly translated entries should now be cached (3-tuple key)
        assert _translation_cache[("ep_pc4", "en", 0)] == "Fresh translation 4"
        assert _translation_cache[("ep_pc5", "en", 0)] == "Fresh translation 5"


class TestCacheManagement:
    """Tests for cache eviction and management."""

    @pytest.mark.asyncio
    async def test_cache_eviction_at_max_size(self) -> None:
        """Cache evicts oldest entries when exceeding max size."""
        router = _make_router(response_text="translated")

        # Fill cache beyond max (3-tuple: entry_id, target_lang, user_id)
        for i in range(_MAX_CACHE_SIZE + 10):
            _translation_cache[(f"ep_{i}", "xx", 0)] = f"cached_{i}"

        # Trigger eviction by translating one more
        await translate_entry(
            entry_id="ep_overflow",
            content="Ich mag Delfine",
            target_lang="en",
            provider_router=router,
        )

        assert cache_size() <= _MAX_CACHE_SIZE

    def test_clear_cache(self) -> None:
        """clear_cache() empties the cache."""
        _translation_cache[("ep_x", "en", 0)] = "cached"
        assert cache_size() == 1
        clear_cache()
        assert cache_size() == 0
