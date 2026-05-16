"""Tests for application.memory_translation_service (T26).

Covers:
    - Same-language short-circuit (no LLM call)
    - Successful translation with cache
    - Cache hit on repeated call
    - Provider failure fallback to original
    - Empty entries
    - translate_entries batch
    - Cache eviction at max size
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.memory_translation_service import (
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
        router = _make_router(response_text="translated text")
        entries = [
            {"id": "ep_a", "content": "Ich mag Katzen"},
            {"id": "ep_b", "content": "Ich mag Hunde"},
        ]
        result = await translate_entries(entries, "en", router)
        assert len(result) == 2
        assert all(e["content"] == "translated text" for e in result)

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


class TestCacheManagement:
    """Tests for cache eviction and management."""

    @pytest.mark.asyncio
    async def test_cache_eviction_at_max_size(self) -> None:
        """Cache evicts oldest entries when exceeding max size."""
        router = _make_router(response_text="translated")

        # Fill cache beyond max
        for i in range(_MAX_CACHE_SIZE + 10):
            # Force different source/target to bypass same-language check
            _translation_cache[(f"ep_{i}", "xx")] = f"cached_{i}"

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
        _translation_cache[("ep_x", "en")] = "cached"
        assert cache_size() == 1
        clear_cache()
        assert cache_size() == 0
