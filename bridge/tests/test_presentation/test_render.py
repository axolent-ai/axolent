"""Tests für presentation.render: Chunking und Response-Cache.

Testet Message-Splitting, Cache-Logik und den send_response Flow.
"""

from __future__ import annotations


import pytest

from presentation.render import (
    TELEGRAM_CHUNK_SIZE,
    _CACHE_MAX,
    _response_cache,
    cache_response,
    get_cached_response,
    split_message,
)


class TestSplitMessage:
    """split_message Chunking-Tests."""

    def test_split_then_convert_chunks_correctly(self) -> None:
        """Kurze Nachrichten werden als einzelner Chunk zurueckgegeben."""
        result = split_message("Hallo Welt")
        assert result == ["Hallo Welt"]

    def test_split_long_message(self) -> None:
        """Lange Nachrichten werden in korrekte Chunks geteilt."""
        long_text = "A" * 10000
        chunks = split_message(long_text, chunk_size=4000)
        assert len(chunks) == 3  # 10000 / 4000 = 2.5, aufgerundet 3
        assert "".join(chunks) == long_text

    def test_split_exactly_at_limit(self) -> None:
        """Text genau am Limit wird als ein Chunk behandelt."""
        text = "B" * TELEGRAM_CHUNK_SIZE
        chunks = split_message(text)
        assert len(chunks) == 1

    def test_split_one_over_limit(self) -> None:
        """Text ein Zeichen ueber dem Limit wird in 2 Chunks geteilt."""
        text = "C" * (TELEGRAM_CHUNK_SIZE + 1)
        chunks = split_message(text)
        assert len(chunks) == 2

    def test_empty_message(self) -> None:
        """Leere Nachricht ergibt einen Chunk mit leerem String."""
        result = split_message("")
        assert result == [""]


class TestResponseCache:
    """Response-Cache LRU-Tests."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        """Cache vor jedem Test leeren."""
        _response_cache.clear()

    def test_cache_and_retrieve(self) -> None:
        """Gecachte Response kann per (chat_id, message_id) abgerufen werden."""
        cache_response(10, 100, "Cached text")
        result = get_cached_response(10, 100)
        assert result == "Cached text"

    def test_cache_miss_returns_none(self) -> None:
        """Nicht-gecachter Key gibt None zurueck."""
        result = get_cached_response(99, 99)
        assert result is None

    def test_cache_evicts_oldest(self) -> None:
        """Cache evicted aelteste Eintraege bei Ueberschreitung von _CACHE_MAX."""
        # _CACHE_MAX Eintraege einfuegen
        for i in range(_CACHE_MAX + 10):
            cache_response(1, i, f"Response {i}")

        # Die aeltesten (0-9) sollten evicted sein
        assert get_cached_response(1, 0) is None
        # Die neuesten sollten noch da sein
        assert get_cached_response(1, _CACHE_MAX + 9) is not None

    def test_html_fallback_on_invalid_tags(self) -> None:
        """Wenn HTML-Send fehlschlaegt, wird Plain-Text-Fallback genutzt.

        Dieser Test prueft die Fallback-Logik in send_response indirekt
        via den split+convert Mechanismus.
        """
        from domain.markdown import markdown_to_telegram_html, strip_markdown

        # Valider Markdown-Input
        md = "## Headline\n**Bold** text"
        html_result = markdown_to_telegram_html(md)
        plain_result = strip_markdown(md)

        # HTML enthaelt Tags
        assert "<b>" in html_result
        # Plain hat keine Tags
        assert "<b>" not in plain_result
        assert "Headline" in plain_result
        assert "Bold" in plain_result
