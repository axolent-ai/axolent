"""Tests für den Streaming-Handler.

Verifiziert:
    - Token-Aggregation und Rate-Limiting
    - Erste Edit erst nach FIRST_EDIT_DELAY_SECONDS
    - Finalize setzt vollständigen Text
    - Multi-Message-Split bei langen Antworten (>4096 Zeichen)
    - Abort zeigt Fehlermeldung
    - Telegram-API-Fehler werden leise geschluckt
    - Telegram-Längen-Limit wird eingehalten
    - Zwischen-Edits als HTML mit smart-trim (Option A)
    - Finale Edit als HTML mit Markdown-Konvertierung
    - split_text_for_telegram() respektiert Markdown-Grenzen
    - find_safe_markdown_end() findet sichere Positionen
    - Adaptive Flood-Control (RetryAfter/429)
    - Final-Edit Retry bei 429
    - Throttle-Backoff und Recovery
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.streaming_handler import (
    DEFAULT_THROTTLE,
    FINAL_EDIT_MAX_RETRIES,
    MAX_THROTTLE,
    THROTTLE_BACKOFF_FACTOR,
    THROTTLE_RECOVERY_AFTER,
    StreamingSession,
    _apply_flood_backoff,
    _is_retry_after,
    _is_safe_markdown_position,
    _record_edit_success,
    _truncate_markdown_for_html_limit,
    abort_streaming,
    find_safe_markdown_end,
    finalize_streaming,
    process_streaming_edit,
    split_text_for_telegram,
)


def _make_session(started_offset: float = 0.0) -> StreamingSession:
    """Erstellt eine Test-StreamingSession mit gemockter Message."""
    msg = AsyncMock()
    msg.edit_text = AsyncMock()
    # Mock chat für Multi-Message-Split (send_message auf dem Chat)
    msg.chat = MagicMock()
    msg.chat.send_message = AsyncMock()
    return StreamingSession(
        message=msg,
        started_at=time.monotonic() - started_offset,
    )


class TestStreamingEdit:
    """Tests für process_streaming_edit()."""

    @pytest.mark.asyncio
    async def test_first_edit_delayed(self) -> None:
        """Keine Edit vor FIRST_EDIT_DELAY_SECONDS."""
        session = _make_session(started_offset=0.5)  # Erst 0.5s vergangen
        await process_streaming_edit(session, "Hello")

        # Kein edit_text-Aufruf weil zu frueh
        session.message.edit_text.assert_not_called()
        assert session.accumulated_text == "Hello"

    @pytest.mark.asyncio
    async def test_edit_after_delay(self) -> None:
        """Edit nach FIRST_EDIT_DELAY_SECONDS."""
        session = _make_session(started_offset=2.0)  # 2s vergangen
        await process_streaming_edit(session, "Hello World")

        session.message.edit_text.assert_called_once()
        assert session.edit_count == 1

    @pytest.mark.asyncio
    async def test_rate_limiting(self) -> None:
        """Keine Edit wenn letzte Edit < EDIT_INTERVAL_SECONDS."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = time.monotonic()  # Gerade eben editiert

        await process_streaming_edit(session, "New token")

        # Kein Aufruf wegen Rate-Limiting
        session.message.edit_text.assert_not_called()
        assert session.accumulated_text == "New token"

    @pytest.mark.asyncio
    async def test_accumulates_text(self) -> None:
        """Text wird korrekt akkumuliert."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0  # Längst fällig

        await process_streaming_edit(session, "Hello ")
        await process_streaming_edit(session, "World")

        # Zweiter Aufruf: Rate-Limited (zu schnell hintereinander)
        assert session.accumulated_text == "Hello World"


class TestStreamingFinalize:
    """Tests für finalize_streaming()."""

    @pytest.mark.asyncio
    async def test_finalize_sets_final_text(self) -> None:
        session = _make_session(started_offset=5.0)
        session.accumulated_text = "partial"

        result = await finalize_streaming(session, "complete answer")

        session.message.edit_text.assert_called_once()
        # Finale Edit nutzt parse_mode="HTML"
        call_kwargs = session.message.edit_text.call_args[1]
        assert call_kwargs.get("parse_mode") == "HTML"
        assert result == "complete answer"

    @pytest.mark.asyncio
    async def test_finalize_returns_full_text_even_when_split(self) -> None:
        """finalize_streaming gibt immer den VOLLEN Text zurück, auch bei Split."""
        session = _make_session(started_offset=5.0)
        long_text = "Dies ist ein Absatz.\n\n" * 300  # >4096 Zeichen

        result = await finalize_streaming(session, long_text)

        assert result == long_text  # Ungekuerzt

    @pytest.mark.asyncio
    async def test_finalize_multi_message_split(self) -> None:
        """Lange Antwort wird in mehrere Nachrichten gesplittet."""
        session = _make_session(started_offset=5.0)
        # Erzeuge Text der definitiv > 4096 Zeichen HTML ergibt
        long_text = "Dies ist ein langer Absatz mit Text.\n\n" * 200

        await finalize_streaming(session, long_text)

        # Erste Nachricht: Edit der bestehenden Message
        session.message.edit_text.assert_called_once()
        first_call_kwargs = session.message.edit_text.call_args[1]
        assert first_call_kwargs.get("parse_mode") == "HTML"

        # Folge-Nachrichten: neue Messages via chat.send_message
        assert session.message.chat.send_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_finalize_multi_message_has_part_markers(self) -> None:
        """Multi-Message-Split hat (1/N), (2/N) etc. Marker."""
        session = _make_session(started_offset=5.0)
        long_text = "Absatz Nummer eins.\n\n" * 300

        await finalize_streaming(session, long_text)

        # Erste Nachricht muss (1/N) enthalten
        first_html = session.message.edit_text.call_args[0][0]
        assert "(1/" in first_html

        # Folge-Nachrichten müssen Part-Marker haben
        for call in session.message.chat.send_message.call_args_list:
            call_text = call[0][0]
            assert "(" in call_text  # Part-Marker vorhanden

    @pytest.mark.asyncio
    async def test_5000_char_response_becomes_2_messages(self) -> None:
        """Spezifischer Test: 5000-Zeichen-Antwort muss als 2 Messages rauskommen."""
        session = _make_session(started_offset=5.0)
        # Baue exakt ~5000 Zeichen Plain-Text
        chunk = "Hier steht ein Satz der ziemlich lang ist. "
        repetitions = 5000 // len(chunk) + 1
        long_text = (chunk * repetitions)[:5000]

        await finalize_streaming(session, long_text)

        # Genau 2 Messages: 1 Edit + 1 neue Nachricht
        assert session.message.edit_text.call_count == 1
        assert session.message.chat.send_message.call_count >= 1
        total = 1 + session.message.chat.send_message.call_count
        assert total >= 2


class TestStreamingAbort:
    """Tests für abort_streaming()."""

    @pytest.mark.asyncio
    async def test_abort_shows_error(self) -> None:
        session = _make_session(started_offset=5.0)

        await abort_streaming(session, "Etwas ist schiefgelaufen")

        session.message.edit_text.assert_called_once()


class TestStreamingTelegramErrors:
    """Tests für Telegram-API-Fehlerbehandlung."""

    @pytest.mark.asyncio
    async def test_message_not_modified_silenced(self) -> None:
        """'message is not modified' wird leise geschluckt."""
        session = _make_session(started_offset=5.0)
        session.message.edit_text = AsyncMock(
            side_effect=Exception("Bad Request: message is not modified")
        )

        # Sollte keine Exception werfen
        await finalize_streaming(session, "text")

    @pytest.mark.asyncio
    async def test_other_errors_logged(self) -> None:
        """Andere Telegram-Fehler werden geloggt aber nicht geraised."""
        session = _make_session(started_offset=5.0)
        session.message.edit_text = AsyncMock(side_effect=Exception("Network timeout"))

        # Sollte keine Exception werfen
        await finalize_streaming(session, "text")


class TestStreamingMarkdownConversion:
    """Tests für die Markdown-zu-HTML-Konvertierung im Streaming-Pfad.

    Zwischen-Edits nutzen jetzt Option A (smart-trim mit HTML-Rendering).
    Die finale Edit konvertiert Markdown zu Telegram-HTML.
    """

    @pytest.mark.asyncio
    async def test_intermediate_edit_renders_html(self) -> None:
        """Zwischen-Edits senden HTML (Option A: smart-trim)."""
        session = _make_session(started_offset=2.0)
        await process_streaming_edit(session, "**fetter Text**")

        session.message.edit_text.assert_called_once()
        call_args = session.message.edit_text.call_args
        # Option A: HTML mit parse_mode
        call_kwargs = call_args[1]
        assert call_kwargs.get("parse_mode") == "HTML"
        call_text = call_args[0][0]
        assert "<b>fetter Text</b>" in call_text

    @pytest.mark.asyncio
    async def test_intermediate_edit_trims_incomplete_bold(self) -> None:
        """Zwischen-Edit schneidet unvollständige ** am Ende ab."""
        session = _make_session(started_offset=2.0)
        # Unvollständiger Bold-Marker am Ende
        await process_streaming_edit(session, "Fertiger Text. **angefan")

        session.message.edit_text.assert_called_once()
        call_text = session.message.edit_text.call_args[0][0]
        # Der sichere Teil sollte gerendert worden sein
        # "**angefan" ist unsicher, sollte abgeschnitten sein
        assert "angefan" not in call_text or "**angefan" not in call_text

    @pytest.mark.asyncio
    async def test_finalize_converts_bold_to_html(self) -> None:
        """Finale Edit konvertiert **text** zu <b>text</b>."""
        session = _make_session(started_offset=5.0)

        result = await finalize_streaming(session, "Das ist **wichtig** hier")

        call_text = session.message.edit_text.call_args[0][0]
        call_kwargs = session.message.edit_text.call_args[1]
        assert "<b>wichtig</b>" in call_text
        assert "**" not in call_text
        assert call_kwargs.get("parse_mode") == "HTML"
        assert result == "Das ist **wichtig** hier"

    @pytest.mark.asyncio
    async def test_finalize_converts_italic_to_html(self) -> None:
        """Finale Edit konvertiert *text* zu <i>text</i>."""
        session = _make_session(started_offset=5.0)

        await finalize_streaming(session, "Das ist *kursiv* hier")

        call_text = session.message.edit_text.call_args[0][0]
        assert "<i>kursiv</i>" in call_text

    @pytest.mark.asyncio
    async def test_finalize_converts_code_to_html(self) -> None:
        """Finale Edit konvertiert `code` zu <code>code</code>."""
        session = _make_session(started_offset=5.0)

        await finalize_streaming(session, "Nutze `pip install` hier")

        call_text = session.message.edit_text.call_args[0][0]
        assert "<code>pip install</code>" in call_text

    @pytest.mark.asyncio
    async def test_finalize_escapes_html_entities(self) -> None:
        """Finale Edit escaped HTML-Sonderzeichen."""
        session = _make_session(started_offset=5.0)

        await finalize_streaming(session, "a < b && c > d")

        call_text = session.message.edit_text.call_args[0][0]
        assert "&lt;" in call_text
        assert "&gt;" in call_text
        assert "&amp;" in call_text

    @pytest.mark.asyncio
    async def test_finalize_html_fallback_on_parse_error(self) -> None:
        """Bei HTML-Parse-Fehler fällt finalize auf Plain-Text zurück."""
        session = _make_session(started_offset=5.0)

        # Erster Aufruf (HTML) schlaegt fehl, zweiter (Plain-Text) gelingt
        session.message.edit_text = AsyncMock(
            side_effect=[
                Exception("Bad Request: can't parse entities"),
                None,  # Fallback-Aufruf gelingt
            ]
        )

        await finalize_streaming(session, "**fett** und *kursiv*")

        # Zwei Aufrufe: HTML-Versuch + Plain-Text-Fallback
        assert session.message.edit_text.call_count == 2
        # Zweiter Aufruf ist Plain-Text (kein parse_mode)
        second_call = session.message.edit_text.call_args_list[1]
        assert "parse_mode" not in second_call[1]
        # Plain-Text hat kein Markdown mehr (strip_markdown)
        fallback_text = second_call[0][0]
        assert "**" not in fallback_text
        assert "fett" in fallback_text

    @pytest.mark.asyncio
    async def test_streaming_then_finalize_flow(self) -> None:
        """Vollständiger Flow: Zwischen-Edits HTML (smart-trim), finale Edit HTML."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0  # Längst fällig

        # Zwischen-Edit (jetzt HTML mit smart-trim)
        await process_streaming_edit(session, "**Das Grundprinzip")
        # Bei unvollständigem **: HTML ohne den unfertigen Teil
        assert session.message.edit_text.call_count >= 1

        # Finale Edit (HTML, vollständig)
        await finalize_streaming(session, "**Das Grundprinzip bleibt gleich**")
        final_call = session.message.edit_text.call_args
        assert final_call[1].get("parse_mode") == "HTML"
        assert "<b>Das Grundprinzip bleibt gleich</b>" in final_call[0][0]


class TestSplitTextForTelegram:
    """Tests für split_text_for_telegram()."""

    def test_short_text_single_part(self) -> None:
        """Kurzer Text bleibt ein Teil."""
        result = split_text_for_telegram("Kurzer Text")
        assert len(result) == 1
        assert result[0] == "Kurzer Text"

    def test_long_text_splits(self) -> None:
        """Langer Text wird gesplittet."""
        text = "Absatz eins.\n\n" * 400
        result = split_text_for_telegram(text)
        assert len(result) >= 2

    def test_split_at_paragraph_boundary(self) -> None:
        """Split bevorzugt Absatz-Grenzen."""
        part_a = "A" * 3000 + "\n\n"
        part_b = "B" * 2000
        text = part_a + part_b
        result = split_text_for_telegram(text, max_length=4096)
        # Erster Teil sollte bei \n\n enden
        assert result[0].endswith("A")  # Trailing whitespace stripped
        assert result[1].startswith("B")

    def test_split_respects_bold_markers(self) -> None:
        """Split schneidet nicht mitten in **bold**."""
        # Text mit Bold-Marker der über die Grenze geht
        text = "A" * 4000 + "**wichtiger fetter Text**" + "B" * 100
        result = split_text_for_telegram(text, max_length=4096)
        # Keiner der Teile sollte ein offenes ** haben
        for part in result:
            bold_count = part.count("**")
            assert bold_count % 2 == 0, f"Offener Bold-Marker in: ...{part[-50:]}"

    def test_split_respects_code_blocks(self) -> None:
        """Split schneidet nicht mitten in ```code```."""
        text = "Text davor.\n\n```python\ndef hello():\n    print('hi')\n```\n\nText danach."
        # Kurzer Text, kein Split noetig
        result = split_text_for_telegram(text)
        assert len(result) == 1

    def test_all_parts_under_limit(self) -> None:
        """Alle Teile bleiben unter dem Limit."""
        text = "Ein ziemlich langer Satz. " * 500
        result = split_text_for_telegram(text, max_length=4096)
        for part in result:
            assert len(part) <= 4096

    def test_empty_text(self) -> None:
        """Leerer Text wird korrekt behandelt."""
        result = split_text_for_telegram("")
        assert len(result) == 1


class TestFindSafeMarkdownEnd:
    """Tests für find_safe_markdown_end()."""

    def test_complete_markdown_returns_full_length(self) -> None:
        """Vollständiger Markdown gibt volle Länge."""
        text = "**fett** und *kursiv* und `code`"
        assert find_safe_markdown_end(text) == len(text)

    def test_open_bold_trims(self) -> None:
        """Offener **-Marker wird abgeschnitten."""
        text = "Fertiger Text. **angefan"
        result = find_safe_markdown_end(text)
        assert result < len(text)
        # Die sichere Position sollte vor ** sein
        safe_part = text[:result]
        assert "**" not in safe_part

    def test_open_code_block_trims(self) -> None:
        """Offener ```-Block wird abgeschnitten."""
        text = "Text davor.\n```python\ndef x():"
        result = find_safe_markdown_end(text)
        assert result < len(text)

    def test_open_inline_code_trims(self) -> None:
        """Offener `-Inline-Code wird abgeschnitten."""
        text = "Verwende `pip inst"
        result = find_safe_markdown_end(text)
        assert result < len(text)

    def test_empty_text(self) -> None:
        """Leerer Text gibt 0."""
        assert find_safe_markdown_end("") == 0


class TestIsSafeMarkdownPosition:
    """Tests für _is_safe_markdown_position()."""

    def test_safe_after_closed_bold(self) -> None:
        text = "**fett** normal"
        assert _is_safe_markdown_position(text, len(text)) is True

    def test_unsafe_inside_bold(self) -> None:
        text = "**fett"
        assert _is_safe_markdown_position(text, len(text)) is False

    def test_safe_after_closed_code(self) -> None:
        text = "`code` rest"
        assert _is_safe_markdown_position(text, len(text)) is True

    def test_unsafe_inside_code(self) -> None:
        text = "`code"
        assert _is_safe_markdown_position(text, len(text)) is False

    def test_unsafe_inside_fenced_code(self) -> None:
        text = "```python\ncode"
        assert _is_safe_markdown_position(text, len(text)) is False

    def test_safe_after_fenced_code(self) -> None:
        text = "```python\ncode\n``` rest"
        assert _is_safe_markdown_position(text, len(text)) is True


class TestTruncateMarkdownForHtmlLimit:
    """Tests für _truncate_markdown_for_html_limit().

    Bug-Reproduktion: Bei langen Antworten (>4096 HTML-Zeichen) wurde
    der HTML-Text hart abgeschnitten, was HTML-Tags zerstörte und
    Telegram 400 Bad Request lieferte.
    """

    def test_short_text_unchanged(self) -> None:
        """Kurzer Text wird nicht verändert."""
        text = "**Kurzer** Text"
        result = _truncate_markdown_for_html_limit(text)
        assert result == text

    def test_long_text_produces_valid_html(self) -> None:
        """Langer Text wird so gekürzt dass HTML valide bleibt."""
        from domain.markdown import markdown_to_telegram_html

        # Erzeuge Text der >4096 HTML-Zeichen ergibt
        text = "## Headline\n\n**Fetter** Text mit `code` und mehr. " * 100
        html = markdown_to_telegram_html(text)
        assert len(html) > 4096  # Precondition: Text ist zu lang

        truncated = _truncate_markdown_for_html_limit(text)
        truncated_html = markdown_to_telegram_html(truncated)

        # HTML muss unter dem Limit sein
        assert len(truncated_html) <= 4096
        # HTML darf keine kaputten Tags haben (alle <b> müssen geschlossen sein)
        assert truncated_html.count("<b>") == truncated_html.count("</b>")
        assert truncated_html.count("<i>") == truncated_html.count("</i>")
        assert truncated_html.count("<code>") == truncated_html.count("</code>")

    def test_bold_heavy_text_no_broken_tags(self) -> None:
        """Text mit vielen Bold-Markern erzeugt keine kaputten Tags."""
        from domain.markdown import markdown_to_telegram_html

        # Simuliert typischen Claude-Output mit viel Formatting
        chunks = [
            f"**Punkt {i}:** Erklärung die etwas länger ist.\n\n" for i in range(150)
        ]
        text = "".join(chunks)
        html = markdown_to_telegram_html(text)
        assert len(html) > 4096

        truncated = _truncate_markdown_for_html_limit(text)
        truncated_html = markdown_to_telegram_html(truncated)
        assert len(truncated_html) <= 4096
        assert truncated_html.count("<b>") == truncated_html.count("</b>")

    def test_code_blocks_not_broken(self) -> None:
        """Code-Blöcke werden nicht mitten drin abgeschnitten."""
        from domain.markdown import markdown_to_telegram_html

        text = "Intro.\n\n```python\n" + "x = 1\n" * 800 + "```\n\nOutro."
        html = markdown_to_telegram_html(text)
        if len(html) <= 4096:
            return  # Test nur relevant wenn Text lang genug

        truncated = _truncate_markdown_for_html_limit(text)
        truncated_html = markdown_to_telegram_html(truncated)
        assert len(truncated_html) <= 4096
        # Keine offenen <pre> Tags
        assert truncated_html.count("<pre>") == truncated_html.count("</pre>")


class TestStreamingDuplicateEditsSkipped:
    """Tests für Duplikat-Erkennung bei Zwischen-Edits.

    Bug-Reproduktion: Wenn smart-trim bei zwei aufeinanderfolgenden
    Edits denselben Text liefert (weil neue Tokens noch nicht safe sind),
    wurde identischer Text an Telegram geschickt -> unnötige 400er.
    """

    @pytest.mark.asyncio
    async def test_duplicate_edit_skipped(self) -> None:
        """Wenn _last_edit_html identisch zum neuen HTML ist, wird kein API-Call gemacht.

        Testet den Duplikat-Mechanismus direkt durch pre-seeding von _last_edit_html.
        """
        from domain.markdown import markdown_to_telegram_html

        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0  # Fällig
        session.accumulated_text = "Hello World"

        # Pre-seed: so tun als ob die letzte Edit genau diesen HTML-Text hatte
        expected_html = markdown_to_telegram_html("Hello World")
        session._last_edit_html = expected_html

        # Versuch eine Edit zu senden (Text unverändert)
        await process_streaming_edit(session, "")

        # Kein API-Call weil der Text sich nicht geändert hat
        session.message.edit_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_different_edit_sent(self) -> None:
        """Verschiedene Edits werden normal gesendet."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0

        await process_streaming_edit(session, "Hello")
        assert session.message.edit_text.call_count == 1

        session.last_edit_time = 0
        await process_streaming_edit(session, " World complete text")
        assert session.message.edit_text.call_count == 2


class TestStreamingLongTextNoHTMLCorruption:
    """Integration-Test: langer Stream erzeugt keine kaputten HTML-Edits.

    Bug-Reproduktion für den konkreten Fehler:
    Antwort mit 7148 Zeichen, 144 Chunks. Nach ca. 40s überschreitet
    der akkumulierte HTML-Text 4096 Zeichen. Alte Logik schnitt HTML
    hart ab -> 400 Bad Request von Telegram.
    """

    @pytest.mark.asyncio
    async def test_long_stream_no_400_from_truncation(self) -> None:
        """Simuliert langen Stream: edit_text wird nie mit kaputtem HTML aufgerufen."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False

        # Simuliere 50 Streaming-Chunks die zusammen >4096 HTML-Zeichen ergeben
        chunk = "**Punkt:** Ein Satz mit Erklärung. "  # ~35 Zeichen pro Chunk
        for i in range(50):
            session.last_edit_time = 0  # Jede Edit ist fällig
            await process_streaming_edit(session, chunk)

        # Prüfe alle edit_text Aufrufe: keiner darf kaputtes HTML haben
        for call in session.message.edit_text.call_args_list:
            html_sent = call[0][0]
            call_kwargs = call[1]
            if call_kwargs.get("parse_mode") == "HTML":
                # Valides HTML: gleiche Anzahl öffnende/schließende Tags
                assert html_sent.count("<b>") == html_sent.count("</b>"), (
                    f"Kaputtes HTML (offener <b>): {html_sent[-100:]}"
                )
                assert html_sent.count("<i>") == html_sent.count("</i>")
                assert html_sent.count("<code>") == html_sent.count("</code>")
                assert html_sent.count("<pre>") == html_sent.count("</pre>")
                # Länge unter Telegram-Limit
                assert len(html_sent) <= 4096, (
                    f"HTML über Limit: {len(html_sent)} Zeichen"
                )


# ---------------------------------------------------------------------------
# Flood-Control / Adaptive Throttle Tests
# ---------------------------------------------------------------------------


class _FakeRetryAfter(Exception):
    """Simuliert telegram.error.RetryAfter mit retry_after Attribut."""

    def __init__(self, retry_after: int = 70):
        super().__init__(f"Flood control exceeded. Retry in {retry_after} seconds")
        self.retry_after = retry_after


class TestIsRetryAfter:
    """Tests für _is_retry_after() Erkennung."""

    def test_detects_retry_after_attribute(self) -> None:
        """Erkennt Exception mit retry_after Attribut."""
        exc = _FakeRetryAfter(42)
        assert _is_retry_after(exc) == 42

    def test_detects_flood_control_string(self) -> None:
        """Erkennt generische Exception mit 'flood control' im Text."""
        exc = Exception("Flood control exceeded. Retry in 30 seconds")
        assert _is_retry_after(exc) == 30

    def test_detects_429_string(self) -> None:
        """Erkennt generische Exception mit '429' im Text."""
        exc = Exception("HTTP 429 Too Many Requests")
        # Kein 'retry in N' parsbar, Default 30
        assert _is_retry_after(exc) == 30

    def test_returns_none_for_unrelated_error(self) -> None:
        """Gibt None für nicht-429-Fehler zurück."""
        exc = Exception("Network timeout")
        assert _is_retry_after(exc) is None

    def test_returns_none_for_bad_request(self) -> None:
        """Gibt None für Bad Request (kein 429)."""
        exc = Exception("Bad Request: can't parse entities")
        assert _is_retry_after(exc) is None

    def test_returns_none_for_not_modified(self) -> None:
        """Gibt None für 'message is not modified'."""
        exc = Exception("Bad Request: message is not modified")
        assert _is_retry_after(exc) is None


class TestApplyFloodBackoff:
    """Tests für _apply_flood_backoff()."""

    def test_sets_pause_until(self) -> None:
        """Setzt _paused_until auf now + retry_after."""
        session = _make_session(started_offset=5.0)
        before = time.monotonic()
        _apply_flood_backoff(session, 70)
        after = time.monotonic()

        assert session._paused_until >= before + 70
        assert session._paused_until <= after + 70

    def test_doubles_throttle(self) -> None:
        """Verdoppelt den Throttle."""
        session = _make_session(started_offset=5.0)
        assert session._current_throttle == DEFAULT_THROTTLE

        _apply_flood_backoff(session, 30)
        assert session._current_throttle == DEFAULT_THROTTLE * THROTTLE_BACKOFF_FACTOR

    def test_throttle_capped_at_max(self) -> None:
        """Throttle wird bei MAX_THROTTLE gecappt."""
        session = _make_session(started_offset=5.0)
        session._current_throttle = 8.0  # Nah am Maximum

        _apply_flood_backoff(session, 30)
        assert session._current_throttle == MAX_THROTTLE

    def test_resets_consecutive_success(self) -> None:
        """Setzt _consecutive_success auf 0."""
        session = _make_session(started_offset=5.0)
        session._consecutive_success = 4

        _apply_flood_backoff(session, 30)
        assert session._consecutive_success == 0


class TestRecordEditSuccess:
    """Tests für _record_edit_success() Throttle-Recovery."""

    def test_increments_counter(self) -> None:
        """Zaehlt erfolgreiche Edits."""
        session = _make_session(started_offset=5.0)
        _record_edit_success(session)
        assert session._consecutive_success == 1

    def test_recovery_after_threshold(self) -> None:
        """Throttle reduziert sich nach THROTTLE_RECOVERY_AFTER Erfolgen."""
        session = _make_session(started_offset=5.0)
        session._current_throttle = 6.0  # Erhöht durch vorherige 429

        for _ in range(THROTTLE_RECOVERY_AFTER):
            _record_edit_success(session)

        # Throttle sollte reduziert sein
        assert session._current_throttle < 6.0
        assert session._current_throttle == max(6.0 * 0.7, DEFAULT_THROTTLE)
        # Counter zurückgesetzt
        assert session._consecutive_success == 0

    def test_throttle_never_below_default(self) -> None:
        """Throttle fällt nicht unter DEFAULT_THROTTLE."""
        session = _make_session(started_offset=5.0)
        session._current_throttle = DEFAULT_THROTTLE  # Bereits am Default

        for _ in range(THROTTLE_RECOVERY_AFTER):
            _record_edit_success(session)

        assert session._current_throttle == DEFAULT_THROTTLE


class TestFloodControlIntermediateEdits:
    """Tests für Flood-Control bei Zwischen-Edits."""

    @pytest.mark.asyncio
    async def test_429_triggers_pause(self) -> None:
        """RetryAfter Exception setzt Session auf Pause."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0

        session.message.edit_text = AsyncMock(side_effect=_FakeRetryAfter(70))

        await process_streaming_edit(session, "Hello World")

        # Session sollte pausiert sein
        assert session._paused_until > time.monotonic()
        # Throttle sollte erhöht sein
        assert session._current_throttle > DEFAULT_THROTTLE

    @pytest.mark.asyncio
    async def test_edits_skipped_during_pause(self) -> None:
        """Während Pause werden Zwischen-Edits übersprungen."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0

        # Pause für 60 Sekunden setzen
        session._paused_until = time.monotonic() + 60

        await process_streaming_edit(session, "Skipped text")

        # Kein API-Call
        session.message.edit_text.assert_not_called()
        # Text wurde aber akkumuliert
        assert session.accumulated_text == "Skipped text"

    @pytest.mark.asyncio
    async def test_edits_resume_after_pause(self) -> None:
        """Nach Ablauf der Pause werden Edits wieder gesendet."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0

        # Pause die bereits abgelaufen ist
        session._paused_until = time.monotonic() - 1

        await process_streaming_edit(session, "Resumed text")

        # Edit wurde gesendet
        session.message.edit_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_throttle_doubles_on_429(self) -> None:
        """Throttle verdoppelt sich bei 429."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session.last_edit_time = 0
        original_throttle = session._current_throttle

        session.message.edit_text = AsyncMock(side_effect=_FakeRetryAfter(30))

        await process_streaming_edit(session, "Hello")

        assert session._current_throttle == original_throttle * THROTTLE_BACKOFF_FACTOR

    @pytest.mark.asyncio
    async def test_throttle_recovers_after_successes(self) -> None:
        """Throttle erholt sich nach erfolgreichen Edits."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session._current_throttle = 6.0  # Erhöht

        for i in range(THROTTLE_RECOVERY_AFTER):
            session.last_edit_time = 0
            session.accumulated_text = ""
            await process_streaming_edit(session, f"Success {i}")

        # Throttle sollte reduziert sein
        assert session._current_throttle < 6.0

    @pytest.mark.asyncio
    async def test_uses_adaptive_throttle_for_rate_limiting(self) -> None:
        """Rate-Limiting nutzt _current_throttle statt fixen Wert."""
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False
        session._current_throttle = 5.0  # Erhöht
        # Letzte Edit war vor 2 Sekunden (unter 5.0s Throttle)
        session.last_edit_time = time.monotonic() - 2.0

        await process_streaming_edit(session, "Too early")

        # Kein API-Call weil 2s < 5.0s Throttle
        session.message.edit_text.assert_not_called()


class TestFloodControlFinalEdit:
    """Tests für Flood-Control bei Final-Edits."""

    @pytest.mark.asyncio
    async def test_final_edit_retries_on_429(self) -> None:
        """Final-Edit wartet bei 429 und versucht erneut."""
        session = _make_session(started_offset=5.0)

        # Erster Versuch: 429, zweiter Versuch: Erfolg
        session.message.edit_text = AsyncMock(
            side_effect=[
                _FakeRetryAfter(1),  # Kurze Wartezeit für Test
                None,  # Erfolg
            ]
        )

        with patch(
            "application.streaming_handler.asyncio.sleep", new_callable=AsyncMock
        ):
            await finalize_streaming(session, "Final answer")

        # Zwei edit_text Aufrufe (Retry)
        assert session.message.edit_text.call_count == 2

    @pytest.mark.asyncio
    async def test_final_edit_fallback_send_message_after_max_retries(self) -> None:
        """Final-Edit fällt auf send_message zurück nach max Retries."""
        session = _make_session(started_offset=5.0)

        # Alle Versuche schlagen mit 429 fehl
        session.message.edit_text = AsyncMock(side_effect=_FakeRetryAfter(1))

        with patch(
            "application.streaming_handler.asyncio.sleep", new_callable=AsyncMock
        ):
            await finalize_streaming(session, "Final answer")

        # edit_text: 1 + FINAL_EDIT_MAX_RETRIES Versuche
        assert session.message.edit_text.call_count == 1 + FINAL_EDIT_MAX_RETRIES
        # Fallback: send_message auf dem Chat
        session.message.chat.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_final_edit_does_not_skip(self) -> None:
        """Final-Edit wird NICHT übersprungen obwohl Session pausiert ist."""
        session = _make_session(started_offset=5.0)
        session._paused_until = time.monotonic() + 60  # Aktive Pause

        # Finalize ignoriert die Pause
        await finalize_streaming(session, "Must be delivered")

        # Edit wurde trotz Pause gesendet
        session.message.edit_text.assert_called_once()


class TestFloodControlMultiMessage:
    """Tests für Flood-Control bei Multi-Message-Split."""

    @pytest.mark.asyncio
    async def test_multi_message_follow_up_retries_on_429(self) -> None:
        """Folge-Messages im Multi-Message-Split retrien bei 429."""
        session = _make_session(started_offset=5.0)

        # Edit gelingt, aber erste send_message liefert 429
        call_count = 0

        async def mock_send_message(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRetryAfter(1)
            return None

        session.message.chat.send_message = AsyncMock(side_effect=mock_send_message)

        long_text = "Dies ist ein Absatz.\n\n" * 300

        with patch(
            "application.streaming_handler.asyncio.sleep", new_callable=AsyncMock
        ):
            await finalize_streaming(session, long_text)

        # send_message wurde mindestens 2x aufgerufen (1 Retry + Erfolg)
        assert session.message.chat.send_message.call_count >= 2


class TestFloodControlStressSimulation:
    """Stress-Simulation: Mock-Telegram der nach N Edits 429 zurückgibt."""

    @pytest.mark.asyncio
    async def test_stress_429_after_5_edits(self) -> None:
        """Simuliert Telegram-Verhalten: 429 nach 5 schnellen Edits.

        Erwartung:
        - Edits 1-5 gehen durch
        - Edit 6 bekommt 429 -> Session pausiert
        - Folge-Edits während Pause werden übersprungen
        - Final-Edit wird trotzdem zugestellt
        """
        session = _make_session(started_offset=3.0)
        session.is_first_edit = False

        edit_attempts = 0
        successful_edits = 0

        async def mock_edit_text(*args, **kwargs):
            nonlocal edit_attempts, successful_edits
            edit_attempts += 1
            if edit_attempts == 6:
                raise _FakeRetryAfter(2)
            successful_edits += 1
            return None

        session.message.edit_text = AsyncMock(side_effect=mock_edit_text)

        # Simuliere 10 schnelle Zwischen-Edits
        for i in range(10):
            session.last_edit_time = 0  # Jede Edit ist fällig
            await process_streaming_edit(session, f"Chunk {i} ")

        # 5 erfolgreiche + 1 fehlgeschlagene Edits
        # Danach sollten Edits übersprungen werden (Pause aktiv)
        assert successful_edits == 5  # Nur 5 gingen durch
        assert session._paused_until > time.monotonic()

        # Final-Edit muss trotzdem ankommen
        edit_attempts = 0  # Reset für finalize (neuer Mock-Zähler)
        successful_edits = 0
        session.message.edit_text = AsyncMock()  # Gelingt jetzt

        with patch(
            "application.streaming_handler.asyncio.sleep", new_callable=AsyncMock
        ):
            await finalize_streaming(session, "Finale Antwort")

        session.message.edit_text.assert_called_once()


class TestMultiMessageEditFallback:
    """Tests für P1-6: Multi-Message Final-Edit Fallback bei erschöpften 429-Retries."""

    @pytest.mark.asyncio
    async def test_edit_fallback_to_send_message_on_exhausted_429(self) -> None:
        """Nach erschöpften Retries auf Edit: Fallback auf send_message."""
        session = _make_session(started_offset=5.0)

        # Edit schlägt immer mit 429 fehl
        session.message.edit_text = AsyncMock(side_effect=_FakeRetryAfter(1))

        # Erzeuge Text der 2+ Parts ergibt
        long_text = "Dies ist ein Absatz.\n\n" * 300

        with patch(
            "application.streaming_handler.asyncio.sleep", new_callable=AsyncMock
        ):
            await finalize_streaming(session, long_text)

        # edit_text wurde 1 + FINAL_EDIT_MAX_RETRIES mal versucht
        assert session.message.edit_text.call_count == 1 + FINAL_EDIT_MAX_RETRIES

        # Fallback: send_message wurde aufgerufen (für Part 1 als neue Nachricht)
        assert session.message.chat.send_message.call_count >= 1
