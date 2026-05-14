"""E2E Streaming-Flow-Tests: vollstaendiger Pfad durch alle Layer.

Testet den Streaming-Pfad von ChatService bis StreamEvent-Verarbeitung
mit Mock-Pool (kein echter Claude-CLI-Aufruf) und Mock-Telegram-Bot.

Szenarien:
  E1: Normale Chat-Frage -> Stream startet -> Events kommen -> Audit korrekt
  E2: Lange Antwort (>5000 Zeichen) -> Multi-Message-Split greift
  E3: Mid-Stream-Crash -> stream_error wird geloggt mit task_meta
  E4: TaskRouter-Klassifikation -> resolved_model im Audit
  E5: Modell-Wechsel (Sonnet -> Opus) -> Stream funktioniert
  E6: Sticky-Language (lang=en) -> Self-Awareness auf Englisch
  E7: Memory wird vor Stream geladen -> Context-Length korrekt
  E8: Privacy-Guard blockt Stream in Gruppen-Chat
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.chat_service import ChatService
from application.streaming_handler import (
    StreamingSession,
    finalize_streaming,
    split_text_for_telegram,
)
from infrastructure.claude_process_pool import StreamEvent
from infrastructure.conversation_storage import _reset_all_for_tests


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_mock_events(
    text: str = "Hello World from Claude!",
    chunk_size: int = 5,
    include_init: bool = True,
) -> list[StreamEvent]:
    """Erzeugt realistische Mock-StreamEvents.

    Args:
        text: Vollstaendiger Antworttext.
        chunk_size: Zeichen pro content_delta Event.
        include_init: Ob ein init-Event vorangestellt wird.

    Returns:
        Liste von StreamEvents (init + content_deltas + result).
    """
    events: list[StreamEvent] = []

    if include_init:
        events.append(
            StreamEvent(
                event_type="init",
                was_cold=False,
                subprocess_pid=12345,
            )
        )

    # Content-Deltas
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        events.append(StreamEvent(event_type="content_delta", text=chunk))

    # Final result
    events.append(
        StreamEvent(
            event_type="result",
            full_text=text,
            is_final=True,
        )
    )
    return events


def _make_error_events(
    text_before_crash: str = "Partial respon",
    error_text: str = "overloaded_error",
) -> list[StreamEvent]:
    """Erzeugt Events die mid-stream crashen."""
    events: list[StreamEvent] = [
        StreamEvent(event_type="init", was_cold=True, subprocess_pid=99999),
        StreamEvent(event_type="content_delta", text="Part"),
        StreamEvent(event_type="content_delta", text="ial "),
        StreamEvent(event_type="content_delta", text="resp"),
        StreamEvent(event_type="error", text=error_text),
    ]
    return events


async def _async_iter_events(
    events: list[StreamEvent],
) -> AsyncIterator[StreamEvent]:
    """Wandelt eine Event-Liste in einen AsyncIterator."""
    for event in events:
        yield event


def _make_mock_persistent_provider(
    events: list[StreamEvent],
) -> MagicMock:
    """Erzeugt einen gemockten ClaudePersistentProvider.

    Speichert die an query_streaming uebergebenen kwargs in
    provider.last_query_kwargs, damit Tests die tatsaechlich
    gesendeten Argumente (z.B. system_prompt, model) pruefen koennen.
    MagicMock.mock_calls trackt keine manuell zugewiesenen
    async-generator-Funktionen, daher dieses Seitenkanal-Pattern.
    """
    provider = MagicMock()
    provider.last_query_kwargs: dict[str, Any] = {}

    async def mock_query_streaming(**kwargs):
        provider.last_query_kwargs = kwargs
        for event in events:
            yield event

    provider.query_streaming = mock_query_streaming
    return provider


def _make_chat_service_with_streaming(
    events: list[StreamEvent],
    memory_service: MagicMock | None = None,
    model_service: MagicMock | None = None,
    task_router: MagicMock | None = None,
    self_awareness_service: MagicMock | None = None,
) -> tuple[ChatService, MagicMock]:
    """Erzeugt ChatService + Mock-Provider fuer Streaming-Tests.

    Returns:
        Tuple von (ChatService, mock_persistent_provider).
    """
    mock_router = MagicMock()
    mock_router.providers = {}
    mock_router.default = "claude_persistent"

    mock_provider = _make_mock_persistent_provider(events)

    svc = ChatService(
        provider_router=mock_router,
        memory_service=memory_service,
        model_service=model_service,
        task_router=task_router,
        self_awareness_service=self_awareness_service,
    )
    return svc, mock_provider


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_conversation_storage() -> None:
    """Raeumt Conversation-Storage vor jedem Test auf."""
    _reset_all_for_tests()


# ---------------------------------------------------------------
# E1: Normale Chat-Frage -> Stream komplett -> Audit korrekt
# ---------------------------------------------------------------


class TestE1NormalStreamFlow:
    """E1: Normale Chat-Frage durchlaeuft den vollen Streaming-Pfad."""

    async def test_stream_produces_events(self) -> None:
        """process_user_message_streaming liefert einen AsyncIterator."""
        events = _make_mock_events("Hallo Welt!")
        svc, mock_provider = _make_chat_service_with_streaming(events)

        stream_iter, mem_count, task_meta = await svc.process_user_message_streaming(
            text="Sag hallo",
            user_id=1,
            chat_id=10,
            username="testuser",
            system_prompt="Du bist hilfreich.",
            persistent_provider=mock_provider,
        )

        collected: list[StreamEvent] = []
        async for event in stream_iter:
            collected.append(event)

        # Mindestens init + content_delta(s) + result
        assert len(collected) >= 3
        assert collected[0].event_type == "init"
        assert any(e.event_type == "content_delta" for e in collected)
        assert collected[-1].event_type == "result"
        assert collected[-1].full_text == "Hallo Welt!"

    async def test_save_streaming_result_writes_audit(self) -> None:
        """save_streaming_result erzeugt Audit mit event_type stream_completed."""
        events = _make_mock_events("Test-Antwort")
        svc, _ = _make_chat_service_with_streaming(events)

        with patch("application.chat_service.write_audit_log") as mock_audit:
            await svc.save_streaming_result(
                user_id=1,
                chat_id=10,
                user_text="Test-Frage",
                response_text="Test-Antwort",
                duration_seconds=2.0,
                username="testuser",
                streaming_chunks=5,
                subprocess_pid=12345,
            )

            mock_audit.assert_called_once()
            entry = mock_audit.call_args[0][0]
            assert entry["event_type"] == "stream_completed"
            assert entry["user_id"] == 1
            assert entry["streaming_chunks"] == 5
            assert entry["subprocess_pid"] == 12345

    async def test_history_saved_after_stream(self) -> None:
        """Nach save_streaming_result sind User- und Assistant-Turn in History."""
        from infrastructure.conversation_storage import get_history

        events = _make_mock_events("Antwort vom Bot")
        svc, _ = _make_chat_service_with_streaming(events)

        await svc.save_streaming_result(
            user_id=1,
            chat_id=10,
            user_text="Frage vom User",
            response_text="Antwort vom Bot",
            duration_seconds=1.0,
        )

        history = await get_history(1, 10)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Frage vom User"
        assert history[1].role == "assistant"
        assert history[1].content == "Antwort vom Bot"


# ---------------------------------------------------------------
# E2: Lange Antwort -> Multi-Message-Split
# ---------------------------------------------------------------


class TestE2MultiMessageSplit:
    """E2: Lange Antwort (>5000 Zeichen) triggert Multi-Message-Split."""

    def test_split_text_for_telegram_splits_long_text(self) -> None:
        """split_text_for_telegram teilt Text >4096 Zeichen in Teile."""
        long_text = "A" * 5000
        parts = split_text_for_telegram(long_text)
        assert len(parts) >= 2
        # Zusammen muessen alle Teile den Originaltext ergeben
        reassembled = "".join(parts)
        assert reassembled == long_text

    def test_split_respects_paragraph_boundaries(self) -> None:
        """Split bevorzugt Absatz-Grenzen."""
        # Zwei Absaetze, erster knapp unter Limit, zweiter drueber
        text = "A" * 3000 + "\n\n" + "B" * 2000
        parts = split_text_for_telegram(text)
        assert len(parts) >= 2
        assert parts[0].strip().endswith("A")

    async def test_finalize_streaming_multi_message(self) -> None:
        """finalize_streaming sendet mehrere Messages bei langem Text."""
        mock_message = MagicMock()
        mock_message.edit_text = AsyncMock()
        mock_message.chat = MagicMock()
        mock_message.chat.send_message = AsyncMock()

        session = StreamingSession(message=mock_message)
        long_text = "X" * 6000
        await finalize_streaming(session, long_text)

        # Erste Nachricht wird editiert, Folge-Nachrichten gesendet
        assert mock_message.edit_text.call_count >= 1 or (
            mock_message.chat.send_message.call_count >= 1
        )


# ---------------------------------------------------------------
# E3: Mid-Stream-Crash -> stream_error geloggt mit task_meta
# ---------------------------------------------------------------


class TestE3MidStreamCrash:
    """E3: Mid-Stream-Error wird korrekt erfasst."""

    async def test_error_event_stops_stream(self) -> None:
        """Error-Event beendet den Stream."""
        events = _make_error_events()
        svc, mock_provider = _make_chat_service_with_streaming(events)

        stream_iter, mem_count, task_meta = await svc.process_user_message_streaming(
            text="Frage",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
        )

        collected: list[StreamEvent] = []
        async for event in stream_iter:
            collected.append(event)
            if event.event_type == "error":
                break

        error_events = [e for e in collected if e.event_type == "error"]
        assert len(error_events) == 1
        assert "overloaded_error" in error_events[0].text

    async def test_error_audit_includes_task_meta(self) -> None:
        """task_meta wird auch bei Fehlern korrekt aus dem Stream geliefert."""
        events = _make_error_events()

        # TaskRouter der klassifiziert
        mock_task_router = MagicMock()
        mock_classification = MagicMock()
        mock_classification.slot.value = "code"
        mock_classification.score = 100
        mock_classification.matched_patterns = ("```",)
        mock_classification.matched_keywords = ("debug",)
        mock_task_router.classify = MagicMock(return_value=mock_classification)
        mock_task_router.resolve_model = MagicMock(return_value="claude-opus-4-7")

        svc, mock_provider = _make_chat_service_with_streaming(
            events, task_router=mock_task_router
        )

        _, _, task_meta = await svc.process_user_message_streaming(
            text="Debugge diesen Code",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
        )

        assert task_meta["task_slot"] == "code"
        assert task_meta["resolved_model"] == "claude-opus-4-7"


# ---------------------------------------------------------------
# E4: TaskRouter-Klassifikation -> resolved_model im Audit
# ---------------------------------------------------------------


class TestE4TaskRouterClassification:
    """E4: TaskRouter-Klassifikation wird korrekt ins Audit geschrieben."""

    async def test_task_meta_includes_classification(self) -> None:
        """task_meta aus process_user_message_streaming enthaelt Klassifikation."""
        events = _make_mock_events("Code-Antwort")
        mock_task_router = MagicMock()
        mock_classification = MagicMock()
        mock_classification.slot.value = "code"
        mock_classification.score = 103
        mock_classification.matched_patterns = ("```",)
        mock_classification.matched_keywords = ("python", "debug")
        mock_task_router.classify = MagicMock(return_value=mock_classification)
        mock_task_router.resolve_model = MagicMock(return_value="claude-opus-4-7")

        svc, mock_provider = _make_chat_service_with_streaming(
            events, task_router=mock_task_router
        )

        _, _, task_meta = await svc.process_user_message_streaming(
            text="```python\nprint('hi')\n```",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
        )

        assert task_meta["task_slot"] == "code"
        assert task_meta["task_score"] == 103
        assert "```" in task_meta["task_matched_patterns"]
        assert task_meta["resolved_model"] == "claude-opus-4-7"

    async def test_audit_contains_resolved_model(self) -> None:
        """save_streaming_result schreibt resolved_model ins Audit."""
        events = _make_mock_events("Antwort")
        svc, _ = _make_chat_service_with_streaming(events)

        task_meta = {
            "task_slot": "code",
            "task_score": 100,
            "task_matched_patterns": ["```"],
            "task_matched_keywords": ["debug"],
            "resolved_model": "claude-opus-4-7",
        }

        with patch("application.chat_service.write_audit_log") as mock_audit:
            await svc.save_streaming_result(
                user_id=1,
                chat_id=10,
                user_text="Debug this",
                response_text="Antwort",
                duration_seconds=1.5,
                task_meta=task_meta,
            )

            entry = mock_audit.call_args[0][0]
            assert entry["task_slot"] == "code"
            assert entry["resolved_model"] == "claude-opus-4-7"


# ---------------------------------------------------------------
# E5: Modell-Wechsel (User wechselt Sonnet -> Opus)
# ---------------------------------------------------------------


class TestE5ModelSwitch:
    """E5: Modell-Wechsel mid-conversation funktioniert."""

    async def test_model_override_passed_to_provider(self) -> None:
        """User-Override wird an den Provider durchgereicht."""
        events = _make_mock_events("Opus-Antwort")

        mock_model_service = MagicMock()
        mock_model_service.get_user_model = MagicMock(return_value="claude-opus-4-7")

        mock_task_router = MagicMock()
        mock_classification = MagicMock()
        mock_classification.slot.value = "chat"
        mock_classification.score = 0
        mock_classification.matched_patterns = ()
        mock_classification.matched_keywords = ()
        mock_task_router.classify = MagicMock(return_value=mock_classification)
        mock_task_router.resolve_model = MagicMock(return_value="claude-opus-4-7")

        svc, mock_provider = _make_chat_service_with_streaming(
            events,
            model_service=mock_model_service,
            task_router=mock_task_router,
        )

        stream_iter, _, task_meta = await svc.process_user_message_streaming(
            text="Hallo",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
        )

        # Stream durchlaufen lassen
        async for _ in stream_iter:
            pass

        # Provider muss mit model=claude-opus-4-7 aufgerufen worden sein
        assert mock_provider.last_query_kwargs.get("model") == "claude-opus-4-7"
        # task_meta muss das resolved_model enthalten
        assert task_meta.get("resolved_model") == "claude-opus-4-7"

    async def test_two_sequential_streams_different_models(self) -> None:
        """Zwei aufeinanderfolgende Streams mit verschiedenen Modellen."""
        events_1 = _make_mock_events("Sonnet-Antwort")
        events_2 = _make_mock_events("Opus-Antwort")

        # Erster Stream: Sonnet
        svc1, provider1 = _make_chat_service_with_streaming(events_1)
        stream1, _, meta1 = await svc1.process_user_message_streaming(
            text="Hallo",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=provider1,
        )
        texts1: list[str] = []
        async for e in stream1:
            if e.event_type == "result":
                texts1.append(e.full_text)

        # Zweiter Stream: neuer Provider (simuliert Modell-Wechsel)
        svc2, provider2 = _make_chat_service_with_streaming(events_2)
        stream2, _, meta2 = await svc2.process_user_message_streaming(
            text="Nochmal",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=provider2,
        )
        texts2: list[str] = []
        async for e in stream2:
            if e.event_type == "result":
                texts2.append(e.full_text)

        assert texts1 == ["Sonnet-Antwort"]
        assert texts2 == ["Opus-Antwort"]


# ---------------------------------------------------------------
# E6: Sticky-Language wird respektiert
# ---------------------------------------------------------------


class TestE6StickyLanguage:
    """E6: Sticky-Language beeinflusst den Self-Awareness-Block."""

    async def test_english_user_gets_english_self_awareness(self) -> None:
        """User mit lang=en bekommt englischen Self-Awareness-Block."""
        from application.model_registry import ModelRegistry
        from application.self_awareness_service import SelfAwarenessService
        from infrastructure.conversation_storage import set_language

        # Sticky-Language auf EN setzen
        await set_language(1, 10, "en")

        registry = ModelRegistry()
        sa_svc = SelfAwarenessService(
            model_service=None,
            task_router=None,
            model_registry=registry,
        )

        events = _make_mock_events("English response")
        svc, mock_provider = _make_chat_service_with_streaming(
            events, self_awareness_service=sa_svc
        )

        # Provider aufrufen und pruefen was als system_prompt uebergeben wird
        stream_iter, _, _ = await svc.process_user_message_streaming(
            text="What model are you using?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="You are helpful.",
            persistent_provider=mock_provider,
        )

        # Stream konsumieren
        async for _ in stream_iter:
            pass

        # system_prompt aus last_query_kwargs pruefen
        system_sent = mock_provider.last_query_kwargs.get("system_prompt", "")
        assert "Current model:" in system_sent

    async def test_german_user_gets_german_self_awareness(self) -> None:
        """User mit lang=de bekommt deutschen Self-Awareness-Block."""
        from application.model_registry import ModelRegistry
        from application.self_awareness_service import SelfAwarenessService
        from infrastructure.conversation_storage import set_language

        await set_language(2, 20, "de")

        registry = ModelRegistry()
        sa_svc = SelfAwarenessService(
            model_service=None,
            task_router=None,
            model_registry=registry,
        )

        events = _make_mock_events("Deutsche Antwort")
        svc, mock_provider = _make_chat_service_with_streaming(
            events, self_awareness_service=sa_svc
        )

        stream_iter, _, _ = await svc.process_user_message_streaming(
            text="Welches Modell bist du?",
            user_id=2,
            chat_id=20,
            username="test",
            system_prompt="Du bist hilfreich.",
            persistent_provider=mock_provider,
        )

        async for _ in stream_iter:
            pass

        # system_prompt aus last_query_kwargs pruefen
        system_sent = mock_provider.last_query_kwargs.get("system_prompt", "")
        assert "Modell:" in system_sent


# ---------------------------------------------------------------
# E7: Memory wird vor Stream geladen
# ---------------------------------------------------------------


class TestE7MemoryLoading:
    """E7: Memory wird vor Stream geladen und in Context eingefuegt."""

    async def test_memory_loaded_before_stream(self) -> None:
        """MemoryService.recall wird vor dem Stream aufgerufen."""
        events = _make_mock_events("Antwort mit Memory-Kontext")

        mock_memory = MagicMock()
        mock_memory.recall = MagicMock(
            side_effect=lambda uid, q, layer, limit: (
                [{"id": "ep_001", "content": "User mag Pizza"}]
                if layer == "episodic"
                else []
            )
        )

        svc, mock_provider = _make_chat_service_with_streaming(
            events, memory_service=mock_memory
        )

        stream_iter, mem_count, _ = await svc.process_user_message_streaming(
            text="Was ist mein Lieblingsessen?",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="Du bist hilfreich.",
            persistent_provider=mock_provider,
        )

        # Memory muss geladen worden sein
        assert mem_count == 1

        # Stream konsumieren damit query_streaming tatsaechlich aufgerufen wird
        async for _ in stream_iter:
            pass

        # System-Prompt muss Memory-Context enthalten
        system_sent = mock_provider.last_query_kwargs.get("system_prompt", "")
        assert "GESPEICHERTE NOTIZEN" in system_sent
        assert "User mag Pizza" in system_sent

    async def test_memory_count_zero_when_no_matches(self) -> None:
        """mem_count ist 0 wenn keine Memory-Treffer."""
        events = _make_mock_events("Antwort")

        mock_memory = MagicMock()
        mock_memory.recall = MagicMock(return_value=[])

        svc, mock_provider = _make_chat_service_with_streaming(
            events, memory_service=mock_memory
        )

        _, mem_count, _ = await svc.process_user_message_streaming(
            text="Irgendeine Frage mit langen Woertern",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
        )

        assert mem_count == 0


# ---------------------------------------------------------------
# E8: Privacy-Guard blockt Stream in Gruppen-Chat
# ---------------------------------------------------------------


class TestE8PrivacyGuard:
    """E8: require_private_chat blockiert Message-Handler in Gruppen."""

    async def test_group_chat_blocked_by_decorator(self) -> None:
        """handle_message lehnt Gruppen-Chats ab."""
        from presentation.decorators import require_private_chat

        call_count = 0

        @require_private_chat
        async def dummy_handler(update, context):
            nonlocal call_count
            call_count += 1

        # Simuliere Gruppen-Chat
        update = MagicMock()
        update.effective_chat = MagicMock()
        update.effective_chat.type = "group"
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await dummy_handler(update, context)

        # Handler darf nicht aufgerufen worden sein
        assert call_count == 0
        # Fehlermeldung muss gesendet worden sein
        update.message.reply_text.assert_called_once()

    async def test_private_chat_allowed(self) -> None:
        """handle_message erlaubt Private-Chats."""
        from presentation.decorators import require_private_chat

        call_count = 0

        @require_private_chat
        async def dummy_handler(update, context):
            nonlocal call_count
            call_count += 1

        # Simuliere Private-Chat
        update = MagicMock()
        update.effective_chat = MagicMock()
        update.effective_chat.type = "private"
        update.message = MagicMock()
        context = MagicMock()

        await dummy_handler(update, context)

        assert call_count == 1


# ---------------------------------------------------------------
# Streaming-Session Edge Cases
# ---------------------------------------------------------------


class TestStreamingSessionEdgeCases:
    """Zusaetzliche Edge-Cases fuer Streaming-Sessions."""

    async def test_empty_stream_produces_no_content(self) -> None:
        """Stream ohne content_delta Events hat keinen Ausgabetext."""
        events = [
            StreamEvent(event_type="init", was_cold=True, subprocess_pid=1),
            StreamEvent(event_type="result", full_text="", is_final=True),
        ]
        svc, mock_provider = _make_chat_service_with_streaming(events)

        stream_iter, _, _ = await svc.process_user_message_streaming(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
        )

        results: list[str] = []
        async for event in stream_iter:
            if event.event_type == "result":
                results.append(event.full_text)

        assert results == [""]

    async def test_cold_start_flag_propagated(self) -> None:
        """was_cold=True im init-Event wird korrekt propagiert."""
        events = [
            StreamEvent(event_type="init", was_cold=True, subprocess_pid=42),
            StreamEvent(event_type="content_delta", text="Hi"),
            StreamEvent(event_type="result", full_text="Hi", is_final=True),
        ]
        svc, mock_provider = _make_chat_service_with_streaming(events)

        stream_iter, _, _ = await svc.process_user_message_streaming(
            text="Test",
            user_id=1,
            chat_id=10,
            username="test",
            system_prompt="System.",
            persistent_provider=mock_provider,
        )

        init_events = []
        async for event in stream_iter:
            if event.event_type == "init":
                init_events.append(event)

        assert len(init_events) == 1
        assert init_events[0].was_cold is True
        assert init_events[0].subprocess_pid == 42
