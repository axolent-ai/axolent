"""Tests für ClaudeProcessPool.

Verifiziert:
    - Process-Spawn und Reuse (warm vs cold) mit (user_id, chat_id) Tuple-Routing
    - Inaktivitäts-Timeout terminiert idle Processes
    - Crash-Recovery: toter Process wird neu gestartet
    - Multi-User-Isolation: User A's Process != User B's Process
    - Graceful Shutdown terminiert alle Processes
    - Health-Check erkennt tote Processes
    - Lock verhindert gleichzeitigen Zugriff auf eine Pipe
    - CLI-Flags enthalten --include-partial-messages für Streaming
    - _read_response parsed echtes CLI-Event-Format korrekt
    - Race-Condition-Schutz: parallele get_or_create erzeugen nur 1 Spawn
    - Cleanup übersprungen aktive (gelockte) Processes
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.claude_process_pool import (
    INACTIVITY_TIMEOUT_SECONDS,
    ClaudeProcessPool,
    ManagedProcess,
)


def _make_mock_process(alive: bool = True, pid: int = 12345) -> AsyncMock:
    """Erstellt einen gemockten asyncio.subprocess.Process."""
    proc = AsyncMock()
    proc.pid = pid
    proc.returncode = None if alive else 1
    proc.stdin = AsyncMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = AsyncMock()
    proc.stderr = AsyncMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


class TestProcessPoolSpawnAndReuse:
    """Tests für Process-Spawn und Wiederverwendung."""

    @pytest.mark.asyncio
    async def test_get_or_create_spawns_new_process(self) -> None:
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, was_cold = await pool.get_or_create(user_id=1, chat_id=100)

        assert was_cold is True
        assert managed.routing_key == (1, 100)
        assert managed.pid == 12345
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_get_or_create_reuses_existing(self) -> None:
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed1, cold1 = await pool.get_or_create(user_id=1, chat_id=200)
                managed2, cold2 = await pool.get_or_create(user_id=1, chat_id=200)

        assert cold1 is True
        assert cold2 is False
        assert managed1 is managed2
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_different_users_get_different_processes(self) -> None:
        pool = ClaudeProcessPool()
        proc_a = _make_mock_process(pid=111)
        proc_b = _make_mock_process(pid=222)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return proc_a if call_count == 1 else proc_b

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                managed_a, _ = await pool.get_or_create(user_id=1, chat_id=1001)
                managed_b, _ = await pool.get_or_create(user_id=2, chat_id=1001)

        assert managed_a.pid != managed_b.pid
        assert managed_a.routing_key == (1, 1001)
        assert managed_b.routing_key == (2, 1001)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_same_user_different_chats_get_different_processes(self) -> None:
        """Gleicher User in verschiedenen Chats bekommt verschiedene Processes."""
        pool = ClaudeProcessPool()
        proc_a = _make_mock_process(pid=111)
        proc_b = _make_mock_process(pid=222)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return proc_a if call_count == 1 else proc_b

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                managed_a, _ = await pool.get_or_create(user_id=1, chat_id=100)
                managed_b, _ = await pool.get_or_create(user_id=1, chat_id=200)

        assert managed_a.pid != managed_b.pid
        await pool.shutdown()


class TestProcessPoolRaceCondition:
    """Tests für Race-Condition-Schutz bei parallelen get_or_create."""

    @pytest.mark.asyncio
    async def test_parallel_get_or_create_spawns_only_once(self) -> None:
        """Zwei parallele get_or_create mit gleichem Key dürfen nur 1 Spawn erzeugen."""
        pool = ClaudeProcessPool()
        spawn_count = 0

        async def slow_mock_create(*args, **kwargs):
            nonlocal spawn_count
            spawn_count += 1
            await asyncio.sleep(0.05)  # Simuliere langsamen Spawn
            return _make_mock_process(pid=spawn_count * 100)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=slow_mock_create):
                # Zwei parallele Anfragen für den gleichen Key
                results = await asyncio.gather(
                    pool.get_or_create(user_id=1, chat_id=500),
                    pool.get_or_create(user_id=1, chat_id=500),
                )

        # Nur 1 Spawn (Double-Check-Locking schützt)
        assert spawn_count == 1
        # Beide bekommen den gleichen ManagedProcess
        managed1 = results[0][0]
        managed2 = results[1][0]
        assert managed1.pid == managed2.pid
        await pool.shutdown()


class TestProcessPoolCrashRecovery:
    """Tests für Crash-Recovery."""

    @pytest.mark.asyncio
    async def test_dead_process_triggers_respawn(self) -> None:
        pool = ClaudeProcessPool()
        dead_proc = _make_mock_process(alive=False, pid=111)
        new_proc = _make_mock_process(alive=True, pid=222)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return dead_proc if call_count == 1 else new_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                # Erster Aufruf: erstellt dead_proc
                managed1, cold1 = await pool.get_or_create(user_id=1, chat_id=300)
                assert cold1 is True
                assert managed1.pid == 111

                # Markiere als tot
                dead_proc.returncode = 1

                # Zweiter Aufruf: erkennt toten Process, erstellt neuen
                managed2, cold2 = await pool.get_or_create(user_id=1, chat_id=300)
                assert cold2 is True
                assert managed2.pid == 222

        await pool.shutdown()


class TestProcessPoolTimeout:
    """Tests für Inaktivitäts-Timeout."""

    @pytest.mark.asyncio
    async def test_cleanup_terminates_expired_processes(self) -> None:
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, _ = await pool.get_or_create(user_id=1, chat_id=400)

        # Simuliere: last_used ist länger als Timeout her
        import time

        managed.last_used = time.monotonic() - INACTIVITY_TIMEOUT_SECONDS - 10

        await pool._cleanup_expired()

        # Process sollte entfernt worden sein
        async with pool._pool_lock:
            assert (1, 400) not in pool._processes

    @pytest.mark.asyncio
    async def test_cleanup_skips_locked_processes(self) -> None:
        """Cleanup terminiert NICHT Processes deren Lock aktiv ist."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, _ = await pool.get_or_create(user_id=1, chat_id=401)

        import time

        managed.last_used = time.monotonic() - INACTIVITY_TIMEOUT_SECONDS - 10

        # Lock akquirieren (simuliert laufenden Stream)
        await managed.lock.acquire()
        try:
            await pool._cleanup_expired()

            # Process sollte NICHT entfernt worden sein
            async with pool._pool_lock:
                assert (1, 401) in pool._processes
        finally:
            managed.lock.release()

        await pool.shutdown()


class TestProcessPoolShutdown:
    """Tests für Graceful Shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_terminates_all(self) -> None:
        pool = ClaudeProcessPool()
        proc1 = _make_mock_process(pid=11)
        proc2 = _make_mock_process(pid=22)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return proc1 if call_count == 1 else proc2

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                await pool.get_or_create(user_id=1, chat_id=501)
                await pool.get_or_create(user_id=2, chat_id=502)

        await pool.shutdown()

        async with pool._pool_lock:
            assert len(pool._processes) == 0

    @pytest.mark.asyncio
    async def test_shutdown_prevents_new_spawns(self) -> None:
        pool = ClaudeProcessPool()
        await pool.shutdown()

        with pytest.raises(RuntimeError, match="Shutdown"):
            await pool.get_or_create(user_id=1, chat_id=600)


class TestProcessPoolHealthCheck:
    """Tests für Health-Check (_is_alive)."""

    def test_alive_process(self) -> None:
        proc = _make_mock_process(alive=True)
        managed = ManagedProcess(
            routing_key=(1, 700),
            process=proc,
            lock=asyncio.Lock(),
            last_used=0,
            pid=12345,
        )
        assert ClaudeProcessPool._is_alive(managed) is True

    def test_dead_process(self) -> None:
        proc = _make_mock_process(alive=False)
        managed = ManagedProcess(
            routing_key=(1, 700),
            process=proc,
            lock=asyncio.Lock(),
            last_used=0,
            pid=12345,
        )
        assert ClaudeProcessPool._is_alive(managed) is False


class TestProcessPoolStats:
    """Tests für get_stats()."""

    @pytest.mark.asyncio
    async def test_stats_returns_correct_info(self) -> None:
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=999)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                await pool.get_or_create(user_id=42, chat_id=800)

        stats = pool.get_stats()
        assert stats["active_processes"] == 1
        assert len(stats["processes"]) == 1
        assert stats["processes"][0]["user_id"] == 42
        assert stats["processes"][0]["chat_id"] == 800
        assert stats["processes"][0]["pid"] == 999
        assert stats["processes"][0]["is_alive"] is True
        assert stats["processes"][0]["is_locked"] is False
        await pool.shutdown()


class TestProcessPoolCLICheck:
    """Tests für CLI-Verfügbarkeitsprüfung."""

    def test_cli_available(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/claude"):
            assert ClaudeProcessPool.is_cli_available() is True

    def test_cli_not_available(self) -> None:
        with patch("shutil.which", return_value=None):
            assert ClaudeProcessPool.is_cli_available() is False


class TestSpawnProcessFlags:
    """Verifiziert dass _spawn_process die richtigen CLI-Flags setzt."""

    @pytest.mark.asyncio
    async def test_include_partial_messages_flag_present(self) -> None:
        """--include-partial-messages MUSS gesetzt sein, sonst kein Streaming."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()
        captured_cmd: list[str] = []

        async def capture_cmd(*args, **kwargs):
            captured_cmd.extend(args)
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=capture_cmd,
            ):
                await pool.get_or_create(user_id=1, chat_id=900)

        assert "--include-partial-messages" in captured_cmd, (
            "CLI-Flags müssen --include-partial-messages enthalten, "
            "sonst liefert die CLI keine stream_event/content_block_delta Events"
        )
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_stream_json_flags_present(self) -> None:
        """--output-format stream-json und --input-format stream-json müssen gesetzt sein."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()
        captured_cmd: list[str] = []

        async def capture_cmd(*args, **kwargs):
            captured_cmd.extend(args)
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=capture_cmd,
            ):
                await pool.get_or_create(user_id=1, chat_id=901)

        assert "--output-format" in captured_cmd
        assert "--input-format" in captured_cmd
        assert "stream-json" in captured_cmd
        await pool.shutdown()


class TestReadResponseParsing:
    """Tests für _read_response: verifiziert Parsing des echten CLI-Event-Formats.

    Das echte Format (mit --include-partial-messages) liefert:
        system, rate_limit_event, stream_event (message_start,
        content_block_start, content_block_delta*, content_block_stop,
        message_delta, message_stop), assistant, result
    """

    @staticmethod
    def _build_managed_with_mock_stdout(lines: list[str]) -> ManagedProcess:
        """Erstellt ManagedProcess mit gemocktem stdout das die gegebenen Zeilen liefert."""
        proc = _make_mock_process()
        encoded_lines = [line.encode("utf-8") for line in lines]
        # readline() liefert die Zeilen nacheinander, dann b"" für EOF
        proc.stdout.readline = AsyncMock(
            side_effect=encoded_lines + [b""],
        )
        return ManagedProcess(
            routing_key=(1, 1000),
            process=proc,
            lock=asyncio.Lock(),
            last_used=0,
            pid=54321,
        )

    @pytest.mark.asyncio
    async def test_parses_content_block_delta_events(self) -> None:
        """content_block_delta Events werden als content_delta StreamEvents geyielded."""
        lines = [
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "Hallo "},
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "Welt"},
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "type": "result",
                    "result": "Hallo Welt",
                }
            )
            + "\n",
        ]

        managed = self._build_managed_with_mock_stdout(lines)
        pool = ClaudeProcessPool()
        events = [e async for e in pool._read_response(managed)]

        assert len(events) == 3
        assert events[0].event_type == "content_delta"
        assert events[0].text == "Hallo "
        assert events[1].event_type == "content_delta"
        assert events[1].text == "Welt"
        assert events[2].event_type == "result"
        assert events[2].full_text == "Hallo Welt"
        assert events[2].is_final is True

    @pytest.mark.asyncio
    async def test_ignores_signature_delta_events(self) -> None:
        """signature_delta Events (Thinking) dürfen kein content_delta yielden."""
        lines = [
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "signature_delta",
                            "signature": "EoQC...",
                        },
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "Antwort"},
                    },
                }
            )
            + "\n",
            json.dumps({"type": "result", "result": "Antwort"}) + "\n",
        ]

        managed = self._build_managed_with_mock_stdout(lines)
        pool = ClaudeProcessPool()
        events = [e async for e in pool._read_response(managed)]

        # signature_delta hat kein "text" => wird nicht als content_delta geyielded
        content_deltas = [e for e in events if e.event_type == "content_delta"]
        assert len(content_deltas) == 1
        assert content_deltas[0].text == "Antwort"

    @pytest.mark.asyncio
    async def test_ignores_non_streaming_event_types(self) -> None:
        """system, rate_limit_event, assistant werden ignoriert."""
        lines = [
            json.dumps({"type": "system", "subtype": "init"}) + "\n",
            json.dumps({"type": "rate_limit_event", "rate_limit_info": {}}) + "\n",
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "message_start",
                        "message": {"model": "claude-opus-4-7"},
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "Test"},
                    },
                }
            )
            + "\n",
            json.dumps({"type": "assistant", "message": {"content": []}}) + "\n",
            json.dumps({"type": "result", "result": "Test"}) + "\n",
        ]

        managed = self._build_managed_with_mock_stdout(lines)
        pool = ClaudeProcessPool()
        events = [e async for e in pool._read_response(managed)]

        assert len(events) == 2
        assert events[0].event_type == "content_delta"
        assert events[1].event_type == "result"

    @pytest.mark.asyncio
    async def test_result_as_dict_with_content_blocks(self) -> None:
        """Result kann als dict mit content-Array kommen."""
        lines = [
            json.dumps(
                {
                    "type": "result",
                    "result": {
                        "content": [
                            {"type": "text", "text": "Teil eins "},
                            {"type": "text", "text": "Teil zwei"},
                        ],
                    },
                }
            )
            + "\n",
        ]

        managed = self._build_managed_with_mock_stdout(lines)
        pool = ClaudeProcessPool()
        events = [e async for e in pool._read_response(managed)]

        assert len(events) == 1
        assert events[0].event_type == "result"
        assert events[0].full_text == "Teil eins Teil zwei"

    @pytest.mark.asyncio
    async def test_accumulated_text_fallback(self) -> None:
        """Wenn result leer ist, wird accumulated text verwendet."""
        lines = [
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "Akkumuliert"},
                    },
                }
            )
            + "\n",
            json.dumps({"type": "result", "result": ""}) + "\n",
        ]

        managed = self._build_managed_with_mock_stdout(lines)
        pool = ClaudeProcessPool()
        events = [e async for e in pool._read_response(managed)]

        result_events = [e for e in events if e.event_type == "result"]
        assert len(result_events) == 1
        assert result_events[0].full_text == "Akkumuliert"

    @pytest.mark.asyncio
    async def test_error_event_yields_error(self) -> None:
        """Error-Events werden als error StreamEvent geyielded."""
        lines = [
            json.dumps(
                {
                    "type": "error",
                    "error": {"message": "Rate limit exceeded"},
                }
            )
            + "\n",
        ]

        managed = self._build_managed_with_mock_stdout(lines)
        pool = ClaudeProcessPool()
        events = [e async for e in pool._read_response(managed)]

        assert len(events) == 1
        assert events[0].event_type == "error"
        assert "Rate limit" in events[0].text
        assert events[0].is_final is True


class TestSendMessageWaitsForReady:
    """Tests für Task 5: send_message() wartet immer auf is_ready."""

    @pytest.mark.asyncio
    async def test_send_message_waits_even_when_not_cold(self) -> None:
        """send_message wartet auf is_ready auch wenn was_cold=False ist.

        Reproduziert den Pre-Warm-Bug: get_or_create wird vorab aufgerufen,
        dann liefert der zweite Aufruf was_cold=False obwohl is_ready=False.
        """
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()
        # stdout liefert Init-Events und dann Result
        init_line = json.dumps({"type": "system", "version": "1.0"}) + "\n"
        result_line = json.dumps({"type": "result", "result": "OK"}) + "\n"
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[
                init_line.encode("utf-8"),
                result_line.encode("utf-8"),
                b"",
            ]
        )

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                # Pre-Warm: erster Aufruf setzt was_cold=True
                managed, was_cold_1 = await pool.get_or_create(user_id=1, chat_id=500)
                assert was_cold_1 is True
                # Simuliere: is_ready ist noch False (Init nicht abgeschlossen)
                managed.is_ready = False

                # Zweiter Aufruf: was_cold=False, aber is_ready noch False
                managed2, was_cold_2 = await pool.get_or_create(user_id=1, chat_id=500)
                assert was_cold_2 is False
                assert managed2.is_ready is False

        # send_message muss trotzdem auf is_ready warten (nicht nur bei was_cold)
        # _wait_for_init setzt is_ready=True nach Init-Events
        # Wir patchen _wait_for_init um zu verifizieren dass es aufgerufen wird
        wait_called = False

        async def track_wait(m: ManagedProcess) -> None:
            nonlocal wait_called
            wait_called = True
            m.is_ready = True  # Simuliere erfolgreiche Init

        pool._wait_for_init = track_wait  # type: ignore[assignment]

        # Mock stdout für die eigentliche Message-Response
        result_line_2 = json.dumps({"type": "result", "result": "Response"}) + "\n"
        mock_proc.stdout.readline = AsyncMock(
            side_effect=[
                result_line_2.encode("utf-8"),
                b"",
            ]
        )

        async for _event in pool.send_message(user_id=1, chat_id=500, prompt="Test"):
            pass

        assert wait_called is True, (
            "send_message muss _wait_for_init aufrufen wenn is_ready=False, "
            "auch wenn was_cold=False"
        )
        await pool.shutdown()


class TestTTLConfiguration:
    """Tests für die TTL-Konfiguration via Umgebungsvariable."""

    def test_default_ttl_is_one_hour(self) -> None:
        """Ohne CLAUDE_SUBPROCESS_TTL_SECONDS gilt 3600s (1 Stunde)."""
        assert INACTIVITY_TIMEOUT_SECONDS == 3600.0

    def test_ttl_configurable_via_env(self) -> None:
        """CLAUDE_SUBPROCESS_TTL_SECONDS setzt den TTL-Wert."""
        import importlib

        import infrastructure.claude_process_pool as mod

        with patch.dict("os.environ", {"CLAUDE_SUBPROCESS_TTL_SECONDS": "1800"}):
            importlib.reload(mod)
            assert mod.INACTIVITY_TIMEOUT_SECONDS == 1800.0

        # Restore
        with patch.dict("os.environ", {}, clear=False):
            if "CLAUDE_SUBPROCESS_TTL_SECONDS" in __import__("os").environ:
                del __import__("os").environ["CLAUDE_SUBPROCESS_TTL_SECONDS"]
            importlib.reload(mod)
            assert mod.INACTIVITY_TIMEOUT_SECONDS == 3600.0
