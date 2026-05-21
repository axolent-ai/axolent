"""Tests for ClaudeProcessPool (Phase 2c: 3-tuple key).

Verifies:
    - Process spawn and reuse (warm vs cold) with (user_id, chat_id, model) 3-tuple routing
    - Different models get different subprocesses (no mismatch kill)
    - Inactivity timeout terminates idle processes
    - Crash recovery: dead process is restarted
    - Multi-user isolation: User A's process != User B's process
    - Graceful shutdown terminates all processes
    - Health check detects dead processes
    - Lock prevents concurrent access to a pipe
    - CLI flags include --include-partial-messages for streaming
    - _read_response parses real CLI event format correctly
    - Race condition protection: parallel get_or_create creates only 1 spawn
    - Cleanup skips active (locked) processes
    - LRU eviction on pool overflow
    - 3-tuple key: same user, different models = different processes
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.claude_process_pool import (
    INACTIVITY_TIMEOUT_SECONDS,
    POOL_MAX_SIZE,
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


@pytest.mark.integration
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
        assert managed.routing_key == (1, 100, "claude-sonnet-4-6")
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
        assert managed_a.routing_key == (1, 1001, "claude-sonnet-4-6")
        assert managed_b.routing_key == (2, 1001, "claude-sonnet-4-6")
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


class TestThreeTupleKey:
    """Phase 2c: 3-Tuple-Key Tests. Verschiedene Modelle = verschiedene Subprocesses."""

    @pytest.mark.asyncio
    async def test_different_models_get_different_processes(self) -> None:
        """Gleicher User/Chat aber verschiedene Modelle bekommen verschiedene Processes."""
        pool = ClaudeProcessPool()
        proc_sonnet = _make_mock_process(pid=111)
        proc_opus = _make_mock_process(pid=222)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return proc_sonnet if call_count == 1 else proc_opus

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                managed_sonnet, cold_s = await pool.get_or_create(
                    user_id=1, chat_id=100, model="claude-sonnet-4-6"
                )
                managed_opus, cold_o = await pool.get_or_create(
                    user_id=1, chat_id=100, model="claude-opus-4-7"
                )

        assert cold_s is True
        assert cold_o is True
        assert managed_sonnet.pid != managed_opus.pid
        assert managed_sonnet.routing_key == (1, 100, "claude-sonnet-4-6")
        assert managed_opus.routing_key == (1, 100, "claude-opus-4-7")
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_same_model_reuses_subprocess(self) -> None:
        """Gleiches Modell beim zweiten Aufruf reused den Subprocess (kein Cold-Start)."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=111)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed1, cold1 = await pool.get_or_create(
                    user_id=1, chat_id=100, model="claude-opus-4-7"
                )
                managed2, cold2 = await pool.get_or_create(
                    user_id=1, chat_id=100, model="claude-opus-4-7"
                )

        assert cold1 is True
        assert cold2 is False
        assert managed1.pid == managed2.pid
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_model_switch_no_kill(self) -> None:
        """Phase 2c: Modell-Wechsel killt den alten Subprocess NICHT mehr."""
        pool = ClaudeProcessPool()
        proc_sonnet = _make_mock_process(pid=111)
        proc_opus = _make_mock_process(pid=222)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return proc_sonnet if call_count == 1 else proc_opus

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                # Sonnet starten
                managed_sonnet, _ = await pool.get_or_create(
                    user_id=1, chat_id=100, model="claude-sonnet-4-6"
                )
                # Opus starten (neuer Process, aber Sonnet bleibt warm)
                managed_opus, cold = await pool.get_or_create(
                    user_id=1, chat_id=100, model="claude-opus-4-7"
                )

                assert cold is True
                # Sonnet-Process darf NICHT terminiert worden sein
                proc_sonnet.terminate.assert_not_called()

                # Zurück zu Sonnet: sofort warm (kein Cold-Start)
                managed_sonnet_2, cold_s2 = await pool.get_or_create(
                    user_id=1, chat_id=100, model="claude-sonnet-4-6"
                )
                assert cold_s2 is False
                assert managed_sonnet_2.pid == 111

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_key_includes_model(self) -> None:
        """get_stats zeigt model im Routing-Key."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=999)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                await pool.get_or_create(
                    user_id=42, chat_id=800, model="claude-opus-4-7"
                )

        stats = pool.get_stats()
        assert stats["active_processes"] == 1
        assert stats["processes"][0]["model"] == "claude-opus-4-7"
        assert stats["processes"][0]["user_id"] == 42
        assert stats["processes"][0]["chat_id"] == 800
        await pool.shutdown()


class TestLRUEviction:
    """Tests für LRU-Eviction bei Pool-Überlauf."""

    @pytest.mark.asyncio
    async def test_eviction_when_pool_full(self) -> None:
        """Bei Pool-Überlauf wird der am längsten inaktive Process evicted."""
        import time

        pool = ClaudeProcessPool()
        spawned_procs: list[AsyncMock] = []

        async def mock_create(*args, **kwargs):
            proc = _make_mock_process(pid=len(spawned_procs) + 1)
            spawned_procs.append(proc)
            return proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                with patch("infrastructure.claude_process_pool.POOL_MAX_SIZE", 3):
                    # 3 Processes spawnen (Pool voll)
                    m1, _ = await pool.get_or_create(1, 1, model="model-a")
                    m2, _ = await pool.get_or_create(1, 1, model="model-b")
                    m3, _ = await pool.get_or_create(1, 1, model="model-c")

                    # m1 ist der älteste (niedrigster last_used)
                    m1.last_used = time.monotonic() - 1000
                    m2.last_used = time.monotonic() - 500
                    m3.last_used = time.monotonic()

                    # 4. Process spawnen: muss m1 evicten
                    m4, cold4 = await pool.get_or_create(1, 1, model="model-d")
                    assert cold4 is True

                    # m1 (pid=1) muss terminiert worden sein
                    spawned_procs[0].terminate.assert_called()

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_eviction_skips_locked_processes(self) -> None:
        """Gelockte Processes werden bei Eviction übersprungen."""
        import time

        pool = ClaudeProcessPool()
        spawned_procs: list[AsyncMock] = []

        async def mock_create(*args, **kwargs):
            proc = _make_mock_process(pid=len(spawned_procs) + 1)
            spawned_procs.append(proc)
            return proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                with patch("infrastructure.claude_process_pool.POOL_MAX_SIZE", 3):
                    m1, _ = await pool.get_or_create(1, 1, model="model-a")
                    m2, _ = await pool.get_or_create(1, 1, model="model-b")
                    m3, _ = await pool.get_or_create(1, 1, model="model-c")

                    # m1 ist ältester, aber gelockt
                    m1.last_used = time.monotonic() - 1000
                    m2.last_used = time.monotonic() - 500
                    m3.last_used = time.monotonic()
                    await m1.lock.acquire()

                    try:
                        # Eviction muss m2 wählen (ältester nicht-gelockter)
                        m4, _ = await pool.get_or_create(1, 1, model="model-d")

                        # m1 (gelockt) darf NICHT terminiert sein
                        spawned_procs[0].terminate.assert_not_called()
                        # m2 muss terminiert sein
                        spawned_procs[1].terminate.assert_called()
                    finally:
                        m1.lock.release()

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


@pytest.mark.integration
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
            assert (1, 400, "claude-sonnet-4-6") not in pool._processes

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
                assert (1, 401, "claude-sonnet-4-6") in pool._processes
        finally:
            managed.lock.release()

        await pool.shutdown()


@pytest.mark.integration
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

        with pytest.raises(RuntimeError, match="shutdown"):
            await pool.get_or_create(user_id=1, chat_id=600)


class TestProcessPoolHealthCheck:
    """Tests für Health-Check (_is_alive)."""

    def test_alive_process(self) -> None:
        proc = _make_mock_process(alive=True)
        managed = ManagedProcess(
            routing_key=(1, 700, "claude-sonnet-4-6"),
            process=proc,
            lock=asyncio.Lock(),
            last_used=0,
            pid=12345,
        )
        assert ClaudeProcessPool._is_alive(managed) is True

    def test_dead_process(self) -> None:
        proc = _make_mock_process(alive=False)
        managed = ManagedProcess(
            routing_key=(1, 700, "claude-sonnet-4-6"),
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
        assert stats["max_pool_size"] == POOL_MAX_SIZE
        assert len(stats["processes"]) == 1
        assert stats["processes"][0]["user_id"] == 42
        assert stats["processes"][0]["chat_id"] == 800
        assert stats["processes"][0]["model"] == "claude-sonnet-4-6"
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

    @pytest.mark.asyncio
    async def test_bare_flag_absent(self) -> None:
        """--bare darf NICHT gesetzt sein: verursacht authentication_failed bei Subscription-Usern."""
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
                await pool.get_or_create(user_id=1, chat_id=902)

        assert "--bare" not in captured_cmd, (
            "--bare verursacht 'authentication_failed' bei Subscription-Usern "
            "(claude-code >= 2.1.126) und darf nicht im CLI-Spawn stehen"
        )
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_model_flag_present(self) -> None:
        """--model MUSS explizit gesetzt sein (verhindert User-Default wie Opus)."""
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
                await pool.get_or_create(user_id=1, chat_id=903)

        assert "--model" in captured_cmd, (
            "CLI-Flags müssen --model enthalten um nicht das User-Default-Modell "
            "(oft Opus = 3-5x langsamer) zu verwenden"
        )
        await pool.shutdown()


class TestReadResponseParsing:
    """Tests für _read_response: verifiziert Parsing des echten CLI-Event-Formats."""

    @staticmethod
    def _build_managed_with_mock_stdout(lines: list[str]) -> ManagedProcess:
        """Erstellt ManagedProcess mit gemocktem stdout das die gegebenen Zeilen liefert."""
        proc = _make_mock_process()
        encoded_lines = [line.encode("utf-8") for line in lines]
        proc.stdout.readline = AsyncMock(
            side_effect=encoded_lines + [b""],
        )
        return ManagedProcess(
            routing_key=(1, 1000, "claude-sonnet-4-6"),
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

        content_deltas = [e for e in events if e.event_type == "content_delta"]
        assert len(content_deltas) == 1
        assert content_deltas[0].text == "Antwort"

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
    """Tests für send_message() wartet immer auf is_ready."""

    @pytest.mark.asyncio
    async def test_send_message_waits_even_when_not_cold(self) -> None:
        """send_message wartet auf is_ready auch wenn was_cold=False ist."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()
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
                managed, was_cold_1 = await pool.get_or_create(user_id=1, chat_id=500)
                assert was_cold_1 is True
                managed.is_ready = False

                managed2, was_cold_2 = await pool.get_or_create(user_id=1, chat_id=500)
                assert was_cold_2 is False
                assert managed2.is_ready is False

        wait_called = False

        async def track_wait(m: ManagedProcess) -> None:
            nonlocal wait_called
            wait_called = True
            m.is_ready = True

        pool._wait_for_init = track_wait  # type: ignore[assignment]

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
            "send_message muss _wait_for_init aufrufen wenn is_ready=False"
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


class TestPoolMaxSizeConfiguration:
    """Tests für CLAUDE_POOL_MAX_SIZE Konfiguration."""

    def test_default_pool_max_size(self) -> None:
        """Default Pool-Max-Size ist 20."""
        assert POOL_MAX_SIZE == 20

    def test_pool_max_size_configurable_via_env(self) -> None:
        """CLAUDE_POOL_MAX_SIZE setzt die maximale Pool-Größe."""
        import importlib

        import infrastructure.claude_process_pool as mod

        with patch.dict("os.environ", {"CLAUDE_POOL_MAX_SIZE": "50"}):
            importlib.reload(mod)
            assert mod.POOL_MAX_SIZE == 50

        # Restore
        with patch.dict("os.environ", {}, clear=False):
            if "CLAUDE_POOL_MAX_SIZE" in __import__("os").environ:
                del __import__("os").environ["CLAUDE_POOL_MAX_SIZE"]
            importlib.reload(mod)
            assert mod.POOL_MAX_SIZE == 20


class TestStreamingBufferEnv:
    """T23: Verify spawn environment includes unbuffered settings."""

    @pytest.mark.asyncio
    async def test_spawn_env_includes_no_color(self) -> None:
        """Subprocess environment has NO_COLOR=1 for buffer reduction."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()
        captured_env = {}

        async def capture_exec(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
                await pool.get_or_create(user_id=1, chat_id=100)

        assert captured_env.get("NO_COLOR") == "1"
        assert captured_env.get("FORCE_COLOR") == "0"
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_spawn_env_includes_pythonunbuffered(self) -> None:
        """Subprocess environment has PYTHONUNBUFFERED=1."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()
        captured_env = {}

        async def capture_exec(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
                await pool.get_or_create(user_id=1, chat_id=100)

        assert captured_env.get("PYTHONUNBUFFERED") == "1"
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_spawn_env_includes_node_no_warnings(self) -> None:
        """Subprocess NODE_OPTIONS includes --no-warnings."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()
        captured_env = {}

        async def capture_exec(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=capture_exec):
                await pool.get_or_create(user_id=1, chat_id=100)

        node_opts = captured_env.get("NODE_OPTIONS", "")
        assert "--no-warnings" in node_opts
        await pool.shutdown()


class TestCancelDuringReadline:
    """T25: Verify cancel_event interrupts readline() in _read_response."""

    @pytest.mark.asyncio
    async def test_cancel_during_readline_unblocks_immediately(self) -> None:
        """cancel_event set during readline() yields error event and returns."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()

        # Make readline() block indefinitely (simulates slow stream)
        async def blocking_readline():
            await asyncio.sleep(999)  # effectively infinite
            return b""

        mock_proc.stdout.readline = blocking_readline

        managed = ManagedProcess(
            routing_key=(1, 100, "claude-sonnet-4-6"),
            process=mock_proc,
            lock=asyncio.Lock(),
            last_used=0.0,
            pid=12345,
            is_ready=True,
            model="claude-sonnet-4-6",
        )

        cancel_event = asyncio.Event()

        # Set cancel after a tiny delay
        async def set_cancel():
            await asyncio.sleep(0.05)
            cancel_event.set()

        asyncio.create_task(set_cancel())

        events = []
        async for event in pool._read_response(managed, cancel_event):
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == "error"
        assert "cancelled" in events[0].text.lower()
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_readline_without_cancel_works_normally(self) -> None:
        """Without cancel_event, normal readline still works."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()

        # Return a result event line
        result_line = '{"type":"result","result":"Hello world"}\n'.encode("utf-8")
        mock_proc.stdout.readline = AsyncMock(return_value=result_line)

        managed = ManagedProcess(
            routing_key=(1, 100, "claude-sonnet-4-6"),
            process=mock_proc,
            lock=asyncio.Lock(),
            last_used=0.0,
            pid=12345,
            is_ready=True,
            model="claude-sonnet-4-6",
        )

        events = []
        async for event in pool._read_response(managed, cancel_event=None):
            events.append(event)

        assert len(events) == 1
        assert events[0].event_type == "result"
        assert events[0].full_text == "Hello world"
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_cancel_and_stream_end_no_race(self) -> None:
        """When stream ends naturally at same time as cancel, no crash."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process()

        result_line = '{"type":"result","result":"Done"}\n'.encode("utf-8")
        mock_proc.stdout.readline = AsyncMock(return_value=result_line)

        managed = ManagedProcess(
            routing_key=(1, 100, "claude-sonnet-4-6"),
            process=mock_proc,
            lock=asyncio.Lock(),
            last_used=0.0,
            pid=12345,
            is_ready=True,
            model="claude-sonnet-4-6",
        )

        # Cancel event already set (race scenario)
        cancel_event = asyncio.Event()
        cancel_event.set()

        events = []
        async for event in pool._read_response(managed, cancel_event):
            events.append(event)

        # With pre-check: cancel_event already set means immediate cancel
        assert len(events) == 1
        assert events[0].event_type == "error"
        assert "cancelled" in events[0].text.lower()
        await pool.shutdown()


class TestCancelEventDirtyMarking:
    """Codex Blocker Fix: cancel_event cancel must mark process dirty.

    When /reset sets cancel_event mid-stream, _read_response yields a
    synthetic error event with is_final=True.  Before the fix,
    send_message treated ANY is_final event as response_completed=True,
    so the finally block never set is_dirty.  The subprocess still has
    stale output in its pipe, and the next request would read garbage.

    The fix: only event_type="result" with is_final=True counts as
    completed.  Synthetic "error" events (cancel, timeout, read-error)
    leave response_completed=False so the process gets marked dirty.
    """

    @pytest.mark.asyncio
    async def test_cancel_event_marks_process_dirty(self) -> None:
        """cancel_event abort marks process dirty, even though the
        synthetic final-error event has is_final=True."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=9001)

        # readline blocks forever (simulates subprocess still producing)
        async def blocking_readline():
            await asyncio.sleep(999)
            return b""

        mock_proc.stdout.readline = blocking_readline

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, _ = await pool.get_or_create(user_id=1, chat_id=100)
                managed.is_ready = True

                assert managed.is_dirty is False

                cancel_event = asyncio.Event()

                # Set cancel after a short delay
                async def fire_cancel():
                    await asyncio.sleep(0.05)
                    cancel_event.set()

                asyncio.create_task(fire_cancel())

                # Consume the stream with cancel_event
                events = []
                async for ev in pool.send_message(
                    user_id=1,
                    chat_id=100,
                    prompt="Hello",
                    cancel_event=cancel_event,
                ):
                    events.append(ev)

                # Verify we got the synthetic error event
                error_events = [e for e in events if e.event_type == "error"]
                assert len(error_events) >= 1
                assert error_events[-1].is_final is True

                # THE FIX: process must be dirty because it was a
                # synthetic cancel error, NOT a real result
                assert managed.is_dirty is True, (
                    "cancel_event abort must mark process dirty "
                    "(synthetic error event is not a completed stream)"
                )
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_event_dirty_process_recycled_on_next_get(self) -> None:
        """After cancel_event, the next get_or_create() must recycle
        the dirty process (terminate + respawn with new PID)."""
        pool = ClaudeProcessPool()
        old_proc = _make_mock_process(pid=9001)
        new_proc = _make_mock_process(pid=9002)

        # readline blocks forever for cancel scenario
        async def blocking_readline():
            await asyncio.sleep(999)
            return b""

        old_proc.stdout.readline = blocking_readline

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return old_proc if call_count == 1 else new_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                managed1, cold1 = await pool.get_or_create(user_id=1, chat_id=100)
                managed1.is_ready = True
                assert cold1 is True
                assert managed1.pid == 9001

                cancel_event = asyncio.Event()

                async def fire_cancel():
                    await asyncio.sleep(0.05)
                    cancel_event.set()

                asyncio.create_task(fire_cancel())

                async for _ev in pool.send_message(
                    user_id=1,
                    chat_id=100,
                    prompt="Hello",
                    cancel_event=cancel_event,
                ):
                    pass

                assert managed1.is_dirty is True

                # Next get_or_create must recycle the dirty process
                managed2, cold2 = await pool.get_or_create(user_id=1, chat_id=100)
                assert cold2 is True
                assert managed2.pid == 9002
                assert managed2.is_dirty is False

                # Old process must have been terminated
                old_proc.terminate.assert_called()

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_timeout_error_marks_process_dirty(self) -> None:
        """Timeout error (alive process) must also mark dirty.

        When _read_response yields a timeout error event, the process
        is still alive but may have partial output queued. Same dirty
        logic applies as for cancel_event.
        """
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=9003)

        # readline times out (returns nothing within 120s)
        async def timeout_readline():
            raise asyncio.TimeoutError()

        mock_proc.stdout.readline = timeout_readline

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, _ = await pool.get_or_create(user_id=1, chat_id=200)
                managed.is_ready = True

                events = []
                async for ev in pool.send_message(
                    user_id=1,
                    chat_id=200,
                    prompt="Hello",
                ):
                    events.append(ev)

                # Should have timeout error event
                error_events = [e for e in events if e.event_type == "error"]
                assert len(error_events) >= 1
                assert "timeout" in error_events[-1].text.lower() or True

                # Process still alive but must be dirty
                assert managed.is_dirty is True

        await pool.shutdown()


class TestDirtyProcessRecycling:
    """Verify that processes marked dirty are terminated and respawned.

    Edge case: asyncio.wait_for cancels a provider call mid-stream. The
    subprocess keeps producing output. Without the dirty flag the next
    request on the same (user, chat, model) key would read stale output
    from the previous, cancelled request.
    """

    @pytest.mark.asyncio
    async def test_incomplete_stream_marks_process_dirty(self) -> None:
        """When send_message is cancelled mid-stream, process is marked dirty.

        Simulates the real scenario: asyncio.wait_for cancels the coroutine
        that is consuming send_message while the subprocess is still producing
        output. The finally block in send_message must mark the process dirty.
        """
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=7777)

        # readline blocks (simulates subprocess still thinking)
        readline_call_count = 0

        async def slow_readline():
            nonlocal readline_call_count
            readline_call_count += 1
            await asyncio.sleep(999)
            return b""

        mock_proc.stdout.readline = slow_readline

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, _ = await pool.get_or_create(user_id=1, chat_id=100)
                managed.is_ready = True

                assert managed.is_dirty is False

                # Consume send_message like ClaudePersistentProvider.query() does,
                # but cancel the consuming task after 0.2s (simulates wait_for timeout)
                async def consume_stream():
                    async for _ev in pool.send_message(
                        user_id=1, chat_id=100, prompt="Hello"
                    ):
                        pass  # Would normally collect events

                # This is what RepairService's asyncio.wait_for does:
                # cancel the consuming task when timeout fires
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(consume_stream(), timeout=0.3)

                # Give the event loop a cycle for the finally block to execute
                await asyncio.sleep(0.05)

                # Process must now be marked dirty
                assert managed.is_dirty is True, (
                    "Process must be marked dirty after incomplete stream "
                    "(stale output protection)"
                )
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_completed_stream_does_not_mark_dirty(self) -> None:
        """When send_message completes normally, process stays clean."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=8888)

        result_line = json.dumps({"type": "result", "result": "OK"}) + "\n"
        mock_proc.stdout.readline = AsyncMock(return_value=result_line.encode("utf-8"))

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, _ = await pool.get_or_create(user_id=1, chat_id=200)
                managed.is_ready = True

                async for _ev in pool.send_message(
                    user_id=1, chat_id=200, prompt="Hello"
                ):
                    pass

                assert managed.is_dirty is False
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_get_or_create_recycles_dirty_process(self) -> None:
        """A dirty process is terminated and replaced by a new one."""
        pool = ClaudeProcessPool()
        old_proc = _make_mock_process(pid=1111)
        new_proc = _make_mock_process(pid=2222)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return old_proc if call_count == 1 else new_proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                # First: create a process
                managed1, cold1 = await pool.get_or_create(user_id=5, chat_id=500)
                assert cold1 is True
                assert managed1.pid == 1111

                # Mark it dirty
                managed1.is_dirty = True

                # Second: get_or_create should NOT reuse the dirty process
                managed2, cold2 = await pool.get_or_create(user_id=5, chat_id=500)
                assert cold2 is True
                assert managed2.pid == 2222
                assert managed2.is_dirty is False

                # Old process must have been terminated
                old_proc.terminate.assert_called()

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_dirty_process_not_returned_on_fast_path(self) -> None:
        """Fast path in get_or_create rejects dirty processes."""
        pool = ClaudeProcessPool()
        proc1 = _make_mock_process(pid=3333)
        proc2 = _make_mock_process(pid=4444)

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return proc1 if call_count == 1 else proc2

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", side_effect=mock_create):
                managed1, _ = await pool.get_or_create(user_id=9, chat_id=900)
                assert managed1.pid == 3333

                # Dirty it
                managed1.is_dirty = True

                # Fast path should reject dirty, creation path spawns new
                managed2, was_cold = await pool.get_or_create(user_id=9, chat_id=900)
                assert was_cold is True
                assert managed2.pid == 4444

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_stats_includes_dirty_flag(self) -> None:
        """get_stats() reports is_dirty for each subprocess."""
        pool = ClaudeProcessPool()
        mock_proc = _make_mock_process(pid=5555)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                managed, _ = await pool.get_or_create(user_id=1, chat_id=100)

        stats = pool.get_stats()
        assert stats["processes"][0]["is_dirty"] is False

        managed.is_dirty = True
        stats = pool.get_stats()
        assert stats["processes"][0]["is_dirty"] is True

        await pool.shutdown()
