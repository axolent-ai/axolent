"""Claude Process Pool: verwaltet persistente Claude-CLI-Subprocesses.

Pro User-ID (chat_id) wird ein eigener Subprocess gehalten.
Niemals werden Subprocesses zwischen Users geteilt (Context-Leak-Risiko).

Features:
    - Process-per-User Isolation
    - 5-Minuten Inaktivitaetstimeout mit automatischer Terminierung
    - Health-Check vor jedem Send
    - Crash-Recovery: bei totem Subprocess wird ein neuer gestartet
    - Graceful Shutdown: alle Subprocesses werden sauber beendet
    - asyncio.Lock pro Process gegen Race Conditions
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

log = logging.getLogger(__name__)

# Inaktivitaetstimeout: 5 Minuten
INACTIVITY_TIMEOUT_SECONDS: float = 300.0

# Cleanup-Intervall: alle 60 Sekunden auf abgelaufene Processes pruefen
CLEANUP_INTERVAL_SECONDS: float = 60.0

# Maximale Init-Wartezeit beim ersten Start
INIT_TIMEOUT_SECONDS: float = 30.0


@dataclass
class StreamEvent:
    """Ein einzelnes Streaming-Event aus dem Claude-Subprocess.

    Attributes:
        event_type: Art des Events (content_delta, result, error, init).
        text: Inkrementeller Text (bei content_delta).
        full_text: Vollstaendige Antwort (bei result).
        raw: Rohes JSON-Event (fuer Debugging).
        is_final: True wenn dies das letzte Event der Antwort ist.
    """

    event_type: str
    text: str = ""
    full_text: str = ""
    raw: dict = field(default_factory=dict)
    is_final: bool = False


@dataclass
class ManagedProcess:
    """Ein verwalteter Claude-Subprocess fuer einen bestimmten User.

    Attributes:
        chat_id: Telegram-Chat-ID (Routing-Key).
        process: Der asyncio-Subprocess.
        lock: Exclusive-Lock fuer Zugriff auf stdin/stdout.
        last_used: Timestamp der letzten Nutzung (monotonic).
        pid: Process-ID fuer Audit-Logging.
        is_ready: True wenn Init-Phase abgeschlossen.
    """

    chat_id: int
    process: asyncio.subprocess.Process
    lock: asyncio.Lock
    last_used: float
    pid: int
    is_ready: bool = False
    _accumulated_text: str = ""


class ClaudeProcessPool:
    """Verwaltet persistente Claude-CLI-Subprocesses pro User.

    Jeder User (identifiziert durch chat_id) bekommt einen eigenen
    Subprocess der wiederverwendet wird. Nach 5 Minuten Inaktivitaet
    wird der Subprocess terminiert.

    Thread-Safety: Alle Methoden sind async-safe. Jeder ManagedProcess
    hat seinen eigenen asyncio.Lock.
    """

    def __init__(self) -> None:
        self._processes: dict[int, ManagedProcess] = {}
        self._pool_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown = False

    @staticmethod
    def is_cli_available() -> bool:
        """Prueft ob `claude` CLI im PATH ist."""
        return shutil.which("claude") is not None

    async def start(self) -> None:
        """Startet den Cleanup-Background-Task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            log.info(
                "ClaudeProcessPool gestartet (Cleanup-Intervall: %.0fs)",
                CLEANUP_INTERVAL_SECONDS,
            )

    async def shutdown(self) -> None:
        """Graceful Shutdown: terminiert alle aktiven Subprocesses."""
        self._shutdown = True

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        async with self._pool_lock:
            chat_ids = list(self._processes.keys())

        for cid in chat_ids:
            await self._terminate_process(cid, reason="shutdown")

        log.info(
            "ClaudeProcessPool heruntergefahren (%d Processes terminiert)",
            len(chat_ids),
        )

    async def get_or_create(self, chat_id: int) -> tuple[ManagedProcess, bool]:
        """Holt einen existierenden oder erstellt einen neuen Subprocess.

        Args:
            chat_id: Telegram-Chat-ID als Routing-Key.

        Returns:
            Tuple von (ManagedProcess, was_cold: bool).
            was_cold=True wenn ein neuer Subprocess gestartet wurde.

        Raises:
            RuntimeError: Wenn CLI nicht verfuegbar oder Start fehlschlaegt.
        """
        if self._shutdown:
            raise RuntimeError("ProcessPool ist im Shutdown-Modus")

        async with self._pool_lock:
            managed = self._processes.get(chat_id)

            if managed is not None and self._is_alive(managed):
                managed.last_used = time.monotonic()
                return managed, False

            # Process ist tot oder existiert nicht: neuen erstellen
            if managed is not None:
                log.warning(
                    "Subprocess fuer chat_id=%d ist tot (pid=%d), starte neu",
                    chat_id,
                    managed.pid,
                )
                await self._kill_process(managed)

        # Neuen Subprocess ausserhalb des Pool-Locks starten
        new_managed = await self._spawn_process(chat_id)

        async with self._pool_lock:
            self._processes[chat_id] = new_managed

        return new_managed, True

    async def send_message(
        self, chat_id: int, prompt: str, system_prompt: str = ""
    ) -> AsyncIterator[StreamEvent]:
        """Sendet eine Nachricht an den Subprocess und streamt die Antwort.

        Args:
            chat_id: Telegram-Chat-ID.
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt (wird in den Content integriert).

        Yields:
            StreamEvent-Objekte mit inkrementellem Text und finalem Result.

        Raises:
            RuntimeError: Bei Subprocess-Crash oder Pipe-Fehler.
        """
        managed, was_cold = await self.get_or_create(chat_id)

        # Warte auf Init wenn cold
        if was_cold and not managed.is_ready:
            await self._wait_for_init(managed)

        async with managed.lock:
            managed.last_used = time.monotonic()
            managed._accumulated_text = ""

            # Prompt als stream-json User-Message formatieren
            if system_prompt:
                combined_text = f"{system_prompt}\n\n---\n\nUser: {prompt}"
            else:
                combined_text = prompt

            message = json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": combined_text}],
                    },
                },
                ensure_ascii=False,
            )

            # Health-Check: ist der Process noch am Leben?
            if not self._is_alive(managed):
                raise RuntimeError(
                    f"Subprocess fuer chat_id={chat_id} ist waehrend Lock tot"
                )

            # Nachricht senden
            try:
                managed.process.stdin.write((message + "\n").encode("utf-8"))
                await managed.process.stdin.drain()
            except (BrokenPipeError, OSError, ConnectionResetError) as e:
                log.error("Pipe-Fehler beim Senden an chat_id=%d: %s", chat_id, e)
                await self._terminate_process(chat_id, reason="pipe_broken")
                raise RuntimeError(f"Subprocess-Pipe gebrochen: {e}") from e

            # Antwort-Events lesen
            async for event in self._read_response(managed):
                yield event

    async def terminate_user(self, chat_id: int) -> bool:
        """Terminiert den Subprocess eines bestimmten Users.

        Args:
            chat_id: Telegram-Chat-ID.

        Returns:
            True wenn ein Subprocess terminiert wurde.
        """
        return await self._terminate_process(chat_id, reason="user_request")

    def get_stats(self) -> dict:
        """Gibt Pool-Statistiken zurueck (fuer Monitoring/Audit)."""
        now = time.monotonic()
        active = []
        for cid, mp in self._processes.items():
            active.append(
                {
                    "chat_id": cid,
                    "pid": mp.pid,
                    "idle_seconds": round(now - mp.last_used, 1),
                    "is_alive": self._is_alive(mp),
                }
            )
        return {
            "active_processes": len(self._processes),
            "processes": active,
        }

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _spawn_process(self, chat_id: int) -> ManagedProcess:
        """Startet einen neuen Claude-CLI-Subprocess.

        Raises:
            RuntimeError: Wenn CLI nicht verfuegbar.
        """
        if not self.is_cli_available():
            raise RuntimeError("claude CLI nicht im PATH gefunden")

        cmd = [
            "claude",
            "--print",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--no-session-persistence",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        pid = proc.pid or 0
        log.info(
            "Neuer Claude-Subprocess gestartet: chat_id=%d, pid=%d",
            chat_id,
            pid,
        )

        managed = ManagedProcess(
            chat_id=chat_id,
            process=proc,
            lock=asyncio.Lock(),
            last_used=time.monotonic(),
            pid=pid,
            is_ready=False,
        )

        return managed

    async def _wait_for_init(self, managed: ManagedProcess) -> None:
        """Wartet bis der Subprocess seine Init-Phase abgeschlossen hat.

        Init-Events (system, init_session etc.) werden gelesen und verworfen.
        Timeout nach INIT_TIMEOUT_SECONDS.
        """
        deadline = time.monotonic() + INIT_TIMEOUT_SECONDS
        proc = managed.process

        while time.monotonic() < deadline:
            if proc.stdout is None:
                break

            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                # Keine weiteren Init-Events, Subprocess ist bereit
                managed.is_ready = True
                log.debug(
                    "Subprocess pid=%d: Init abgeschlossen (Timeout-basiert)",
                    managed.pid,
                )
                return

            if not line:
                # EOF: Process hat sich beendet
                raise RuntimeError(
                    f"Subprocess pid={managed.pid} hat sich waehrend Init beendet"
                )

            # Init-Events loggen aber ignorieren
            line_str = line.decode("utf-8", "replace").strip()
            if line_str:
                try:
                    event = json.loads(line_str)
                    event_type = event.get("type", "unknown")
                    log.debug("Init-Event pid=%d: type=%s", managed.pid, event_type)
                except json.JSONDecodeError:
                    pass

        managed.is_ready = True
        log.debug(
            "Subprocess pid=%d: Init-Deadline erreicht, markiere als ready", managed.pid
        )

    async def _read_response(
        self, managed: ManagedProcess
    ) -> AsyncIterator[StreamEvent]:
        """Liest Antwort-Events aus stdout bis ein 'result' Event kommt.

        Yields:
            StreamEvent-Objekte.
        """
        proc = managed.process
        if proc.stdout is None:
            raise RuntimeError("Subprocess hat keinen stdout")

        while True:
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=120.0,  # 2 Minuten Timeout pro Zeile
                )
            except asyncio.TimeoutError:
                log.warning("Timeout beim Lesen von pid=%d", managed.pid)
                yield StreamEvent(
                    event_type="error",
                    text="Timeout: keine Antwort vom Subprocess",
                    is_final=True,
                )
                return

            if not line:
                # EOF: Process ist gestorben
                log.error("EOF von pid=%d waehrend Antwort-Lesen", managed.pid)
                yield StreamEvent(
                    event_type="error",
                    text="Subprocess unerwartet beendet",
                    is_final=True,
                )
                return

            line_str = line.decode("utf-8", "replace").strip()
            if not line_str:
                continue

            try:
                event = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "unknown")

            if event_type == "stream_event":
                inner = event.get("event", {})
                inner_type = inner.get("type", "")

                if inner_type == "content_block_delta":
                    delta = inner.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        managed._accumulated_text += text
                        yield StreamEvent(
                            event_type="content_delta",
                            text=text,
                            raw=event,
                        )

            elif event_type == "assistant":
                # Partial assembled message (ignorieren, wir bauen selbst zusammen)
                pass

            elif event_type == "result":
                # Antwort komplett
                result_text = event.get("result", "")
                if isinstance(result_text, str):
                    final_text = result_text
                elif isinstance(result_text, dict):
                    # Manchmal kommt result als dict mit content-Array
                    blocks = result_text.get("content", [])
                    final_text = "".join(
                        b.get("text", "") for b in blocks if b.get("type") == "text"
                    )
                else:
                    final_text = str(result_text)

                # Fallback: wenn result leer aber accumulated text vorhanden
                if not final_text and managed._accumulated_text:
                    final_text = managed._accumulated_text

                yield StreamEvent(
                    event_type="result",
                    full_text=final_text,
                    raw=event,
                    is_final=True,
                )
                return

            elif event_type == "error":
                error_msg = event.get("error", {}).get("message", "Unbekannter Fehler")
                yield StreamEvent(
                    event_type="error",
                    text=error_msg,
                    raw=event,
                    is_final=True,
                )
                return

            # Andere Events (system, rate_limit etc.) werden ignoriert

    @staticmethod
    def _is_alive(managed: ManagedProcess) -> bool:
        """Prueft ob der Subprocess noch laeuft."""
        return managed.process.returncode is None

    async def _terminate_process(self, chat_id: int, reason: str = "") -> bool:
        """Terminiert einen Subprocess sauber.

        Returns:
            True wenn ein Subprocess terminiert wurde.
        """
        async with self._pool_lock:
            managed = self._processes.pop(chat_id, None)

        if managed is None:
            return False

        await self._kill_process(managed)
        log.info(
            "Subprocess terminiert: chat_id=%d, pid=%d, reason=%s",
            chat_id,
            managed.pid,
            reason,
        )
        return True

    @staticmethod
    async def _kill_process(managed: ManagedProcess) -> None:
        """Killt einen Subprocess (terminate, dann kill nach 3s)."""
        proc = managed.process
        if proc.returncode is not None:
            return  # Bereits beendet

        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass  # Process war schon weg

    async def _cleanup_loop(self) -> None:
        """Background-Task: terminiert inaktive Subprocesses."""
        try:
            while not self._shutdown:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            pass

    async def _cleanup_expired(self) -> None:
        """Terminiert alle Subprocesses die laenger als INACTIVITY_TIMEOUT_SECONDS idle sind."""
        now = time.monotonic()
        expired_ids: list[int] = []

        async with self._pool_lock:
            for chat_id, managed in self._processes.items():
                idle_time = now - managed.last_used
                if idle_time > INACTIVITY_TIMEOUT_SECONDS:
                    expired_ids.append(chat_id)

        for chat_id in expired_ids:
            await self._terminate_process(chat_id, reason="inactivity_timeout")

        if expired_ids:
            log.info(
                "Cleanup: %d inaktive Subprocesses terminiert (chat_ids: %s)",
                len(expired_ids),
                expired_ids,
            )
