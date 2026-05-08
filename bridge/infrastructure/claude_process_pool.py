"""Claude Process Pool: verwaltet persistente Claude-CLI-Subprocesses.

Pro (user_id, chat_id)-Tuple wird ein eigener Subprocess gehalten.
Niemals werden Subprocesses zwischen Users geteilt (Context-Leak-Risiko).

Features:
    - Process-per-User Isolation via (user_id, chat_id) Routing-Key
    - 5-Minuten Inaktivitätstimeout mit automatischer Terminierung
    - Health-Check vor jedem Send
    - Crash-Recovery: bei totem Subprocess wird ein neuer gestartet
    - Graceful Shutdown: alle Subprocesses werden sauber beendet
    - asyncio.Lock pro Process gegen Race Conditions
    - Per-Key Creation Lock gegen Race Conditions bei parallelen Erstanfragen
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

# Inaktivitätstimeout: 5 Minuten
INACTIVITY_TIMEOUT_SECONDS: float = 300.0

# Cleanup-Intervall: alle 60 Sekunden auf abgelaufene Processes prüfen
CLEANUP_INTERVAL_SECONDS: float = 60.0

# Maximale Init-Wartezeit beim ersten Start
INIT_TIMEOUT_SECONDS: float = 30.0


@dataclass
class StreamEvent:
    """Ein einzelnes Streaming-Event aus dem Claude-Subprocess.

    Attributes:
        event_type: Art des Events (content_delta, result, error, init).
        text: Inkrementeller Text (bei content_delta).
        full_text: Vollständige Antwort (bei result).
        raw: Rohes JSON-Event (für Debugging).
        is_final: True wenn dies das letzte Event der Antwort ist.
    """

    event_type: str
    text: str = ""
    full_text: str = ""
    raw: dict = field(default_factory=dict)
    is_final: bool = False


@dataclass
class ManagedProcess:
    """Ein verwalteter Claude-Subprocess für einen bestimmten User.

    Attributes:
        routing_key: (user_id, chat_id) Tuple als Routing-Key.
        process: Der asyncio-Subprocess.
        lock: Exclusive-Lock für Zugriff auf stdin/stdout.
        last_used: Timestamp der letzten Nutzung (monotonic).
        pid: Process-ID für Audit-Logging.
        is_ready: True wenn Init-Phase abgeschlossen.
    """

    routing_key: tuple[int, int]
    process: asyncio.subprocess.Process
    lock: asyncio.Lock
    last_used: float
    pid: int
    is_ready: bool = False
    _accumulated_text: str = ""


class ClaudeProcessPool:
    """Verwaltet persistente Claude-CLI-Subprocesses pro User.

    Jeder User (identifiziert durch (user_id, chat_id) Tuple) bekommt
    einen eigenen Subprocess der wiederverwendet wird. Nach 5 Minuten
    Inaktivität wird der Subprocess terminiert.

    Thread-Safety: Alle Methoden sind async-safe. Jeder ManagedProcess
    hat seinen eigenen asyncio.Lock. Per-Key Creation Locks verhindern
    Race Conditions bei parallelen Erstanfragen.
    """

    def __init__(self) -> None:
        self._processes: dict[tuple[int, int], ManagedProcess] = {}
        self._pool_lock = asyncio.Lock()
        self._creation_locks: dict[tuple[int, int], asyncio.Lock] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown = False

    @staticmethod
    def is_cli_available() -> bool:
        """Prüft ob `claude` CLI im PATH ist."""
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
            keys = list(self._processes.keys())

        for key in keys:
            await self._terminate_process(key, reason="shutdown")

        log.info(
            "ClaudeProcessPool heruntergefahren (%d Processes terminiert)",
            len(keys),
        )

    async def get_or_create(
        self, user_id: int, chat_id: int
    ) -> tuple[ManagedProcess, bool]:
        """Holt einen existierenden oder erstellt einen neuen Subprocess.

        Verwendet Double-Check-Locking mit Per-Key Creation Lock,
        um Race Conditions bei parallelen Erstanfragen zu verhindern.

        Args:
            user_id: Telegram-User-ID.
            chat_id: Telegram-Chat-ID.

        Returns:
            Tuple von (ManagedProcess, was_cold: bool).
            was_cold=True wenn ein neuer Subprocess gestartet wurde.

        Raises:
            RuntimeError: Wenn CLI nicht verfügbar oder Start fehlschlägt.
        """
        if self._shutdown:
            raise RuntimeError("ProcessPool ist im Shutdown-Modus")

        key = (user_id, chat_id)

        # Schneller Pfad: existierender, lebendiger Process
        async with self._pool_lock:
            managed = self._processes.get(key)
            if managed is not None and self._is_alive(managed):
                managed.last_used = time.monotonic()
                return managed, False

            # Per-Key Creation Lock holen (oder erstellen)
            if key not in self._creation_locks:
                self._creation_locks[key] = asyncio.Lock()
            creation_lock = self._creation_locks[key]

        # Per-Key Lock: nur ein Spawn pro Key gleichzeitig
        async with creation_lock:
            # Double-Check: vielleicht hat ein anderer Task inzwischen gespawnt
            async with self._pool_lock:
                managed = self._processes.get(key)
                if managed is not None and self._is_alive(managed):
                    managed.last_used = time.monotonic()
                    return managed, False

                # Process ist tot: aufräumen
                if managed is not None:
                    log.warning(
                        "Subprocess für key=%s ist tot (pid=%d), starte neu",
                        key,
                        managed.pid,
                    )
                    await self._kill_process(managed)

            # Neuen Subprocess ausserhalb des Pool-Locks starten
            new_managed = await self._spawn_process(key)

            async with self._pool_lock:
                self._processes[key] = new_managed

            return new_managed, True

    async def send_message(
        self,
        user_id: int,
        chat_id: int,
        prompt: str,
        system_prompt: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Sendet eine Nachricht an den Subprocess und streamt die Antwort.

        Args:
            user_id: Telegram-User-ID.
            chat_id: Telegram-Chat-ID.
            prompt: User-Nachricht.
            system_prompt: Optionaler System-Prompt (wird in den Content integriert).

        Yields:
            StreamEvent-Objekte mit inkrementellem Text und finalem Result.

        Raises:
            RuntimeError: Bei Subprocess-Crash oder Pipe-Fehler.
        """
        key = (user_id, chat_id)
        managed, was_cold = await self.get_or_create(user_id, chat_id)

        # Warte auf Init unabhängig von was_cold (Pre-Warm-Pfad kann
        # was_cold=False liefern obwohl is_ready noch False ist).
        if not managed.is_ready:
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
                raise RuntimeError(f"Subprocess für key={key} ist während Lock tot")

            # Nachricht senden
            try:
                managed.process.stdin.write((message + "\n").encode("utf-8"))
                await managed.process.stdin.drain()
            except (BrokenPipeError, OSError, ConnectionResetError) as e:
                log.error("Pipe-Fehler beim Senden an key=%s: %s", key, e)
                await self._terminate_process(key, reason="pipe_broken")
                raise RuntimeError(f"Subprocess-Pipe gebrochen: {e}") from e

            # Antwort-Events lesen, last_used dabei aktualisieren
            async for event in self._read_response(managed):
                managed.last_used = time.monotonic()
                yield event

    async def terminate_session(self, user_id: int, chat_id: int) -> bool:
        """Terminiert den Subprocess einer bestimmten User-Session.

        Args:
            user_id: Telegram-User-ID.
            chat_id: Telegram-Chat-ID.

        Returns:
            True wenn ein Subprocess terminiert wurde.
        """
        return await self._terminate_process((user_id, chat_id), reason="user_request")

    def get_stats(self) -> dict:
        """Gibt Pool-Statistiken zurück (für Monitoring/Audit)."""
        now = time.monotonic()
        active = []
        for key, mp in self._processes.items():
            active.append(
                {
                    "user_id": key[0],
                    "chat_id": key[1],
                    "pid": mp.pid,
                    "idle_seconds": round(now - mp.last_used, 1),
                    "is_alive": self._is_alive(mp),
                    "is_locked": mp.lock.locked(),
                }
            )
        return {
            "active_processes": len(self._processes),
            "processes": active,
        }

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _spawn_process(self, key: tuple[int, int]) -> ManagedProcess:
        """Startet einen neuen Claude-CLI-Subprocess.

        Args:
            key: (user_id, chat_id) Routing-Key.

        Raises:
            RuntimeError: Wenn CLI nicht verfügbar.
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
            "--include-partial-messages",  # R04 Round 1: ohne dieses Flag emittiert die CLI keine content_block_delta-Events
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
            "Neuer Claude-Subprocess gestartet: key=%s, pid=%d",
            key,
            pid,
        )

        managed = ManagedProcess(
            routing_key=key,
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
        Timeout nach INIT_TIMEOUT_SECONDS. Bei EOF während Init wird eine
        RuntimeError geworfen.
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
                    f"Subprocess pid={managed.pid} hat sich während Init beendet"
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
        """Liest Antwort-Events aus stdout bis ein 'result'-Event kommt.

        Yields:
            StreamEvent-Objekte (content_delta, result, error).
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
                log.error("EOF von pid=%d während Antwort-Lesen", managed.pid)
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
        """Prüft ob der Subprocess noch läuft."""
        return managed.process.returncode is None

    async def _terminate_process(self, key: tuple[int, int], reason: str = "") -> bool:
        """Terminiert einen Subprocess sauber.

        Args:
            key: (user_id, chat_id) Routing-Key.
            reason: Grund für die Terminierung (für Logging).

        Returns:
            True wenn ein Subprocess terminiert wurde.
        """
        async with self._pool_lock:
            managed = self._processes.pop(key, None)

        if managed is None:
            return False

        await self._kill_process(managed)
        log.info(
            "Subprocess terminiert: key=%s, pid=%d, reason=%s",
            key,
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
        """Background-Task: terminiert inaktive Subprocesses regelmäßig."""
        try:
            while not self._shutdown:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            pass

    async def _cleanup_expired(self) -> None:
        """Terminiert idle Subprocesses die länger als INACTIVITY_TIMEOUT_SECONDS inaktiv sind.

        Prüft zusätzlich ob der Process gerade aktiv gelockt ist (laufender
        Stream). Gelockte Processes werden übersprungen und beim nächsten
        Cleanup-Zyklus erneut geprüft.
        """
        now = time.monotonic()
        expired_keys: list[tuple[int, int]] = []

        async with self._pool_lock:
            for key, managed in self._processes.items():
                idle_time = now - managed.last_used
                if idle_time > INACTIVITY_TIMEOUT_SECONDS:
                    # Nicht terminieren wenn gerade ein Stream aktiv ist
                    if managed.lock.locked():
                        log.debug(
                            "Cleanup übersprungen für key=%s: Lock aktiv",
                            key,
                        )
                        continue
                    expired_keys.append(key)

        for key in expired_keys:
            await self._terminate_process(key, reason="inactivity_timeout")

        if expired_keys:
            log.info(
                "Cleanup: %d inaktive Subprocesses terminiert (keys: %s)",
                len(expired_keys),
                expired_keys,
            )
