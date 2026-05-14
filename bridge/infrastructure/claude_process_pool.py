"""Claude Process Pool: manages persistent Claude CLI subprocesses.

Per (user_id, chat_id, model) tuple a separate subprocess is maintained.
Subprocesses are never shared between users (context leak risk).

Features:
    * Process-per-user isolation via (user_id, chat_id, model) 3-tuple routing key
    * Keep all 6 models warm simultaneously (no cold start on slot switch)
    * LRU eviction at max pool size (default: 20, configurable via CLAUDE_POOL_MAX_SIZE)
    * 60-minute inactivity timeout with automatic termination
    * Health check before every send
    * Crash recovery: dead subprocess is replaced with a new one
    * Graceful shutdown: all subprocesses are cleanly terminated
    * asyncio.Lock per process against race conditions
    * Per-key creation lock against race conditions on parallel first requests
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

log = logging.getLogger(__name__)

# 1 hour idle timeout: more practical for solo/small multi-user setups.
# Tradeoff: ~150-300 MB RAM per subprocess while kept warm.
# Uncritical for <10 active users. Configurable via .env.
INACTIVITY_TIMEOUT_SECONDS: float = float(
    os.getenv("CLAUDE_SUBPROCESS_TTL_SECONDS", str(60 * 60))
)

# Cleanup interval: check for expired processes every 60 seconds
CLEANUP_INTERVAL_SECONDS: float = 60.0

# Maximum init wait time on first start
INIT_TIMEOUT_SECONDS: float = 30.0

# Model for the process pool. Default: Sonnet (fast, affordable).
# Without explicit --model the CLI uses user default (often Opus = 3-5x slower).
CLAUDE_POOL_MODEL: str = os.getenv("CLAUDE_POOL_MODEL", "claude-sonnet-4-6")

# Maximum number of concurrent subprocesses in the pool.
# Default 20: comfortable for 6 models per user and ~3 users.
# RAM budget: ~150-300 MB per subprocess, i.e. ~3-6 GB at full utilization.
POOL_MAX_SIZE: int = int(os.getenv("CLAUDE_POOL_MAX_SIZE", "20"))

# Routing key type: (user_id, chat_id, model)
PoolKey = tuple[int, int, str]


@dataclass
class StreamEvent:
    """A single streaming event from the Claude subprocess.

    Attributes:
        event_type: Type of event (content_delta, result, error, init).
        text: Incremental text (for content_delta).
        full_text: Complete response (for result).
        raw: Raw JSON event (for debugging).
        is_final: True if this is the last event of the response.
        was_cold: True if a new subprocess was started (only for init).
        subprocess_pid: PID of the used subprocess (only for init).
    """

    event_type: str
    text: str = ""
    full_text: str = ""
    raw: dict = field(default_factory=dict)
    is_final: bool = False
    was_cold: bool = False
    subprocess_pid: int = 0


@dataclass
class ManagedProcess:
    """A managed Claude subprocess for a specific user/model combination.

    Attributes:
        routing_key: (user_id, chat_id, model) 3-tuple as routing key.
        process: The asyncio subprocess.
        lock: Exclusive lock for stdin/stdout access.
        last_used: Timestamp of last use (monotonic).
        pid: Process ID for audit logging.
        is_ready: True when init phase is complete.
        model: Model ID this subprocess was started with.
    """

    routing_key: PoolKey
    process: asyncio.subprocess.Process
    lock: asyncio.Lock
    last_used: float
    pid: int
    is_ready: bool = False
    _accumulated_text: str = ""
    model: str = ""


class ClaudeProcessPool:
    """Manages persistent Claude CLI subprocesses per user/model.

    Each combination (user_id, chat_id, model) gets its own subprocess
    that is reused. This allows a user to keep all 6 models warm
    simultaneously (no cold start on slot switch).

    When max pool size (default: 20) is reached, the longest-inactive
    subprocess is terminated via LRU eviction.

    Thread safety: all methods are async-safe. Each ManagedProcess
    has its own asyncio.Lock. Per-key creation locks prevent
    race conditions on parallel first requests.
    """

    def __init__(self) -> None:
        self._processes: dict[PoolKey, ManagedProcess] = {}
        self._pool_lock = asyncio.Lock()
        self._creation_locks: dict[PoolKey, asyncio.Lock] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown = False

    @staticmethod
    def is_cli_available() -> bool:
        """Check if `claude` CLI is in PATH."""
        return shutil.which("claude") is not None

    async def start(self) -> None:
        """Start the cleanup background task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            log.info(
                "ClaudeProcessPool started (cleanup interval: %.0fs)",
                CLEANUP_INTERVAL_SECONDS,
            )

    async def shutdown(self) -> None:
        """Graceful shutdown: terminates all active subprocesses."""
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
            "ClaudeProcessPool shut down (%d processes terminated)",
            len(keys),
        )

    async def get_or_create(
        self, user_id: int, chat_id: int, model: str | None = None
    ) -> tuple[ManagedProcess, bool]:
        """Get an existing or create a new subprocess.

        Phase 2c: routing key is (user_id, chat_id, model). Each model
        gets its own subprocess. No model mismatch, no kill on slot switch:
        all models stay warm.

        On pool overflow (> POOL_MAX_SIZE) the longest-inactive subprocess
        is terminated via LRU eviction.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            model: Optional model ID. None = CLAUDE_POOL_MODEL (default).

        Returns:
            Tuple of (ManagedProcess, was_cold: bool).
            was_cold=True if a new subprocess was started.

        Raises:
            RuntimeError: If CLI is not available or start fails.
        """
        if self._shutdown:
            raise RuntimeError("ProcessPool is in shutdown mode")

        effective_model = model or CLAUDE_POOL_MODEL
        key: PoolKey = (user_id, chat_id, effective_model)

        # Fast path: existing, alive process
        async with self._pool_lock:
            managed = self._processes.get(key)
            if managed is not None and self._is_alive(managed):
                managed.last_used = time.monotonic()
                return managed, False

            # Get (or create) per-key creation lock
            if key not in self._creation_locks:
                self._creation_locks[key] = asyncio.Lock()
            creation_lock = self._creation_locks[key]

        # Per-key lock: only one spawn per key at a time
        async with creation_lock:
            # Double-check: another task may have spawned in the meantime
            async with self._pool_lock:
                managed = self._processes.get(key)
                if managed is not None and self._is_alive(managed):
                    managed.last_used = time.monotonic()
                    return managed, False

                # Process is dead: clean up
                old_managed_to_kill: ManagedProcess | None = None
                if managed is not None and not self._is_alive(managed):
                    log.warning(
                        "Subprocess for key=%s is dead (pid=%d), restarting",
                        key,
                        managed.pid,
                    )
                    old_managed_to_kill = managed
                    del self._processes[key]

            # Kill outside the pool lock
            if old_managed_to_kill is not None:
                await self._kill_process(old_managed_to_kill)

            # LRU eviction when pool limit is reached
            await self._evict_if_needed()

            # Start new subprocess
            new_managed = await self._spawn_process(key, model=effective_model)

            async with self._pool_lock:
                self._processes[key] = new_managed

            return new_managed, True

    async def send_message(
        self,
        user_id: int,
        chat_id: int,
        prompt: str,
        system_prompt: str = "",
        model: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Send a message to the subprocess and stream the response.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            prompt: User message.
            system_prompt: Optional system prompt (integrated into the content).
            model: Optional model ID (None = pool default).

        Yields:
            StreamEvent objects with incremental text and final result.

        Raises:
            RuntimeError: On subprocess crash or pipe error.
        """
        effective_model = model or CLAUDE_POOL_MODEL
        key: PoolKey = (user_id, chat_id, effective_model)
        managed, was_cold = await self.get_or_create(user_id, chat_id, model=model)

        # Wait for init regardless of was_cold (pre-warm path can
        # return was_cold=False even when is_ready is still False).
        if not managed.is_ready:
            await self._wait_for_init(managed)

        # Yield init event with process metadata (before the lock,
        # so the caller knows was_cold/pid immediately)
        yield StreamEvent(
            event_type="init",
            was_cold=was_cold,
            subprocess_pid=managed.pid,
        )

        async with managed.lock:
            managed.last_used = time.monotonic()
            managed._accumulated_text = ""

            # Format prompt as stream-json user message
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

            # Health check: is the process still alive?
            if not self._is_alive(managed):
                raise RuntimeError(f"Subprocess for key={key} died while locked")

            # Send message
            try:
                managed.process.stdin.write((message + "\n").encode("utf-8"))
                await managed.process.stdin.drain()
            except (BrokenPipeError, OSError, ConnectionResetError) as e:
                log.error("Pipe error sending to key=%s: %s", key, e)
                await self._terminate_process(key, reason="pipe_broken")
                raise RuntimeError(f"Subprocess pipe broken: {e}") from e

            # Read response events, updating last_used along the way
            async for event in self._read_response(managed):
                managed.last_used = time.monotonic()
                yield event

    async def terminate_session(
        self, user_id: int, chat_id: int, model: str | None = None
    ) -> bool:
        """Terminate subprocess(es) for a specific user session.

        If model is specified: terminate only that specific subprocess.
        If model=None: terminate all subprocesses for this user/chat.

        Args:
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.
            model: Optional model ID. None = all models for this chat.

        Returns:
            True if at least one subprocess was terminated.
        """
        if model is not None:
            effective_model = model or CLAUDE_POOL_MODEL
            return await self._terminate_process(
                (user_id, chat_id, effective_model), reason="user_request"
            )
        # Terminate all subprocesses for this user/chat
        async with self._pool_lock:
            keys_to_kill = [
                k for k in self._processes if k[0] == user_id and k[1] == chat_id
            ]
        terminated = False
        for key in keys_to_kill:
            if await self._terminate_process(key, reason="user_request"):
                terminated = True
        return terminated

    def get_stats(self) -> dict:
        """Return pool statistics (for monitoring/audit)."""
        now = time.monotonic()
        active = []
        for key, mp in self._processes.items():
            active.append(
                {
                    "user_id": key[0],
                    "chat_id": key[1],
                    "model": key[2],
                    "pid": mp.pid,
                    "idle_seconds": round(now - mp.last_used, 1),
                    "is_alive": self._is_alive(mp),
                    "is_locked": mp.lock.locked(),
                }
            )
        return {
            "active_processes": len(self._processes),
            "max_pool_size": POOL_MAX_SIZE,
            "processes": active,
        }

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _evict_if_needed(self) -> None:
        """LRU eviction: terminate the longest-inactive subprocess when pool is full.

        Skipped when the oldest subprocess is currently locked (active stream).
        In that case no eviction is performed (pool may briefly have POOL_MAX_SIZE+1).
        """
        async with self._pool_lock:
            if len(self._processes) < POOL_MAX_SIZE:
                return

            # Find the longest-inactive, unlocked process
            candidate_key: PoolKey | None = None
            oldest_time = float("inf")

            for key, mp in self._processes.items():
                if mp.lock.locked():
                    continue  # Active stream, do not evict
                if mp.last_used < oldest_time:
                    oldest_time = mp.last_used
                    candidate_key = key

            if candidate_key is None:
                log.warning(
                    "Pool full (%d/%d) but all processes are locked. "
                    "No eviction possible.",
                    len(self._processes),
                    POOL_MAX_SIZE,
                )
                return

            evict_managed = self._processes.pop(candidate_key)

        # Kill outside the pool lock
        log.info(
            "LRU-Eviction: key=%s, pid=%d (idle %.1fs)",
            candidate_key,
            evict_managed.pid,
            time.monotonic() - evict_managed.last_used,
        )
        await self._kill_process(evict_managed)

    async def _spawn_process(
        self, key: PoolKey, model: str | None = None
    ) -> ManagedProcess:
        """Start a new Claude CLI subprocess.

        Args:
            key: (user_id, chat_id, model) 3-tuple routing key.
            model: Optional model ID. None = CLAUDE_POOL_MODEL.

        Raises:
            RuntimeError: If CLI is not available.
        """
        if not self.is_cli_available():
            raise RuntimeError("claude CLI not found in PATH")

        effective_model = model or CLAUDE_POOL_MODEL

        cmd = [
            "claude",
            "--print",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",  # R04 Round 1: without this flag the CLI does not emit content_block_delta events
            "--verbose",
            "--no-session-persistence",
            # NOTE: --bare is intentionally NOT set here.
            # R10 introduced --bare to save ~5000 token init overhead,
            # but --bare causes "authentication_failed" for subscription users
            # (tested with claude-code 2.1.126). Without --bare, auth works correctly.
            "--model",
            effective_model,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        pid = proc.pid or 0
        log.info(
            "New Claude subprocess started: key=%s, pid=%d",
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
            model=effective_model,
        )

        return managed

    async def _wait_for_init(self, managed: ManagedProcess) -> None:
        """Wait until the subprocess has completed its init phase.

        Init events (system, init_session etc.) are read and discarded.
        Timeout after INIT_TIMEOUT_SECONDS. On EOF during init a
        RuntimeError is raised.
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
                # No more init events, subprocess is ready
                managed.is_ready = True
                log.debug(
                    "Subprocess pid=%d: init completed (timeout-based)",
                    managed.pid,
                )
                return

            if not line:
                # EOF: process has terminated
                raise RuntimeError(
                    f"Subprocess pid={managed.pid} terminated during init"
                )

            # Log but ignore init events
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
            "Subprocess pid=%d: init deadline reached, marking as ready", managed.pid
        )

    async def _read_response(
        self, managed: ManagedProcess
    ) -> AsyncIterator[StreamEvent]:
        """Read response events from stdout until a 'result' event arrives.

        Yields:
            StreamEvent objects (content_delta, result, error).
        """
        proc = managed.process
        if proc.stdout is None:
            raise RuntimeError("Subprocess has no stdout")

        while True:
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=120.0,  # 2 minute timeout per line
                )
            except asyncio.TimeoutError:
                log.warning("Timeout reading from pid=%d", managed.pid)
                yield StreamEvent(
                    event_type="error",
                    text="Timeout: no response from subprocess",
                    is_final=True,
                )
                return

            if not line:
                # EOF: process has died
                log.error("EOF from pid=%d during response reading", managed.pid)
                yield StreamEvent(
                    event_type="error",
                    text="Subprocess terminated unexpectedly",
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
                # Partial assembled message (ignored, we build our own)
                pass

            elif event_type == "result":
                # Response complete
                result_text = event.get("result", "")
                if isinstance(result_text, str):
                    final_text = result_text
                elif isinstance(result_text, dict):
                    # Sometimes result arrives as a dict with content array
                    blocks = result_text.get("content", [])
                    final_text = "".join(
                        b.get("text", "") for b in blocks if b.get("type") == "text"
                    )
                else:
                    final_text = str(result_text)

                # Fallback: if result is empty but accumulated text exists
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
                error_msg = event.get("error", {}).get("message", "Unknown error")
                yield StreamEvent(
                    event_type="error",
                    text=error_msg,
                    raw=event,
                    is_final=True,
                )
                return

            # Other events (system, rate_limit etc.) are ignored

    @staticmethod
    def _is_alive(managed: ManagedProcess) -> bool:
        """Check if the subprocess is still running."""
        return managed.process.returncode is None

    async def _terminate_process(self, key: PoolKey, reason: str = "") -> bool:
        """Terminate a subprocess cleanly.

        Args:
            key: (user_id, chat_id, model) 3-tuple routing key.
            reason: Reason for termination (for logging).

        Returns:
            True if a subprocess was terminated.
        """
        async with self._pool_lock:
            managed = self._processes.pop(key, None)

        if managed is None:
            return False

        await self._kill_process(managed)
        log.info(
            "Subprocess terminated: key=%s, pid=%d, reason=%s",
            key,
            managed.pid,
            reason,
        )
        return True

    @staticmethod
    async def _kill_process(managed: ManagedProcess) -> None:
        """Kill a subprocess (terminate, then kill after 3s)."""
        proc = managed.process
        if proc.returncode is not None:
            return  # Already terminated

        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass  # Process was already gone

    async def _cleanup_loop(self) -> None:
        """Background task: terminate inactive subprocesses periodically."""
        try:
            while not self._shutdown:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            pass

    async def _cleanup_expired(self) -> None:
        """Terminate idle subprocesses that have been inactive longer than INACTIVITY_TIMEOUT_SECONDS.

        Also checks if the process is currently locked (running stream).
        Locked processes are skipped and rechecked in the next cleanup cycle.
        """
        now = time.monotonic()
        expired_keys: list[PoolKey] = []

        async with self._pool_lock:
            for key, managed in self._processes.items():
                idle_time = now - managed.last_used
                if idle_time > INACTIVITY_TIMEOUT_SECONDS:
                    # Do not terminate while a stream is active
                    if managed.lock.locked():
                        log.debug(
                            "Cleanup skipped for key=%s: lock active",
                            key,
                        )
                        continue
                    expired_keys.append(key)

        for key in expired_keys:
            await self._terminate_process(key, reason="inactivity_timeout")

        if expired_keys:
            log.info(
                "Cleanup: %d inactive subprocesses terminated (keys: %s)",
                len(expired_keys),
                expired_keys,
            )
