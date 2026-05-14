"""Streaming adapter: applies Text Guard to token streams.

Buffers partial words until word boundaries, then runs rules
on complete words. Pass-through for known-clean tokens.

Critical for AXOLENT bot: live token output through the filter.

Pattern:
    * Token chunks accumulate until word boundary (space, newline, punctuation)
    * On complete word: run through rule set
    * On match: yield corrected token
    * On no match: yield original token

Pure domain logic, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from domain.text_guard.guard import TextGuard

# Characters that signal a word boundary
_WORD_BOUNDARY_CHARS = set(" \t\n\r.,;:!?()[]{}\"'`/\\<>@#$%^&*+=|~")

# Pattern to split text into words and non-words
_WORD_SPLIT = re.compile(r"(\s+|[^\w]+)")


class StreamingTextGuard:
    """Apply text-guard rules to token streams.

    Buffers partial words until word boundary, then runs rules
    on complete words. Pass-through for known-clean tokens.

    Usage:
        guard = TextGuard(rule_set)
        stream_guard = StreamingTextGuard(guard)
        async for token in stream_guard.filter(token_stream):
            send_to_user(token)
    """

    def __init__(self, guard: TextGuard) -> None:
        self._guard = guard
        self._buffer = ""
        self._in_code_block = False
        self._backtick_count = 0

    def reset(self) -> None:
        """Reset the streaming state for a new message."""
        self._buffer = ""
        self._in_code_block = False
        self._backtick_count = 0

    async def filter(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """Filter a token stream, correcting diacritics on the fly.

        Yields tokens as soon as word boundaries are detected.
        Buffers partial words to avoid splitting in the middle of
        a correctable word.

        Args:
            token_stream: Async iterator of string tokens from LLM.

        Yields:
            Corrected (or unchanged) string tokens.
        """
        async for token in token_stream:
            self._buffer += token

            # Track code block state
            self._track_code_blocks(token)

            # Try to flush complete words from the buffer
            flushed = self._flush_complete_words()
            if flushed:
                yield flushed

        # Flush any remaining buffer at end of stream
        if self._buffer:
            yield self._process_remaining()

    def process_token(self, token: str) -> str | None:
        """Process a single token synchronously.

        Returns corrected text if a word boundary was found,
        None if the token was buffered (waiting for more input).

        This is the synchronous API for integration with existing
        streaming handlers that are not async-iterator based.

        Args:
            token: A single token string.

        Returns:
            Corrected text to emit, or None if buffered.
        """
        self._buffer += token
        self._track_code_blocks(token)
        flushed = self._flush_complete_words()
        return flushed if flushed else None

    def flush(self) -> str:
        """Flush any remaining buffered text.

        Call this at end of stream to get any remaining content.

        Returns:
            Remaining corrected text.
        """
        if self._buffer:
            result = self._process_remaining()
            return result
        return ""

    def _track_code_blocks(self, token: str) -> None:
        """Track fenced code block state from token content."""
        for char in token:
            if char == "`":
                self._backtick_count += 1
            else:
                if self._backtick_count >= 3:
                    self._in_code_block = not self._in_code_block
                self._backtick_count = 0

    def _flush_complete_words(self) -> str:
        """Extract and correct complete words from the buffer.

        Keeps the last partial word in the buffer.

        Returns:
            Corrected text from complete words, empty string if none.
        """
        if not self._buffer:
            return ""

        # Find the last word boundary position
        last_boundary = -1
        for i in range(len(self._buffer) - 1, -1, -1):
            if self._buffer[i] in _WORD_BOUNDARY_CHARS:
                last_boundary = i
                break

        if last_boundary < 0:
            return ""

        # Everything up to and including the boundary is ready
        ready = self._buffer[: last_boundary + 1]
        self._buffer = self._buffer[last_boundary + 1 :]

        if self._in_code_block:
            return ready

        return self._correct_text(ready)

    def _process_remaining(self) -> str:
        """Process remaining buffer content at end of stream."""
        text = self._buffer
        self._buffer = ""

        if self._in_code_block:
            return text

        return self._correct_text(text)

    def _correct_text(self, text: str) -> str:
        """Apply word-level corrections to a text fragment."""
        parts = _WORD_SPLIT.split(text)
        result: list[str] = []
        for part in parts:
            if part and not _WORD_SPLIT.match(part):
                result.append(self._guard.fix_word(part))
            else:
                result.append(part)
        return "".join(result)
