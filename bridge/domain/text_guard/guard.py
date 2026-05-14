"""Core Text Guard engine: language-agnostic diacritic correction.

Applies word-list based replacement with word-boundary matching
to avoid false positives on loan words.

Pure domain logic: no I/O, no file access.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from domain.text_guard.models import Issue, RuleSet, WordPair

log = logging.getLogger(__name__)

# Pattern to detect fenced code blocks in markdown
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# Pattern to detect inline code in markdown
_INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")


class TextGuard:
    """Language-agnostic text correction engine.

    Loads a RuleSet and applies word-list based replacements
    with word boundaries to avoid false positives on loan words.

    Usage:
        guard = TextGuard(rule_set, mode="fix")
        corrected = guard.fix("Ich erklaere dir das fuer dich.")

        guard = TextGuard(rule_set, mode="check")
        issues = guard.check("Ich erklaere dir das fuer dich.")
    """

    def __init__(
        self,
        rule_set: RuleSet,
        *,
        mode: Literal["check", "fix"] = "fix",
    ) -> None:
        self._rule_set = rule_set
        self._mode = mode
        self._whitelist_lower = {w.lower() for w in rule_set.loan_word_whitelist}

        # Pre-compile regex patterns for each word pair
        self._compiled_pairs: list[tuple[WordPair, re.Pattern[str]]] = []
        for pair in rule_set.word_pairs:
            flags = 0 if pair.case_sensitive else re.IGNORECASE
            pattern = re.compile(r"\b" + re.escape(pair.ascii_form) + r"\b", flags)
            self._compiled_pairs.append((pair, pattern))

    @property
    def language(self) -> str:
        """The language code this guard operates on."""
        return self._rule_set.language

    @property
    def rule_count(self) -> int:
        """Number of word pairs in the rule set."""
        return len(self._rule_set.word_pairs)

    def check(self, text: str) -> list[Issue]:
        """Return list of detected issues without modifying text.

        Args:
            text: Input text to check.

        Returns:
            List of Issue objects, one per detected problem.
        """
        if not text or not text.strip():
            return []

        issues: list[Issue] = []
        lines = text.split("\n")

        # Build set of code-block ranges to skip
        skip_ranges: list[tuple[int, int]] = []
        if self._rule_set.code_block_skip:
            skip_ranges = self._get_skip_ranges(text)

        offset = 0
        for line_num, line in enumerate(lines, start=1):
            line_start = offset
            offset += len(line) + 1  # +1 for newline

            for pair, pattern in self._compiled_pairs:
                for match in pattern.finditer(line):
                    abs_start = line_start + match.start()
                    abs_end = line_start + match.end()

                    # Skip if inside code block
                    if self._in_skip_range(abs_start, abs_end, skip_ranges):
                        continue

                    # Skip if it is a whitelisted loan word
                    matched_word = match.group(0)
                    if matched_word.lower() in self._whitelist_lower:
                        continue

                    issues.append(
                        Issue(
                            line=line_num,
                            column=match.start(),
                            ascii_form=matched_word,
                            correct_form=self._get_replacement(matched_word, pair),
                            excerpt=line.strip(),
                        )
                    )

        return issues

    def fix(self, text: str) -> str:
        """Return corrected text with ASCII diacritics replaced.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text.
        """
        if not text or not text.strip():
            return text

        # Build skip ranges for code blocks
        skip_ranges: list[tuple[int, int]] = []
        if self._rule_set.code_block_skip:
            skip_ranges = self._get_skip_ranges(text)

        # Apply replacements using a single-pass approach per pair
        result = text
        # Track cumulative offset shift from replacements
        for pair, pattern in self._compiled_pairs:
            result = self._apply_pair(result, pair, pattern, skip_ranges)
            # Recompute skip ranges after each replacement pass
            # because positions shift
            if self._rule_set.code_block_skip:
                skip_ranges = self._get_skip_ranges(result)

        return result

    def fix_word(self, word: str) -> str:
        """Fix a single word (used by streaming adapter).

        No code-block detection (caller handles that).

        Args:
            word: A single word token.

        Returns:
            Corrected word, or original if no rule matches.
        """
        if not word:
            return word

        if word.lower() in self._whitelist_lower:
            return word

        for pair, pattern in self._compiled_pairs:
            match = pattern.fullmatch(word)
            if match:
                return self._get_replacement(word, pair)

        return word

    def _apply_pair(
        self,
        text: str,
        pair: WordPair,
        pattern: re.Pattern[str],
        skip_ranges: list[tuple[int, int]],
    ) -> str:
        """Apply a single word pair replacement across the text.

        Processes matches right-to-left so earlier positions stay valid.
        """
        matches = list(pattern.finditer(text))
        if not matches:
            return text

        # Process right-to-left to preserve positions
        result_chars = list(text)
        for match in reversed(matches):
            start, end = match.start(), match.end()

            # Skip if inside code block
            if self._in_skip_range(start, end, skip_ranges):
                continue

            # Skip whitelisted words
            matched_word = match.group(0)
            if matched_word.lower() in self._whitelist_lower:
                continue

            replacement = self._get_replacement(matched_word, pair)
            result_chars[start:end] = list(replacement)

        return "".join(result_chars)

    @staticmethod
    def _get_replacement(matched: str, pair: WordPair) -> str:
        """Compute the replacement preserving the original case pattern.

        If the matched word is ALL CAPS, the replacement is ALL CAPS.
        If the matched word is Title Case, the replacement is Title Case.
        Otherwise, the replacement uses the correct_form as-is.
        """
        if matched.isupper():
            return pair.correct_form.upper()
        if matched[0].isupper() and pair.ascii_form[0].islower():
            return pair.correct_form[0].upper() + pair.correct_form[1:]
        return pair.correct_form

    @staticmethod
    def _get_skip_ranges(text: str) -> list[tuple[int, int]]:
        """Find character ranges that should be skipped (code blocks, inline code)."""
        ranges: list[tuple[int, int]] = []
        for match in _CODE_BLOCK_PATTERN.finditer(text):
            ranges.append((match.start(), match.end()))
        for match in _INLINE_CODE_PATTERN.finditer(text):
            ranges.append((match.start(), match.end()))
        return sorted(ranges)

    @staticmethod
    def _in_skip_range(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
        """Check if a position overlaps with any skip range."""
        for r_start, r_end in ranges:
            if start >= r_start and end <= r_end:
                return True
            if r_start > end:
                break
        return False
