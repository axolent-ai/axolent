"""Prompt injection detection for user-supplied content.

Used by:
  - /remember handler (GAP-05): blocks malicious memory entries
  - ImportOrchestrator (GAP-03): detects injections in imported conversations

This is a lightweight, regex-based detector. It catches the most common
prompt injection patterns without requiring an LLM call. Coverage is
estimated at ~80% of known attack vectors (prefix/suffix/role-play).

NOT a silver bullet. Defense-in-depth requires:
  1. This detector (fast, zero-cost, blocks obvious attacks)
  2. Delimiter wrapping (reduces model confusion)
  3. Model-level instruction hierarchy (Claude's system prompt priority)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class InjectionMatch:
    """Result of an injection detection match.

    Attributes:
        pattern_name: Identifier for the matched pattern.
        matched_text: The text substring that triggered the match.
        severity: Estimated severity ("high", "medium", "low").
    """

    pattern_name: str
    matched_text: str
    severity: str


# Compiled regex patterns for prompt injection detection.
# Each tuple: (pattern_name, compiled_regex, severity)
# Patterns are case-insensitive and applied after NFKC normalization.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Direct override attempts
    (
        "ignore_previous_instructions",
        re.compile(
            r"ignore\s+(the\s+)?(previous|above|all|prior|earlier)(\s+\w+)?\s+instructions?",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "disregard_instructions",
        re.compile(
            r"disregard\s+(the\s+)?(previous|above|all|prior|earlier)(\s+\w+)?\s+instructions?",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "forget_instructions",
        re.compile(
            r"forget\s+(everything|your|all|the)\s*(instructions?|rules?|constraints?|guidelines?)?",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "new_instructions",
        re.compile(
            r"(new|updated?|revised?|override)\s+instructions?\s*:",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Role confusion / impersonation
    (
        "role_tag_injection",
        re.compile(
            r"(system\s*:|assistant\s*:|<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\])",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "act_as_pretend",
        re.compile(
            r"(act\s+as|pretend\s+(you\s+are|to\s+be)|you\s+are\s+now|from\s+now\s+on\s+you\s+are)",
            re.IGNORECASE,
        ),
        "medium",
    ),
    (
        "developer_mode",
        re.compile(
            r"(enable|activate|enter)\s+(developer|admin|debug|god|unrestricted)\s+mode",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Exfiltration attempts
    (
        "reveal_system_prompt",
        re.compile(
            r"(reveal|show|output|display|print|repeat|share)\s+(your\s+)?(system\s+prompt|instructions?|initial\s+prompt|hidden\s+instructions?)",
            re.IGNORECASE,
        ),
        "medium",
    ),
    (
        "repeat_above",
        re.compile(
            r"repeat\s+(everything|all|the\s+text)\s+(above|before)\s+(this|here)",
            re.IGNORECASE,
        ),
        "medium",
    ),
    # Authority impersonation
    (
        "admin_system_update",
        re.compile(
            r"\[(system|admin)\s+(update|message|override|notice)\s*(from)?\s*",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "anthropic_override",
        re.compile(
            r"(anthropic|openai|google|meta)\s+(admin|team|update|override|policy)",
            re.IGNORECASE,
        ),
        "medium",
    ),
    # XML/HTML escape attempts
    (
        "xml_system_tag",
        re.compile(
            r"</?system>|</assistant_response>|<\|system\|>",
            re.IGNORECASE,
        ),
        "high",
    ),
    # DAN and jailbreak variants
    (
        "dan_jailbreak",
        re.compile(
            r"you\s+are\s+(now\s+)?(DAN|evil|unrestricted|jailbroken|unfiltered)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # German variants (i18n injection)
    (
        "ignore_previous_de",
        re.compile(
            r"ignorier(e|en?)\s+(alle\s+)?(vorherigen?|obigen?|bisherigen?)\s+(anweisungen?|instruktionen?|regeln?)",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "new_instructions_de",
        re.compile(
            r"(neue|aktualisierte)\s+(anweisungen?|instruktionen?)\s*:",
            re.IGNORECASE,
        ),
        "high",
    ),
]


class InjectionDetector:
    """Detects prompt injection patterns in user-supplied text.

    Thread-safe: all state is immutable after construction.

    Usage:
        detector = InjectionDetector()
        match = detector.check("Ignore all previous instructions...")
        if match is not None:
            # Block the content
            log.warning("Injection detected: %s", match.pattern_name)
    """

    def __init__(
        self,
        *,
        extra_patterns: list[tuple[str, re.Pattern, str]] | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            extra_patterns: Optional additional patterns to check.
                Each tuple is (name, compiled_regex, severity).
        """
        self._patterns = list(_INJECTION_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def check(self, text: str) -> Optional[InjectionMatch]:
        """Check text for prompt injection patterns.

        Normalizes text (NFKC) before pattern matching to prevent
        homoglyph bypass (GAP-08 defense-in-depth).

        Args:
            text: The text to check for injection patterns.

        Returns:
            InjectionMatch if a pattern matches, None if clean.
        """
        if not text or not text.strip():
            return None

        # NFKC normalization: defeats homoglyph attacks
        normalized = unicodedata.normalize("NFKC", text)

        for pattern_name, regex, severity in self._patterns:
            match = regex.search(normalized)
            if match:
                return InjectionMatch(
                    pattern_name=pattern_name,
                    matched_text=match.group(0),
                    severity=severity,
                )

        return None

    def check_all(self, text: str) -> list[InjectionMatch]:
        """Check text for ALL matching injection patterns.

        Unlike check() which returns on first match, this returns
        all matches. Useful for audit logging.

        Args:
            text: The text to check.

        Returns:
            List of all matches (empty if clean).
        """
        if not text or not text.strip():
            return []

        normalized = unicodedata.normalize("NFKC", text)
        matches: list[InjectionMatch] = []

        for pattern_name, regex, severity in self._patterns:
            match = regex.search(normalized)
            if match:
                matches.append(
                    InjectionMatch(
                        pattern_name=pattern_name,
                        matched_text=match.group(0),
                        severity=severity,
                    )
                )

        return matches

    def is_suspicious(self, text: str) -> bool:
        """Quick boolean check: is text suspicious?

        Args:
            text: The text to check.

        Returns:
            True if any injection pattern matches.
        """
        return self.check(text) is not None
