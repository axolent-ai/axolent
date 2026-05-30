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
from dataclasses import dataclass
from typing import Optional

from application.security.input_normalizer import (
    normalize_aggressive,
    normalize_for_security_check,
)


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
# Patterns are case-insensitive and applied after NFKC + Confusables + Mn-strip.
# Phase 1.5: UTS-39 Confusables folding is now active (Cyrillic/Greek -> Latin).
# Cross-Script bypass (e.g. Cyrillic 'a' U+0430) is closed.
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
    # --- Multilingual override patterns (Finding 9, 20-locale coverage) ---
    # French
    (
        "ignore_previous_fr",
        re.compile(
            r"ignor(e[zr]?|ons)\s+(toutes?\s+)?(les\s+)?(instructions?|consignes?)\s+(pr[eé]c[eé]dentes?|ant[eé]rieures?|pr[eé]c[eé]dentes?)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Spanish
    (
        "ignore_previous_es",
        re.compile(
            r"ignora\s+(todas?\s+)?(las\s+)?(instrucciones?|directrices?)\s+(anteriores?|previas?)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Italian
    (
        "ignore_previous_it",
        re.compile(
            r"ignora\s+(tutte?\s+)?(le\s+)?(istruzioni|direttive)\s+(precedenti|anteriori)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Portuguese
    (
        "ignore_previous_pt",
        re.compile(
            r"ignor(e|ar)\s+(todas?\s+)?(as\s+)?(instru[cç][oõ]es?|diretrizes?)\s+(anteriores?|pr[eé]vias?)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Russian
    (
        "ignore_previous_ru",
        re.compile(
            r"игнорируй\s+(все\s+)?предыдущие\s+инструкции",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Japanese
    (
        "ignore_previous_ja",
        re.compile(
            r"(以前|上記|すべて)(の)?指示(を)?無視",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Chinese
    (
        "ignore_previous_zh",
        re.compile(
            r"忽略(所有|之前|上述)(的)?指令",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Arabic
    (
        "ignore_previous_ar",
        re.compile(
            r"تجاهل\s+(جميع\s+)?التعليمات\s+السابقة",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Hindi
    (
        "ignore_previous_hi",
        re.compile(
            r"(पिछले|पूर्व|सभी)\s+(निर्देशों|अनुदेशों)\s+(को\s+)?अनदेखा\s+कर",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Korean
    (
        "ignore_previous_ko",
        re.compile(
            r"(이전|위|모든)\s*(지시|명령)(을|를)?\s*무시",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Dutch
    (
        "ignore_previous_nl",
        re.compile(
            r"negeer\s+(alle\s+)?(vorige|eerdere|bovenstaande)\s+(instructies?|aanwijzingen?)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Polish
    (
        "ignore_previous_pl",
        re.compile(
            r"ignoruj\s+(wszystkie\s+)?(poprzednie|wcześniejsze)\s+(instrukcje|polecenia)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Swedish
    (
        "ignore_previous_sv",
        re.compile(
            r"ignorera\s+(alla\s+)?(tidigare|föregående)\s+(instruktioner|anvisningar)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Turkish
    (
        "ignore_previous_tr",
        re.compile(
            r"(önceki|tüm)\s+(talimatları?|yönergeleri?)\s+(yok\s+say|görmezden\s+gel|ihmal\s+et)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Ukrainian
    (
        "ignore_previous_uk",
        re.compile(
            r"ігноруй\s+(всі\s+)?попередні\s+інструкції",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Vietnamese
    (
        "ignore_previous_vi",
        re.compile(
            r"bỏ\s+qua\s+(tất\s+cả\s+)?(các\s+)?(hướng\s+dẫn|chỉ\s+thị)\s+(trước|đã\s+cho)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Thai (patterns in NFKC-normalized form since detector normalizes first)
    (
        "ignore_previous_th",
        re.compile(
            r"(ละเลย|เพิกเฉย).{0,20}(คำสั่ง|คําสั่ง|คำแนะนำ|คําแนะนํา)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # Indonesian
    (
        "ignore_previous_id",
        re.compile(
            r"abaikan\s+(semua\s+)?(instruksi|perintah)\s+(sebelumnya|di\s+atas)",
            re.IGNORECASE,
        ),
        "high",
    ),
    # --- Language-agnostic structural patterns (Finding 9/10) ---
    # Role labels that could appear in user input to spoof turns
    (
        "axolent_role_injection",
        re.compile(
            r"^\s*Axolent\s*:",
            re.MULTILINE,
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

        Two-pass normalization (Phase 1.5):
          Pass 1: Basic (NFKC + Cf-strip) -- preserves native scripts
                  for multilingual pattern matching (Russian, Hindi, etc.)
          Pass 2: Aggressive (+ Confusables + Mn-strip) -- catches
                  mixed-script bypass (Cyrillic 'a' in Latin text)

        Args:
            text: The text to check for injection patterns.

        Returns:
            InjectionMatch if a pattern matches, None if clean.
        """
        if not text or not text.strip():
            return None

        # Pass 1: Basic normalization (NFKC + Cf strip)
        # Preserves Cyrillic/Devanagari/Thai for native-script patterns
        basic = normalize_for_security_check(text)

        for pattern_name, regex, severity in self._patterns:
            match = regex.search(basic)
            if match:
                return InjectionMatch(
                    pattern_name=pattern_name,
                    matched_text=match.group(0),
                    severity=severity,
                )

        # Pass 2: Aggressive normalization (+ Confusables + Mn strip)
        # Catches mixed-script bypass (Cyrillic substitutions in Latin)
        aggressive = normalize_aggressive(text)
        if aggressive != basic:
            for pattern_name, regex, severity in self._patterns:
                match = regex.search(aggressive)
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
        all matches. Useful for audit logging. Uses two-pass normalization.

        Args:
            text: The text to check.

        Returns:
            List of all matches (empty if clean).
        """
        if not text or not text.strip():
            return []

        # Two-pass: basic + aggressive (deduplicate by pattern_name)
        basic = normalize_for_security_check(text)
        aggressive = normalize_aggressive(text)
        seen_patterns: set[str] = set()
        matches: list[InjectionMatch] = []

        for normalized in (basic, aggressive):
            for pattern_name, regex, severity in self._patterns:
                if pattern_name in seen_patterns:
                    continue
                match = regex.search(normalized)
                if match:
                    seen_patterns.add(pattern_name)
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
