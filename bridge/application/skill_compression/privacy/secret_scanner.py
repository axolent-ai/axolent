"""Secret Scanner: multi-layered No-Model-Secret enforcement (HC-SC-13).

Prevents hypotheses from storing secrets, PII, or sensitive data.
Two-layer detection approach:

  Layer 1 (regex): 16 regex patterns for typical secret patterns
  Layer 2 (heuristic): Heuristic filter for edge cases

Note: ALLOWED_CLAIM_PATTERNS below is a declared allowlist for future
Layer 0 (positive signal / whitelist). It is NOT consumed in production.
The current implementation is conservative: anything that matches
a secret or heuristic pattern is blocked.

Consolidates and extends the 8 regex patterns from skill_commands.py
(Step 5) into a structured, multi-layered scanner.

HC-SC-13 [BLOCKER]: No-Model-Secret multi-layered. Allowlist + Regex +
  Heuristic. When in doubt: do NOT store.

AG-SC-2 [GUARD]: test_no_secret_patterns_in_hypotheses verifies this
  scanner blocks token/price/IBAN leaks in skill storage.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from application.skill_compression.hypothesis_storage import Hypothesis

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Layer 1: Allowlist for permitted claim content
# ---------------------------------------------------------------

# Claims SHOULD contain these kinds of content.
# If a claim contains ONLY allowlisted content, it passes Layer 1.
# This is a positive signal, not a blocking mechanism.
ALLOWED_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    # Behavioral instructions (do/don't patterns)
    re.compile(
        r"(?:always|never|prefer|avoid|use|verwende|immer|nie|bevorzuge|vermeide)",
        re.IGNORECASE,
    ),
    # Format preferences
    re.compile(
        r"(?:format|style|tone|structure|layout|markdown|bullet|tabelle|formatier|stil)",
        re.IGNORECASE,
    ),
    # Workflow/process descriptions
    re.compile(
        r"(?:first|then|before|after|step|workflow|process|zuerst|dann|bevor|danach|schritt)",
        re.IGNORECASE,
    ),
    # Scope indicators
    re.compile(
        r"(?:for |bei |in |when |wenn |during |client|project|kunde|projekt)",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------
# Layer 2: Secret patterns (regex, extended from Step 5)
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SecretPattern:
    """A named regex pattern for secret detection.

    Attributes:
        name: Human-readable category name.
        pattern: Compiled regex.
        description_de: German description for user feedback.
    """

    name: str
    pattern: re.Pattern[str]
    description_de: str


SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    # --- API tokens and keys ---
    SecretPattern(
        name="api_token",
        pattern=re.compile(
            r"(?:sk-|ghp_|gho_|xox[bpas]-|bearer\s+|token[:\s=]+)\S{8,}",
            re.IGNORECASE,
        ),
        description_de="API-Token oder Secret",
    ),
    SecretPattern(
        name="aws_key",
        pattern=re.compile(
            r"(?:AKIA|ASIA)[A-Z0-9]{16}",
        ),
        description_de="AWS Access Key",
    ),
    SecretPattern(
        name="generic_secret_assignment",
        pattern=re.compile(
            r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|private[_-]?key)"
            r"[:\s=]+\S{8,}",
            re.IGNORECASE,
        ),
        description_de="API-Key oder Secret-Key",
    ),
    # --- Financial data ---
    SecretPattern(
        name="price_currency_symbol",
        pattern=re.compile(
            r"[$€£¥]\s*\d+[.,]?\d*",
        ),
        description_de="Preisangabe",
    ),
    SecretPattern(
        name="price_currency_code",
        pattern=re.compile(
            r"\d+[.,]?\d*\s*(?:EUR|USD|GBP|CHF|JPY|SEK|NOK|DKK|PLN|CZK)",
            re.IGNORECASE,
        ),
        description_de="Preisangabe",
    ),
    SecretPattern(
        name="tax_format",
        pattern=re.compile(
            r"(?:MwSt|USt|VAT|Steuer)[.:]?\s*\d+[.,]?\d*\s*%",
            re.IGNORECASE,
        ),
        description_de="Steuerangabe",
    ),
    SecretPattern(
        name="iban",
        pattern=re.compile(
            r"\b[A-Z]{2}\d{2}\s?(?:[\dA-Z]{4}\s?){2,7}[\dA-Z]{1,4}\b",
        ),
        description_de="IBAN/Kontonummer",
    ),
    SecretPattern(
        name="credit_card",
        pattern=re.compile(
            r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        ),
        description_de="Kreditkartennummer",
    ),
    # --- Personal identifiers ---
    SecretPattern(
        name="email",
        pattern=re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        ),
        description_de="E-Mail-Adresse",
    ),
    SecretPattern(
        name="phone",
        pattern=re.compile(
            r"\+?\d[\d\s\-()]{8,}\d",
        ),
        description_de="Telefonnummer",
    ),
    SecretPattern(
        name="social_security",
        pattern=re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b",
        ),
        description_de="Sozialversicherungsnummer",
    ),
    # --- Passwords ---
    SecretPattern(
        name="password",
        pattern=re.compile(
            r"(?:passwor[td]|kennwort|password|pwd|secret|geheim)[:\s=]+\S+",
            re.IGNORECASE,
        ),
        description_de="Passwort",
    ),
    # --- Long encoded strings (likely tokens/keys) ---
    SecretPattern(
        name="long_hex",
        pattern=re.compile(
            r"\b[a-fA-F0-9]{32,}\b",
        ),
        description_de="Langer Hex-String (vermutlich Token/Key)",
    ),
    SecretPattern(
        name="long_base64",
        pattern=re.compile(
            r"\b[A-Za-z0-9+/]{40,}={0,2}\b",
        ),
        description_de="Langer Base64-String (vermutlich Token/Key)",
    ),
    # --- URLs with credentials ---
    SecretPattern(
        name="url_with_credentials",
        pattern=re.compile(
            r"https?://[^:\s]+:[^@\s]+@",
        ),
        description_de="URL mit eingebetteten Zugangsdaten",
    ),
    # --- IP addresses (private data context) ---
    SecretPattern(
        name="ip_address",
        pattern=re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        ),
        description_de="IP-Adresse",
    ),
)


# ---------------------------------------------------------------
# Layer 3: Heuristic filters
# ---------------------------------------------------------------

# Long digit sequences without clear context (likely account numbers,
# order numbers, or other identifiers that should not be stored)
_LONG_DIGIT_PATTERN: re.Pattern[str] = re.compile(r"\b\d{10,}\b")

# Patterns that look like file paths with user directories
_USER_PATH_PATTERN: re.Pattern[str] = re.compile(
    r"(?:/home/|/Users/|C:\\Users\\|D:\\Users\\)\S+",
    re.IGNORECASE,
)

# Third-party personal data indicators
_THIRD_PARTY_PII_PATTERNS: list[re.Pattern[str]] = [
    # "Person X's [data]"
    re.compile(
        r"(?:name|address|phone|email|birth|age|salary|income|"
        r"adresse|telefon|geburt|alter|gehalt|einkommen)"
        r"\s+(?:of|von|des|der)\s+\S+",
        re.IGNORECASE,
    ),
    # Direct name patterns: "Herr/Frau/Mr/Mrs Lastname"
    re.compile(
        r"(?:Herr|Frau|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
    ),
]


# ---------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SecretMatch:
    """A detected secret/PII match.

    Attributes:
        pattern_name: Name of the matched pattern.
        matched_text: The actual text that matched (truncated).
        description_de: German description for user feedback.
        layer: Which detection layer found it (1-3).
    """

    pattern_name: str
    matched_text: str
    description_de: str
    layer: int


# ---------------------------------------------------------------
# SecretScanner
# ---------------------------------------------------------------


class SecretScanner:
    """Multi-layered secret/PII scanner for hypothesis claims (HC-SC-13).

    Layer 1 (regex): Regex scanner for known secret patterns (16 patterns)
    Layer 2 (heuristic): Heuristic filter for edge cases

    Note: ALLOWED_CLAIM_PATTERNS is declared but not consumed.
    See module docstring for rationale.

    Usage:
        scanner = SecretScanner()
        matches = scanner.scan(text)
        if matches:
            # text contains secrets, block storage

        # Or for hypothesis-level check:
        if scanner.block_if_secrets(hypothesis):
            reason = scanner.get_block_reason(hypothesis)
    """

    def scan(self, text: str) -> list[SecretMatch]:
        """Scan text for secrets and PII.

        Runs Layer 2 (regex) and Layer 3 (heuristic) checks.

        Args:
            text: Text to scan.

        Returns:
            List of SecretMatch objects. Empty = clean.
        """
        matches: list[SecretMatch] = []

        # Layer 2: Regex patterns
        for sp in SECRET_PATTERNS:
            m = sp.pattern.search(text)
            if m:
                # Truncate matched text for logging safety
                matched = m.group()[:30]
                matches.append(
                    SecretMatch(
                        pattern_name=sp.name,
                        matched_text=matched,
                        description_de=sp.description_de,
                        layer=2,
                    )
                )

        # Layer 3: Heuristic checks
        if _LONG_DIGIT_PATTERN.search(text):
            matches.append(
                SecretMatch(
                    pattern_name="long_digit_sequence",
                    matched_text="[redacted]",
                    description_de="Lange Ziffernfolge (vermutlich Kontonummer/ID)",
                    layer=3,
                )
            )

        if _USER_PATH_PATTERN.search(text):
            matches.append(
                SecretMatch(
                    pattern_name="user_file_path",
                    matched_text="[redacted]",
                    description_de="Dateipfad mit Benutzername",
                    layer=3,
                )
            )

        for pii_pattern in _THIRD_PARTY_PII_PATTERNS:
            m = pii_pattern.search(text)
            if m:
                matches.append(
                    SecretMatch(
                        pattern_name="third_party_pii",
                        matched_text=m.group()[:20],
                        description_de="Personenbezogene Daten Dritter",
                        layer=3,
                    )
                )
                break  # One match is enough

        return matches

    def block_if_secrets(self, hypothesis: Hypothesis) -> bool:
        """Check if a hypothesis should be blocked due to secrets.

        HC-SC-13: In doubt, do NOT store.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            True if the hypothesis should be BLOCKED.
        """
        matches = self.scan(hypothesis.claim)
        if matches:
            # Logs only pattern name + description, never the matched value.
            log.info(  # nosemgrep
                "Secret scanner BLOCKED hypothesis %s: %d matches (first: %s / %s)",
                hypothesis.hypothesis_id,
                len(matches),
                matches[0].pattern_name,
                matches[0].description_de,
            )
            return True
        return False

    def get_block_reason(self, hypothesis: Hypothesis) -> Optional[str]:
        """Get the block reason for a hypothesis.

        Returns None if the hypothesis is not blocked.

        Args:
            hypothesis: The hypothesis to check.

        Returns:
            Block reason string, or None if not blocked.
        """
        matches = self.scan(hypothesis.claim)
        if not matches:
            return None

        # Build reason from first match
        first = matches[0]
        return (
            f"Secret/PII detected (Layer {first.layer}): "
            f"{first.description_de} ({first.pattern_name})"
        )
