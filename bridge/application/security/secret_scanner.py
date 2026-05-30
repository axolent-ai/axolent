"""Secret Scanner: multi-layered No-Model-Secret enforcement (HC-SC-13).

Prevents storage of secrets, PII, or sensitive data in memory and hypotheses.
Two-layer detection approach:

  Layer 1 (regex): Regex patterns for typical secret patterns
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

Architecture: Lives in application/security/ alongside InjectionDetector.
Imported by MemoryService (application) and PrivacyPipeline (application).

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from application.security.input_normalizer import normalize_aggressive

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
        pattern_label_key: i18n key for the secret type label
            (e.g. "secret.api_token"). Used by handlers to display
            the secret type in the user's language.
    """

    name: str
    pattern: re.Pattern[str]
    description_de: str
    pattern_label_key: str


SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    # --- API tokens and keys ---
    SecretPattern(
        name="api_token",
        pattern=re.compile(
            r"(?:sk-|ghp_|gho_|xox[bpas]-|bearer\s+|token[:\s=]+)\S{8,}",
            re.IGNORECASE,
        ),
        description_de="API-Token oder Secret",
        pattern_label_key="secret.api_token",
    ),
    SecretPattern(
        name="aws_key",
        pattern=re.compile(
            r"(?:AKIA|ASIA)[A-Z0-9]{16}",
        ),
        description_de="AWS Access Key",
        pattern_label_key="secret.aws_key",
    ),
    SecretPattern(
        name="generic_secret_assignment",
        pattern=re.compile(
            r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|private[_-]?key)"
            r"[:\s=]+\S{8,}",
            re.IGNORECASE,
        ),
        description_de="API-Key oder Secret-Key",
        pattern_label_key="secret.generic_key",
    ),
    # --- Stripe API keys (sk_live_, sk_test_, pk_live_, pk_test_) ---
    SecretPattern(
        name="stripe_key",
        pattern=re.compile(
            r"(?:sk_live_|sk_test_|pk_live_|pk_test_)[A-Za-z0-9]{24,}",
        ),
        description_de="Stripe API Key",
        pattern_label_key="secret.stripe_key",
    ),
    # --- Google API keys (AIza...) ---
    SecretPattern(
        name="google_api_key",
        pattern=re.compile(
            r"AIza[0-9A-Za-z_-]{35}",
        ),
        description_de="Google API Key",
        pattern_label_key="secret.google_key",
    ),
    # --- GitHub modern token prefixes ---
    SecretPattern(
        name="github_modern_token",
        pattern=re.compile(
            r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[usr]_[A-Za-z0-9]{36})",
        ),
        description_de="GitHub Token",
        pattern_label_key="secret.github_token",
    ),
    # --- JWT (three base64url segments) ---
    SecretPattern(
        name="jwt",
        pattern=re.compile(
            r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        ),
        description_de="JSON Web Token (JWT)",
        pattern_label_key="secret.jwt",
    ),
    # --- Financial data ---
    SecretPattern(
        name="price_currency_symbol",
        pattern=re.compile(
            r"[$€£¥]\s*\d+[.,]?\d*",
        ),
        description_de="Preisangabe",
        pattern_label_key="secret.price",
    ),
    SecretPattern(
        name="price_currency_code",
        pattern=re.compile(
            r"\d+[.,]?\d*\s*(?:EUR|USD|GBP|CHF|JPY|SEK|NOK|DKK|PLN|CZK)",
            re.IGNORECASE,
        ),
        description_de="Preisangabe",
        pattern_label_key="secret.price",
    ),
    SecretPattern(
        name="tax_format",
        pattern=re.compile(
            r"(?:MwSt|USt|VAT|Steuer)[.:]?\s*\d+[.,]?\d*\s*%",
            re.IGNORECASE,
        ),
        description_de="Steuerangabe",
        pattern_label_key="secret.tax",
    ),
    SecretPattern(
        name="iban",
        pattern=re.compile(
            r"\b[A-Z]{2}\d{2}\s?(?:[\dA-Z]{4}\s?){2,7}[\dA-Z]{1,4}\b",
        ),
        description_de="IBAN/Kontonummer",
        pattern_label_key="secret.iban",
    ),
    SecretPattern(
        name="credit_card",
        pattern=re.compile(
            r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        ),
        description_de="Kreditkartennummer",
        pattern_label_key="secret.credit_card",
    ),
    # --- Personal identifiers ---
    SecretPattern(
        name="email",
        pattern=re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        ),
        description_de="E-Mail-Adresse",
        pattern_label_key="secret.email",
    ),
    SecretPattern(
        name="phone",
        pattern=re.compile(
            r"\+?\d[\d\s\-()]{8,}\d",
        ),
        description_de="Telefonnummer",
        pattern_label_key="secret.phone",
    ),
    SecretPattern(
        name="social_security",
        pattern=re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b",
        ),
        description_de="Sozialversicherungsnummer",
        pattern_label_key="secret.ssn",
    ),
    # --- Passwords ---
    SecretPattern(
        name="password",
        pattern=re.compile(
            r"(?:passwor[td]|kennwort|password|pwd|secret|geheim)[:\s=]+\S+",
            re.IGNORECASE,
        ),
        description_de="Passwort",
        pattern_label_key="secret.password",
    ),
    # --- Long encoded strings (likely tokens/keys) ---
    SecretPattern(
        name="long_hex",
        pattern=re.compile(
            r"\b[a-fA-F0-9]{32,}\b",
        ),
        description_de="Langer Hex-String (vermutlich Token/Key)",
        pattern_label_key="secret.long_hex",
    ),
    SecretPattern(
        name="long_base64",
        pattern=re.compile(
            r"\b[A-Za-z0-9+/]{40,}={0,2}\b",
        ),
        description_de="Langer Base64-String (vermutlich Token/Key)",
        pattern_label_key="secret.long_base64",
    ),
    # --- URLs with credentials ---
    SecretPattern(
        name="url_with_credentials",
        pattern=re.compile(
            r"https?://[^:\s]+:[^@\s]+@",
        ),
        description_de="URL mit eingebetteten Zugangsdaten",
        pattern_label_key="secret.url_credentials",
    ),
    # --- IP addresses (private data context) ---
    SecretPattern(
        name="ip_address",
        pattern=re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        ),
        description_de="IP-Adresse",
        pattern_label_key="secret.ip_address",
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
        pattern_label_key: i18n key for the secret type label.
    """

    pattern_name: str
    matched_text: str
    description_de: str
    layer: int
    pattern_label_key: str = "secret.generic"


# ---------------------------------------------------------------
# Exception for defense-in-depth gate
# ---------------------------------------------------------------


class SecretBlockedError(Exception):
    """Raised when content contains detected secrets.

    Used by MemoryService.remember_episodic() to signal that
    content was blocked by the SecretScanner gate.

    Attributes:
        matches: List of detected secret matches.
    """

    def __init__(self, matches: list[SecretMatch]) -> None:
        self.matches = matches
        super().__init__(
            f"Secret blocked: {len(matches)} match(es), "
            f"first={matches[0].pattern_name if matches else 'unknown'}"
        )


# ---------------------------------------------------------------
# SecretScanner
# ---------------------------------------------------------------


# Maximum characters of matched text to keep in SecretMatch (logging safety).
MATCHED_TEXT_TRUNCATE: int = 30


class SecretScanner:
    """Multi-layered secret/PII scanner (HC-SC-13).

    Layer 1 (regex): Regex scanner for known secret patterns
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
        Applies central security normalization (NFKC + Cf strip)
        before pattern matching to prevent Zero-Width bypass.

        Args:
            text: Text to scan.

        Returns:
            List of SecretMatch objects. Empty = clean.
        """
        # Aggressive normalization: defeats Zero-Width, Compatibility-Form,
        # Combining-Mark, and Cross-Script Confusable bypasses (Phase 1.5).
        # SecretScanner patterns are all Latin-based (API keys, tokens),
        # so aggressive folding is safe here (no native-script patterns).
        text = normalize_aggressive(text)

        matches: list[SecretMatch] = []

        # Layer 2: Regex patterns
        for sp in SECRET_PATTERNS:
            m = sp.pattern.search(text)
            if m:
                # Truncate matched text for logging safety
                matched = m.group()[:MATCHED_TEXT_TRUNCATE]
                matches.append(
                    SecretMatch(
                        pattern_name=sp.name,
                        matched_text=matched,
                        description_de=sp.description_de,
                        layer=2,
                        pattern_label_key=sp.pattern_label_key,
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
                    pattern_label_key="secret.long_digit",
                )
            )

        if _USER_PATH_PATTERN.search(text):
            matches.append(
                SecretMatch(
                    pattern_name="user_file_path",
                    matched_text="[redacted]",
                    description_de="Dateipfad mit Benutzername",
                    layer=3,
                    pattern_label_key="secret.file_path",
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
                        pattern_label_key="secret.third_party_pii",
                    )
                )
                break  # One match is enough

        return matches

    def block_if_secrets(self, hypothesis: object) -> bool:
        """Check if a hypothesis should be blocked due to secrets.

        HC-SC-13: In doubt, do NOT store.

        Args:
            hypothesis: Object with a .claim attribute (str).

        Returns:
            True if the hypothesis should be BLOCKED.
        """
        claim = getattr(hypothesis, "claim", "")
        matches = self.scan(claim)
        if matches:
            hyp_id = getattr(hypothesis, "hypothesis_id", "unknown")
            # Logs only pattern name + description, never the matched value.
            log.info(  # nosemgrep
                "Secret scanner BLOCKED hypothesis %s: %d matches (first: %s / %s)",
                hyp_id,
                len(matches),
                matches[0].pattern_name,
                matches[0].description_de,
            )
            return True
        return False

    def get_block_reason(self, hypothesis: object) -> Optional[str]:
        """Get the block reason for a hypothesis.

        Returns None if the hypothesis is not blocked.

        Args:
            hypothesis: Object with a .claim attribute (str).

        Returns:
            Block reason string, or None if not blocked.
        """
        claim = getattr(hypothesis, "claim", "")
        matches = self.scan(claim)
        if not matches:
            return None

        # Build reason from first match
        first = matches[0]
        return (
            f"Secret/PII detected (Layer {first.layer}): "
            f"{first.description_de} ({first.pattern_name})"
        )
