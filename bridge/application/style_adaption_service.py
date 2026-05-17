"""Style Adaption Service: learns and mirrors user communication style.

Observes user messages over time to build a style profile per user.
The profile captures: emoji usage, formality level, language mixing,
tonality, and device signals. This profile is injected into the
system prompt so the LLM can adapt its output accordingly.

Part of P3 (Contextual Silence with Style Adaptation).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# Thresholds for device detection
_MOBILE_EMOJI_DENSITY_THRESHOLD = 0.05  # 5% of chars are emojis
_MOBILE_AVG_MSG_LENGTH_THRESHOLD = 60  # chars
_MOBILE_TYPO_DENSITY_THRESHOLD = 0.03  # 3% words look like typos

# Minimum messages before we start making style assertions
_MIN_MESSAGES_FOR_PROFILE = 5

# Emoji regex (simplified, covers most common unicode emoji ranges)
_EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map symbols
    r"\U0001F1E0-\U0001F1FF"  # flags
    r"\U00002702-\U000027B0"  # dingbats
    r"\U0000FE00-\U0000FE0F"  # variation selectors
    r"\U0001F900-\U0001F9FF"  # supplemental symbols
    r"\U00002600-\U000026FF"  # misc symbols
    r"]",
    re.UNICODE,
)

# Common typo patterns (typical smartphone autocorrect artifacts)
_TYPO_INDICATORS = re.compile(
    r"\b[a-z]*[A-Z][a-z]*[A-Z]\b"  # random caps mid-word
    r"|\b\w{1,2}\b(?:\s\b\w{1,2}\b){3,}"  # many very short words in a row
)


@dataclass
class StyleProfile:
    """Captured style signals for a user.

    Attributes:
        emoji_frequency: Ratio of messages containing emojis (0.0 to 1.0).
        avg_message_length: Average character count per message.
        formality: Detected formality ('du', 'sie', or 'unknown').
        uses_code_switching: Whether user mixes languages.
        tonality: 'terse', 'warm', or 'neutral'.
        device_signal: 'mobile', 'desktop', or 'unknown'.
        observed_messages: Number of messages observed so far.
        last_updated: Unix timestamp of last observation.
        custom_words: Frequently used special words or slang.
    """

    emoji_frequency: float = 0.0
    avg_message_length: float = 0.0
    formality: str = "unknown"
    uses_code_switching: bool = False
    tonality: str = "neutral"
    device_signal: str = "unknown"
    observed_messages: int = 0
    last_updated: float = 0.0
    custom_words: list[str] = field(default_factory=list)

    def is_mature(self) -> bool:
        """Check if enough data has been collected for reliable assertions."""
        return self.observed_messages >= _MIN_MESSAGES_FOR_PROFILE

    def to_prompt_block(self, lang: str = "de") -> str:
        """Convert the profile into a system prompt injection block.

        Only produces output if the profile is mature enough.

        Args:
            lang: Language code for the block text.

        Returns:
            Prompt block string, or empty string if profile is immature.
        """
        if not self.is_mature():
            return ""

        lines: list[str] = ["[USER STYLE PROFILE]"]

        # Emoji guidance
        if self.emoji_frequency > 0.3:
            lines.append("User uses emojis frequently. You may mirror this.")
        elif self.emoji_frequency < 0.05:
            lines.append("User rarely uses emojis. Keep your responses emoji-free.")

        # Formality
        if self.formality == "du":
            lines.append("User prefers informal address (Du). Mirror this.")
        elif self.formality == "sie":
            lines.append("User prefers formal address (Sie). Mirror this.")

        # Tonality
        if self.tonality == "terse":
            lines.append(
                "User communicates in terse, direct style. "
                "Match this: short sentences, no filler."
            )
        elif self.tonality == "warm":
            lines.append(
                "User communicates warmly and elaborately. "
                "You may be warmer and slightly more verbose."
            )

        # Code-switching
        if self.uses_code_switching:
            lines.append(
                "User mixes languages (code-switching). "
                "Accept this naturally, do not ask or correct."
            )

        # Device
        if self.device_signal == "mobile":
            lines.append(
                "User appears to be on mobile. "
                "Format responses compactly: shorter paragraphs, less nesting."
            )

        # Custom vocabulary
        if self.custom_words:
            words_str = ", ".join(self.custom_words[:10])
            lines.append(f"User frequently uses: {words_str}. Recognize these.")

        if len(lines) <= 1:
            return ""

        return "\n".join(lines)


class StyleAdaptionService:
    """Observes user messages and maintains style profiles.

    In-memory storage (profiles reset on bot restart). Future versions
    can persist to SQLite for cross-session learning.

    Since Phase 2: includes anti-repetition awareness. The prompt block
    now includes a rule to avoid filler word repetition (e.g. "Gerne").
    """

    # Filler words that the bot tends to overuse (per language)
    REPETITION_FILLERS: dict[str, list[str]] = {
        "de": [
            "Gerne",
            "Sicher",
            "Natürlich",
            "Selbstverständlich",
            "Klar",
            "Absolut",
        ],
        "en": [
            "Sure",
            "Certainly",
            "Of course",
            "Absolutely",
            "Great question",
            "Happy to help",
        ],
    }

    def __init__(self) -> None:
        self._profiles: dict[int, StyleProfile] = {}
        # Rolling buffer of recent message metadata per user
        self._message_buffer: dict[int, list[dict]] = {}
        self._buffer_max_size = 50

    def observe(self, user_id: int, message_text: str) -> None:
        """Observe a user message and update the style profile.

        Called on every incoming user message. Lightweight analysis
        that does not block the response pipeline.

        Args:
            user_id: Telegram user ID.
            message_text: The raw message text from the user.
        """
        if not message_text or not message_text.strip():
            return

        # Get or create profile
        profile = self._profiles.setdefault(user_id, StyleProfile())

        # Get or create message buffer
        buffer = self._message_buffer.setdefault(user_id, [])

        # Analyze this message
        msg_meta = self._analyze_message(message_text)
        buffer.append(msg_meta)

        # Trim buffer
        if len(buffer) > self._buffer_max_size:
            buffer[:] = buffer[-self._buffer_max_size :]

        # Update profile from buffer
        self._update_profile(profile, buffer)
        profile.observed_messages = len(buffer)
        profile.last_updated = time.time()

    def get_profile(self, user_id: int) -> Optional[StyleProfile]:
        """Get the current style profile for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            StyleProfile or None if no observations yet.
        """
        return self._profiles.get(user_id)

    def get_prompt_block(self, user_id: int, lang: str = "de") -> str:
        """Get the style profile as a system prompt block.

        Includes anti-repetition rule for all users (even before
        profile is mature), since filler word overuse is a systemic issue.

        Args:
            user_id: Telegram user ID.
            lang: Language code.

        Returns:
            Prompt block string (may contain only anti-repetition rule
            if profile is not yet mature).
        """
        parts: list[str] = []

        profile = self._profiles.get(user_id)
        if profile is not None:
            profile_block = profile.to_prompt_block(lang)
            if profile_block:
                parts.append(profile_block)

        # Anti-repetition rule (always active, regardless of profile maturity)
        anti_rep = self._get_anti_repetition_block(lang)
        if anti_rep:
            parts.append(anti_rep)

        return "\n\n".join(parts)

    def _get_anti_repetition_block(self, lang: str = "de") -> str:
        """Build the anti-repetition prompt block.

        Args:
            lang: Language code.

        Returns:
            Anti-repetition instruction block.
        """
        fillers = self.REPETITION_FILLERS.get(lang, self.REPETITION_FILLERS["en"])
        filler_str = ", ".join(f"'{w}'" for w in fillers[:4])

        if lang == "de":
            return (
                "[ANTI-REPETITION]\n"
                f"Vermeide repetitive Satzanfaenge und Fuellwoerter wie {filler_str}. "
                "Variiere bewusst. Beginne Antworten NICHT mit Floskeln. "
                "Komme direkt zum Inhalt."
            )
        return (
            "[ANTI-REPETITION]\n"
            f"Avoid repetitive sentence starters and filler words like {filler_str}. "
            "Vary your openings consciously. Do NOT start responses with pleasantries. "
            "Get straight to the content."
        )

    def check_repetition_warning(self, response: str, lang: str = "de") -> str | None:
        """Check a response for filler word overuse.

        Post-response check that can be used by callers to flag
        quality issues. Does NOT modify the response.

        Args:
            response: The LLM response text.
            lang: Language code.

        Returns:
            Warning message if repetition detected, None otherwise.
        """
        fillers = self.REPETITION_FILLERS.get(lang, self.REPETITION_FILLERS["en"])
        response_lower = response.lower()

        hits: list[str] = []
        for filler in fillers:
            count = response_lower.count(filler.lower())
            if count >= 2:
                hits.append(f"{filler} (x{count})")

        if hits:
            return f"Repetition detected: {', '.join(hits)}"
        return None

    def _analyze_message(self, text: str) -> dict:
        """Analyze a single message for style signals.

        Args:
            text: Message text.

        Returns:
            Dict with analysis results.
        """
        emoji_count = len(_EMOJI_PATTERN.findall(text))
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        # Formality detection (German: Du vs Sie)
        formality = "unknown"
        text_lower = text.lower()
        if any(
            w in text_lower.split()
            for w in ("sie", "ihnen", "ihrer", "ihrem")
            if not text_lower.startswith("sie ")  # "sie" as "they" is informal
        ):
            # Only count as formal if used in addressing context
            if "sie" in text_lower.split()[:3] or "ihnen" in text_lower.split():
                formality = "sie"
        if any(w in text_lower.split() for w in ("du", "dir", "dich", "dein")):
            formality = "du"

        # Tonality: terse vs warm
        tonality = "neutral"
        if char_count < 30 and word_count < 8:
            tonality = "terse"
        elif char_count > 150 or word_count > 25:
            tonality = "warm"

        # Code-switching: detect English words in German text
        english_indicators = {
            "the",
            "is",
            "are",
            "was",
            "have",
            "just",
            "actually",
            "like",
            "really",
            "maybe",
            "sorry",
            "sure",
            "okay",
            "nice",
            "cool",
            "whatever",
            "anyway",
            "btw",
            "lol",
        }
        words_lower = {w.lower().strip(".,!?") for w in words}
        has_code_switching = len(words_lower & english_indicators) >= 2

        return {
            "emoji_count": emoji_count,
            "char_count": char_count,
            "word_count": word_count,
            "emoji_density": emoji_count / max(char_count, 1),
            "formality": formality,
            "tonality": tonality,
            "has_code_switching": has_code_switching,
            "timestamp": time.time(),
        }

    def _update_profile(self, profile: StyleProfile, buffer: list[dict]) -> None:
        """Update profile from the message buffer.

        Args:
            profile: The StyleProfile to update.
            buffer: List of message analysis dicts.
        """
        if not buffer:
            return

        n = len(buffer)

        # Emoji frequency: fraction of messages with at least one emoji
        msgs_with_emoji = sum(1 for m in buffer if m["emoji_count"] > 0)
        profile.emoji_frequency = msgs_with_emoji / n

        # Average message length
        profile.avg_message_length = sum(m["char_count"] for m in buffer) / n

        # Formality: majority vote
        formality_votes = [
            m["formality"] for m in buffer if m["formality"] != "unknown"
        ]
        if formality_votes:
            du_count = formality_votes.count("du")
            sie_count = formality_votes.count("sie")
            if du_count > sie_count:
                profile.formality = "du"
            elif sie_count > du_count:
                profile.formality = "sie"
            else:
                profile.formality = "unknown"

        # Tonality: majority vote
        tonality_votes = [m["tonality"] for m in buffer if m["tonality"] != "neutral"]
        if tonality_votes:
            terse_count = tonality_votes.count("terse")
            warm_count = tonality_votes.count("warm")
            if terse_count > warm_count and terse_count > n * 0.4:
                profile.tonality = "terse"
            elif warm_count > terse_count and warm_count > n * 0.4:
                profile.tonality = "warm"
            else:
                profile.tonality = "neutral"

        # Code-switching: if 30%+ of messages show it
        cs_count = sum(1 for m in buffer if m["has_code_switching"])
        profile.uses_code_switching = cs_count / n > 0.3

        # Device signal: based on message length + emoji density
        avg_emoji_density = sum(m["emoji_density"] for m in buffer) / n
        if (
            profile.avg_message_length < _MOBILE_AVG_MSG_LENGTH_THRESHOLD
            and avg_emoji_density > _MOBILE_EMOJI_DENSITY_THRESHOLD
        ):
            profile.device_signal = "mobile"
        elif profile.avg_message_length > 120:
            profile.device_signal = "desktop"
        else:
            profile.device_signal = "unknown"
