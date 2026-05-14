"""Simple language detection for user messages.

No external dependencies. Uses heuristics based on
frequent words and character patterns. Sufficient for
distinguishing the most common languages (de, en, es, fr).

Fallback: "de" (default language from onboarding).
"""

import logging
import re

log = logging.getLogger(__name__)

# Frequent function words per language (lowercase).
# Short, unambiguous words that appear in almost every sentence.
_MARKERS: dict[str, set[str]] = {
    "en": {
        "i",
        "the",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "shall",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "not",
        "but",
        "and",
        "or",
        "for",
        "with",
        "from",
        "about",
        "into",
        "through",
        "you",
        "your",
        "me",
        "my",
        "his",
        "her",
        "its",
        "our",
        "their",
        "i'm",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "can't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "haven't",
        "hasn't",
        "please",
        "thanks",
        "thank",
        "yes",
        "no",
        "okay",
        "here",
        "there",
        "very",
        "just",
        "also",
        "too",
        "need",
        "help",
        "want",
        "know",
        "think",
        "tell",
        "hello",
        "hi",
        "hey",
        "sure",
        "great",
        "good",
        "it",
        "if",
        "of",
        "on",
        "at",
        "to",
        "by",
        "up",
        "so",
        "be",
        "an",
        "am",
        "he",
        "she",
        "we",
        "they",
        "some",
        "any",
        "all",
        "each",
        "every",
        "more",
        "much",
        "get",
        "got",
        "make",
        "take",
        "give",
        "go",
        "come",
        "see",
        "look",
        "find",
        "work",
        "try",
        "use",
        "show",
    },
    "de": {
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "einer",
        "eines",
        "ist",
        "sind",
        "war",
        "waren",
        "hat",
        "haben",
        "hatte",
        "wird",
        "werden",
        "wurde",
        "wurden",
        "kann",
        "konnte",
        "soll",
        "sollte",
        "muss",
        "musste",
        "darf",
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
        "mein",
        "dein",
        "sein",
        "ihr",
        "unser",
        "euer",
        "nicht",
        "aber",
        "und",
        "oder",
        "wenn",
        "weil",
        "dass",
        "mit",
        "von",
        "aus",
        "nach",
        "bei",
        "für",
        "was",
        "wer",
        "wie",
        "wo",
        "warum",
        "wann",
        "auch",
        "noch",
        "schon",
        "nur",
        "sehr",
        "hier",
        "dort",
        "ja",
        "nein",
        "bitte",
        "danke",
        "gut",
        "kein",
        "keine",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "es",
        "son",
        "fue",
        "era",
        "tiene",
        "hay",
        "yo",
        "tu",
        "nosotros",
        "ellos",
        "ellas",
        "no",
        "pero",
        "como",
        "por",
        "para",
        "con",
        "que",
        "muy",
        "bien",
        "gracias",
        "hola",
    },
    "fr": {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "est",
        "sont",
        "suis",
        "avez",
        "ont",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "ne",
        "pas",
        "mais",
        "avec",
        "pour",
        "dans",
        "sur",
        "que",
        "qui",
        "quoi",
        "comment",
        "pourquoi",
        "tres",
        "bien",
        "merci",
        "bonjour",
        "oui",
        "non",
    },
}

# Language-specific character patterns
_CHAR_HINTS: dict[str, re.Pattern] = {
    "de": re.compile(r"[äöüßÄÖÜ]"),
    "fr": re.compile(r"[àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ]"),
    "es": re.compile(r"[áéíóúñ¿¡ÁÉÍÓÚÑ]"),
}


def _detect_language_core(text: str) -> tuple[str, float]:
    """Internal detection logic: returns (language, confidence).

    Confidence is a value between 0.0 and 1.0 indicating
    how certain the detection is. Higher score = more marker matches.

    Args:
        text: User message.

    Returns:
        Tuple of (ISO-639-1 language code, confidence score).
        Fallback: ("de", 0.0).
    """
    if not text or not text.strip():
        return "de", 0.0

    text_lower = text.lower().strip()

    # Normalize smart quotes before marker match
    text_lower = text_lower.replace("‘", "'").replace("’", "'")

    # Step 1: character-based hints (umlauts = German, accents = French, etc.)
    char_scores: dict[str, int] = {}
    for lang, pattern in _CHAR_HINTS.items():
        count = len(pattern.findall(text))
        if count > 0:
            char_scores[lang] = count

    # Step 2: word-based analysis
    # Extract words (alphabetic only + apostrophes for "don't" etc.)
    words = re.findall(r"[a-zA-ZäöüßÄÖÜàâéèêëîïôùûüçáéíóúñ']+", text_lower)

    if not words:
        return "de", 0.0

    word_set = set(words)
    scores: dict[str, float] = {}

    for lang, markers in _MARKERS.items():
        matches = word_set & markers
        # Score = number of matched marker words / total word count
        if matches:
            scores[lang] = len(matches) / len(words)

    # Add character hints as bonus
    for lang, char_count in char_scores.items():
        scores[lang] = scores.get(lang, 0) + (char_count * 0.1)

    if not scores:
        return "de", 0.0

    # Language with highest score wins
    best_lang = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_lang]

    # Minimum threshold: if best score is very low, default to German
    if best_score < 0.05:
        return "de", 0.0

    # Normalize confidence: score of 0.2+ counts as very certain (1.0)
    # Score of 0.05 is minimum (just above threshold) = 0.25
    confidence = min(1.0, best_score / 0.2)

    return best_lang, confidence


def detect_language(text: str) -> str:
    """Detect the language of a short text via heuristic.

    Args:
        text: User message.

    Returns:
        ISO-639-1 language code ("en", "de", "es", "fr").
        Fallback: "de".
    """
    lang, _ = _detect_language_core(text)
    return lang


def detect_language_with_confidence(text: str) -> tuple[str, float]:
    """Detect language AND return confidence score.

    Used by smart language detection to decide whether
    a sticky language override should be replaced.

    Args:
        text: User message.

    Returns:
        Tuple of (ISO-639-1 language code, confidence 0.0..1.0).
        Confidence > 0.7 means: clear language detection.
    """
    return _detect_language_core(text)
