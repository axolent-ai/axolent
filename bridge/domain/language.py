"""Einfache Sprach-Detection für User-Nachrichten.

Keine externen Dependencies. Verwendet Heuristiken basierend auf
häufigen Wörtern und Zeichenmustern. Genügt für die
Unterscheidung der gängigsten Sprachen (de, en, es, fr, it, pt).

Fallback: "de" (Jessicas Hauptsprache).
"""

import logging
import re

log = logging.getLogger(__name__)

# Häufige Funktionswörter pro Sprache (lowercase).
# Kurze, eindeutige Wörter die in fast jedem Satz vorkommen.
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

# Sprach-spezifische Zeichen-Pattern
_CHAR_HINTS: dict[str, re.Pattern] = {
    "de": re.compile(r"[äöüßÄÖÜ]"),
    "fr": re.compile(r"[àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ]"),
    "es": re.compile(r"[áéíóúñ¿¡ÁÉÍÓÚÑ]"),
}


def _detect_language_core(text: str) -> tuple[str, float]:
    """Interne Detection-Logik: gibt (Sprache, Confidence) zurück.

    Confidence ist ein Wert zwischen 0.0 und 1.0 der angibt,
    wie sicher die Erkennung ist. Höherer Score = mehr Marker-Übereinstimmung.

    Args:
        text: User-Nachricht.

    Returns:
        Tuple von (ISO-639-1-Sprachcode, Confidence-Score).
        Fallback: ("de", 0.0).
    """
    if not text or not text.strip():
        return "de", 0.0

    text_lower = text.lower().strip()

    # Smart-Quotes normalisieren vor Marker-Match
    text_lower = text_lower.replace("‘", "’").replace("’", "’")

    # Schritt 1: Zeichen-basierte Hints (Umlaute = Deutsch, Akzente = Französisch, etc.)
    char_scores: dict[str, int] = {}
    for lang, pattern in _CHAR_HINTS.items():
        count = len(pattern.findall(text))
        if count > 0:
            char_scores[lang] = count

    # Schritt 2: Wort-basierte Analyse
    # Wörter extrahieren (nur alphabetisch + Apostrophe für "don’t" etc.)
    words = re.findall(r"[a-zA-ZäöüßÄÖÜàâéèêëîïôùûüçáéíóúñ’]+", text_lower)

    if not words:
        return "de", 0.0

    word_set = set(words)
    scores: dict[str, float] = {}

    for lang, markers in _MARKERS.items():
        matches = word_set & markers
        # Score = Anzahl gematchter Marker-Wörter / Gesamtzahl Wörter
        if matches:
            scores[lang] = len(matches) / len(words)

    # Zeichen-Hints als Bonus addieren
    for lang, char_count in char_scores.items():
        scores[lang] = scores.get(lang, 0) + (char_count * 0.1)

    if not scores:
        return "de", 0.0

    # Sprache mit höchstem Score gewinnt
    best_lang = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_lang]

    # Mindest-Schwelle: wenn der beste Score sehr niedrig ist, Default Deutsch
    if best_score < 0.05:
        return "de", 0.0

    # Confidence normalisieren: Score von 0.2+ gilt als sehr sicher (1.0)
    # Score von 0.05 ist Minimum (knapp über Schwelle) = 0.25
    confidence = min(1.0, best_score / 0.2)

    return best_lang, confidence


def detect_language(text: str) -> str:
    """Erkennt die Sprache eines kurzen Textes via Heuristik.

    Args:
        text: User-Nachricht.

    Returns:
        ISO-639-1 Sprachcode ("en", "de", "es", "fr").
        Fallback: "de".
    """
    lang, _ = _detect_language_core(text)
    return lang


def detect_language_with_confidence(text: str) -> tuple[str, float]:
    """Erkennt Sprache UND gibt Confidence-Score zurück.

    Wird von der Smart-Language-Detection genutzt um zu entscheiden
    ob ein Sticky-Language-Override überschrieben werden soll.

    Args:
        text: User-Nachricht.

    Returns:
        Tuple von (ISO-639-1-Sprachcode, Confidence 0.0..1.0).
        Confidence > 0.7 bedeutet: klare Spracherkennung.
    """
    return _detect_language_core(text)
