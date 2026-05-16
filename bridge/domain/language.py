"""Language detection for user messages.

Supports 20 languages via Unicode script detection and
marker-word heuristics. No external dependencies.

Detection strategy:
1. Non-Latin scripts (Arabic, Chinese, Japanese, Korean, Hindi,
   Thai, Cyrillic) are detected deterministically via Unicode ranges.
2. Cyrillic text is further classified as Russian vs Ukrainian
   via distinctive markers.
3. Latin-script languages are scored by frequency of marker words.

Fallback: "en" (international default).
"""

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

# --- Unicode script patterns for non-Latin detection ---

_SCRIPT_PATTERNS: dict[str, re.Pattern] = {
    "ar": re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]+"),
    "zh": re.compile(r"[一-鿿㐀-䶿]+"),
    "ja": re.compile(r"[぀-ゟ゠-ヿ]+"),
    "ko": re.compile(r"[가-힯ᄀ-ᇿ]+"),
    "hi": re.compile(r"[ऀ-ॿ]+"),
    "th": re.compile(r"[฀-๿]+"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]+"),
}

# Ukrainian-specific characters and markers (for Cyrillic disambiguation)
_UKRAINIAN_CHARS = re.compile(r"[їієґЇІЄҐ]")
_UKRAINIAN_MARKERS = {"i", "та", "що", "як", "це", "але", "або", "ще", "вiн"}
_RUSSIAN_MARKERS = {"и", "в", "не", "на", "что", "как", "это", "он", "она", "они"}

# --- Marker words for Latin-script languages ---

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
    "nl": {
        "het",
        "een",
        "van",
        "voor",
        "niet",
        "dat",
        "zijn",
        "naar",
        "ook",
        "aan",
        "maar",
        "nog",
        "wel",
        "deze",
        "werd",
        "wordt",
        "hebben",
        "heeft",
        "kunnen",
        "moeten",
        "willen",
        "zullen",
        "deze",
        "meer",
        "veel",
        "heel",
        "goed",
        "waar",
        "hier",
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
    "it": {
        "il",
        "lo",
        "la",
        "gli",
        "le",
        "un",
        "una",
        "uno",
        "di",
        "che",
        "non",
        "sono",
        "per",
        "questo",
        "quella",
        "come",
        "anche",
        "con",
        "ma",
        "piu",
        "molto",
        "bene",
        "grazie",
        "ciao",
        "cosa",
        "dove",
        "quando",
        "perche",
        "ancora",
        "sempre",
        "tutto",
        "ogni",
    },
    "pt": {
        "o",
        "a",
        "os",
        "as",
        "um",
        "uma",
        "de",
        "que",
        "do",
        "da",
        "em",
        "para",
        "com",
        "nao",
        "uma",
        "por",
        "mais",
        "como",
        "mas",
        "foi",
        "ao",
        "ele",
        "ela",
        "seu",
        "sua",
        "ou",
        "ser",
        "quando",
        "muito",
        "nos",
        "ja",
        "eu",
        "tambem",
        "obrigado",
    },
    "pl": {
        "jest",
        "nie",
        "to",
        "na",
        "tak",
        "ale",
        "jak",
        "bardzo",
        "dobrze",
        "gdzie",
        "kiedy",
        "dlaczego",
        "jestem",
        "masz",
        "moze",
        "tutaj",
        "tam",
        "teraz",
        "jeszcze",
        "tylko",
        "tego",
        "jego",
        "jej",
        "nasz",
        "wasz",
        "wszystko",
        "kazdy",
    },
    "sv": {
        "och",
        "att",
        "det",
        "som",
        "med",
        "av",
        "den",
        "har",
        "inte",
        "till",
        "var",
        "jag",
        "kan",
        "ska",
        "alla",
        "mycket",
        "denna",
        "efter",
        "bara",
        "hur",
        "vad",
        "finns",
        "sedan",
        "utan",
        "eller",
        "men",
        "helt",
        "mellan",
        "redan",
    },
    "tr": {
        "bir",
        "bu",
        "olan",
        "gibi",
        "daha",
        "ancak",
        "sonra",
        "bunu",
        "onun",
        "kadar",
        "olarak",
        "nasil",
        "neden",
        "evet",
        "hayir",
        "burada",
        "orada",
        "simdi",
        "sadece",
        "hepsi",
        "bazi",
        "cok",
        "iyi",
        "tesekkur",
    },
    "id": {
        "yang",
        "dan",
        "untuk",
        "dengan",
        "ini",
        "itu",
        "adalah",
        "dari",
        "pada",
        "akan",
        "sudah",
        "bisa",
        "ada",
        "tidak",
        "juga",
        "seperti",
        "mereka",
        "kami",
        "saya",
        "kita",
        "sangat",
        "baik",
        "bagaimana",
        "kenapa",
        "dimana",
        "kapan",
        "terima",
        "kasih",
    },
    "vi": {
        "cua",
        "khong",
        "trong",
        "nhung",
        "nhu",
        "cac",
        "mot",
        "nay",
        "duoc",
        "khi",
        "cung",
        "rat",
        "tot",
        "bao",
        "tai",
        "sao",
        "noi",
        "day",
        "hom",
        "nay",
    },
}

# Diacritical character patterns for Latin-script disambiguation
_CHAR_HINTS: dict[str, re.Pattern] = {
    "de": re.compile(r"[äöüßÄÖÜ]"),
    "fr": re.compile(r"[àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ]"),
    "es": re.compile(r"[áéíóúñ¿¡ÁÉÍÓÚÑ]"),
    "pt": re.compile(r"[ãõâêôàáéíóúçÃÕÂÊÔÀÁÉÍÓÚÇ]"),
    "pl": re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]"),
    "sv": re.compile(r"[åÅ]"),
    "tr": re.compile(r"[çğışöüÇĞİŞÖÜ]"),
    "vi": re.compile(
        r"[àáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ]"
    ),
}

# Word extraction pattern (includes accented chars and apostrophes)
_WORD_PATTERN = re.compile(
    r"[a-zA-ZäöüßÄÖÜàâéèêëîïôùûüçáéíóúñãõåąćęłńśźżğışđ"
    r"ắằặẳẵấầậẩẫếềệểễốồộổỗớờợởỡứừựửữỳýỵỷỹ'']+"
)


def _detect_script(text: str) -> Optional[str]:
    """Detect language from non-Latin Unicode scripts.

    Returns language code if a non-Latin script dominates,
    None otherwise (meaning Latin-based detection should proceed).

    Args:
        text: Raw user message.

    Returns:
        Language code or None.
    """
    # Count characters per script
    script_counts: dict[str, int] = {}
    for script, pattern in _SCRIPT_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            char_count = sum(len(m) for m in matches)
            script_counts[script] = char_count

    if not script_counts:
        return None

    # Find dominant script
    best_script = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
    best_count = script_counts[best_script]

    # Require at least 2 characters of the script
    if best_count < 2:
        return None

    # Special case: Cyrillic needs ru vs uk disambiguation
    if best_script == "cyrillic":
        return _disambiguate_cyrillic(text)

    # Special case: CJK could be Chinese or Japanese
    # If hiragana/katakana present, it's Japanese
    if best_script == "zh" and "ja" in script_counts:
        return "ja"

    return best_script


def _disambiguate_cyrillic(text: str) -> str:
    """Distinguish Russian from Ukrainian in Cyrillic text.

    Args:
        text: Text containing Cyrillic characters.

    Returns:
        "uk" for Ukrainian, "ru" for Russian.
    """
    # Check for Ukrainian-specific characters first
    if _UKRAINIAN_CHARS.search(text):
        return "uk"

    # Check marker words
    words = set(text.lower().split())
    uk_hits = len(words & _UKRAINIAN_MARKERS)
    ru_hits = len(words & _RUSSIAN_MARKERS)

    if uk_hits > ru_hits:
        return "uk"
    return "ru"


def _detect_language_core(text: str) -> tuple[str, float]:
    """Internal detection logic: returns (language, confidence).

    Strategy:
    1. Check for non-Latin scripts (deterministic).
    2. Score Latin-script languages by marker words + char hints.
    3. Return best match with confidence score.

    Confidence is a value between 0.0 and 1.0 indicating
    how certain the detection is.

    Args:
        text: User message.

    Returns:
        Tuple of (ISO-639-1 language code, confidence score).
        Fallback: ("en", 0.0).
    """
    if not text or not text.strip():
        return "en", 0.0

    # Step 1: Non-Latin script detection
    script_lang = _detect_script(text)
    if script_lang:
        return script_lang, 1.0

    text_lower = text.lower().strip()

    # Normalize smart quotes before marker match
    text_lower = text_lower.replace("‘", "'").replace("’", "'")

    # Step 2: Character-based hints (diacritics)
    char_scores: dict[str, int] = {}
    for lang, pattern in _CHAR_HINTS.items():
        count = len(pattern.findall(text))
        if count > 0:
            char_scores[lang] = count

    # Step 3: Word-based analysis
    words = _WORD_PATTERN.findall(text_lower)

    if not words:
        return "en", 0.0

    word_set = set(words)
    scores: dict[str, float] = {}

    for lang, markers in _MARKERS.items():
        matches = word_set & markers
        if matches:
            scores[lang] = len(matches) / len(words)

    # Add character hints as bonus (weighted)
    for lang, char_count in char_scores.items():
        scores[lang] = scores.get(lang, 0) + (char_count * 0.15)

    if not scores:
        return "en", 0.0

    # Language with highest score wins
    best_lang = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_lang]

    # Minimum threshold
    if best_score < 0.05:
        return "en", 0.0

    # Normalize confidence: score of 0.2+ counts as very certain (1.0)
    confidence = min(1.0, best_score / 0.2)

    return best_lang, confidence


def detect_language(text: str) -> str:
    """Detect the language of a short text via heuristic.

    Supports 20 languages: de, en, nl, fr, es, it, pt, pl, ru, tr,
    sv, ja, ko, zh, uk, ar, hi, id, th, vi.

    Args:
        text: User message.

    Returns:
        ISO-639-1 language code.
        Fallback: "en".
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
