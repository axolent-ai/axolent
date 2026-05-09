"""Leakage-Filter: Prüft LLM-Responses auf System-Prompt-Leakage (C-3).

Naive Heuristik: Sucht nach signifikanten Substrings des System-Prompts
(>= MIN_SUBSTRING_LENGTH Zeichen) in der LLM-Response. Wenn gefunden,
wird eine bereinigte Response zurückgegeben.

Design-Prinzipien:
    - Konservativ: lieber kein False Positive als zu strikt
    - Keine Regex-Bomben: einfacher Substring-Match
    - Schnell: O(n*m) im Worst Case, aber mit kurzen Chunks
    - Application-Layer: Business-Regel, kein Telegram-Code
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# Minimale Länge eines Substrings um als Leak zu gelten.
# Kurze Fragmente (<40 Zeichen) könnten zufällig in normalen
# Antworten vorkommen (False Positives).
MIN_SUBSTRING_LENGTH: int = 40

# Schrittweite für die Substring-Extraktion aus dem System-Prompt.
# Schrittweite 1 = lückenlos, keine Boundary-False-Negatives.
# Performance: Bei 2000-Zeichen-Prompt ca. 2000 Chunks, jeder 40 Zeichen.
# Substring-in-Checks in CPython sind O(n+m) dank Boyer-Moore.
_CHUNK_STEP: int = 1

# Replacement-Text für gefundene Leaks
_REDACTED_TEXT: str = "[Inhalt redacted]"

# Generische Refusal-Antwort wenn ein Leak erkannt wird
REFUSAL_RESPONSE: str = (
    "Ich kann meine internen Instruktionen nicht teilen. "
    "Was kann ich sonst für dich tun?"
)


def _extract_fingerprints(system_prompt: str) -> list[str]:
    """Extrahiert überlappende Substrings aus dem System-Prompt.

    Normalisiert den Text (lowercase, Whitespace-Reduktion) und
    extrahiert Chunks der Länge MIN_SUBSTRING_LENGTH mit Schrittweite
    _CHUNK_STEP.

    Args:
        system_prompt: Der vollständige System-Prompt.

    Returns:
        Liste von normalisierten Substring-Fingerprints.
    """
    # Normalisieren: lowercase, Whitespace zusammenfassen
    normalized = " ".join(system_prompt.lower().split())
    if len(normalized) < MIN_SUBSTRING_LENGTH:
        return []

    fingerprints: list[str] = []
    for i in range(0, len(normalized) - MIN_SUBSTRING_LENGTH + 1, _CHUNK_STEP):
        chunk = normalized[i : i + MIN_SUBSTRING_LENGTH]
        fingerprints.append(chunk)
    return fingerprints


def check_for_system_prompt_leakage(response: str, system_prompt: str) -> Optional[str]:
    """Prüft ob die LLM-Response Teile des System-Prompts enthält.

    Vergleicht normalisierte Substrings des System-Prompts gegen die
    normalisierte Response. Bei Treffer wird die Refusal-Response
    zurückgegeben.

    Args:
        response: Die LLM-Response die geprüft werden soll.
        system_prompt: Der aktive System-Prompt.

    Returns:
        None wenn kein Leak erkannt wurde.
        REFUSAL_RESPONSE wenn ein Leak erkannt wurde.
    """
    if not response or not system_prompt:
        return None

    fingerprints = _extract_fingerprints(system_prompt)
    if not fingerprints:
        return None

    # Response normalisieren (identisch zum Fingerprint)
    normalized_response = " ".join(response.lower().split())

    for fp in fingerprints:
        if fp in normalized_response:
            log.warning(
                "System-Prompt-Leakage erkannt: %d-Zeichen Match gefunden. "
                "Response wird durch Refusal ersetzt.",
                len(fp),
            )
            return REFUSAL_RESPONSE

    return None
