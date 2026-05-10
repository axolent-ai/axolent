"""Debate-Orchestrator: Multi-AI-Debate Feature (R10).

Fragt mehrere Provider parallel mit derselben Frage und sammelt Antworten.
Crash-resilient: ein crashender Provider stoppt nicht die anderen.
Optional: Konsens/Dissens-Analyse via Heuristik (Phase 1) oder LLM-Judge (Phase 1+).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)

# Konfiguration via Environment
DEBATE_TIMEOUT_SECONDS: int = 60
_DEBATE_PROVIDERS_RAW: str = os.getenv("DEBATE_PROVIDERS", "")


def _get_configured_providers() -> list[str] | None:
    """Parst DEBATE_PROVIDERS env-var. None = alle verfuegbaren nutzen."""
    if not _DEBATE_PROVIDERS_RAW.strip():
        return None
    return [p.strip() for p in _DEBATE_PROVIDERS_RAW.split(",") if p.strip()]


@dataclass(frozen=True)
class DebateResult:
    """Ergebnis einer Multi-AI-Debate.

    Attributes:
        question: Die gestellte Frage.
        responses: Provider-Name -> Antworttext (erfolgreiche Provider).
        errors: Provider-Name -> Fehlermeldung (gecrashte Provider).
        consensus_analysis: Konsens/Dissens-Analyse (optional).
        duration_seconds: Gesamtdauer der Debate.
        providers_queried: Liste aller angefragten Provider.
    """

    question: str
    responses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    consensus_analysis: Optional[str] = None
    duration_seconds: float = 0.0
    providers_queried: list[str] = field(default_factory=list)


class DebateOrchestrator:
    """Orchestriert Multi-AI-Debates ueber mehrere Provider.

    Fragt alle verfuegbaren (oder konfigurierte) Provider parallel,
    sammelt Antworten mit Timeout-Schutz und erstellt eine Konsens-Analyse.

    Args:
        provider_router: Der ProviderRouter mit registrierten Providern.
        timeout_seconds: Max Wartezeit pro Provider (Default: 60s).
    """

    def __init__(
        self,
        provider_router: ProviderRouter,
        timeout_seconds: int = DEBATE_TIMEOUT_SECONDS,
    ) -> None:
        self.provider_router = provider_router
        self.timeout_seconds = timeout_seconds

    def _select_providers(self) -> list[str]:
        """Bestimmt welche Provider fuer die Debate genutzt werden.

        Prioritaet:
        1. DEBATE_PROVIDERS env-var (wenn gesetzt)
        2. Alle verfuegbaren Provider

        Returns:
            Liste der Provider-Namen die genutzt werden sollen.
        """
        configured = _get_configured_providers()
        if configured is not None:
            # Nur konfigurierte Provider die auch verfuegbar sind
            available = set(self.provider_router.list_available())
            selected = [p for p in configured if p in available]
            if not selected:
                log.warning(
                    "Keine konfigurierten DEBATE_PROVIDERS verfuegbar: %s. "
                    "Verfuegbar: %s",
                    configured,
                    list(available),
                )
            return selected

        return self.provider_router.list_available()

    async def _query_provider(
        self,
        provider_name: str,
        question: str,
        user_id: int,
        chat_id: int,
    ) -> tuple[str, str | None, str | None]:
        """Fragt einen einzelnen Provider mit Timeout.

        Returns:
            Tuple: (provider_name, response_text_or_None, error_or_None)
        """
        try:
            response = await asyncio.wait_for(
                self.provider_router.route(
                    prompt=question,
                    system_prompt=(
                        "Antworte praegnant und informativ. "
                        "Halte dich an 2-4 Saetze wenn moeglich."
                    ),
                    provider_name=provider_name,
                    timeout_seconds=self.timeout_seconds,
                    user_id=user_id,
                    chat_id=chat_id,
                ),
                timeout=self.timeout_seconds + 5,  # Asyncio-Timeout als Fallback
            )
            if response.success:
                return (provider_name, response.text, None)
            else:
                return (provider_name, None, response.error or "Unbekannter Fehler")
        except asyncio.TimeoutError:
            return (provider_name, None, f"Timeout nach {self.timeout_seconds}s")
        except Exception as exc:
            return (provider_name, None, str(exc))

    def _analyze_consensus(self, responses: dict[str, str]) -> str:
        """Einfache Konsens-Heuristik (Phase 1, kein LLM-Judge).

        Vergleicht Antwortlaengen und einfache Wort-Overlap-Analyse.

        Args:
            responses: Provider-Name -> Antworttext.

        Returns:
            Kurze Konsens/Dissens-Einschaetzung.
        """
        if len(responses) < 2:
            return "Nur ein Provider hat geantwortet. Kein Vergleich moeglich."

        texts = list(responses.values())

        # Wortmengen fuer Overlap-Analyse
        word_sets: list[set[str]] = []
        for text in texts:
            words = set(text.lower().split())
            # Nur signifikante Woerter (>3 Zeichen)
            significant = {w for w in words if len(w) > 3}
            word_sets.append(significant)

        # Paarweisen Overlap berechnen (Jaccard-Similarity)
        overlaps: list[float] = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                if not union:
                    overlaps.append(0.0)
                    continue
                intersection = word_sets[i] & word_sets[j]
                overlaps.append(len(intersection) / len(union))

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # Entscheidungslogik
        if avg_overlap > 0.35:
            return (
                f"Die Provider stimmen inhaltlich weitgehend ueberein "
                f"(Wort-Overlap: {avg_overlap:.0%}). "
                f"Hohe Uebereinstimmung in den Kernaussagen."
            )
        elif avg_overlap > 0.20:
            return (
                f"Die Provider zeigen teilweise Uebereinstimmung "
                f"(Wort-Overlap: {avg_overlap:.0%}). "
                f"Kernaussagen aehnlich, aber unterschiedliche Schwerpunkte."
            )
        else:
            return (
                f"Die Provider geben deutlich unterschiedliche Antworten "
                f"(Wort-Overlap: {avg_overlap:.0%}). "
                f"Vergleiche die Antworten oben fuer verschiedene Perspektiven."
            )

    async def debate(
        self,
        question: str,
        user_id: int,
        chat_id: int,
    ) -> DebateResult:
        """Fuehrt eine Multi-AI-Debate durch.

        1. Identifiziert verfuegbare Provider
        2. Fragt alle parallel (asyncio.gather)
        3. Sammelt Antworten + Fehler
        4. Erstellt Konsens-Analyse

        Args:
            question: Die User-Frage.
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.

        Returns:
            DebateResult mit allen Antworten, Fehlern und Analyse.
        """
        t_start = time.monotonic()

        providers = self._select_providers()
        if not providers:
            return DebateResult(
                question=question,
                responses={},
                errors={"system": "Keine Provider verfuegbar"},
                consensus_analysis=None,
                duration_seconds=time.monotonic() - t_start,
                providers_queried=[],
            )

        log.info(
            "Debate gestartet: %d Provider (%s), Frage: %s",
            len(providers),
            providers,
            question[:80],
        )

        # Alle Provider parallel abfragen
        tasks = [
            self._query_provider(name, question, user_id, chat_id) for name in providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: dict[str, str] = {}
        errors: dict[str, str] = {}

        for result in results:
            if isinstance(result, Exception):
                # Sollte nicht passieren (Exceptions werden in _query_provider gefangen)
                errors["unknown"] = str(result)
                continue
            provider_name, text, error = result
            if text is not None:
                responses[provider_name] = text
            elif error is not None:
                errors[provider_name] = error

        # Konsens-Analyse
        consensus: str | None = None
        if responses:
            consensus = self._analyze_consensus(responses)

        duration = time.monotonic() - t_start

        log.info(
            "Debate abgeschlossen: %d Antworten, %d Fehler, %.1fs",
            len(responses),
            len(errors),
            duration,
        )

        return DebateResult(
            question=question,
            responses=responses,
            errors=errors,
            consensus_analysis=consensus,
            duration_seconds=duration,
            providers_queried=providers,
        )
