"""Debate-Orchestrator: Multi-AI-Debate Feature (R10).

Fragt mehrere Provider parallel mit derselben Frage und sammelt Antworten.
Crash-resilient: ein crashender Provider stoppt nicht die anderen.
Optional: Konsens/Dissens-Analyse via Heuristik (Phase 1) oder LLM-Judge (Phase 1+).

Provider-Deduplizierung (seit R10-Fix):
Wenn mehrere Provider dasselbe Backend-Modell nutzen (z.B. claude_persistent
und claude nutzen beide die Claude CLI), wird nur einer pro Gruppe im Debate
verwendet. Das verhindert verzerrte Konsens-Analysen und Token-Verschwendung.
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

# Provider-Gruppen: Provider die dasselbe Backend-Modell nutzen.
# Pro Gruppe wird nur der erste verfügbare Provider im Debate verwendet.
# Reihenfolge = Priorität (erster Eintrag wird bevorzugt).
PROVIDER_GROUPS: dict[str, list[str]] = {
    "claude": ["claude_persistent", "claude"],  # beide nutzen Claude CLI
}

# Reverse-Lookup: provider_name -> group_name (oder None wenn standalone)
_PROVIDER_TO_GROUP: dict[str, str] = {}
for _group_name, _members in PROVIDER_GROUPS.items():
    for _member in _members:
        _PROVIDER_TO_GROUP[_member] = _group_name


def _get_configured_providers() -> list[str] | None:
    """Parst DEBATE_PROVIDERS env-var. None = alle verfügbaren nutzen."""
    if not _DEBATE_PROVIDERS_RAW.strip():
        return None
    return [p.strip() for p in _DEBATE_PROVIDERS_RAW.split(",") if p.strip()]


def deduplicate_providers(available: list[str]) -> list[str]:
    """Dedupliziert Provider die dasselbe Backend-Modell nutzen.

    Pro PROVIDER_GROUPS-Gruppe wird nur der erste verfügbare Provider behalten.
    Standalone-Provider (nicht in einer Gruppe) werden immer behalten.

    Args:
        available: Liste verfügbarer Provider-Namen.

    Returns:
        Deduplizierte Liste (Reihenfolge bleibt erhalten).
    """
    selected: list[str] = []
    used_groups: set[str] = set()

    for provider in available:
        group = _PROVIDER_TO_GROUP.get(provider)
        if group:
            if group not in used_groups:
                selected.append(provider)
                used_groups.add(group)
                log.debug("Provider-Dedup: %s vertritt Gruppe '%s'", provider, group)
            else:
                log.debug(
                    "Provider-Dedup: %s übersprungen (Gruppe '%s' bereits vertreten)",
                    provider,
                    group,
                )
        else:
            # Standalone-Provider: immer behalten
            selected.append(provider)

    if len(selected) < len(available):
        log.info(
            "Provider-Dedup: %d -> %d Provider (%s)",
            len(available),
            len(selected),
            selected,
        )

    return selected


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
    """Orchestriert Multi-AI-Debates über mehrere Provider.

    Fragt alle verfügbaren (oder konfigurierte) Provider parallel,
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
        """Bestimmt welche Provider für die Debate genutzt werden.

        Priorität:
        1. DEBATE_PROVIDERS env-var (wenn gesetzt)
        2. Alle verfügbaren Provider

        In beiden Fällen wird anschließend dedupliziert: Provider die dasselbe
        Backend-Modell nutzen werden auf einen pro Gruppe reduziert.

        Returns:
            Deduplizierte Liste der Provider-Namen.
        """
        configured = _get_configured_providers()
        if configured is not None:
            # Nur konfigurierte Provider die auch verfügbar sind
            available = set(self.provider_router.list_available())
            selected = [p for p in configured if p in available]
            if not selected:
                log.warning(
                    "Keine konfigurierten DEBATE_PROVIDERS verfügbar: %s. "
                    "Verfügbar: %s",
                    configured,
                    list(available),
                )
            return deduplicate_providers(selected)

        all_available = self.provider_router.list_available()
        return deduplicate_providers(all_available)

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
                        "Antworte prägnant und informativ. "
                        "Halte dich an 2-4 Sätze wenn möglich."
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

        Vergleicht Antwortlängen und einfache Wort-Overlap-Analyse.

        Args:
            responses: Provider-Name -> Antworttext.

        Returns:
            Kurze Konsens/Dissens-Einschätzung.
        """
        if len(responses) < 2:
            return "Nur ein Provider hat geantwortet. Kein Vergleich möglich."

        texts = list(responses.values())

        # Wortmengen für Overlap-Analyse
        word_sets: list[set[str]] = []
        for text in texts:
            words = set(text.lower().split())
            # Nur signifikante Wörter (>3 Zeichen)
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
                f"Die Provider stimmen inhaltlich weitgehend überein "
                f"(Wort-Overlap: {avg_overlap:.0%}). "
                f"Hohe Übereinstimmung in den Kernaussagen."
            )
        elif avg_overlap > 0.20:
            return (
                f"Die Provider zeigen teilweise Übereinstimmung "
                f"(Wort-Overlap: {avg_overlap:.0%}). "
                f"Kernaussagen ähnlich, aber unterschiedliche Schwerpunkte."
            )
        else:
            return (
                f"Die Provider geben deutlich unterschiedliche Antworten "
                f"(Wort-Overlap: {avg_overlap:.0%}). "
                f"Vergleiche die Antworten oben für verschiedene Perspektiven."
            )

    async def debate(
        self,
        question: str,
        user_id: int,
        chat_id: int,
    ) -> DebateResult:
        """Führt eine Multi-AI-Debate durch.

        1. Identifiziert verfügbare Provider (mit Deduplizierung)
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
                errors={"system": "Keine Provider verfügbar"},
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
