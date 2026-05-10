"""Debate-Orchestrator: Multi-AI-Debate Feature (R10).

Fragt mehrere Provider parallel mit derselben Frage und sammelt Antworten.
Crash-resilient: ein crashender Provider stoppt nicht die anderen.
Konsens/Dissens-Analyse via Heuristik + LLM-as-Judge Final Review.

Provider-Deduplizierung (seit R10-Fix):
Wenn mehrere Provider dasselbe Backend-Modell nutzen (z.B. claude_persistent
und claude nutzen beide die Claude CLI), wird nur einer pro Gruppe im Debate
verwendet. Das verhindert verzerrte Konsens-Analysen und Token-Verschwendung.

Final-Review-Layer (seit R10-Erweiterung):
Nach den parallelen Antworten wird ein LLM-as-Judge Call gemacht der alle
Antworten evaluiert und eine eindeutige Empfehlung mit Pro/Contra abgibt.
Judge-Provider: claude_persistent (Fallback: ollama_local mit Qualitätswarnung).
Bias-Mitigation: Provider-Namen werden im Judge-Prompt anonymisiert.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)

# Konfiguration via Environment
DEBATE_TIMEOUT_SECONDS: int = 60
_DEBATE_PROVIDERS_RAW: str = os.getenv("DEBATE_PROVIDERS", "")

# Sentinel Chat-ID fuer Judge-Calls: separater Konversationskontext
# damit der Judge nicht die Debate-Antwort im Kontext hat.
_JUDGE_CHAT_ID_OFFSET: int = 900_000_000

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
class ProviderEvaluation:
    """Bewertung einer einzelnen Provider-Antwort durch den Judge.

    Attributes:
        provider: Provider-Name (realer Name, nach De-Anonymisierung).
        pros: Liste positiver Aspekte der Antwort.
        cons: Liste negativer Aspekte der Antwort.
    """

    provider: str
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FinalVerdict:
    """Ergebnis des LLM-as-Judge Final Reviews.

    Attributes:
        winner: Provider-Name des Gewinners (oder "tie" bei Gleichstand).
        recommendation: Kurzer Satz mit der finalen Empfehlung.
        synthesis: Inhaltliche Synthese die das Beste aller Antworten vereint.
        evaluations: Pro/Contra-Bewertung je Provider-Antwort.
        reasoning: 1-2 Sätze warum dieser Winner gewählt wurde.
        judge_provider: Welcher Provider den Judge-Call gemacht hat.
        judge_quality_warning: Warnung wenn ein schwächerer Judge genutzt wurde.
    """

    winner: str
    recommendation: str
    synthesis: str = ""
    evaluations: list[ProviderEvaluation] = field(default_factory=list)
    reasoning: str = ""
    judge_provider: str = ""
    judge_quality_warning: str | None = None


@dataclass(frozen=True)
class DebateResult:
    """Ergebnis einer Multi-AI-Debate.

    Attributes:
        question: Die gestellte Frage.
        responses: Provider-Name -> Antworttext (erfolgreiche Provider).
        errors: Provider-Name -> Fehlermeldung (gecrashte Provider).
        consensus_analysis: Konsens/Dissens-Analyse (optional).
        final_verdict: LLM-as-Judge Bewertung (optional, None wenn Judge fehlschlägt).
        duration_seconds: Gesamtdauer der Debate.
        providers_queried: Liste aller angefragten Provider.
    """

    question: str
    responses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    consensus_analysis: Optional[str] = None
    final_verdict: Optional[FinalVerdict] = None
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

    def _build_judge_prompt(
        self,
        question: str,
        responses: dict[str, str],
    ) -> tuple[str, dict[str, str]]:
        """Baut den Judge-Prompt mit anonymisierten Provider-Namen.

        Bias-Mitigation: Provider-Namen werden durch neutrale Labels ersetzt.
        Der Judge sieht nur "Antwort A", "Antwort B" etc.

        Args:
            question: Die Original-Frage.
            responses: Provider-Name -> Antworttext.

        Returns:
            Tuple: (prompt_text, label_to_provider_mapping)
                label_to_provider_mapping: z.B. {"A": "claude_persistent", "B": "ollama_local"}
        """
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        label_to_provider: dict[str, str] = {}
        answer_blocks: list[str] = []

        for i, (provider_name, text) in enumerate(responses.items()):
            label = labels[i] if i < len(labels) else f"Z{i}"
            label_to_provider[label] = provider_name
            answer_blocks.append(f"--- Antwort {label} ---\n{text.strip()}\n")

        answers_text = "\n".join(answer_blocks)

        prompt = (
            f"Frage des Users:\n{question}\n\n"
            f"Die folgenden Antworten wurden von verschiedenen KI-Modellen generiert.\n"
            f"Bewerte sie neutral und objektiv.\n\n"
            f"{answers_text}\n"
            f"Deine Aufgabe:\n"
            f"1. Identifiziere die Staerken und Schwaechen jeder Antwort\n"
            f"2. Erstelle eine SYNTHESE die das Beste aller Antworten vereint\n"
            f"3. Die Synthese soll eine eigenstaendige, vollstaendige Antwort sein "
            f"(nicht nur 'A ist besser')\n\n"
            f"WICHTIG: Deine GESAMTE Antwort muss EIN EINZIGES JSON-Objekt sein.\n"
            f"Kein Text davor, kein Text danach, kein Markdown, keine Erklaerung.\n"
            f"Starte direkt mit {{ und ende mit }}.\n\n"
            f"JSON-Schema (exakt einhalten):\n"
            f'{{"winner": "<Buchstabe der besten Antwort oder tie>", '
            f'"synthesis": "<Vollstaendige synthetisierte Antwort die das Beste vereint, '
            f'2-5 Saetze, NIEMALS leer lassen>", '
            f'"recommendation": "<1 Satz: klare Empfehlung>", '
            f'"evaluations": ['
            f'{{"label": "<Buchstabe>", "pros": ["..."], "cons": ["..."]}}, ...'
            f"], "
            f'"reasoning": "<1-2 Saetze warum dieser Winner>"}}'
        )

        return prompt, label_to_provider

    @staticmethod
    def _extract_json_object(raw_text: str) -> str | None:
        """Extrahiert ein JSON-Objekt aus beliebigem Text.

        Strategien (in Reihenfolge):
        1. Gesamter Text ist valides JSON
        2. Markdown-Codeblock entfernen (```json ... ``` oder ``` ... ```)
        3. Erstes { bis letztes } extrahieren (Brace-Matching)

        Args:
            raw_text: Beliebiger Text der ein JSON-Objekt enthalten kann.

        Returns:
            Extrahierter JSON-String oder None wenn kein JSON gefunden.
        """
        text = raw_text.strip()

        # Strategie 1: Gesamter Text ist bereits valides JSON
        if text.startswith("{"):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass

        # Strategie 2: Markdown-Codeblock (```json\n...\n``` oder ```\n...\n```)
        # Auch wenn Text VOR dem Codeblock steht
        codeblock_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if codeblock_match:
            candidate = codeblock_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Strategie 3: Erstes { bis zum matchenden } (Brace-Counting)
        first_brace = text.find("{")
        if first_brace == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        for i in range(first_brace, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                if in_string:
                    escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        return None

        return None

    def _parse_judge_response(
        self,
        raw_text: str,
        label_to_provider: dict[str, str],
    ) -> FinalVerdict | None:
        """Parst die JSON-Antwort des Judges und mapped Labels zu Provider-Namen.

        Graceful: gibt None zurück bei Parse-Fehlern.
        Robust: extrahiert JSON auch wenn der Judge Prosa drumherum schreibt
        oder einen Markdown-Codeblock verwendet.

        Args:
            raw_text: Roh-Antwort des Judge-LLMs.
            label_to_provider: Mapping Label -> Provider-Name.

        Returns:
            FinalVerdict oder None wenn Parsing fehlschlägt.
        """
        log.debug("Judge raw response (%d chars): %s", len(raw_text), raw_text[:500])

        json_text = self._extract_json_object(raw_text)
        if json_text is None:
            log.warning(
                "Judge-Response: kein JSON-Objekt extrahierbar. Erste 300 Zeichen: %s",
                raw_text[:300],
            )
            return None

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            log.warning("Judge-Response ist kein valides JSON: %s", json_text[:200])
            return None

        # Pflichtfelder prüfen
        if not isinstance(data, dict):
            log.warning("Judge-Response ist kein Dict: %s", type(data))
            return None

        winner_label = data.get("winner", "")
        recommendation = data.get("recommendation", "")
        synthesis = data.get("synthesis", "")
        reasoning = data.get("reasoning", "")
        evaluations_raw = data.get("evaluations", [])

        # Winner-Label -> Provider-Name
        if winner_label.lower() == "tie":
            winner = "tie"
        else:
            winner = label_to_provider.get(winner_label, winner_label)

        # Evaluations parsen
        evaluations: list[ProviderEvaluation] = []
        for eval_item in evaluations_raw:
            if not isinstance(eval_item, dict):
                continue
            label = eval_item.get("label", "")
            provider_name = label_to_provider.get(label, label)
            pros = eval_item.get("pros", [])
            cons = eval_item.get("cons", [])
            if not isinstance(pros, list):
                pros = [str(pros)]
            if not isinstance(cons, list):
                cons = [str(cons)]
            evaluations.append(
                ProviderEvaluation(
                    provider=provider_name,
                    pros=pros,
                    cons=cons,
                )
            )

        return FinalVerdict(
            winner=winner,
            recommendation=str(recommendation),
            synthesis=str(synthesis),
            evaluations=evaluations,
            reasoning=str(reasoning),
        )

    async def final_review(
        self,
        question: str,
        responses: dict[str, str],
        user_id: int,
        chat_id: int,
    ) -> FinalVerdict | None:
        """Führt den LLM-as-Judge Final Review durch.

        Strategie:
        1. Versuche claude_persistent als Judge (höchste Qualität)
        2. Fallback auf ollama_local mit Qualitätswarnung
        3. Bei komplettem Fehler: None (Caller fällt auf Heuristik zurück)

        Bias-Mitigation: Provider-Namen werden anonymisiert im Judge-Prompt.

        Args:
            question: Die Original-Frage.
            responses: Provider-Name -> Antworttext (mind. 2 Einträge).
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.

        Returns:
            FinalVerdict oder None wenn Judge komplett fehlschlägt.
        """
        if len(responses) < 2:
            log.debug("Final Review übersprungen: weniger als 2 Antworten")
            return None

        prompt, label_to_provider = self._build_judge_prompt(question, responses)

        judge_system_prompt = (
            "Du bist ein neutraler Schiedsrichter der KI-Antworten bewertet. "
            "Du kennst die Provider-Namen nicht und bewertest rein nach Qualität: "
            "Korrektheit, Vollständigkeit, Klarheit und Relevanz. "
            "Antworte IMMER mit validem JSON, niemals mit Fliesstext."
        )

        # Judge-Provider-Auswahl: claude_persistent > ollama_local
        judge_candidates = ["claude_persistent", "claude", "ollama_local"]
        available = set(self.provider_router.list_available())

        # Provider die an der Debate teilgenommen haben ausschliessen
        # um Selbst-Bewertungs-Bias zu minimieren?
        # Nein: Bei nur 2 Providern würde keiner übrig bleiben.
        # Stattdessen: Anonymisierung reicht als Bias-Mitigation.

        judge_provider: str | None = None
        quality_warning: str | None = None

        for candidate in judge_candidates:
            if candidate in available:
                judge_provider = candidate
                break

        if judge_provider is None:
            log.warning("Kein Judge-Provider verfügbar, Final Review entfällt")
            return None

        if judge_provider == "ollama_local":
            quality_warning = "Lokaler Judge (Ollama), Bewertungsqualität reduziert"

        log.info("Final Review: Judge-Provider = %s", judge_provider)

        # Isolierter Konversationskontext fuer den Judge:
        # Offset auf die chat_id damit der Judge-Call NICHT in die
        # User-Konversation geht (wuerde Bias durch vorherige Debate-Antwort erzeugen
        # und kann JSON-Output stoeren weil Claude im Chat-Modus antwortet).
        judge_chat_id = chat_id + _JUDGE_CHAT_ID_OFFSET

        try:
            response = await asyncio.wait_for(
                self.provider_router.route(
                    prompt=prompt,
                    system_prompt=judge_system_prompt,
                    provider_name=judge_provider,
                    timeout_seconds=self.timeout_seconds,
                    user_id=user_id,
                    chat_id=judge_chat_id,
                ),
                timeout=self.timeout_seconds + 5,
            )

            if not response.success:
                log.warning(
                    "Judge-Call fehlgeschlagen: %s (text=%r)",
                    response.error or "kein Text",
                    (response.text or "")[:200],
                )
                return None

            log.debug(
                "Judge-Response erhalten (%d chars, %.1fs): %s",
                len(response.text),
                response.duration_seconds,
                response.text[:300],
            )

            verdict = self._parse_judge_response(response.text, label_to_provider)
            if verdict is None:
                log.warning(
                    "Judge-Response konnte nicht geparst werden. "
                    "Vollstaendige Response (%d chars): %s",
                    len(response.text),
                    response.text[:500],
                )
                return None

            # Judge-Metadaten hinzufügen (frozen dataclass, neues Objekt)
            return FinalVerdict(
                winner=verdict.winner,
                recommendation=verdict.recommendation,
                synthesis=verdict.synthesis,
                evaluations=verdict.evaluations,
                reasoning=verdict.reasoning,
                judge_provider=judge_provider,
                judge_quality_warning=quality_warning,
            )

        except asyncio.TimeoutError:
            log.warning("Judge-Call Timeout nach %ds", self.timeout_seconds)
            return None
        except Exception as exc:
            log.warning("Judge-Call Exception: %s", exc)
            return None

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

        # Final Review (LLM-as-Judge)
        verdict: FinalVerdict | None = None
        if len(responses) >= 2:
            verdict = await self.final_review(
                question=question,
                responses=responses,
                user_id=user_id,
                chat_id=chat_id,
            )

        duration = time.monotonic() - t_start

        log.info(
            "Debate abgeschlossen: %d Antworten, %d Fehler, verdict=%s, %.1fs",
            len(responses),
            len(errors),
            verdict.winner if verdict else "none",
            duration,
        )

        return DebateResult(
            question=question,
            responses=responses,
            errors=errors,
            consensus_analysis=consensus,
            final_verdict=verdict,
            duration_seconds=duration,
            providers_queried=providers,
        )
