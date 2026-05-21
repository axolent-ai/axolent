"""Layer 6: Skill Explainer for Skill-Compression.

Answers user questions about skill decisions using 8 question types
(HC-SC-18, HC-EXPLAIN-1). Inspired by the RCA structure used internally.

Question types:
  1. WHAT_RECOGNIZED     - "Was hat das Pattern erkannt?"
  2. WHY_NOT_SKILL       - "Warum wurde das nicht zum Skill?"
  3. WHY_PROMOTED        - "Warum wurde es zum Skill promotet?"
  4. WHEN_DRIFT          - "Wann wurde Drift erkannt?"
  5. WHAT_NEEDED         - "Was wäre nötig damit Pattern vertrauenswürdig wird?"
  6. LESSONS_LEARNED     - "Welche Lessons hat das System gelernt?"
  7. SCOPE_BOUNDARIES    - "Wo gilt dieser Skill NICHT?"
  8. COUNTER_EVIDENCE    - "Welche Belege sprechen GEGEN diesen Skill?"

HC-EXPLAIN-1 [BLOCKER]: All 8 types supported. Missing data returns an
  honest "no data" answer, never hallucinated content.

HC-LAYER2-1: Explainer reads evidence and hypothesis state. It does NOT
  modify any data.

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from application.skill_compression.evidence_ledger import (
    NEGATIVE_SIGNALS,
    POSITIVE_SIGNALS,
    EvidenceLedger,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.pattern_judge import (
    STATUS_ACTIVE,
    STATUS_CANDIDATE,
    STATUS_CONFIRMED,
    STATUS_NEEDS_REVIEW,
    STATUS_SUGGESTED,
    SUGGEST_MIN_EVIDENCE,
    SUGGEST_MIN_SESSIONS,
    THRESHOLDS,
    _resolve_threshold_key,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Question types (8 types, HC-SC-18)
# ---------------------------------------------------------------


class ExplainerQuestionType(Enum):
    """The 8 explainer question types from the Spec."""

    WHAT_RECOGNIZED = "what_recognized"
    WHY_NOT_SKILL = "why_not_skill"
    WHY_PROMOTED = "why_promoted"
    WHEN_DRIFT = "when_drift"
    WHAT_NEEDED = "what_needed"
    LESSONS_LEARNED = "lessons_learned"
    SCOPE_BOUNDARIES = "scope_boundaries"
    COUNTER_EVIDENCE = "counter_evidence"


# ---------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExplainerResponse:
    """Response from the SkillExplainer.

    Attributes:
        question_type: Which question was asked.
        hypothesis_id: The hypothesis being explained.
        title: Short title for the explanation.
        explanation: Detailed answer text.
        has_data: Whether sufficient data was available for the answer.
        evidence_count: Number of evidence records consulted.
    """

    question_type: ExplainerQuestionType
    hypothesis_id: str
    title: str
    explanation: str
    has_data: bool
    evidence_count: int = 0


# ---------------------------------------------------------------
# No-data response helper
# ---------------------------------------------------------------

# i18n: ok (German user-facing text, IC-EXPLAIN-1)
_NO_DATA_TITLE = "Keine Daten vorhanden"
_NO_DATA_TEMPLATE = (
    "Zu dieser Frage liegen keine ausreichenden Daten vor. "
    "Das System hat noch nicht genug Belege gesammelt, "
    "um eine fundierte Antwort zu geben."
)

_NO_HYPOTHESIS_TITLE = "Hypothese nicht gefunden"  # i18n: ok
_NO_HYPOTHESIS_MSG = (  # i18n: ok
    "Die Hypothese mit der ID '{hyp_id}' wurde nicht gefunden. "
    "Nutze /skills um eine Liste aller Skills anzuzeigen."
)


def _no_data_response(
    question_type: ExplainerQuestionType,
    hypothesis_id: str,
    detail: str = "",
) -> ExplainerResponse:
    """Build a response when no data is available (HC-EXPLAIN-1).

    Never hallucinate. If data is missing, say so honestly.

    Args:
        question_type: The question type asked.
        hypothesis_id: The hypothesis ID.
        detail: Optional additional detail.

    Returns:
        ExplainerResponse with has_data=False.
    """
    explanation = _NO_DATA_TEMPLATE
    if detail:
        explanation = f"{explanation}\n\n{detail}"

    return ExplainerResponse(
        question_type=question_type,
        hypothesis_id=hypothesis_id,
        title=_NO_DATA_TITLE,
        explanation=explanation,
        has_data=False,
        evidence_count=0,
    )


# ---------------------------------------------------------------
# Scope description helpers
# ---------------------------------------------------------------


def _describe_scope(scope: HypothesisScope) -> str:
    """Describe a scope in human-readable German.

    Args:
        scope: The hypothesis scope.

    Returns:
        Human-readable scope description.
    """
    # i18n: ok
    parts: list[str] = []
    if scope.client:
        parts.append(f"Kunde: {scope.client}")
    if scope.project:
        parts.append(f"Projekt: {scope.project}")
    if scope.context:
        parts.append(f"Kontext: {', '.join(scope.context)}")
    if not parts:
        return "Global (keine Einschränkung)"  # i18n: ok
    return ", ".join(parts)


def _describe_status(status: str) -> str:
    """Translate hypothesis status to German for users.

    Args:
        status: Internal status string.

    Returns:
        German status name.
    """
    # i18n: ok
    status_map = {
        "candidate": "Kandidat (intern, noch nicht sichtbar)",
        "suggested": "Vorgeschlagen (wartet auf Bestätigung)",
        "confirmed": "Bestätigt (wird bei Anfrage angewendet)",
        "active": "Aktiv (wird automatisch angewendet)",
        "needs_review": "Überprüfung nötig (Widersprüche erkannt)",
        "paused": "Pausiert (manuell gestoppt)",
        "archived": "Archiviert (lange nicht genutzt)",
        "retired": "Vergessen (Tombstone aktiv)",
    }
    return status_map.get(status, status)


# ---------------------------------------------------------------
# SkillExplainer
# ---------------------------------------------------------------


class SkillExplainer:
    """Layer 6: Answers user questions about skill decisions.

    Reads hypothesis state, evidence records, and version history
    to construct explanations. Purely read-only: never modifies data.

    HC-SC-18: All 8 question types supported.
    HC-EXPLAIN-1: Missing data yields honest "no data" response.

    Usage:
        explainer = SkillExplainer(storage, ledger)
        response = explainer.explain("hyp_abc123", ExplainerQuestionType.WHY_PROMOTED)
    """

    def __init__(
        self,
        storage: HypothesisStorage,
        ledger: Optional[EvidenceLedger] = None,
    ) -> None:
        """Initialize the SkillExplainer.

        Args:
            storage: Hypothesis storage for DB access.
            ledger: Evidence ledger for in-memory evidence. If None,
                    only DB evidence is used.
        """
        self._storage = storage
        self._ledger = ledger

    def explain(
        self,
        hypothesis_id: str,
        question_type: ExplainerQuestionType,
    ) -> ExplainerResponse:
        """Answer a user question about a hypothesis.

        Dispatches to the appropriate handler for each question type.
        Returns an honest "no data" response if data is insufficient.

        Args:
            hypothesis_id: The hypothesis to explain.
            question_type: Which of the 8 question types.

        Returns:
            ExplainerResponse with the answer.
        """
        hyp = self._storage.get_hypothesis(hypothesis_id)
        if hyp is None:
            return ExplainerResponse(
                question_type=question_type,
                hypothesis_id=hypothesis_id,
                title=_NO_HYPOTHESIS_TITLE,
                explanation=_NO_HYPOTHESIS_MSG.format(hyp_id=hypothesis_id),
                has_data=False,
            )

        evidence_rows = self._storage.get_evidence_for_hypothesis(hypothesis_id)

        handlers = {
            ExplainerQuestionType.WHAT_RECOGNIZED: self._explain_what_recognized,
            ExplainerQuestionType.WHY_NOT_SKILL: self._explain_why_not_skill,
            ExplainerQuestionType.WHY_PROMOTED: self._explain_why_promoted,
            ExplainerQuestionType.WHEN_DRIFT: self._explain_when_drift,
            ExplainerQuestionType.WHAT_NEEDED: self._explain_what_needed,
            ExplainerQuestionType.LESSONS_LEARNED: self._explain_lessons_learned,
            ExplainerQuestionType.SCOPE_BOUNDARIES: self._explain_scope_boundaries,
            ExplainerQuestionType.COUNTER_EVIDENCE: self._explain_counter_evidence,
        }

        handler = handlers[question_type]
        return handler(hyp, evidence_rows)

    def list_question_types(self) -> list[tuple[str, str]]:
        """List all available question types with descriptions.

        Returns:
            List of (type_value, description_de) tuples.
        """
        # i18n: ok
        return [
            ("what_recognized", "Was hat das Pattern erkannt?"),
            ("why_not_skill", "Warum wurde das nicht zum Skill?"),
            ("why_promoted", "Warum wurde es zum Skill promotet?"),
            ("when_drift", "Wann wurde Drift erkannt?"),
            (
                "what_needed",
                "Was wäre nötig damit das Pattern wieder vertrauenswürdig wird?",
            ),
            ("lessons_learned", "Welche Lessons hat das System gelernt?"),
            ("scope_boundaries", "Wo gilt dieser Skill NICHT?"),
            ("counter_evidence", "Welche Belege sprechen GEGEN diesen Skill?"),
        ]

    # ── Type 1: What was recognized ─────────────────────────────

    def _explain_what_recognized(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 1: What did the pattern recognize?

        Answers from evidence + hypothesis claim.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records from DB.

        Returns:
            ExplainerResponse.
        """
        if not hyp.claim:
            return _no_data_response(
                ExplainerQuestionType.WHAT_RECOGNIZED,
                hyp.hypothesis_id,
            )

        # i18n: ok
        lines: list[str] = []
        lines.append(f"Pattern: {hyp.claim}")
        lines.append(f"Typ: {hyp.type}")
        lines.append(f"Geltungsbereich: {_describe_scope(hyp.scope)}")
        lines.append(f"Status: {_describe_status(hyp.status)}")
        lines.append(f"Version: v{hyp.version}")
        lines.append(f"Elo-Rating: {hyp.elo_rating:.0f}")

        if evidence:
            pos_count = sum(
                1 for e in evidence if e.get("signal_type") in POSITIVE_SIGNALS
            )
            neg_count = sum(
                1 for e in evidence if e.get("signal_type") in NEGATIVE_SIGNALS
            )
            lines.append("")
            lines.append(
                f"Belege: {len(evidence)} gesamt ({pos_count} positiv, {neg_count} negativ)"
            )
            lines.append(
                f"Erstellt: {hyp.created_at[:10] if hyp.created_at else 'unbekannt'}"
            )
            if hyp.last_applied:
                lines.append(f"Zuletzt angewendet: {hyp.last_applied[:10]}")

        return ExplainerResponse(
            question_type=ExplainerQuestionType.WHAT_RECOGNIZED,
            hypothesis_id=hyp.hypothesis_id,
            title=f"Pattern: {hyp.claim[:50]}",  # i18n: ok
            explanation="\n".join(lines),
            has_data=True,
            evidence_count=len(evidence),
        )

    # ── Type 2: Why not a skill ─────────────────────────────────

    def _explain_why_not_skill(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 2: Why was this NOT promoted to a skill?

        5-Why analysis for negative decision. Identifies which trigger
        conditions are not met.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records.

        Returns:
            ExplainerResponse with 5-Why chain.
        """
        if hyp.status in (STATUS_CONFIRMED, STATUS_ACTIVE):
            # Already a skill
            return ExplainerResponse(
                question_type=ExplainerQuestionType.WHY_NOT_SKILL,
                hypothesis_id=hyp.hypothesis_id,
                title="Bereits ein Skill",  # i18n: ok
                explanation=(  # i18n: ok
                    f"Dieses Pattern ist bereits ein Skill "
                    f"(Status: {_describe_status(hyp.status)})."
                ),
                has_data=True,
                evidence_count=len(evidence),
            )

        # i18n: ok
        reasons: list[str] = []

        # Check evidence count
        if len(evidence) < SUGGEST_MIN_EVIDENCE:
            reasons.append(
                f"1. Zu wenige Belege: {len(evidence)} von mindestens "
                f"{SUGGEST_MIN_EVIDENCE} benötigten."
            )

        # Check session diversity
        sessions = {e.get("episode_id") for e in evidence if e.get("episode_id")}
        if len(sessions) < SUGGEST_MIN_SESSIONS:
            reasons.append(
                f"2. Zu wenige Sessions: {len(sessions)} von mindestens "
                f"{SUGGEST_MIN_SESSIONS} benötigten."
            )

        # Check Elo rating
        threshold_key = _resolve_threshold_key(hyp)
        threshold = THRESHOLDS.get(threshold_key, THRESHOLDS["default"])
        if hyp.elo_rating < threshold.min_elo_rating:
            reasons.append(
                f"3. Elo-Rating zu niedrig: {hyp.elo_rating:.0f} "
                f"(Schwelle: {threshold.min_elo_rating:.0f} für {threshold_key})."
            )

        # Check BKT confidence
        if hyp.bayes_confidence < threshold.min_bkt_confidence:
            reasons.append(
                f"4. Bayes-Confidence zu niedrig: {hyp.bayes_confidence:.3f} "
                f"(Schwelle: {threshold.min_bkt_confidence:.3f})."
            )

        # Check contradictions
        neg_count = sum(1 for e in evidence if e.get("signal_type") in NEGATIVE_SIGNALS)
        if neg_count > 0:
            reasons.append(f"5. Widersprüche vorhanden: {neg_count} negative Belege.")

        if not reasons:
            reasons.append(
                "Keine spezifischen Gründe identifizierbar. "
                "Möglicherweise fehlt die explizite User-Bestätigung."
            )

        explanation = "Gründe warum dieses Pattern kein Skill ist:\n\n" + "\n".join(
            reasons
        )

        return ExplainerResponse(
            question_type=ExplainerQuestionType.WHY_NOT_SKILL,
            hypothesis_id=hyp.hypothesis_id,
            title="Noch kein Skill",  # i18n: ok
            explanation=explanation,
            has_data=True,
            evidence_count=len(evidence),
        )

    # ── Type 3: Why promoted ────────────────────────────────────

    def _explain_why_promoted(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 3: Why was this promoted to a skill?

        Shows evidence, confidence scores, and threshold met.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records.

        Returns:
            ExplainerResponse with promotion rationale.
        """
        if hyp.status in (STATUS_CANDIDATE, STATUS_SUGGESTED):
            return _no_data_response(
                ExplainerQuestionType.WHY_PROMOTED,
                hyp.hypothesis_id,
                "Dieses Pattern wurde noch nicht zum Skill promotet.",  # i18n: ok
            )

        # i18n: ok
        lines: list[str] = []
        lines.append(f"Skill '{hyp.claim[:50]}' wurde promotet weil:")
        lines.append("")

        pos_count = sum(1 for e in evidence if e.get("signal_type") in POSITIVE_SIGNALS)
        neg_count = sum(1 for e in evidence if e.get("signal_type") in NEGATIVE_SIGNALS)
        sessions = {e.get("episode_id") for e in evidence if e.get("episode_id")}

        lines.append(f"1. Positive Belege: {pos_count}")
        lines.append(f"2. Sessions: {len(sessions)}")
        lines.append(f"3. Elo-Rating: {hyp.elo_rating:.0f}")
        lines.append(f"4. Bayes-Confidence: {hyp.bayes_confidence:.3f}")
        lines.append(f"5. Widersprüche: {neg_count}")

        threshold_key = _resolve_threshold_key(hyp)
        threshold = THRESHOLDS.get(threshold_key, THRESHOLDS["default"])
        lines.append("")
        lines.append(f"Angewendete Schwelle ({threshold_key}):")
        lines.append(f"  Bestätigungen: >= {threshold.min_confirmations}")
        lines.append(f"  Elo: >= {threshold.min_elo_rating:.0f}")
        lines.append(f"  BKT: >= {threshold.min_bkt_confidence:.3f}")
        lines.append(f"  Sessions: >= {threshold.min_sessions}")

        if hyp.source_type == "learn_command":
            lines.append("")
            lines.append(
                "Hinweis: Dieser Skill wurde per /learn erstellt und ist decay-immun."
            )

        return ExplainerResponse(
            question_type=ExplainerQuestionType.WHY_PROMOTED,
            hypothesis_id=hyp.hypothesis_id,
            title=f"Promotion: {hyp.claim[:40]}",  # i18n: ok
            explanation="\n".join(lines),
            has_data=True,
            evidence_count=len(evidence),
        )

    # ── Type 4: When was drift detected ─────────────────────────

    def _explain_when_drift(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 4: When was drift detected?

        Shows timeline of contradictions in the evidence.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records.

        Returns:
            ExplainerResponse with contradiction timeline.
        """
        contradictions = [
            e for e in evidence if e.get("signal_type") in NEGATIVE_SIGNALS
        ]

        if not contradictions:
            return _no_data_response(
                ExplainerQuestionType.WHEN_DRIFT,
                hyp.hypothesis_id,
                "Kein Drift erkannt. Alle Belege sind positiv.",  # i18n: ok
            )

        # i18n: ok
        lines: list[str] = []
        lines.append(f"Drift-Timeline für '{hyp.claim[:40]}':")
        lines.append("")

        for i, c in enumerate(contradictions, 1):
            ts = c.get("created_at", "unbekannt")[:16]
            sig_type = c.get("signal_type", "unbekannt")
            strength = c.get("signal_strength", 0.0)
            lines.append(f"  {i}. [{ts}] {sig_type} (Stärke: {strength:.2f})")

        lines.append("")
        lines.append(
            f"Widersprüche gesamt: {len(contradictions)} von {len(evidence)} Belegen"
        )

        if hyp.last_contradiction_at:
            lines.append(f"Letzter Widerspruch: {hyp.last_contradiction_at[:16]}")

        if hyp.status == STATUS_NEEDS_REVIEW:
            lines.append("")
            lines.append(
                "Status: Überprüfung nötig. Zu viele Widersprüche in letzter Zeit."
            )

        return ExplainerResponse(
            question_type=ExplainerQuestionType.WHEN_DRIFT,
            hypothesis_id=hyp.hypothesis_id,
            title=f"Drift-Analyse: {hyp.claim[:40]}",  # i18n: ok
            explanation="\n".join(lines),
            has_data=True,
            evidence_count=len(evidence),
        )

    # ── Type 5: What is needed ──────────────────────────────────

    def _explain_what_needed(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 5: What would be needed to make the pattern trustworthy?

        Analyzes current deficits and recommends concrete actions.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records.

        Returns:
            ExplainerResponse with action recommendations.
        """
        # i18n: ok
        lines: list[str] = []
        lines.append(f"Massnahmen für '{hyp.claim[:40]}':")
        lines.append("")

        actions: list[str] = []
        threshold_key = _resolve_threshold_key(hyp)
        threshold = THRESHOLDS.get(threshold_key, THRESHOLDS["default"])

        pos_count = sum(1 for e in evidence if e.get("signal_type") in POSITIVE_SIGNALS)
        sessions = {e.get("episode_id") for e in evidence if e.get("episode_id")}

        # Evidence deficit
        if pos_count < threshold.min_confirmations:
            needed = threshold.min_confirmations - pos_count
            actions.append(
                f"Noch {needed} positive Bestätigung(en) nötig "
                f"(aktuell: {pos_count}, Schwelle: {threshold.min_confirmations})."
            )

        # Session deficit
        if len(sessions) < threshold.min_sessions:
            needed = threshold.min_sessions - len(sessions)
            actions.append(
                f"Noch {needed} weitere Session(s) nötig "
                f"(aktuell: {len(sessions)}, Schwelle: {threshold.min_sessions})."
            )

        # Elo deficit
        if hyp.elo_rating < threshold.min_elo_rating:
            deficit = threshold.min_elo_rating - hyp.elo_rating
            actions.append(
                f"Elo-Rating muss um {deficit:.0f} Punkte steigen "
                f"(aktuell: {hyp.elo_rating:.0f}, Schwelle: {threshold.min_elo_rating:.0f})."
            )

        # BKT deficit
        if hyp.bayes_confidence < threshold.min_bkt_confidence:
            actions.append(
                f"Bayes-Confidence muss steigen "
                f"(aktuell: {hyp.bayes_confidence:.3f}, "
                f"Schwelle: {threshold.min_bkt_confidence:.3f})."
            )

        # Needs review
        if hyp.status == STATUS_NEEDS_REVIEW:
            actions.append(
                "Das Pattern ist in Überprüfung wegen Widersprüchen. "
                "Bestätige oder korrigiere es bei der nächsten Anwendung."
            )

        if not actions:
            actions.append(
                "Keine konkreten Defizite identifiziert. "
                "Das Pattern erfüllt alle Schwellenwerte."
            )

        for i, action in enumerate(actions, 1):
            lines.append(f"  {i}. {action}")

        return ExplainerResponse(
            question_type=ExplainerQuestionType.WHAT_NEEDED,
            hypothesis_id=hyp.hypothesis_id,
            title=f"Nötige Schritte: {hyp.claim[:35]}",  # i18n: ok
            explanation="\n".join(lines),
            has_data=True,
            evidence_count=len(evidence),
        )

    # ── Type 6: Lessons learned ─────────────────────────────────

    def _explain_lessons_learned(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 6: What lessons did the system learn from this pattern?

        Summarizes version history, evidence trends, and corrections.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records.

        Returns:
            ExplainerResponse with lessons.
        """
        version_history = self._storage.get_version_history(hyp.hypothesis_id)

        # i18n: ok
        lines: list[str] = []
        lines.append(f"Lessons für Pattern '{hyp.claim[:40]}':")
        lines.append("")

        lessons: list[str] = []

        # Lesson from version changes
        if version_history:
            lines.append(f"Versionshistorie ({len(version_history)} Versionen):")
            for vh in version_history:
                v_num = vh.get("version", "?")
                reason = vh.get("change_reason", "")
                old_claim = vh.get("claim", "")
                if reason:
                    lines.append(f"  v{v_num}: {old_claim[:50]} (Grund: {reason})")
                    lessons.append(f"v{v_num} -> v{hyp.version}: {reason}")
            lines.append("")

        # Lesson from contradictions
        neg_count = sum(1 for e in evidence if e.get("signal_type") in NEGATIVE_SIGNALS)
        if neg_count > 0:
            ratio = neg_count / max(len(evidence), 1)
            lessons.append(
                f"{neg_count} Widersprüche ({ratio:.0%} der Belege) zeigen "
                f"dass das Pattern nicht immer zutrifft."
            )

        # Lesson from decay immunity
        if hyp.decay_immune:
            lessons.append("Vom User explizit als wichtig markiert (decay-immun).")

        # Lesson from source type
        if hyp.source_type == "learn_command":
            lessons.append(
                "Per /learn manuell erstellt. Starkes User-Signal für Relevanz."
            )
        elif hyp.source_type == "import":
            lessons.append(
                "Aus importierten Daten erkannt. "
                "Startet als 'suggested', braucht Live-Bestätigung."
            )

        if not lessons:
            return _no_data_response(
                ExplainerQuestionType.LESSONS_LEARNED,
                hyp.hypothesis_id,
                "Noch keine konkreten Lessons vorhanden. "  # i18n: ok
                "Das Pattern ist zu jung für eine Analyse.",
            )

        lines.append("Erkenntnisse:")
        for i, lesson in enumerate(lessons, 1):
            lines.append(f"  {i}. {lesson}")

        return ExplainerResponse(
            question_type=ExplainerQuestionType.LESSONS_LEARNED,
            hypothesis_id=hyp.hypothesis_id,
            title=f"Lessons: {hyp.claim[:40]}",  # i18n: ok
            explanation="\n".join(lines),
            has_data=True,
            evidence_count=len(evidence),
        )

    # ── Type 7: Scope boundaries ────────────────────────────────

    def _explain_scope_boundaries(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 7: Where does this skill NOT apply?

        Describes the defined scope and its boundaries.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records.

        Returns:
            ExplainerResponse with scope analysis.
        """
        # i18n: ok
        lines: list[str] = []
        lines.append(f"Geltungsbereich von '{hyp.claim[:40]}':")
        lines.append("")

        scope = hyp.scope
        lines.append(f"Definierter Scope: {_describe_scope(scope)}")
        lines.append("")

        # Describe where it does NOT apply
        lines.append("Gilt NICHT:")

        boundaries: list[str] = []

        if scope.client:
            boundaries.append(f"Bei anderen Kunden (nur für '{scope.client}').")
        else:
            boundaries.append("Keine kundenspezifische Einschränkung (global).")

        if scope.project:
            boundaries.append(f"In anderen Projekten (nur für '{scope.project}').")
        else:
            boundaries.append("Keine projektspezifische Einschränkung (global).")

        if scope.context:
            boundaries.append(f"Außerhalb des Kontexts: {', '.join(scope.context)}.")

        # Check for contradictions that might indicate scope issues
        neg_evidence = [e for e in evidence if e.get("signal_type") in NEGATIVE_SIGNALS]
        if neg_evidence:
            boundaries.append(
                f"{len(neg_evidence)} Widerspruch/Widersprüche deuten auf "
                "mögliche Scope-Probleme hin. Das Pattern trifft "
                "möglicherweise nicht in allen Situationen zu."
            )

        for i, boundary in enumerate(boundaries, 1):
            lines.append(f"  {i}. {boundary}")

        return ExplainerResponse(
            question_type=ExplainerQuestionType.SCOPE_BOUNDARIES,
            hypothesis_id=hyp.hypothesis_id,
            title=f"Scope: {hyp.claim[:40]}",  # i18n: ok
            explanation="\n".join(lines),
            has_data=True,
            evidence_count=len(evidence),
        )

    # ── Type 8: Counter evidence ────────────────────────────────

    def _explain_counter_evidence(
        self,
        hyp: Hypothesis,
        evidence: list[dict],
    ) -> ExplainerResponse:
        """Type 8: What evidence speaks AGAINST this skill?

        Lists all negative evidence with timestamps and details.

        Args:
            hyp: The hypothesis.
            evidence: Evidence records.

        Returns:
            ExplainerResponse with counter-evidence listing.
        """
        neg_evidence = [e for e in evidence if e.get("signal_type") in NEGATIVE_SIGNALS]

        if not neg_evidence:
            return _no_data_response(
                ExplainerQuestionType.COUNTER_EVIDENCE,
                hyp.hypothesis_id,
                "Keine Gegenbelege vorhanden. "  # i18n: ok
                "Alle bisherigen Belege sind positiv.",
            )

        # i18n: ok
        lines: list[str] = []
        lines.append(f"Gegenbelege für '{hyp.claim[:40]}':")
        lines.append("")

        for i, e in enumerate(neg_evidence, 1):
            ts = e.get("created_at", "unbekannt")[:16]
            sig_type = e.get("signal_type", "unbekannt")
            strength = e.get("signal_strength", 0.0)
            version = e.get("hypothesis_version", "?")

            type_desc = "Korrektur" if sig_type == "correction" else "Ablehnung"
            lines.append(
                f"  {i}. [{ts}] {type_desc} "
                f"(Stärke: {strength:.2f}, Version: v{version})"
            )

        lines.append("")
        total = len(evidence)
        ratio = len(neg_evidence) / max(total, 1)
        lines.append(
            f"Zusammenfassung: {len(neg_evidence)} von {total} Belegen "
            f"sind negativ ({ratio:.0%})."
        )

        if ratio > 0.3:
            lines.append("")
            lines.append(
                "Warnung: Mehr als 30% der Belege sind negativ. "
                "Dieses Pattern könnte angepasst oder enger begrenzt werden."
            )

        return ExplainerResponse(
            question_type=ExplainerQuestionType.COUNTER_EVIDENCE,
            hypothesis_id=hyp.hypothesis_id,
            title=f"Gegenbelege: {hyp.claim[:35]}",  # i18n: ok
            explanation="\n".join(lines),
            has_data=True,
            evidence_count=len(evidence),
        )
