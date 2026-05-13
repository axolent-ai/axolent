"""TaskRouter: Klassifiziert User-Nachrichten in 6 Task-Slots.

Dreistufige Heuristik:
  1. Explizite Slot-Marker (/code, /reason, etc.)
  2. Pattern + Keyword Matching mit Score
  3. Fallback auf CHAT

Pro Slot wird das passende Modell resolved:
  User-Override > Slot-Default > System-Default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from domain.task_slot import SLOT_PRIORITY, TaskSlot

if TYPE_CHECKING:
    from application.model_service import ModelService

# Lazy import helper to avoid circular import at module level
_resolve_alias = None


def _get_resolve_alias():
    """Lazy-load resolve_alias from model_service to avoid circular imports."""
    global _resolve_alias
    if _resolve_alias is None:
        from application.model_service import resolve_alias

        _resolve_alias = resolve_alias
    return _resolve_alias


log = logging.getLogger(__name__)

# Default YAML path
_DEFAULT_YAML = Path(__file__).parent.parent / "config" / "task_slots.yaml"

# Code-Block ist ein extrem starkes Signal
_CODE_BLOCK_SCORE = 100

# Pattern-Match Score (pro Pattern)
_PATTERN_SCORE = 3

# Keyword-Match Score (pro Keyword)
_KEYWORD_SCORE = 1


# ──────────────────────────────────────────────────────────────
# German ASCII-to-Umlaut Normalisierung
# ──────────────────────────────────────────────────────────────

# Reihenfolge: Großbuchstaben vor Kleinbuchstaben, damit z.B. "Ae" nicht
# versehentlich als "a" + "e" gematcht wird.
_UMLAUT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Ae", "Ä"),
    ("Oe", "Ö"),
    ("Ue", "Ü"),
    ("ae", "ä"),
    ("oe", "ö"),
    ("ue", "ü"),
)

# ss -> ß: Whitelist-Ansatz statt Regex, weil ein generisches Vokal-ss-Vokal
# Pattern zu viele False Positives erzeugt (z.B. "processing" -> "proceßing",
# "Klassifizier" -> "Klaßifizier"). Nur eindeutig deutsche Wortstämme.
_SS_TO_ESZETT_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b([Ss])trass", re.UNICODE), r"\1traß"),
    (re.compile(r"\b([Ss])chliess", re.UNICODE), r"\1chließ"),
    (re.compile(r"\b([Aa])usserdem", re.UNICODE), r"\1ußerdem"),
    (re.compile(r"\b([Aa])usserhalb", re.UNICODE), r"\1ußerhalb"),
    (re.compile(r"\b([Gg])emaess", re.UNICODE), r"\1emäß"),
    (re.compile(r"\b([Gg])ross", re.UNICODE), r"\1roß"),
    (re.compile(r"\b([Ww])eiss", re.UNICODE), r"\1eiß"),
    (re.compile(r"\b([Hh])eiss", re.UNICODE), r"\1eiß"),
)


def _normalize_german_input(text: str) -> str:
    """Normalisiert ASCII-Umlaut-Umschreibungen zu echten deutschen Umlauten.

    Zweck: User-Eingaben mit ASCII-Umlaut-Umschreibungen sollen dieselben
    Keyword-Matches auslösen wie Eingaben mit echten Umlauten (ä, ö, ü, ß).

    Risiko der Übergeneralisierung ist gering, weil:
    1. TaskRouter matcht nur gegen deutsche Keywords/Patterns
    2. Englische Begriffe mit ae/oe/ue (queue, phoenix, aesthetic) matchen
       ohnehin nicht gegen deutsche Slot-Keywords
    3. Die Normalisierung ändert nur den internen Klassifikations-Input,
       nicht die angezeigte User-Nachricht

    ss -> ß: Bewusst per Whitelist statt generischem Regex, weil ein
    Vokal-ss-Vokal-Pattern zu viele englische Wörter trifft (processing,
    message, klassifizier etc.). Nur eindeutige deutsche Wortstämme.

    Performance: Reine String-Operationen + wenige Regex-Subs. Bei 1000 Zeichen < 0.1ms.
    """
    for ascii_form, umlaut in _UMLAUT_REPLACEMENTS:
        text = text.replace(ascii_form, umlaut)

    for pattern, replacement in _SS_TO_ESZETT_WORDS:
        text = pattern.sub(replacement, text)

    return text


@dataclass(frozen=True)
class SlotConfig:
    """Konfiguration für einen einzelnen Task-Slot.

    Immutable nach Konstruktion. Geladene Werte aus YAML.
    """

    slot: TaskSlot
    default_model: str
    patterns: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    min_keyword_matches: int = 2
    min_word_count: int | None = None
    max_word_count: int | None = None
    fallback: bool = False


@dataclass(frozen=True)
class ClassificationResult:
    """Ergebnis einer Task-Klassifikation.

    Enthält den erkannten Slot, den Score und die gematchten Indikatoren.
    """

    slot: TaskSlot
    score: int
    matched_patterns: tuple[str, ...] = ()
    matched_keywords: tuple[str, ...] = ()


class TaskRouter:
    """Klassifiziert User-Nachrichten in Task-Slots und resolved Modelle.

    Nutzt YAML-konfigurierte Heuristiken für deterministische Klassifikation.
    Kein ML, kein LLM-Call für die Klassifikation.

    Args:
        slot_configs: Liste von SlotConfig-Objekten.
        model_service: ModelService für User-Override-Resolution.
    """

    def __init__(
        self,
        slot_configs: list[SlotConfig],
        model_service: Optional["ModelService"] = None,
    ) -> None:
        self._slots: dict[TaskSlot, SlotConfig] = {
            cfg.slot: cfg for cfg in slot_configs
        }
        self._model_service = model_service
        self._fallback_slot = TaskSlot.CHAT

        # Finde expliziten Fallback
        for cfg in slot_configs:
            if cfg.fallback:
                self._fallback_slot = cfg.slot
                break

    def classify(self, text: str) -> ClassificationResult:
        """Klassifiziert eine User-Nachricht in einen Task-Slot.

        Dreistufige Heuristik:
          1. Explizite Slot-Marker (Präfix /code, /reason, etc.)
          2. Pattern + Keyword Matching mit Score
          3. Fallback auf CHAT

        Bei Gleichstand gilt Priorität: CODE > REASON > RESEARCH > CREATIVE > QUICK > CHAT

        Args:
            text: User-Nachricht.

        Returns:
            ClassificationResult mit Slot, Score und Match-Details.
        """
        if not text or not text.strip():
            return ClassificationResult(slot=self._fallback_slot, score=0)

        # Stufe 0: ASCII-Umlaut-Normalisierung für deutsches Keyword-Matching
        text = _normalize_german_input(text)

        # Stufe 1: Explizite Marker
        stripped = text.strip()
        lower_stripped = stripped.lower()
        for slot in TaskSlot:
            prefix = f"/{slot.value}"
            if lower_stripped.startswith(prefix) and (
                len(lower_stripped) == len(prefix)
                or lower_stripped[len(prefix)] in (" ", "\n")
            ):
                return ClassificationResult(slot=slot, score=1000)

        # Stufe 2: Pattern + Keyword Matching
        text_lower = text.lower()
        word_count = len(text.split())

        best_result: ClassificationResult | None = None
        best_score = 0

        for slot in SLOT_PRIORITY:
            cfg = self._slots.get(slot)
            if cfg is None or cfg.fallback:
                continue

            # Word-Count-Filter
            if cfg.min_word_count is not None and word_count < cfg.min_word_count:
                continue
            if cfg.max_word_count is not None and word_count > cfg.max_word_count:
                continue

            score = 0
            matched_patterns: list[str] = []
            matched_keywords: list[str] = []

            # Pattern-Matching
            for pattern in cfg.patterns:
                if pattern in text:
                    # Code-Block (```) ist ein besonders starkes Signal
                    if pattern == "```":
                        score += _CODE_BLOCK_SCORE
                    else:
                        score += _PATTERN_SCORE
                    matched_patterns.append(pattern)

            # Keyword-Matching (case-insensitive)
            for keyword in cfg.keywords:
                if keyword.lower() in text_lower:
                    score += _KEYWORD_SCORE
                    matched_keywords.append(keyword)

            # Minimum-Keyword-Threshold prüfen
            if len(matched_keywords) < cfg.min_keyword_matches and not matched_patterns:
                continue

            if score > best_score:
                best_score = score
                best_result = ClassificationResult(
                    slot=slot,
                    score=score,
                    matched_patterns=tuple(matched_patterns),
                    matched_keywords=tuple(matched_keywords),
                )

        if best_result is not None:
            return best_result

        # Stufe 3: Fallback
        return ClassificationResult(slot=self._fallback_slot, score=0)

    def resolve_model(self, user_id: int, slot: TaskSlot) -> str | None:
        """Resolved das Modell für einen User und Slot.

        Priorität:
          1. User-Override pro Slot (via ModelService/SQLite)
          2. User-Override global
          3. Slot-Default (aus YAML, kanonisch aufgelöst)
          4. None (caller nutzt System-Default)

        Alle Rückgaben sind kanonische Model-IDs (z.B. 'claude-opus-4-7'),
        nie Aliase. Das verhindert Pool-Key-Duplikate.

        Args:
            user_id: Telegram-User-ID.
            slot: Erkannter Task-Slot.

        Returns:
            Kanonische Modell-ID oder None.
        """
        if self._model_service is not None:
            # 1. Slot-spezifischer Override (bereits kanonisch aus ModelService)
            slot_override = self._model_service.get_user_model(user_id, slot=slot.value)
            if slot_override is not None:
                return slot_override

            # 2. Globaler Override (bereits kanonisch aus ModelService)
            global_override = self._model_service.get_user_model(user_id, slot="global")
            if global_override is not None:
                return global_override

        # 3. Slot-Default aus Config (Alias -> kanonische ID)
        cfg = self._slots.get(slot)
        if cfg is not None and cfg.default_model:
            resolve = _get_resolve_alias()
            canonical = resolve(cfg.default_model)
            if canonical is not None:
                return canonical
            # Fallback: default_model ist bereits eine volle ID oder unbekannt
            return cfg.default_model

        return None

    def get_default_for_slot(self, slot: TaskSlot) -> str:
        """Liefert die kanonische Model-ID für den Slot-Default (Single Source of Truth).

        Fällt auf DEFAULT_MODEL zurück wenn der Slot keinen eigenen Default hat.

        Args:
            slot: Task-Slot.

        Returns:
            Kanonische Model-ID (garantiert nicht-leer).
        """
        from application.model_service import DEFAULT_MODEL

        cfg = self._slots.get(slot)
        if cfg is not None and cfg.default_model:
            resolve = _get_resolve_alias()
            canonical = resolve(cfg.default_model)
            if canonical is not None:
                return canonical
        return DEFAULT_MODEL

    def get_slot_defaults(self) -> dict[TaskSlot, str]:
        """Gibt die Default-Modelle pro Slot zurück.

        Returns:
            Dict von TaskSlot -> default_model alias.
        """
        return {
            slot: cfg.default_model
            for slot, cfg in self._slots.items()
            if cfg.default_model
        }


def load_slot_configs(yaml_path: Path | str | None = None) -> list[SlotConfig]:
    """Lädt SlotConfigs aus YAML.

    Bei Fehler: Fallback auf Hardcoded-Defaults (CHAT-only).

    Args:
        yaml_path: Pfad zur task_slots.yaml (None = Default).

    Returns:
        Liste von SlotConfig-Objekten.
    """
    if yaml_path is None:
        yaml_path = _DEFAULT_YAML
    path = Path(yaml_path)

    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except (FileNotFoundError, yaml.YAMLError, OSError) as exc:
        log.warning(
            "task_slots.yaml konnte nicht geladen werden (%s). "
            "Fallback auf CHAT-only Default.",
            exc,
        )
        return [
            SlotConfig(
                slot=TaskSlot.CHAT,
                default_model="sonnet",
                fallback=True,
            )
        ]

    if not isinstance(data, dict) or "slots" not in data:
        log.warning(
            "task_slots.yaml hat kein 'slots'-Key. Fallback auf CHAT-only Default."
        )
        return [
            SlotConfig(
                slot=TaskSlot.CHAT,
                default_model="sonnet",
                fallback=True,
            )
        ]

    configs: list[SlotConfig] = []
    for slot_name, slot_data in data["slots"].items():
        task_slot = TaskSlot.from_string(slot_name)
        if task_slot is None:
            log.warning("Unbekannter Slot '%s' in YAML, wird übersprungen.", slot_name)
            continue

        if not isinstance(slot_data, dict):
            log.warning("Slot '%s' hat keine gültige Konfiguration.", slot_name)
            continue

        configs.append(
            SlotConfig(
                slot=task_slot,
                default_model=str(slot_data.get("default_model", "sonnet")),
                patterns=tuple(str(p) for p in slot_data.get("patterns", [])),
                keywords=tuple(str(k) for k in slot_data.get("keywords", [])),
                min_keyword_matches=int(slot_data.get("min_keyword_matches", 2)),
                min_word_count=(
                    int(slot_data["min_word_count"])
                    if "min_word_count" in slot_data
                    else None
                ),
                max_word_count=(
                    int(slot_data["max_word_count"])
                    if "max_word_count" in slot_data
                    else None
                ),
                fallback=bool(slot_data.get("fallback", False)),
            )
        )

    if not configs:
        log.warning("Keine Slots in YAML geladen. Fallback auf CHAT-only Default.")
        return [
            SlotConfig(
                slot=TaskSlot.CHAT,
                default_model="sonnet",
                fallback=True,
            )
        ]

    log.info(
        "TaskRouter: %d Slots geladen aus %s: %s",
        len(configs),
        path.name,
        ", ".join(c.slot.value for c in configs),
    )
    return configs
