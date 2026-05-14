"""Task router: classifies user messages into 6 task slots.

Three-stage heuristic:
  1. Explicit slot markers (/code, /reason, etc.)
  2. Pattern + keyword matching with score
  3. Fallback to CHAT

Per slot the appropriate model is resolved:
  User override > slot default > system default.
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

# Code block is an extremely strong signal
_CODE_BLOCK_SCORE = 100

# Pattern match score (per pattern)
_PATTERN_SCORE = 3

# Keyword match score (per keyword)
_KEYWORD_SCORE = 1


# ──────────────────────────────────────────────────────────────
# German ASCII-to-umlaut normalization
# ──────────────────────────────────────────────────────────────

# Order: uppercase before lowercase, so e.g. "Ae" is not
# accidentally matched as "a" + "e".
_UMLAUT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Ae", "Ä"),
    ("Oe", "Ö"),
    ("Ue", "Ü"),
    ("ae", "ä"),
    ("oe", "ö"),
    ("ue", "ü"),
)

# ss -> ß: whitelist approach instead of regex, because a generic
# vowel-ss-vowel pattern produces too many false positives
# (e.g. "processing" -> "proceßing", "Klassifizier" -> "Klaßifizier").
# Only unambiguously German stems.
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
    """Normalize ASCII umlaut representations to real German umlauts.

    Purpose: user inputs with ASCII umlaut substitutions should trigger
    the same keyword matches as inputs with real umlauts (ä, ö, ü, ß).

    The risk of over-generalization is low because:
    1. TaskRouter only matches against German keywords/patterns
    2. English terms with ae/oe/ue (queue, phoenix, aesthetic) do not
       match German slot keywords anyway
    3. Normalization only changes the internal classification input,
       not the displayed user message

    ss -> ß: deliberately uses a whitelist instead of a generic regex,
    because a vowel-ss-vowel pattern hits too many English words
    (processing, message, etc.). Only unambiguous German stems.

    Performance: pure string operations + a few regex subs. For 1000 chars < 0.1ms.
    """
    for ascii_form, umlaut in _UMLAUT_REPLACEMENTS:
        text = text.replace(ascii_form, umlaut)

    for pattern, replacement in _SS_TO_ESZETT_WORDS:
        text = pattern.sub(replacement, text)

    return text


@dataclass(frozen=True)
class SlotConfig:
    """Configuration for a single task slot.

    Immutable after construction. Values loaded from YAML.
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
    """Result of a task classification.

    Contains the detected slot, the score, and the matched indicators.
    """

    slot: TaskSlot
    score: int
    matched_patterns: tuple[str, ...] = ()
    matched_keywords: tuple[str, ...] = ()


class TaskRouter:
    """Classifies user messages into task slots and resolves models.

    Uses YAML-configured heuristics for deterministic classification.
    No ML, no LLM call for classification.

    Args:
        slot_configs: List of SlotConfig objects.
        model_service: ModelService for user override resolution.
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

        # Find explicit fallback
        for cfg in slot_configs:
            if cfg.fallback:
                self._fallback_slot = cfg.slot
                break

    def classify(self, text: str) -> ClassificationResult:
        """Classify a user message into a task slot.

        Three-stage heuristic:
          1. Explicit slot markers (prefix /code, /reason, etc.)
          2. Pattern + keyword matching with score
          3. Fallback to CHAT

        On tie, priority applies: CODE > REASON > RESEARCH > CREATIVE > QUICK > CHAT

        Args:
            text: User message.

        Returns:
            ClassificationResult with slot, score, and match details.
        """
        if not text or not text.strip():
            return ClassificationResult(slot=self._fallback_slot, score=0)

        # Stage 0: ASCII umlaut normalization for German keyword matching
        text = _normalize_german_input(text)

        # Stage 1: explicit markers
        stripped = text.strip()
        lower_stripped = stripped.lower()
        for slot in TaskSlot:
            prefix = f"/{slot.value}"
            if lower_stripped.startswith(prefix) and (
                len(lower_stripped) == len(prefix)
                or lower_stripped[len(prefix)] in (" ", "\n")
            ):
                return ClassificationResult(slot=slot, score=1000)

        # Stage 2: pattern + keyword matching
        text_lower = text.lower()
        word_count = len(text.split())

        best_result: ClassificationResult | None = None
        best_score = 0

        for slot in SLOT_PRIORITY:
            cfg = self._slots.get(slot)
            if cfg is None or cfg.fallback:
                continue

            # Word count filter
            if cfg.min_word_count is not None and word_count < cfg.min_word_count:
                continue
            if cfg.max_word_count is not None and word_count > cfg.max_word_count:
                continue

            score = 0
            matched_patterns: list[str] = []
            matched_keywords: list[str] = []

            # Pattern matching
            for pattern in cfg.patterns:
                if pattern in text:
                    # Code block (```) is a particularly strong signal
                    if pattern == "```":
                        score += _CODE_BLOCK_SCORE
                    else:
                        score += _PATTERN_SCORE
                    matched_patterns.append(pattern)

            # Keyword matching (case-insensitive)
            for keyword in cfg.keywords:
                if keyword.lower() in text_lower:
                    score += _KEYWORD_SCORE
                    matched_keywords.append(keyword)

            # Minimum keyword threshold check
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

        # Stage 3: fallback
        return ClassificationResult(slot=self._fallback_slot, score=0)

    def resolve_model(self, user_id: int, slot: TaskSlot) -> str | None:
        """Resolve the model for a user and slot.

        Priority:
          1. User override per slot (via ModelService/SQLite)
          2. User override global
          3. Slot default (from YAML, canonically resolved)
          4. None (caller uses system default)

        All return values are canonical model IDs (e.g. 'claude-opus-4-7'),
        never aliases. This prevents pool key duplicates.

        Args:
            user_id: Telegram user ID.
            slot: Detected task slot.

        Returns:
            Canonical model ID or None.
        """
        if self._model_service is not None:
            # 1. Slot-specific override (already canonical from ModelService)
            slot_override = self._model_service.get_user_model(user_id, slot=slot.value)
            if slot_override is not None:
                return slot_override

            # 2. Global override (already canonical from ModelService)
            global_override = self._model_service.get_user_model(user_id, slot="global")
            if global_override is not None:
                return global_override

        # 3. Slot default from config (alias -> canonical ID)
        cfg = self._slots.get(slot)
        if cfg is not None and cfg.default_model:
            resolve = _get_resolve_alias()
            canonical = resolve(cfg.default_model)
            if canonical is not None:
                return canonical
            # Fallback: default_model is already a full ID or unknown
            return cfg.default_model

        return None

    def get_default_for_slot(self, slot: TaskSlot) -> str:
        """Return the canonical model ID for the slot default (single source of truth).

        Falls back to DEFAULT_MODEL if the slot has no own default.

        Args:
            slot: Task slot.

        Returns:
            Canonical model ID (guaranteed non-empty).
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
        """Return the default models per slot.

        Returns:
            Dict of TaskSlot -> default_model alias.
        """
        return {
            slot: cfg.default_model
            for slot, cfg in self._slots.items()
            if cfg.default_model
        }


def load_slot_configs(yaml_path: Path | str | None = None) -> list[SlotConfig]:
    """Load SlotConfigs from YAML.

    On error: fallback to hardcoded defaults (CHAT-only).

    Args:
        yaml_path: Path to task_slots.yaml (None = default).

    Returns:
        List of SlotConfig objects.
    """
    if yaml_path is None:
        yaml_path = _DEFAULT_YAML
    path = Path(yaml_path)

    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except (FileNotFoundError, yaml.YAMLError, OSError) as exc:
        log.warning(
            "Could not load task_slots.yaml (%s). Falling back to CHAT-only default.",
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
            "task_slots.yaml has no 'slots' key. Falling back to CHAT-only default."
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
            log.warning("Unknown slot '%s' in YAML, skipping.", slot_name)
            continue

        if not isinstance(slot_data, dict):
            log.warning("Slot '%s' has no valid configuration.", slot_name)
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
        log.warning("No slots loaded from YAML. Falling back to CHAT-only default.")
        return [
            SlotConfig(
                slot=TaskSlot.CHAT,
                default_model="sonnet",
                fallback=True,
            )
        ]

    log.info(
        "TaskRouter: %d slots loaded from %s: %s",
        len(configs),
        path.name,
        ", ".join(c.slot.value for c in configs),
    )
    return configs
