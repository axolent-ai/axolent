"""Memory Conflict Detector: identifies contradictory memory entries.

Scans a list of memory entries for subject-value conflicts using
DE/EN regex patterns (e.g. "Meine Lieblingsfarbe ist blau" vs
"Meine Lieblingsfarbe ist gruen").

Architecture: Application-layer service. Used by ChatService when
building the memory context for the system prompt.

Heuristic approach: pattern-based subject-value extraction with
predicate-type grouping. Conflicts are only raised when subject AND
predicate type match (e.g. "Mein Auto ist rot" vs "Mein Auto ist blau"
conflict because both use the "property" predicate, but "Mein Auto ist rot"
vs "Mein Auto heisst Tesla" do NOT conflict because "property" and "name"
are different predicate types).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------
# Predicate types
# ---------------------------------------------------------------

PREDICATE_PROPERTY = "property"  # ist/sind/war/were/is/are
PREDICATE_NAME = "name"  # heisst/nennt sich/is called


# ---------------------------------------------------------------
# Subject-Value extraction patterns (DE + EN)
# Each pattern returns (subject, value) in groups 1 and 2,
# and is tagged with a predicate_type.
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SubjectValuePattern:
    """Internal: a regex pattern tagged with a predicate type."""

    pattern: re.Pattern[str]
    predicate_type: str


_SUBJECT_VALUE_PATTERNS: list[_SubjectValuePattern] = [
    # DE compound: "Meine Lieblingsfarbe ist blau" (property)
    _SubjectValuePattern(
        pattern=re.compile(
            r"(?:mein[e]?\s+)?lieblings(\w+)\s+(?:ist|sind|war|waren)\s+(.+)",
            re.IGNORECASE,
        ),
        predicate_type=PREDICATE_PROPERTY,
    ),
    # DE name predicate: "Mein Auto heisst Tesla", "Meine Katze nennt sich Luna"
    _SubjectValuePattern(
        pattern=re.compile(
            r"mein[e]?\s+(\w+)\s+(?:heisst|heißt|nennt\s+sich)\s+(.+)",
            re.IGNORECASE,
        ),
        predicate_type=PREDICATE_NAME,
    ),
    # DE property predicate: "Mein Auto ist rot"
    _SubjectValuePattern(
        pattern=re.compile(
            r"mein[e]?\s+(\w+)\s+(?:ist|sind|war|waren)\s+(.+)",
            re.IGNORECASE,
        ),
        predicate_type=PREDICATE_PROPERTY,
    ),
    # EN name predicate: "My car is called Tesla"
    _SubjectValuePattern(
        pattern=re.compile(
            r"my\s+(?:favorite\s+|favourite\s+)?(\w+)\s+is\s+called\s+(.+)",
            re.IGNORECASE,
        ),
        predicate_type=PREDICATE_NAME,
    ),
    # EN property predicate: "My favorite color is blue", "My car is red"
    _SubjectValuePattern(
        pattern=re.compile(
            r"my\s+(?:favorite\s+|favourite\s+)?(\w+)\s+(?:is|are|was|were)\s+(.+)",
            re.IGNORECASE,
        ),
        predicate_type=PREDICATE_PROPERTY,
    ),
]


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """A detected memory conflict.

    Attributes:
        subject: The conflicting subject (e.g. "farbe", "color").
        values: List of conflicting values.
        entry_ids: List of entry IDs with conflicting values.
        predicate_type: The predicate type that conflicts share.
    """

    subject: str
    values: list[str]
    entry_ids: list[str]
    predicate_type: str = PREDICATE_PROPERTY


def is_conflict_relevant_to_intent(
    conflict: MemoryConflict,
    skill_trigger: str | None,
    user_input: str,
) -> bool:
    """Determine if a memory conflict is relevant given a matched skill.

    Round-4 rewrite (Codex briefing 2026-05-27):
    The Round-3 logic used "trigger in conflict.values" which is semantically
    wrong. Example: Skill "rot -> RGB erklären" + Memory conflict with
    values=[blau, grün, rot] would wrongly show the conflict because "rot"
    appears in values, even though the skill has nothing to do with
    Lieblingsfarbe.

    New logic (subject/intent match):
    - If NO skill matched: all conflicts are relevant (existing behavior).
    - If a skill matched: conflict is relevant ONLY if:
      1. conflict.subject appears in the skill claim/trigger text, OR
      2. conflict.subject appears in the user input.
    - Otherwise: conflict is irrelevant (skill takes priority).

    Args:
        conflict: A detected MemoryConflict.
        skill_trigger: The full skill claim text (None = no skill matched).
        user_input: The original user input text.

    Returns:
        True if the conflict should be included in the prompt, False if it
        should be suppressed because it is irrelevant to the active skill.
    """
    # No skill matched: all conflicts are relevant (existing behavior)
    if not skill_trigger:
        return True

    # Normalize for comparison
    subject_lower = conflict.subject.strip().lower()
    skill_text_lower = skill_trigger.strip().lower()
    input_lower = user_input.strip().lower()

    # Relevant only if conflict.subject appears in the skill text
    if subject_lower in skill_text_lower:
        return True

    # OR if conflict.subject appears in the user input
    if subject_lower in input_lower:
        return True

    # Otherwise: NOT relevant. Skill has priority over this conflict.
    return False


class MemoryConflictDetector:
    """Detects conflicting memory entries based on subject-value patterns.

    Heuristic approach: extracts (subject, predicate_type, value) triples
    from memory entries using regex patterns, then groups by
    (subject, predicate_type). If a group has multiple different values,
    it is a conflict.

    Key improvement: different predicate types (property vs name) on the
    same subject do NOT conflict. "Mein Auto ist rot" and "Mein Auto
    heisst Tesla" are NOT a conflict because they describe different
    attributes.

    Limitations:
        - Only DE and EN patterns are supported.
        - Pattern-based, not semantic understanding.

    Usage:
        detector = MemoryConflictDetector()
        conflicts = detector.detect(entries)
        # conflicts is a list of MemoryConflict
    """

    def detect(self, entries: list[dict]) -> list[MemoryConflict]:
        """Detect conflicts in a list of memory entries.

        Args:
            entries: List of memory entry dicts (must have 'id' and 'content').

        Returns:
            List of MemoryConflict objects. Empty = no conflicts.
        """
        # Map: (subject, predicate_type) -> {value: [entry_ids]}
        subject_map: dict[tuple[str, str], dict[str, list[str]]] = {}

        for entry in entries:
            content = entry.get("content", "")
            entry_id = entry.get("id", "")

            for svp in _SUBJECT_VALUE_PATTERNS:
                m = svp.pattern.search(content)
                if m:
                    subject = m.group(1).strip().lower()
                    value = m.group(2).strip().lower()
                    # Truncate value to first 100 chars to avoid noise
                    value = value[:100]
                    key = (subject, svp.predicate_type)

                    if key not in subject_map:
                        subject_map[key] = {}
                    if value not in subject_map[key]:
                        subject_map[key][value] = []
                    subject_map[key][value].append(entry_id)
                    break  # First matching pattern wins

        # Build conflicts: groups with 2+ distinct values
        conflicts: list[MemoryConflict] = []
        for (subject, predicate_type), value_map in subject_map.items():
            if len(value_map) >= 2:
                all_values = list(value_map.keys())
                all_ids: list[str] = []
                for ids in value_map.values():
                    all_ids.extend(ids)
                conflicts.append(
                    MemoryConflict(
                        subject=subject,
                        values=all_values,
                        entry_ids=all_ids,
                        predicate_type=predicate_type,
                    )
                )

        return conflicts
