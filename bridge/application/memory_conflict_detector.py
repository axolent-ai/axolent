"""Memory Conflict Detector: identifies contradictory memory entries.

Scans a list of memory entries for subject-value conflicts using
DE/EN regex patterns (e.g. "Meine Lieblingsfarbe ist blau" vs
"Meine Lieblingsfarbe ist gruen").

Architecture: Application-layer service. Used by ChatService when
building the memory context for the system prompt.

Heuristic approach: pattern-based subject-value extraction with
simple grouping. NOT semantic understanding. False positives are
possible (e.g. "Mein Auto ist rot" vs "Mein Auto heisst Tesla"
have different predicates but the same extracted subject).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------
# Subject-Value extraction patterns (DE + EN)
# ---------------------------------------------------------------

_SUBJECT_VALUE_PATTERNS: list[re.Pattern[str]] = [
    # DE compound: "Meine Lieblingsfarbe ist blau"
    re.compile(
        r"(?:mein[e]?\s+)?lieblings(\w+)\s+(?:ist|sind|war|waren)\s+(.+)",
        re.IGNORECASE,
    ),
    # DE generic: "Mein Auto ist rot", "Meine Katze heisst Luna"
    re.compile(
        r"mein[e]?\s+(\w+)\s+(?:ist|sind|heisst|heißt|war|waren)\s+(.+)",
        re.IGNORECASE,
    ),
    # EN: "My favorite color is blue", "My car is red"
    re.compile(
        r"my\s+(?:favorite\s+|favourite\s+)?(\w+)\s+(?:is|are|was|were)\s+(.+)",
        re.IGNORECASE,
    ),
]


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """A detected memory conflict.

    Attributes:
        subject: The conflicting subject (e.g. "farbe", "color").
        values: List of conflicting values.
        entry_ids: List of entry IDs with conflicting values.
    """

    subject: str
    values: list[str]
    entry_ids: list[str]


class MemoryConflictDetector:
    """Detects conflicting memory entries based on subject-value patterns.

    Heuristic approach: extracts (subject, value) pairs from memory
    entries using regex patterns, then groups by subject. If a subject
    has multiple different values, it is a conflict.

    Limitations:
        - Only DE and EN patterns are supported.
        - Cannot distinguish semantic predicates (color vs name).
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
        # Map: subject -> {value: [entry_ids]}
        subject_map: dict[str, dict[str, list[str]]] = {}

        for entry in entries:
            content = entry.get("content", "")
            entry_id = entry.get("id", "")

            for pattern in _SUBJECT_VALUE_PATTERNS:
                m = pattern.search(content)
                if m:
                    subject = m.group(1).strip().lower()
                    value = m.group(2).strip().lower()
                    # Truncate value to first 100 chars to avoid noise
                    value = value[:100]

                    if subject not in subject_map:
                        subject_map[subject] = {}
                    if value not in subject_map[subject]:
                        subject_map[subject][value] = []
                    subject_map[subject][value].append(entry_id)
                    break  # First matching pattern wins

        # Build conflicts: subjects with 2+ distinct values
        conflicts: list[MemoryConflict] = []
        for subject, value_map in subject_map.items():
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
                    )
                )

        return conflicts
