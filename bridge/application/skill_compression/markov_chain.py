"""Layer 2: Per-User Markov Chain for Skill-Compression.

Builds a first-order transition matrix between action types
(intent + domain). Allows next-action prediction for pattern
candidate generation.

IC-MARKOV-1: First-order Markov chain for v1. The state is
defined as "domain.intent" (e.g. "marketing.create_ad_copy").
Second-order (bigram states) is a v2 extension.

HC-LAYER2-1: Markov predictions are candidate signals, NOT truth.
They feed into the Evidence Ledger, never directly into SkillMatcher.

The chain is fully incremental: each new event updates the transition
matrix without recomputing from scratch. No GPU, no external dependency.

State representation matches N-Gram's event key format for consistency
across Layer 2 algorithms.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from application.skill_compression.event_normalizer import NormalizedEvent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarkovTransition:
    """A single state transition with its probability.

    Attributes:
        from_state: Source state (e.g. "marketing.create_ad_copy").
        to_state: Target state (e.g. "marketing.analyze").
        probability: Transition probability [0.0, 1.0].
        observations: Number of times this transition was observed.
    """

    from_state: str
    to_state: str
    probability: float
    observations: int


# ---------------------------------------------------------------
# Markov Chain (mutable, not frozen: it accumulates state)
# ---------------------------------------------------------------


class MarkovChain:
    """First-order Markov chain for per-user action prediction.

    Maintains a transition count matrix and computes probabilities
    on-the-fly. Fully incremental: call update() with each new event.

    The chain does NOT store raw events, only transition counts.
    This is privacy-friendly (no message content stored).

    Usage:
        chain = MarkovChain()
        chain.update(event_1)
        chain.update(event_2)
        chain.update(event_3)
        predictions = chain.predict_next("marketing.create_ad_copy")
    """

    def __init__(self) -> None:
        """Initialize an empty Markov chain."""
        # _counts[from_state][to_state] = observation count
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_state: str | None = None
        self._total_transitions: int = 0

    @staticmethod
    def _event_key(event: NormalizedEvent) -> str:
        """Derive canonical state key from a NormalizedEvent.

        Matches N-Gram's _event_key for cross-algorithm consistency.

        Args:
            event: A normalized event.

        Returns:
            State key string (e.g. 'marketing.create_ad_copy').
        """
        return f"{event.domain}.{event.intent}"

    def update(self, event: NormalizedEvent) -> None:
        """Update the chain with a new event (incremental).

        If there is a previous state, records a transition from
        the previous state to the current state.

        Args:
            event: The new normalized event.
        """
        current_state = self._event_key(event)

        if self._last_state is not None:
            self._counts[self._last_state][current_state] += 1
            self._total_transitions += 1
            log.debug(
                "Markov transition: %s -> %s (count=%d)",
                self._last_state,
                current_state,
                self._counts[self._last_state][current_state],
            )

        self._last_state = current_state

    def update_batch(self, events: list[NormalizedEvent]) -> None:
        """Update the chain with a batch of chronologically ordered events.

        Convenience method for processing historical event sequences.

        Args:
            events: Chronologically ordered list of events.
        """
        for event in events:
            self.update(event)

    def predict_next(
        self,
        current_state: str,
        *,
        top_k: int = 5,
    ) -> list[MarkovTransition]:
        """Predict the most likely next states from the current state.

        Args:
            current_state: The current state key.
            top_k: Maximum number of predictions to return.

        Returns:
            List of MarkovTransition objects sorted by probability (descending).
        """
        if current_state not in self._counts:
            return []

        transitions = self._counts[current_state]
        total = sum(transitions.values())

        if total == 0:
            return []

        results: list[MarkovTransition] = []
        for to_state, count in transitions.items():
            results.append(
                MarkovTransition(
                    from_state=current_state,
                    to_state=to_state,
                    probability=count / total,
                    observations=count,
                )
            )

        # Sort by probability descending
        results.sort(key=lambda t: t.probability, reverse=True)

        return results[:top_k]

    def get_transition_probability(
        self,
        from_state: str,
        to_state: str,
    ) -> float:
        """Get the probability of a specific transition.

        Args:
            from_state: Source state.
            to_state: Target state.

        Returns:
            Probability [0.0, 1.0]. Returns 0.0 if transition was never observed.
        """
        if from_state not in self._counts:
            return 0.0

        transitions = self._counts[from_state]
        total = sum(transitions.values())
        if total == 0:
            return 0.0

        return transitions.get(to_state, 0) / total

    def get_all_states(self) -> set[str]:
        """Return all states that have been observed.

        Returns:
            Set of state keys.
        """
        states: set[str] = set()
        for from_state, transitions in self._counts.items():
            states.add(from_state)
            states.update(transitions.keys())
        return states

    @property
    def total_transitions(self) -> int:
        """Total number of recorded transitions."""
        return self._total_transitions

    def get_state_counts(self, from_state: str) -> dict[str, int]:
        """Get raw transition counts from a specific state.

        Useful for serialization and debugging.

        Args:
            from_state: Source state.

        Returns:
            Dict mapping to_state -> count. Empty dict if state unknown.
        """
        if from_state not in self._counts:
            return {}
        return dict(self._counts[from_state])

    def to_dict(self) -> dict:
        """Serialize the chain to a dict for persistence.

        Returns:
            Dict with counts and last_state.
        """
        return {
            "counts": {
                from_s: dict(to_counts) for from_s, to_counts in self._counts.items()
            },
            "last_state": self._last_state,
            "total_transitions": self._total_transitions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MarkovChain:
        """Restore a chain from a serialized dict.

        Args:
            data: Dict as produced by to_dict().

        Returns:
            Restored MarkovChain instance.
        """
        chain = cls()
        for from_s, to_counts in data.get("counts", {}).items():
            for to_s, count in to_counts.items():
                chain._counts[from_s][to_s] = count
        chain._last_state = data.get("last_state")
        chain._total_transitions = data.get("total_transitions", 0)
        return chain

    def reset(self) -> None:
        """Clear all recorded transitions.

        Resets the chain to its initial empty state.
        """
        self._counts.clear()
        self._last_state = None
        self._total_transitions = 0
