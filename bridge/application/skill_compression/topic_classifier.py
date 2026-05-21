"""Layer 2 adjunct: BERTopic background topic classification.

Classifies user messages into topics for pattern discovery. Runs in
batch mode (not real-time) as a suggestion provider for the Algorithmic
Candidate Layer.

HC-LAYER2-1 [BLOCKER]: BERTopic is a suggestion provider, NOT a
  truth source. Pattern Judge decides final.

HC-BERT-1 [BLOCKER]: BERTopic is an optional dependency. Missing
  installation MUST NOT crash the application.

IC-BERT-1: Trigger frequency is daily batch or /classify-topics command.
IC-BERT-2: If bertopic is not installed, show install hint.

Usage:
    classifier = TopicClassifier()
    if classifier.is_available():
        assignments = classifier.classify_batch(events)
        candidates = classifier.suggest_pattern_candidates(topic_id, events)
    else:
        # Graceful skip, log info

No hard dependencies. bertopic is imported lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from application.skill_compression.event_normalizer import NormalizedEvent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Availability check (HC-BERT-1)
# ---------------------------------------------------------------

_BERTOPIC_AVAILABLE: Optional[bool] = None

# IC-BERT-2: install hint
INSTALL_HINT = (
    "BERTopic ist nicht installiert. "  # i18n: ok
    "Für Topic-Klassifikation: pip install bertopic"
)


def _check_bertopic_available() -> bool:
    """Check if bertopic is importable (lazy, cached).

    HC-BERT-1: This check is used everywhere before BERTopic access.
    If not available, all operations gracefully skip.

    Returns:
        True if bertopic can be imported.
    """
    global _BERTOPIC_AVAILABLE  # noqa: PLW0603
    if _BERTOPIC_AVAILABLE is not None:
        return _BERTOPIC_AVAILABLE

    try:
        import bertopic  # noqa: F401

        _BERTOPIC_AVAILABLE = True
        log.info(
            "BERTopic available (version: %s)", getattr(bertopic, "__version__", "?")
        )
    except ImportError:
        _BERTOPIC_AVAILABLE = False
        log.info("BERTopic not installed. Topic classification disabled.")

    return _BERTOPIC_AVAILABLE


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TopicAssignment:
    """Result of assigning a topic to an event.

    Attributes:
        event_id: The classified event ID.
        topic_id: Assigned topic ID (-1 for outlier/noise).
        topic_label: Human-readable topic label.
        confidence: Classification confidence [0.0, 1.0].
        keywords: Top keywords for this topic.
    """

    event_id: str
    topic_id: int
    topic_label: str
    confidence: float
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidatePattern:
    """A pattern candidate suggested by topic clustering.

    HC-LAYER2-1: This is a SUGGESTION only. Pattern Judge decides.

    Attributes:
        topic_id: Source topic ID.
        suggested_claim: Proposed claim text.
        supporting_event_ids: Events that support this pattern.
        topic_keywords: Keywords that define this topic.
        event_count: Number of events in this topic.
        suggestion_confidence: How confident the suggestion is.
    """

    topic_id: int
    suggested_claim: str
    supporting_event_ids: tuple[str, ...] = ()
    topic_keywords: tuple[str, ...] = ()
    event_count: int = 0
    suggestion_confidence: float = 0.0


@dataclass
class TopicModelState:
    """Internal state for the topic model.

    Not frozen because the model state changes between fits.

    Attributes:
        model: The BERTopic model instance (or None).
        last_fit_event_count: Number of events at last fit.
        topic_count: Number of topics found.
        fitted: Whether the model has been fitted.
    """

    model: object = None
    last_fit_event_count: int = 0
    topic_count: int = 0
    fitted: bool = False


# ---------------------------------------------------------------
# TopicClassifier
# ---------------------------------------------------------------


class TopicClassifier:
    """BERTopic-based background topic classifier.

    Runs in batch mode to discover topic clusters in user messages.
    Suggests pattern candidates but NEVER promotes them directly.

    HC-LAYER2-1: Suggestion provider only.
    HC-BERT-1: All methods gracefully skip if bertopic is not installed.

    Thread safety: NOT thread-safe. Designed for single-threaded context.

    Usage:
        classifier = TopicClassifier()
        if classifier.is_available():
            assignments = classifier.classify_batch(events)
    """

    def __init__(
        self,
        *,
        min_topic_size: int = 3,
        nr_topics: Optional[int] = None,
        language: str = "multilingual",
    ) -> None:
        """Initialize the TopicClassifier.

        Args:
            min_topic_size: Minimum cluster size for a topic.
            nr_topics: Target number of topics (None = auto).
            language: Language for the embedding model.
        """
        self._min_topic_size = min_topic_size
        self._nr_topics = nr_topics
        self._language = language
        self._state = TopicModelState()

    def is_available(self) -> bool:
        """Check if BERTopic is available for use.

        HC-BERT-1: Returns False if bertopic is not installed.

        Returns:
            True if BERTopic can be used.
        """
        return _check_bertopic_available()

    def classify_batch(
        self,
        events: list[NormalizedEvent],
    ) -> list[TopicAssignment]:
        """Classify a batch of events into topics.

        Fits a new BERTopic model on the provided events and returns
        topic assignments. This is designed for periodic batch runs,
        not real-time classification.

        HC-BERT-1: Returns empty list if bertopic not installed.
        HC-LAYER2-1: Results are suggestions only.

        Args:
            events: List of normalized events to classify.

        Returns:
            List of TopicAssignment objects.
        """
        if not self.is_available():
            log.info("BERTopic not available, skipping classification")
            return []

        if not events:
            return []

        # Minimum events needed for meaningful clustering
        if len(events) < self._min_topic_size:
            log.info(
                "Too few events for topic classification: %d < %d",
                len(events),
                self._min_topic_size,
            )
            return []

        try:
            return self._fit_and_classify(events)
        except Exception:
            log.exception("BERTopic classification failed")
            return []

    def suggest_pattern_candidates(
        self,
        topic_id: int,
        recent_events: list[NormalizedEvent],
    ) -> list[CandidatePattern]:
        """Suggest pattern candidates from a specific topic.

        Takes events assigned to a topic and proposes pattern
        candidates based on keyword overlap and frequency.

        HC-LAYER2-1: These are SUGGESTIONS. Pattern Judge decides.
        HC-BERT-1: Returns empty list if bertopic not installed.

        Args:
            topic_id: The topic to generate candidates from.
            recent_events: Events assigned to this topic.

        Returns:
            List of CandidatePattern suggestions.
        """
        if not self.is_available():
            return []

        if not self._state.fitted or self._state.model is None:
            log.info("Topic model not fitted, no candidates to suggest")
            return []

        if topic_id < 0:
            # -1 is the outlier/noise topic in BERTopic
            return []

        if not recent_events:
            return []

        try:
            return self._generate_candidates(topic_id, recent_events)
        except Exception:
            log.exception("Failed to generate candidates for topic %d", topic_id)
            return []

    def get_topic_info(self) -> list[dict]:
        """Get information about all discovered topics.

        HC-BERT-1: Returns empty list if not available or not fitted.

        Returns:
            List of dicts with topic_id, count, and keywords.
        """
        if not self.is_available() or not self._state.fitted:
            return []

        if self._state.model is None:
            return []

        try:
            model = self._state.model
            # BERTopic.get_topic_info() returns a DataFrame
            info_df = model.get_topic_info()
            result = []
            for _, row in info_df.iterrows():
                tid = row.get("Topic", -1)
                if tid == -1:
                    continue  # skip outlier topic
                result.append(
                    {
                        "topic_id": int(tid),
                        "count": int(row.get("Count", 0)),
                        "name": str(row.get("Name", "")),
                        "representation": str(row.get("Representation", "")),
                    }
                )
            return result
        except Exception:
            log.exception("Failed to get topic info")
            return []

    def reset(self) -> None:
        """Reset the topic model state.

        Useful for re-fitting with new data.
        """
        self._state = TopicModelState()
        log.info("Topic classifier state reset")

    # ── Private methods ──────────────────────────────────────────

    def _fit_and_classify(
        self,
        events: list[NormalizedEvent],
    ) -> list[TopicAssignment]:
        """Fit BERTopic model and classify events.

        Args:
            events: Events to classify.

        Returns:
            Topic assignments.
        """
        from bertopic import BERTopic

        # Prepare document texts
        documents = [e.raw_text for e in events]

        # Create and fit model
        model = BERTopic(
            min_topic_size=self._min_topic_size,
            nr_topics=self._nr_topics,
            language=self._language,
            calculate_probabilities=True,
            verbose=False,
        )

        topics, probabilities = model.fit_transform(documents)

        # Store state
        self._state.model = model
        self._state.last_fit_event_count = len(events)
        self._state.topic_count = len(set(topics)) - (1 if -1 in topics else 0)
        self._state.fitted = True

        log.info(
            "BERTopic fitted: %d events -> %d topics",
            len(events),
            self._state.topic_count,
        )

        # Build assignments
        assignments: list[TopicAssignment] = []
        for i, event in enumerate(events):
            topic_id = int(topics[i])

            # Get topic keywords
            keywords: tuple[str, ...] = ()
            if topic_id >= 0:
                try:
                    topic_words = model.get_topic(topic_id)
                    if topic_words:
                        keywords = tuple(w for w, _ in topic_words[:5])
                except Exception:  # nosec B110
                    log.debug("Failed to get keywords for topic %d", topic_id)

            # Get confidence
            confidence = 0.0
            if probabilities is not None and len(probabilities) > i:
                prob_row = probabilities[i]
                if hasattr(prob_row, "__len__") and len(prob_row) > 0:
                    confidence = float(max(prob_row))
                elif isinstance(prob_row, (int, float)):
                    confidence = float(prob_row)

            # Generate topic label
            label = f"Topic {topic_id}"
            if keywords:
                label = (
                    f"{keywords[0]} / {keywords[1]}"
                    if len(keywords) > 1
                    else keywords[0]
                )

            assignments.append(
                TopicAssignment(
                    event_id=event.event_id,
                    topic_id=topic_id,
                    topic_label=label,
                    confidence=confidence,
                    keywords=keywords,
                )
            )

        return assignments

    def _generate_candidates(
        self,
        topic_id: int,
        events: list[NormalizedEvent],
    ) -> list[CandidatePattern]:
        """Generate pattern candidates from a topic cluster.

        Args:
            topic_id: The topic to extract candidates from.
            events: Events in this topic.

        Returns:
            CandidatePattern suggestions.
        """
        if self._state.model is None:
            return []

        model = self._state.model

        # Get topic keywords
        try:
            topic_words = model.get_topic(topic_id)
        except Exception:
            return []

        if not topic_words:
            return []

        keywords = tuple(w for w, _ in topic_words[:7])
        event_ids = tuple(e.event_id for e in events)

        # Suggest a claim based on the top keywords and common intents
        intents = {}
        domains = {}
        for e in events:
            intents[e.intent] = intents.get(e.intent, 0) + 1
            domains[e.domain] = domains.get(e.domain, 0) + 1

        top_intent = max(intents, key=intents.get) if intents else "general"
        top_domain = max(domains, key=domains.get) if domains else "general"

        # Build a suggested claim from topic keywords and common patterns
        keyword_str = ", ".join(keywords[:3])
        suggested_claim = (
            f"User requests about {keyword_str} "
            f"(intent: {top_intent}, domain: {top_domain})"
        )

        confidence = min(1.0, len(events) / 10.0)

        return [
            CandidatePattern(
                topic_id=topic_id,
                suggested_claim=suggested_claim,
                supporting_event_ids=event_ids,
                topic_keywords=keywords,
                event_count=len(events),
                suggestion_confidence=confidence,
            )
        ]
