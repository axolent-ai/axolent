"""LanguageContext: immutable language resolution result.

Canonical location for the frozen dataclass that represents
THE language decision for an entire request lifecycle.

Re-exported from application.language_resolver for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True, slots=True)
class LanguageContext:
    """Immutable language resolution result.

    Once created, this object represents THE language decision
    for an entire request lifecycle. All consumers (ChatService,
    DebateOrchestrator, StatusSession, PromptComposer) must use
    this context rather than resolving language independently.

    Attributes:
        code: ISO-639-1 language code, guaranteed non-empty.
        source: How the language was determined.
        confidence: Detection confidence (0.0..1.0). For sticky/override: 1.0.
        switched_from: Previous sticky value if a smart-switch occurred.
        request_id: Unique ID for audit correlation.
    """

    code: str
    source: Literal["override", "sticky", "detected", "default"]
    confidence: float
    switched_from: Optional[str]
    request_id: str

    def effective_lang(self) -> str:
        """Return the effective language code.

        Convenience method for contexts that only need the code.
        """
        return self.code

    @property
    def was_smart_switched(self) -> bool:
        """True if a smart-switch occurred (user implicitly changed language)."""
        return self.switched_from is not None
