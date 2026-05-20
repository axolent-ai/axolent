"""Language Control Plane: dedicated subsystem for language enforcement.

This package consolidates all language-related logic into a single
coherent subsystem. Components:

- LanguageContext: immutable per-request language decision
- LanguageResolver: single-entry-point resolution
- LanguageContract: dynamic natural-language contract builder
- ModelAdherenceProfile: per-model enforcement levels
- ResponseLanguageVerifier: post-hoc output language verification
- RepairService: automatic re-query on language violations
- StreamGuard: early streaming abort on wrong-language output
- LanguageDetectorBackend: protocol for pluggable detection backends
- VerificationStatus: three-level verification outcome (PASS/WARN/FAIL)
"""

from application.language.backends import (
    DomainLanguageBackend,
    LangdetectBackend,
    LanguageDetectorBackend,
)
from application.language.context import LanguageContext
from application.language.contract import LanguageContract
from application.language.enforcement import EnforcementResult, LanguageEnforcement
from application.language.model_profiles import (
    ModelAdherenceProfile,
    get_profile,
)
from application.language.repair_service import RepairService
from application.language.resolver import LanguageResolver
from application.language.stream_guard import StreamGuard
from application.language.verifier import (
    ResponseLanguageVerifier,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "DomainLanguageBackend",
    "EnforcementResult",
    "LangdetectBackend",
    "LanguageContext",
    "LanguageContract",
    "LanguageDetectorBackend",
    "LanguageEnforcement",
    "LanguageResolver",
    "ModelAdherenceProfile",
    "RepairService",
    "ResponseLanguageVerifier",
    "StreamGuard",
    "VerificationResult",
    "VerificationStatus",
    "get_profile",
]
