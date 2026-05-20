"""Language Control Plane: dedicated subsystem for language enforcement.

This package consolidates all language-related logic into a single
coherent subsystem. Components:

Phase 1:
- LanguageContext: immutable per-request language decision
- LanguageResolver: single-entry-point resolution
- LanguageContract: dynamic natural-language contract builder
- ModelAdherenceProfile: per-model enforcement levels
- ResponseLanguageVerifier: post-hoc output language verification
- RepairService: automatic re-query on language violations
- StreamGuard: early streaming abort on wrong-language output
- LanguageDetectorBackend: protocol for pluggable detection backends
- VerificationStatus: three-level verification outcome (PASS/WARN/FAIL)

Phase 2 additions:
- LanguageRegistry: central, read-only source of truth for language metadata
- DetectionOrchestrator: multi-backend language detection with fallback logic
- DetectionTier, LanguageRegistryEntry: registry data model
- DetectionCandidate, OrchestratedDetection: orchestrator data model

Phase 2 Add-on 3:
- DetectionAuditEvent: frozen, JSON-serialisable audit event per detection
- DetectionAuditLogger: optional structured logger for audit events
- build_audit_event: factory to create audit events from context + detection
"""

from application.language.audit import (
    DetectionAuditEvent,
    DetectionAuditLogger,
    build_audit_event,
)
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
from application.language.orchestrator import (
    DetectionCandidate,
    DetectionOrchestrator,
    DetectionOrchestratorProtocol,
    OrchestratedDetection,
)
from application.language.registry import (
    DetectionTier,
    InMemoryLanguageRegistry,
    LanguageRegistryEntry,
    LanguageRegistryProtocol,
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
    # Phase 1
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
    # Phase 2: Registry
    "DetectionTier",
    "InMemoryLanguageRegistry",
    "LanguageRegistryEntry",
    "LanguageRegistryProtocol",
    # Phase 2: Orchestrator
    "DetectionCandidate",
    "DetectionOrchestrator",
    "DetectionOrchestratorProtocol",
    "OrchestratedDetection",
    # Phase 2 Add-on 3: Audit
    "DetectionAuditEvent",
    "DetectionAuditLogger",
    "build_audit_event",
]
