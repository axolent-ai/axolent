"""Architecture guards for LCP Cleanup Bundle (Codex Findings 4-8).

Tests enforce structural invariants introduced by the cleanup:

1. LanguageContext.with_request_id() preserves all Phase 2 fields
   (Codex Finding 4).
2. enforcement.py does NOT import write_audit_log from infrastructure
   directly (Codex Finding 5, hexagonal rule).
3. Mutable dicts in frozen dataclasses are wrapped in MappingProxyType
   (Claude Issue 2).
"""

from __future__ import annotations

import ast
import re
import types
from pathlib import Path

import pytest

from application.language.context import LanguageContext
from application.language.orchestrator import (
    DetectionCandidate,
    OrchestratedDetection,
)


# bridge/ root
_BRIDGE_ROOT = Path(__file__).resolve().parents[2]


def _read_source(relative_path: str) -> str:
    """Read a source file relative to bridge root."""
    full = _BRIDGE_ROOT / relative_path
    return full.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Finding 4: LanguageContext.with_request_id preserves Phase 2 fields
# ---------------------------------------------------------------------------


class TestLanguageContextWithRequestId:
    """Codex Finding 4: with_request_id() must preserve ALL fields."""

    def _full_context(self) -> LanguageContext:
        """Build a LanguageContext with all Phase 2 fields populated."""
        return LanguageContext(
            code="de",
            source="detected",
            confidence=0.93,
            switched_from="en",
            request_id="original-id",
            detection_distribution={"de": 0.93, "nl": 0.05, "en": 0.02},
            reliability_score=0.88,
            confidence_history=(("langdetect", 0.93), ("domain_heuristic", 0.7)),
            detection_tier="high",
            text_length_bucket="medium",
            backends_consulted=frozenset({"langdetect", "domain_heuristic"}),
        )

    def test_request_id_is_replaced(self) -> None:
        """The new request_id is set correctly."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.request_id == "new-id"

    def test_code_preserved(self) -> None:
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.code == ctx.code

    def test_source_preserved(self) -> None:
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.source == ctx.source

    def test_confidence_preserved(self) -> None:
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.confidence == ctx.confidence

    def test_switched_from_preserved(self) -> None:
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.switched_from == ctx.switched_from

    def test_detection_distribution_preserved(self) -> None:
        """Phase 2 field that was LOST before this fix."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert dict(new_ctx.detection_distribution) == dict(ctx.detection_distribution)

    def test_reliability_score_preserved(self) -> None:
        """Phase 2 field that was LOST before this fix."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.reliability_score == ctx.reliability_score

    def test_confidence_history_preserved(self) -> None:
        """Phase 2 field that was LOST before this fix."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.confidence_history == ctx.confidence_history

    def test_detection_tier_preserved(self) -> None:
        """Phase 2 field that was LOST before this fix."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.detection_tier == ctx.detection_tier

    def test_text_length_bucket_preserved(self) -> None:
        """Phase 2 field that was LOST before this fix."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.text_length_bucket == ctx.text_length_bucket

    def test_backends_consulted_preserved(self) -> None:
        """Phase 2 field that was LOST before this fix."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        assert new_ctx.backends_consulted == ctx.backends_consulted

    def test_result_is_frozen(self) -> None:
        """The returned context must be frozen (immutable)."""
        ctx = self._full_context()
        new_ctx = ctx.with_request_id("new-id")
        with pytest.raises(AttributeError):
            new_ctx.code = "en"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Finding 4 cont'd: execution/resolvers.py uses with_request_id
# ---------------------------------------------------------------------------


class TestExecutionResolverPreservesPhase2Fields:
    """Codex Finding 4: resolvers.py must NOT rebuild LanguageContext manually."""

    def test_resolvers_uses_with_request_id(self) -> None:
        """execution/resolvers.py must use with_request_id, not LanguageContext(...)."""
        source = _read_source("application/execution/resolvers.py")
        # Must contain with_request_id call
        assert "with_request_id(" in source, (
            "execution/resolvers.py does not use with_request_id(). "
            "Phase 2 fields will be lost when re-wrapping LanguageContext."
        )

    def test_resolvers_no_manual_langctx_rebuild(self) -> None:
        """execution/resolvers.py must NOT manually build LanguageContext(code=...)."""
        source = _read_source("application/execution/resolvers.py")
        # The old pattern was: LanguageContext(code=lang_ctx.code, ...)
        # with only Phase 1 fields. This should not exist anymore.
        # We check that there is no LanguageContext( with code= as first kwarg
        # after the import section (which is fine).
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = node.func
                name = ""
                if isinstance(callee, ast.Name):
                    name = callee.id
                elif isinstance(callee, ast.Attribute):
                    name = callee.attr
                if name == "LanguageContext" and node.keywords:
                    # Should not be called with keyword args in resolvers.py
                    # (only import + with_request_id usage is allowed)
                    kw_names = [kw.arg for kw in node.keywords if kw.arg]
                    if "code" in kw_names:
                        pytest.fail(
                            "resolvers.py manually constructs LanguageContext "
                            "with code=... kwarg. Use with_request_id() instead "
                            "to preserve Phase 2 fields."
                        )


# ---------------------------------------------------------------------------
# Finding 5: No direct infrastructure.audit_log import in application/
# ---------------------------------------------------------------------------


class TestNoDirectInfrastructureAuditLogImport:
    """Codex Finding 5: write_audit_log must not be imported in application/."""

    # Files that are allowed to import write_audit_log directly.
    # Finding 5 scope: enforcement.py must NOT import it.
    # Other application files that import it are pre-existing and
    # out of scope for this cleanup (tracked for future hexagonal hardening).
    _ALLOWED_BASENAMES = frozenset(
        {
            "main.py",
            "chat_service.py",
            "audit_service.py",
            "debate_orchestrator.py",
        }
    )

    def test_enforcement_no_direct_infrastructure_audit_log_import(self) -> None:
        """enforcement.py must NOT import write_audit_log from infrastructure."""
        source = _read_source("application/language/enforcement.py")
        pattern = re.compile(
            r"^from infrastructure\.audit_log import|"
            r"^import infrastructure\.audit_log",
            re.MULTILINE,
        )
        assert not pattern.search(source), (
            "enforcement.py still imports infrastructure.audit_log directly. "
            "Per Codex Finding 5, it must use the AuditLogPort protocol."
        )

    def test_no_new_direct_infrastructure_audit_log_in_lcp(self) -> None:
        """No LCP module should import write_audit_log directly."""
        pattern = re.compile(
            r"^from infrastructure\.audit_log import|"
            r"^import infrastructure\.audit_log",
            re.MULTILINE,
        )
        forbidden_hits: list[str] = []

        lcp_dir = _BRIDGE_ROOT / "application" / "language"
        for py_file in lcp_dir.rglob("*.py"):
            rel = str(py_file.relative_to(_BRIDGE_ROOT)).replace("\\", "/")
            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            if pattern.search(content):
                forbidden_hits.append(rel)

        assert not forbidden_hits, (
            f"infrastructure.audit_log imported directly in LCP modules: "
            f"{forbidden_hits}. LCP modules must use AuditLogPort."
        )


# ---------------------------------------------------------------------------
# Claude Issue 2: Mutable dicts in frozen dataclasses
# ---------------------------------------------------------------------------


class TestImmutableDistributions:
    """Claude Issue 2: detection_distribution must be read-only."""

    def test_language_context_distribution_is_mapping_proxy(self) -> None:
        """LanguageContext.detection_distribution must be MappingProxyType."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9, "en": 0.1},
        )
        assert isinstance(ctx.detection_distribution, types.MappingProxyType)

    def test_language_context_distribution_rejects_mutation(self) -> None:
        """Mutating detection_distribution must raise TypeError."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9},
        )
        with pytest.raises(TypeError):
            ctx.detection_distribution["en"] = 0.1  # type: ignore[index]

    def test_language_context_distribution_read_access_works(self) -> None:
        """Read operations (.get, .items, []) must still work."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
            detection_distribution={"de": 0.9, "en": 0.1},
        )
        assert ctx.detection_distribution["de"] == 0.9
        assert ctx.detection_distribution.get("en") == 0.1
        assert len(list(ctx.detection_distribution.items())) == 2

    def test_detection_candidate_distribution_is_mapping_proxy(self) -> None:
        """DetectionCandidate.distribution must be MappingProxyType."""
        candidate = DetectionCandidate(
            backend_name="test",
            distribution={"de": 0.9},
            top_lang="de",
            top_confidence=0.9,
            latency_ms=1.0,
        )
        assert isinstance(candidate.distribution, types.MappingProxyType)

    def test_detection_candidate_distribution_rejects_mutation(self) -> None:
        """Mutating DetectionCandidate.distribution must raise TypeError."""
        candidate = DetectionCandidate(
            backend_name="test",
            distribution={"de": 0.9},
            top_lang="de",
            top_confidence=0.9,
            latency_ms=1.0,
        )
        with pytest.raises(TypeError):
            candidate.distribution["en"] = 0.1  # type: ignore[index]

    def test_orchestrated_detection_distribution_is_mapping_proxy(self) -> None:
        """OrchestratedDetection.distribution must be MappingProxyType."""
        result = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={"de": 0.9},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
        )
        assert isinstance(result.distribution, types.MappingProxyType)

    def test_orchestrated_detection_distribution_rejects_mutation(self) -> None:
        """Mutating OrchestratedDetection.distribution must raise TypeError."""
        result = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={"de": 0.9},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
        )
        with pytest.raises(TypeError):
            result.distribution["en"] = 0.1  # type: ignore[index]

    def test_default_distribution_is_empty_mapping_proxy(self) -> None:
        """Default detection_distribution (no arg) is empty MappingProxyType."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
        )
        assert isinstance(ctx.detection_distribution, types.MappingProxyType)
        assert len(ctx.detection_distribution) == 0


# ---------------------------------------------------------------------------
# Finding 6: Orchestrator distribution consistency in dissent
# ---------------------------------------------------------------------------


class TestOrchestratorDissentDistributionConsistency:
    """Codex Finding 6: distribution must match winner in dissent cases."""

    def test_had_dissent_field_exists(self) -> None:
        """OrchestratedDetection must have had_dissent field."""
        result = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={"de": 0.9},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
            had_dissent=True,
        )
        assert result.had_dissent is True

    def test_had_dissent_default_is_false(self) -> None:
        """had_dissent defaults to False."""
        result = OrchestratedDetection(
            code="de",
            confidence=0.9,
            distribution={"de": 0.9},
            reliability_score=0.85,
            candidates=(),
            decision_reason="test",
            text_length_bucket="medium",
        )
        assert result.had_dissent is False


# ---------------------------------------------------------------------------
# Finding 5: LanguageEnforcement uses AuditLogPort
# ---------------------------------------------------------------------------


class TestEnforcementUsesAuditPort:
    """Codex Finding 5: LanguageEnforcement must accept audit_log parameter."""

    def test_enforcement_constructor_has_audit_log_param(self) -> None:
        """LanguageEnforcement.__init__ must accept audit_log keyword."""
        import inspect

        from application.language.enforcement import LanguageEnforcement

        sig = inspect.signature(LanguageEnforcement.__init__)
        assert "audit_log" in sig.parameters, (
            "LanguageEnforcement.__init__ does not have audit_log parameter. "
            "Codex Finding 5 requires AuditLogPort injection."
        )

    def test_enforcement_source_has_audit_log_port_protocol(self) -> None:
        """enforcement.py must define AuditLogPort protocol."""
        source = _read_source("application/language/enforcement.py")
        assert "class AuditLogPort" in source, (
            "enforcement.py does not define AuditLogPort protocol."
        )

    def test_enforcement_source_uses_write_audit_via_port(self) -> None:
        """enforcement.py must call self._audit_log, not write_audit_log."""
        source = _read_source("application/language/enforcement.py")
        assert "self._audit_log" in source or "self._write_audit" in source, (
            "enforcement.py does not use injected audit_log port."
        )
