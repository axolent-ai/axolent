"""Tests for Phase 2 Core Migration (Step 4/4).

Covers:
1. Architecture Guards (post-migration invariants):
   - Guard 1: contract.py no longer has _LANGUAGE_NAMES dict
   - Guard 2: resolver.py no longer imports domain.language
   - Guard 3: no raw backend codes (e.g. "no", "zh-cn") leak through
   - Guard 5: backends.py _normalize() delegates to Registry

2. Migration Tests (functional correctness):
   - Resolver produces LanguageContext with Phase 2 fields populated
   - Contract produces correct language names via Registry
   - Backends normalization works via Registry delegation

3. End-to-End Integration:
   - Dutch text through full pipeline (original Phase-1 RCA fail-case)

Test naming: test_<guard|migration|e2e>_<subject>_<expected>.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from application.language.resolver import LanguageResolver

from application.language.context import LanguageContext
from application.language.contract import LanguageContract
from application.language.registry import InMemoryLanguageRegistry
from infrastructure.conversation_storage import _reset_all_for_tests


def _has_langdetect() -> bool:
    """Check if langdetect is importable."""
    try:
        import langdetect  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


# Path to bridge/ root (three levels up from this test file)
_BRIDGE_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clear_storage() -> None:
    """Reset conversation storage before each test."""
    _reset_all_for_tests()


# ---------------------------------------------------------------------------
# 1. Architecture Guards
# ---------------------------------------------------------------------------


class TestGuard1ContractNoLocalLanguageNames:
    """Guard 1: contract.py must not maintain its own _LANGUAGE_NAMES dict.

    HC-R7: After migration, all language metadata comes from the Registry.
    """

    def test_contract_has_no_language_names_dict(self) -> None:
        """contract.py must not define _LANGUAGE_NAMES as a local dict literal."""
        contract_path = _BRIDGE_ROOT / "application" / "language" / "contract.py"
        source = contract_path.read_text(encoding="utf-8")

        # Match a dict assignment like _LANGUAGE_NAMES: dict[...] = { or _LANGUAGE_NAMES = {
        # but NOT a comment or docstring referencing it.
        pattern = re.compile(
            r"^_LANGUAGE_NAMES\s*(?::\s*dict[^=]*)?\s*=\s*\{",
            re.MULTILINE,
        )
        matches = pattern.findall(source)
        assert not matches, (
            f"HC-R7 violation: contract.py still defines _LANGUAGE_NAMES "
            f"as a local dict: {matches}"
        )

    def test_contract_imports_registry(self) -> None:
        """contract.py must import from the registry module."""
        contract_path = _BRIDGE_ROOT / "application" / "language" / "contract.py"
        source = contract_path.read_text(encoding="utf-8")

        assert "from application.language.registry import" in source, (
            "HC-R7: contract.py must import LanguageRegistry for name lookups"
        )


class TestGuard2ResolverNoDirectDomainLanguageImport:
    """Guard 2: resolver.py must not import from domain.language.

    HC-O7: After Phase 2, resolver uses DetectionOrchestrator exclusively.
    """

    def test_resolver_does_not_import_domain_language(self) -> None:
        """resolver.py must not have any domain.language import statements."""
        resolver_path = _BRIDGE_ROOT / "application" / "language" / "resolver.py"
        source = resolver_path.read_text(encoding="utf-8")

        # Match only real import lines (beginning of line), not docstring prose
        pattern = re.compile(
            r"^(?:from domain\.language|import domain\.language)",
            re.MULTILINE,
        )
        matches = pattern.findall(source)
        assert not matches, (
            f"HC-O7 violation: resolver.py imports domain.language directly: {matches}"
        )

    def test_resolver_imports_orchestrator(self) -> None:
        """resolver.py must import DetectionOrchestrator."""
        resolver_path = _BRIDGE_ROOT / "application" / "language" / "resolver.py"
        source = resolver_path.read_text(encoding="utf-8")

        assert "DetectionOrchestrator" in source, (
            "HC-O7: resolver.py must use DetectionOrchestrator"
        )


class TestGuard3NoRawBackendCodesLeak:
    """Guard 3: no raw backend codes in OrchestratedDetection or LanguageContext.

    HC-O4: All codes normalized via Registry.resolve_backend_code().

    Note: _normalize is a @staticmethod, so we call it on the class
    directly without instantiation (avoids importing langdetect).
    """

    def test_norwegian_code_normalized_through_backends(self) -> None:
        """LangdetectBackend._normalize() maps 'no' -> 'nb' via Registry."""
        from application.language.backends import LangdetectBackend

        assert LangdetectBackend._normalize("no") == "nb"

    def test_chinese_simplified_normalized_through_backends(self) -> None:
        """LangdetectBackend._normalize() maps 'zh-cn' -> 'zh' via Registry."""
        from application.language.backends import LangdetectBackend

        assert LangdetectBackend._normalize("zh-cn") == "zh"

    def test_chinese_traditional_normalized_through_backends(self) -> None:
        """LangdetectBackend._normalize() maps 'zh-tw' -> 'zh' via Registry."""
        from application.language.backends import LangdetectBackend

        assert LangdetectBackend._normalize("zh-tw") == "zh"

    def test_unknown_code_passes_through(self) -> None:
        """Unknown backend codes pass through unchanged."""
        from application.language.backends import LangdetectBackend

        assert LangdetectBackend._normalize("en") == "en"
        assert LangdetectBackend._normalize("de") == "de"


class TestGuard5BackendsNormalizationDelegatesToRegistry:
    """Guard 5: LangdetectBackend._normalize() delegates to Registry.

    HC-R4: No hardcoded mapping dict in backends.py.
    """

    def test_backends_no_hardcoded_mapping_dict(self) -> None:
        """backends.py _normalize() must not contain a hardcoded mapping dict."""
        backends_path = _BRIDGE_ROOT / "application" / "language" / "backends.py"
        source = backends_path.read_text(encoding="utf-8")

        # Find the _normalize method and check for inline dict literals
        # within it. We look for patterns like: mapping = { or {"zh-cn":
        # that indicate a local mapping dict.
        normalize_match = re.search(
            r"def _normalize\(.*?\).*?(?=\n    def |\nclass |\Z)",
            source,
            re.DOTALL,
        )
        assert normalize_match is not None, "Could not find _normalize method"
        normalize_body = normalize_match.group()

        # Should NOT contain a dict literal with backend mappings
        has_local_dict = re.search(r'\{[^}]*"(?:zh-cn|zh-tw|no)"\s*:', normalize_body)
        assert has_local_dict is None, (
            "HC-R4 violation: _normalize() still has hardcoded mapping dict"
        )

        # Should contain a call to resolve_backend_code or _registry
        assert (
            "resolve_backend_code" in normalize_body or "_registry" in normalize_body
        ), "HC-R4: _normalize() must delegate to Registry"

    def test_backends_imports_registry(self) -> None:
        """backends.py must import from the registry module."""
        backends_path = _BRIDGE_ROOT / "application" / "language" / "backends.py"
        source = backends_path.read_text(encoding="utf-8")

        assert "from application.language.registry import" in source, (
            "HC-R4: backends.py must import Registry for code normalization"
        )


# ---------------------------------------------------------------------------
# 2. Migration Tests (functional correctness)
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal stub implementing LanguageDetectorBackend protocol."""

    def __init__(self, distribution: dict[str, float] | None = None) -> None:
        self._distribution = distribution or {}

    def detect_distribution(self, text: str) -> dict[str, float]:
        return dict(self._distribution)


def _make_resolver_with_stubs(
    primary_dist: dict[str, float] | None = None,
    fallback_dist: dict[str, float] | None = None,
    default_lang: str = "de",
) -> "LanguageResolver":
    """Build a LanguageResolver with stub backends (no langdetect needed)."""
    from application.language.orchestrator import DetectionOrchestrator
    from application.language.resolver import LanguageResolver

    orch = DetectionOrchestrator(
        primary_backend=_StubBackend(primary_dist or {"en": 0.92, "de": 0.08}),
        fallback_backend=_StubBackend(fallback_dist or {"en": 0.85}),
        registry=InMemoryLanguageRegistry(),
    )
    return LanguageResolver(default_lang=default_lang, orchestrator=orch)


class TestResolverProducesPhase2Context:
    """Resolver must populate Phase 2 fields when detection occurs."""

    async def test_resolve_detected_has_detection_metadata(self) -> None:
        """Detected language context has Phase 2 fields populated."""
        resolver = _make_resolver_with_stubs(
            primary_dist={"en": 0.92, "de": 0.08},
            fallback_dist={"en": 0.85},
        )
        ctx = await resolver.resolve(
            user_id=900,
            chat_id=900,
            text="What is the weather like today in London?",
        )

        assert ctx.code == "en"
        assert ctx.source == "detected"
        assert ctx.confidence > 0.0
        # Phase 2 fields
        assert ctx.has_detection_metadata is True
        assert len(ctx.detection_distribution) > 0
        assert ctx.reliability_score > 0.0
        assert len(ctx.confidence_history) > 0
        assert ctx.text_length_bucket is not None
        assert len(ctx.backends_consulted) > 0

    async def test_resolve_override_has_empty_phase2_fields(self) -> None:
        """Override contexts do NOT have detection metadata."""
        resolver = _make_resolver_with_stubs()
        ctx = await resolver.resolve(
            user_id=901, chat_id=901, text="anything", override="fr"
        )

        assert ctx.code == "fr"
        assert ctx.source == "override"
        # Phase 2 fields empty for override
        assert ctx.has_detection_metadata is False
        assert ctx.detection_distribution == {}

    async def test_resolve_sticky_has_empty_phase2_fields(self) -> None:
        """Sticky contexts (no switch) do NOT have detection metadata."""
        from infrastructure.conversation_storage import set_language

        await set_language(902, 902, "it")

        # Stub returns low-confidence so it stays sticky
        resolver = _make_resolver_with_stubs(
            primary_dist={"it": 0.30, "en": 0.20},
            fallback_dist={"it": 0.25},
        )
        ctx = await resolver.resolve(user_id=902, chat_id=902, text="ok")

        assert ctx.code == "it"
        assert ctx.source == "sticky"
        assert ctx.has_detection_metadata is False

    async def test_resolve_readonly_has_detection_metadata(self) -> None:
        """Read-only resolution also populates Phase 2 fields."""
        resolver = _make_resolver_with_stubs(
            primary_dist={"en": 0.90, "de": 0.10},
        )
        ctx = await resolver.resolve_readonly(
            user_id=903,
            chat_id=903,
            text="This is an English sentence for testing purposes.",
        )

        assert ctx.code == "en"
        assert ctx.source == "detected"
        assert ctx.has_detection_metadata is True

    def test_from_code_has_empty_phase2_fields(self) -> None:
        """from_code() backward compat: Phase 2 fields empty."""
        from application.language.resolver import LanguageResolver

        ctx = LanguageResolver.from_code("fr")
        assert ctx.code == "fr"
        assert ctx.has_detection_metadata is False
        assert ctx.detection_distribution == {}


class TestContractUsesRegistryNames:
    """Contract must produce correct language names via Registry lookup."""

    def test_contract_german_name(self) -> None:
        """German code resolves to 'German' via Registry."""
        ctx = LanguageContext(
            code="de",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="test",
        )
        contract = LanguageContract.build(ctx, model_id="claude-opus-4-7")
        assert "German" in contract

    def test_contract_dutch_name(self) -> None:
        """Dutch code resolves to 'Dutch' via Registry."""
        ctx = LanguageContext(
            code="nl",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="test",
        )
        contract = LanguageContract.build(ctx, model_id="claude-opus-4-7")
        assert "Dutch" in contract

    def test_contract_all_registry_languages_have_names(self) -> None:
        """Every language in the Registry produces a named contract."""
        registry = InMemoryLanguageRegistry()
        for code in registry.list_codes():
            ctx = LanguageContext(
                code=code,
                source="sticky",
                confidence=1.0,
                switched_from=None,
                request_id="test",
            )
            contract = LanguageContract.build(ctx, model_id="claude-opus-4-7")
            # Should contain the English name, not just the code
            entry = registry.get(code)
            assert entry.name in contract, (
                f"Contract for {code} does not contain name '{entry.name}'"
            )

    def test_contract_unknown_code_falls_back_to_code(self) -> None:
        """Unknown codes use the code itself (graceful degradation)."""
        ctx = LanguageContext(
            code="xx",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="test",
        )
        contract = LanguageContract.build(ctx, model_id="claude-opus-4-7")
        assert "xx" in contract

    def test_repair_contract_uses_registry_names(self) -> None:
        """Repair contract also uses Registry for language names."""
        ctx = LanguageContext(
            code="de",
            source="detected",
            confidence=0.9,
            switched_from=None,
            request_id="test",
        )
        contract = LanguageContract.build_repair_contract(ctx, "en")
        assert "German" in contract
        assert "English" in contract


class TestBackendsNormalizationViaRegistry:
    """Backends normalization produces correct results via Registry."""

    @pytest.mark.skipif(
        not _has_langdetect(),
        reason="langdetect not installed",
    )
    def test_langdetect_backend_distribution_normalized(self) -> None:
        """Real langdetect backend produces normalized codes."""
        from application.language.backends import LangdetectBackend

        backend = LangdetectBackend()
        # Norwegian text: langdetect returns "no" internally
        text = (
            "Dette er en norsk tekst som er lang nok til at den "
            "burde bli gjenkjent som norsk av deteksjonsbiblioteket"
        )
        distribution = backend.detect_distribution(text)
        # If Norwegian detected, it should be "nb" not "no"
        assert "no" not in distribution, (
            f"Raw 'no' code leaked in distribution: {distribution}"
        )


# ---------------------------------------------------------------------------
# 3. End-to-End Integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_langdetect(),
    reason="langdetect not installed; skipping E2E",
)
class TestEndToEndIntegration:
    """Full pipeline E2E: text -> Resolver -> Orchestrator -> Backend."""

    async def test_dutch_text_through_full_pipeline(self) -> None:
        """Dutch text detected correctly (original Phase-1 RCA fail-case).

        This is THE critical integration test: Dutch text must be
        detected as 'nl', not misclassified as 'en' or 'de'.
        """
        from application.language.resolver import LanguageResolver

        resolver = LanguageResolver(default_lang="de")
        ctx = await resolver.resolve(
            user_id=950,
            chat_id=950,
            text=(
                "Dit is een Nederlandse tekst die lang genoeg is zodat "
                "de taaldetectie het betrouwbaar als Nederlands kan herkennen "
                "en een hoge betrouwbaarheidsscore teruggeeft."
            ),
        )

        assert ctx.code == "nl", f"Expected 'nl', got '{ctx.code}'"
        assert ctx.source == "detected"
        assert ctx.confidence > 0.5
        # Phase 2 fields populated
        assert ctx.has_detection_metadata is True
        assert "nl" in ctx.detection_distribution

    async def test_german_text_through_full_pipeline(self) -> None:
        """German text through full Phase 2 pipeline."""
        from application.language.resolver import LanguageResolver

        resolver = LanguageResolver(default_lang="de")
        ctx = await resolver.resolve(
            user_id=951,
            chat_id=951,
            text=(
                "Dies ist ein deutscher Text der lang genug ist damit "
                "die Spracherkennung ihn zuverlaessig als Deutsch erkennen kann."
            ),
        )

        assert ctx.code == "de"
        assert ctx.confidence > 0.5
        assert ctx.has_detection_metadata is True


# ---------------------------------------------------------------------------
# 4. __init__.py exports
# ---------------------------------------------------------------------------


class TestInitExports:
    """Verify that Phase 2 types are exported from application.language."""

    def test_registry_types_exported(self) -> None:
        """Registry types available via application.language."""
        from application.language import (
            DetectionTier,
            InMemoryLanguageRegistry,
            LanguageRegistryEntry,
            LanguageRegistryProtocol,
        )

        assert DetectionTier is not None
        assert InMemoryLanguageRegistry is not None
        assert LanguageRegistryEntry is not None
        assert LanguageRegistryProtocol is not None

    def test_orchestrator_types_exported(self) -> None:
        """Orchestrator types available via application.language."""
        from application.language import (
            DetectionCandidate,
            DetectionOrchestrator,
            DetectionOrchestratorProtocol,
            OrchestratedDetection,
        )

        assert DetectionCandidate is not None
        assert DetectionOrchestrator is not None
        assert DetectionOrchestratorProtocol is not None
        assert OrchestratedDetection is not None
