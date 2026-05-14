"""Tests for ModelRegistry: YAML loading, alias lookup, best-for-dimension.

Tests:
  - YAML loading with correct data
  - Alias resolution (case-insensitive, whitespace-tolerant)
  - Provider filtering
  - best_for_dimension (with and without provider filter)
  - Edge cases: YAML missing, YAML corrupt, duplicate alias
  - get_display_name
  - all_ids / all_aliases
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from application.model_registry import ModelRegistry


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


MINIMAL_YAML = dedent("""\
    models:
      - id: test-model-a
        display_name: Test A
        provider: test_provider
        aliases: [testa, a]
        context_window: 100000
        pricing_input_per_mtok: 1.0
        pricing_output_per_mtok: 5.0
        scores:
          coding: 80.0
          reasoning: 70.0
          knowledge: 60.0
          speed: 90
        supports_thinking: true
        supports_effort: false
        is_open_source: false

      - id: test-model-b
        display_name: Test B
        provider: other_provider
        aliases: [testb, b]
        context_window: 50000
        pricing_input_per_mtok: 0.5
        pricing_output_per_mtok: 2.5
        scores:
          coding: 60.0
          reasoning: 90.0
          knowledge: 85.0
          speed: 50
        supports_thinking: false
        supports_effort: true
        is_open_source: true

      - id: test-model-c
        display_name: Test C
        provider: test_provider
        aliases: [testc]
        context_window: 200000
        pricing_input_per_mtok: 3.0
        pricing_output_per_mtok: 15.0
        scores:
          coding: 95.0
          reasoning: 85.0
          knowledge: 75.0
          speed: 40
        supports_thinking: true
        supports_effort: true
        is_open_source: false
""")


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    """Write minimal YAML to a temp file and return path."""
    path = tmp_path / "models.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    return path


@pytest.fixture
def registry(yaml_file: Path) -> ModelRegistry:
    """ModelRegistry loaded from minimal test YAML."""
    return ModelRegistry(yaml_path=yaml_file)


# ──────────────────────────────────────────────────────────────
# Loading Tests
# ──────────────────────────────────────────────────────────────


class TestLoading:
    """Tests für YAML-Loading und Parsing."""

    def test_loads_all_models(self, registry: ModelRegistry) -> None:
        """Alle Modelle aus YAML werden geladen."""
        assert len(registry.all()) == 3  # nosemgrep: len-all-count

    def test_model_metadata_fields(self, registry: ModelRegistry) -> None:
        """Alle Felder werden korrekt geparst."""
        meta = registry.get("test-model-a")
        assert meta is not None
        assert meta.id == "test-model-a"
        assert meta.display_name == "Test A"
        assert meta.provider == "test_provider"
        assert meta.aliases == ("testa", "a")
        assert meta.context_window == 100000
        assert meta.pricing_input_per_mtok == 1.0
        assert meta.pricing_output_per_mtok == 5.0
        assert meta.scores["coding"] == 80.0
        assert meta.supports_thinking is True
        assert meta.supports_effort is False
        assert meta.is_open_source is False

    def test_model_is_frozen(self, registry: ModelRegistry) -> None:
        """ModelMetadata ist immutable (frozen dataclass)."""
        meta = registry.get("test-model-a")
        assert meta is not None
        with pytest.raises(AttributeError):
            meta.id = "changed"  # type: ignore[misc]

    def test_missing_yaml_raises(self, tmp_path: Path) -> None:
        """Fehlende YAML-Datei wirft FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ModelRegistry(yaml_path=tmp_path / "nonexistent.yaml")

    def test_corrupt_yaml_raises(self, tmp_path: Path) -> None:
        """Korruptes YAML wirft yaml.YAMLError."""
        path = tmp_path / "bad.yaml"
        path.write_text(": : : [invalid yaml{{{", encoding="utf-8")
        with pytest.raises(Exception):  # yaml.YAMLError
            ModelRegistry(yaml_path=path)

    def test_missing_models_key_raises(self, tmp_path: Path) -> None:
        """YAML ohne 'models'-Key wirft ValueError."""
        path = tmp_path / "no_models.yaml"
        path.write_text("something_else: true\n", encoding="utf-8")
        with pytest.raises(ValueError, match="models"):
            ModelRegistry(yaml_path=path)

    def test_models_not_list_raises(self, tmp_path: Path) -> None:
        """'models' als String statt Liste wirft ValueError."""
        path = tmp_path / "bad_type.yaml"
        path.write_text("models: not_a_list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="list"):
            ModelRegistry(yaml_path=path)

    def test_invalid_entry_skipped(self, tmp_path: Path) -> None:
        """Einträge mit fehlenden Pflichtfeldern werden übersprungen."""
        yaml_content = dedent("""\
            models:
              - id: good-model
                display_name: Good
                provider: prov
                aliases: [good]
                context_window: 100000
                pricing_input_per_mtok: 1.0
                pricing_output_per_mtok: 5.0
                scores: {}
                supports_thinking: false
                supports_effort: false
                is_open_source: false
              - display_name: Missing ID
                provider: prov
                aliases: []
                context_window: 100000
                pricing_input_per_mtok: 1.0
                pricing_output_per_mtok: 5.0
        """)
        path = tmp_path / "partial.yaml"
        path.write_text(yaml_content, encoding="utf-8")
        reg = ModelRegistry(yaml_path=path)
        assert len(reg.all()) == 1  # nosemgrep: len-all-count
        assert reg.get("good-model") is not None


# ──────────────────────────────────────────────────────────────
# Alias Lookup Tests
# ──────────────────────────────────────────────────────────────


class TestAliasLookup:
    """Tests für Alias-Resolution."""

    def test_lookup_by_alias(self, registry: ModelRegistry) -> None:
        """Alias wird korrekt aufgelöst."""
        meta = registry.get("testa")
        assert meta is not None
        assert meta.id == "test-model-a"

    def test_lookup_by_short_alias(self, registry: ModelRegistry) -> None:
        """Kurzer Alias funktioniert."""
        meta = registry.get("a")
        assert meta is not None
        assert meta.id == "test-model-a"

    def test_lookup_by_full_id(self, registry: ModelRegistry) -> None:
        """Lookup per vollständiger ID funktioniert."""
        meta = registry.get("test-model-b")
        assert meta is not None
        assert meta.id == "test-model-b"

    def test_case_insensitive(self, registry: ModelRegistry) -> None:
        """Alias-Lookup ist case-insensitive."""
        assert registry.get("TestA") is not None
        assert registry.get("TESTA") is not None
        assert registry.get("TestA") == registry.get("testa")

    def test_whitespace_stripped(self, registry: ModelRegistry) -> None:
        """Whitespace wird beim Lookup entfernt."""
        assert registry.get("  testa  ") is not None
        assert registry.get("  testa  ") == registry.get("testa")

    def test_unknown_returns_none(self, registry: ModelRegistry) -> None:
        """Unbekannter Alias gibt None zurück."""
        assert registry.get("nonexistent") is None
        assert registry.get("") is None

    def test_resolve_id(self, registry: ModelRegistry) -> None:
        """resolve_id gibt die kanonische ID zurück."""
        assert registry.resolve_id("testa") == "test-model-a"
        assert registry.resolve_id("test-model-b") == "test-model-b"
        assert registry.resolve_id("unknown") is None

    def test_duplicate_alias_keeps_first(self, tmp_path: Path) -> None:
        """Bei doppeltem Alias gewinnt das erste Modell."""
        yaml_content = dedent("""\
            models:
              - id: model-first
                display_name: First
                provider: prov
                aliases: [shared_alias]
                context_window: 100000
                pricing_input_per_mtok: 1.0
                pricing_output_per_mtok: 5.0
                scores: {}
                supports_thinking: false
                supports_effort: false
                is_open_source: false
              - id: model-second
                display_name: Second
                provider: prov
                aliases: [shared_alias]
                context_window: 100000
                pricing_input_per_mtok: 1.0
                pricing_output_per_mtok: 5.0
                scores: {}
                supports_thinking: false
                supports_effort: false
                is_open_source: false
        """)
        path = tmp_path / "dup_alias.yaml"
        path.write_text(yaml_content, encoding="utf-8")
        reg = ModelRegistry(yaml_path=path)
        meta = reg.get("shared_alias")
        assert meta is not None
        assert meta.id == "model-first"


# ──────────────────────────────────────────────────────────────
# Provider Filter Tests
# ──────────────────────────────────────────────────────────────


class TestProviderFilter:
    """Tests für Provider-Filterung."""

    def test_for_provider(self, registry: ModelRegistry) -> None:
        """for_provider gibt nur Modelle des Providers zurück."""
        result = registry.for_provider("test_provider")
        assert len(result) == 2
        ids = {m.id for m in result}
        assert ids == {"test-model-a", "test-model-c"}

    def test_for_provider_case_insensitive(self, registry: ModelRegistry) -> None:
        """Provider-Filter ist case-insensitive."""
        result = registry.for_provider("OTHER_PROVIDER")
        assert len(result) == 1
        assert result[0].id == "test-model-b"

    def test_for_unknown_provider(self, registry: ModelRegistry) -> None:
        """Unbekannter Provider gibt leere Liste zurück."""
        assert registry.for_provider("unknown") == []


# ──────────────────────────────────────────────────────────────
# Best-for-Dimension Tests
# ──────────────────────────────────────────────────────────────


class TestBestForDimension:
    """Tests für best_for_dimension."""

    def test_best_coding(self, registry: ModelRegistry) -> None:
        """Bestes Coding-Modell wird korrekt ermittelt."""
        best = registry.best_for_dimension("coding")
        assert best is not None
        assert best.id == "test-model-c"  # 95.0

    def test_best_reasoning(self, registry: ModelRegistry) -> None:
        """Bestes Reasoning-Modell wird korrekt ermittelt."""
        best = registry.best_for_dimension("reasoning")
        assert best is not None
        assert best.id == "test-model-b"  # 90.0

    def test_best_speed(self, registry: ModelRegistry) -> None:
        """Schnellstes Modell wird korrekt ermittelt."""
        best = registry.best_for_dimension("speed")
        assert best is not None
        assert best.id == "test-model-a"  # 90

    def test_best_with_provider_filter(self, registry: ModelRegistry) -> None:
        """Provider-Filter schraenkt Auswahl ein."""
        best = registry.best_for_dimension("coding", providers=["test_provider"])
        assert best is not None
        assert best.id == "test-model-c"  # 95.0 (test_provider)

        best2 = registry.best_for_dimension("coding", providers=["other_provider"])
        assert best2 is not None
        assert best2.id == "test-model-b"  # 60.0 (only one)

    def test_unknown_dimension_returns_none(self, registry: ModelRegistry) -> None:
        """Unbekannte Dimension gibt None zurück."""
        assert registry.best_for_dimension("nonexistent") is None


# ──────────────────────────────────────────────────────────────
# Utility Method Tests
# ──────────────────────────────────────────────────────────────


class TestUtilities:
    """Tests für Hilfsmethoden."""

    def test_all_ids(self, registry: ModelRegistry) -> None:
        """all_ids gibt alle Modell-IDs zurück."""
        ids = registry.all_ids()
        assert ids == {"test-model-a", "test-model-b", "test-model-c"}

    def test_all_aliases(self, registry: ModelRegistry) -> None:
        """all_aliases gibt Alias->ID Mapping zurück."""
        aliases = registry.all_aliases()
        assert aliases["testa"] == "test-model-a"
        assert aliases["a"] == "test-model-a"
        assert aliases["testb"] == "test-model-b"
        assert aliases["b"] == "test-model-b"
        assert aliases["testc"] == "test-model-c"

    def test_get_display_name_known(self, registry: ModelRegistry) -> None:
        """get_display_name gibt korrekten Namen zurück."""
        assert registry.get_display_name("test-model-a") == "Test A"
        assert registry.get_display_name("test-model-b") == "Test B"

    def test_get_display_name_unknown(self, registry: ModelRegistry) -> None:
        """Unbekannte ID wird als Display-Name zurückgegeben."""
        assert registry.get_display_name("unknown-model") == "unknown-model"

    def test_get_score(self, registry: ModelRegistry) -> None:
        """ModelMetadata.get_score funktioniert korrekt."""
        meta = registry.get("test-model-a")
        assert meta is not None
        assert meta.get_score("coding") == 80.0
        assert meta.get_score("nonexistent") is None


# ──────────────────────────────────────────────────────────────
# Production YAML Validation
# ──────────────────────────────────────────────────────────────


class TestProductionYaml:
    """Validiert die echte models.yaml Datei."""

    @pytest.fixture
    def prod_registry(self) -> ModelRegistry:
        """Laedt die Produktions-YAML."""
        prod_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
        return ModelRegistry(yaml_path=prod_path)

    def test_minimum_model_count(self, prod_registry: ModelRegistry) -> None:
        """Produktions-YAML enthaelt mindestens 10 Modelle."""
        assert len(prod_registry.all()) >= 10  # nosemgrep: len-all-count

    def test_anthropic_models_present(self, prod_registry: ModelRegistry) -> None:
        """Alle drei Anthropic-Modelle sind vorhanden."""
        assert prod_registry.get("opus") is not None
        assert prod_registry.get("sonnet") is not None
        assert prod_registry.get("haiku") is not None

    def test_backward_compat_aliases(self, prod_registry: ModelRegistry) -> None:
        """Bestehende Aliase (opus/sonnet/haiku) loesen korrekt auf."""
        assert prod_registry.resolve_id("opus") == "claude-opus-4-7"
        assert prod_registry.resolve_id("sonnet") == "claude-sonnet-4-6"
        assert prod_registry.resolve_id("haiku") == "claude-haiku-4-5-20251001"

    def test_all_models_have_required_scores(
        self, prod_registry: ModelRegistry
    ) -> None:
        """Alle Modelle haben die vier Score-Dimensionen."""
        required = {"coding", "reasoning", "knowledge", "speed"}
        for model in prod_registry.all():
            missing = required - set(model.scores.keys())
            assert not missing, f"Model '{model.id}' missing scores: {missing}"

    def test_no_negative_pricing(self, prod_registry: ModelRegistry) -> None:
        """Keine negativen Preise."""
        for model in prod_registry.all():
            assert model.pricing_input_per_mtok >= 0, (
                f"{model.id} has negative input pricing"
            )
            assert model.pricing_output_per_mtok >= 0, (
                f"{model.id} has negative output pricing"
            )
