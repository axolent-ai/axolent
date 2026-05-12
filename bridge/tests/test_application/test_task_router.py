"""Tests fuer TaskRouter: Heuristik-Klassifikation, Modell-Resolution, YAML-Loading.

25+ Tests die alle Slot-Klassifikationen, Edge-Cases, Prioritaeten und Fallback abdecken.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from application.task_router import (
    SlotConfig,
    TaskRouter,
    load_slot_configs,
)
from domain.task_slot import TaskSlot


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def full_configs() -> list[SlotConfig]:
    """Vollstaendige SlotConfig-Liste analog zur Produktions-YAML."""
    return [
        SlotConfig(
            slot=TaskSlot.CODE,
            default_model="opus",
            patterns=("```", "def ", "function ", "class ", "import "),
            keywords=(
                "debug",
                "refactor",
                "implementier",
                "implement",
                "programmier",
                "bug",
                "fehler im code",
                "error",
                "syntax",
                "traceback",
                "exception",
                "TypeError",
                "ValueError",
            ),
            min_keyword_matches=2,
        ),
        SlotConfig(
            slot=TaskSlot.REASON,
            default_model="opus",
            keywords=(
                "analysier",
                "vergleich",
                "pro und contra",
                "step by step",
                "schritt fuer schritt",
                "berechn",
                "kalkulier",
                "strategie",
                "hypothese",
            ),
            min_keyword_matches=2,
            min_word_count=50,
        ),
        SlotConfig(
            slot=TaskSlot.RESEARCH,
            default_model="opus",
            keywords=(
                "recherchier",
                "research",
                "zusammenfass",
                "markt",
                "wettbewerb",
                "benchmark",
                "trend",
                "branche",
                "deep dive",
            ),
            patterns=("http://", "https://"),
            min_keyword_matches=2,
        ),
        SlotConfig(
            slot=TaskSlot.CREATIVE,
            default_model="sonnet",
            keywords=(
                "brainstorm",
                "ideen",
                "vorschlaege",
                "varianten",
                "alternativen",
                "schreib",
                "formulier",
                "verfass",
                "headline",
                "slogan",
                "ad copy",
                "email",
                "blog",
                "artikel",
            ),
            min_keyword_matches=2,
        ),
        SlotConfig(
            slot=TaskSlot.QUICK,
            default_model="haiku",
            keywords=(
                "klassifizier",
                "kategorisier",
                "extrahier",
                "ist das",
                "ja oder nein",
                "uebersetz",
                "formatier",
                "konvertier",
            ),
            min_keyword_matches=1,
            max_word_count=50,
        ),
        SlotConfig(
            slot=TaskSlot.CHAT,
            default_model="sonnet",
            fallback=True,
        ),
    ]


@pytest.fixture
def router(full_configs: list[SlotConfig]) -> TaskRouter:
    """TaskRouter mit vollstaendiger Konfiguration (ohne ModelService)."""
    return TaskRouter(slot_configs=full_configs)


# ──────────────────────────────────────────────────────────────
# CODE-Slot Tests
# ──────────────────────────────────────────────────────────────


class TestCodeSlotClassification:
    """Tests fuer CODE-Slot-Erkennung."""

    def test_code_block_triple_backtick(self, router: TaskRouter) -> None:
        """Code-Block (```) wird als CODE erkannt (starkes Signal)."""
        result = router.classify("```python\nprint('hello')\n```")
        assert result.slot == TaskSlot.CODE
        assert result.score >= 100  # CODE_BLOCK_SCORE

    def test_code_keywords_debug_and_error(self, router: TaskRouter) -> None:
        """Zwei Code-Keywords ergeben CODE-Slot."""
        result = router.classify("Ich habe einen debug error in meinem Programm")
        assert result.slot == TaskSlot.CODE

    def test_code_pattern_def(self, router: TaskRouter) -> None:
        """Python-Funktionsdefinition wird erkannt."""
        result = router.classify("def calculate_total(items): pass\n\nBitte debug das")
        assert result.slot == TaskSlot.CODE

    def test_code_single_keyword_insufficient(self, router: TaskRouter) -> None:
        """Ein einzelnes Code-Keyword reicht nicht (min_keyword_matches=2)."""
        result = router.classify("Es gibt einen error.")
        assert result.slot == TaskSlot.CHAT  # Fallback

    def test_code_pattern_import(self, router: TaskRouter) -> None:
        """Import-Statement wird als CODE-Indikator erkannt."""
        result = router.classify("import os\nimport sys\n\nWas macht das?")
        assert result.slot == TaskSlot.CODE

    def test_code_mixed_with_keywords(self, router: TaskRouter) -> None:
        """Code-Block + Keywords ergibt hohen Score."""
        text = "```\ndef foo(): pass\n```\nBitte debug und refactor das"
        result = router.classify(text)
        assert result.slot == TaskSlot.CODE
        assert result.score > 100

    def test_traceback_and_exception(self, router: TaskRouter) -> None:
        """Traceback + Exception ergeben CODE."""
        result = router.classify("Ich bekomme einen traceback mit einer exception")
        assert result.slot == TaskSlot.CODE


# ──────────────────────────────────────────────────────────────
# REASON-Slot Tests
# ──────────────────────────────────────────────────────────────


class TestReasonSlotClassification:
    """Tests fuer REASON-Slot-Erkennung."""

    def test_reason_keywords_long_text(self, router: TaskRouter) -> None:
        """Analyse-Keywords in langem Text ergeben REASON."""
        # Erzeuge Text > 50 Woerter mit Reason-Keywords
        filler = "dies ist ein langer Text " * 10
        text = f"Bitte analysier die strategie. {filler}"
        result = router.classify(text)
        assert result.slot == TaskSlot.REASON

    def test_reason_too_short(self, router: TaskRouter) -> None:
        """Reason-Keywords in kurzem Text (<50 Woerter) fallen nicht auf REASON."""
        result = router.classify("Analysier die Strategie kurz.")
        assert result.slot != TaskSlot.REASON

    def test_reason_berechnung(self, router: TaskRouter) -> None:
        """Berechnung + Kalkulation in langem Text ergibt REASON."""
        filler = "additional context for the analysis " * 10
        text = f"Berechn die Marge und kalkulier den ROI. {filler}"
        result = router.classify(text)
        assert result.slot == TaskSlot.REASON


# ──────────────────────────────────────────────────────────────
# RESEARCH-Slot Tests
# ──────────────────────────────────────────────────────────────


class TestResearchSlotClassification:
    """Tests fuer RESEARCH-Slot-Erkennung."""

    def test_research_keywords(self, router: TaskRouter) -> None:
        """Recherche-Keywords ergeben RESEARCH."""
        result = router.classify(
            "Recherchier die aktuellen Markt-trends in der Branche"
        )
        assert result.slot == TaskSlot.RESEARCH

    def test_research_with_url(self, router: TaskRouter) -> None:
        """URL-Pattern + Keyword ergibt RESEARCH."""
        result = router.classify(
            "Zusammenfass den Artikel: https://example.com/article"
        )
        assert result.slot == TaskSlot.RESEARCH

    def test_research_deep_dive(self, router: TaskRouter) -> None:
        """Deep Dive + Markt ergibt RESEARCH."""
        result = router.classify("Mach einen deep dive zum Wettbewerb im Markt")
        assert result.slot == TaskSlot.RESEARCH


# ──────────────────────────────────────────────────────────────
# CREATIVE-Slot Tests
# ──────────────────────────────────────────────────────────────


class TestCreativeSlotClassification:
    """Tests fuer CREATIVE-Slot-Erkennung."""

    def test_creative_brainstorm(self, router: TaskRouter) -> None:
        """Brainstorm + Ideen ergibt CREATIVE."""
        result = router.classify("Brainstorm mal ein paar Ideen fuer ein Logo")
        assert result.slot == TaskSlot.CREATIVE

    def test_creative_ad_copy(self, router: TaskRouter) -> None:
        """Ad Copy + Headline ergibt CREATIVE."""
        result = router.classify("Schreib eine ad copy mit einer guten headline")
        assert result.slot == TaskSlot.CREATIVE

    def test_creative_varianten(self, router: TaskRouter) -> None:
        """Varianten + Vorschlaege ergibt CREATIVE."""
        result = router.classify("Gib mir 5 Varianten und Vorschlaege fuer den Slogan")
        assert result.slot == TaskSlot.CREATIVE


# ──────────────────────────────────────────────────────────────
# QUICK-Slot Tests
# ──────────────────────────────────────────────────────────────


class TestQuickSlotClassification:
    """Tests fuer QUICK-Slot-Erkennung."""

    def test_quick_klassifizier(self, router: TaskRouter) -> None:
        """Klassifizier in kurzem Text ergibt QUICK."""
        result = router.classify("Klassifizier das als Spam")
        assert result.slot == TaskSlot.QUICK

    def test_quick_uebersetze(self, router: TaskRouter) -> None:
        """Uebersetze in kurzem Text ergibt QUICK."""
        result = router.classify("Uebersetz das ins Englische")
        assert result.slot == TaskSlot.QUICK

    def test_quick_too_long(self, router: TaskRouter) -> None:
        """QUICK-Keywords in langem Text (>50 Woerter) ergeben NICHT Quick."""
        long_text = "Bitte extrahier " + "ein sehr langer Text " * 20
        result = router.classify(long_text)
        assert result.slot != TaskSlot.QUICK


# ──────────────────────────────────────────────────────────────
# CHAT-Fallback Tests
# ──────────────────────────────────────────────────────────────


class TestChatFallback:
    """Tests fuer CHAT-Fallback-Verhalten."""

    def test_simple_question(self, router: TaskRouter) -> None:
        """Einfache Frage faellt auf CHAT."""
        result = router.classify("Was ist die Hauptstadt von Frankreich?")
        assert result.slot == TaskSlot.CHAT

    def test_empty_text(self, router: TaskRouter) -> None:
        """Leerer Text faellt auf CHAT."""
        result = router.classify("")
        assert result.slot == TaskSlot.CHAT

    def test_whitespace_only(self, router: TaskRouter) -> None:
        """Nur Whitespace faellt auf CHAT."""
        result = router.classify("   \n  ")
        assert result.slot == TaskSlot.CHAT

    def test_greeting(self, router: TaskRouter) -> None:
        """Begruessung faellt auf CHAT."""
        result = router.classify("Hallo, wie geht es dir?")
        assert result.slot == TaskSlot.CHAT


# ──────────────────────────────────────────────────────────────
# Explizite Marker Tests (Stufe 1)
# ──────────────────────────────────────────────────────────────


class TestExplicitMarkers:
    """Tests fuer explizite Slot-Marker (/code, /reason, etc.)."""

    def test_slash_code(self, router: TaskRouter) -> None:
        """/code Praefix triggert CODE-Slot."""
        result = router.classify("/code Schreib eine Funktion")
        assert result.slot == TaskSlot.CODE
        assert result.score == 1000

    def test_slash_reason(self, router: TaskRouter) -> None:
        """/reason Praefix triggert REASON-Slot."""
        result = router.classify("/reason Analysiere das Problem")
        assert result.slot == TaskSlot.REASON

    def test_slash_creative(self, router: TaskRouter) -> None:
        """/creative Praefix triggert CREATIVE-Slot."""
        result = router.classify("/creative Schreib einen Slogan")
        assert result.slot == TaskSlot.CREATIVE

    def test_slash_quick(self, router: TaskRouter) -> None:
        """/quick Praefix triggert QUICK-Slot."""
        result = router.classify("/quick Ist das richtig?")
        assert result.slot == TaskSlot.QUICK

    def test_slash_research(self, router: TaskRouter) -> None:
        """/research Praefix triggert RESEARCH-Slot."""
        result = router.classify("/research Suche nach Trends")
        assert result.slot == TaskSlot.RESEARCH

    def test_slash_chat(self, router: TaskRouter) -> None:
        """/chat Praefix triggert CHAT-Slot."""
        result = router.classify("/chat Einfach reden")
        assert result.slot == TaskSlot.CHAT

    def test_slash_marker_case_insensitive(self, router: TaskRouter) -> None:
        """Marker sind case-insensitive."""
        result = router.classify("/CODE Schreib was")
        assert result.slot == TaskSlot.CODE

    def test_slash_marker_not_partial(self, router: TaskRouter) -> None:
        """Marker matchen nicht als Teilwort (z.B. /coder)."""
        result = router.classify("/coder ist ein Beruf")
        assert result.slot == TaskSlot.CHAT  # Fallback


# ──────────────────────────────────────────────────────────────
# Prioritaet bei Gleichstand
# ──────────────────────────────────────────────────────────────


class TestPriorityTiebreaking:
    """Tests fuer Prioritaet bei Score-Gleichstand."""

    def test_code_beats_creative(self, router: TaskRouter) -> None:
        """Bei Overlap gewinnt CODE ueber CREATIVE."""
        # "schreib" ist in CREATIVE, "debug" + "error" sind in CODE
        result = router.classify("Schreib mir einen debug fix fuer den error")
        assert result.slot == TaskSlot.CODE


# ──────────────────────────────────────────────────────────────
# ClassificationResult Tests
# ──────────────────────────────────────────────────────────────


class TestClassificationResult:
    """Tests fuer das ClassificationResult-Objekt."""

    def test_result_attributes(self, router: TaskRouter) -> None:
        """ClassificationResult hat alle erwarteten Attribute."""
        result = router.classify("```python\nprint('hello')\n```")
        assert hasattr(result, "slot")
        assert hasattr(result, "score")
        assert hasattr(result, "matched_patterns")
        assert hasattr(result, "matched_keywords")

    def test_matched_patterns_populated(self, router: TaskRouter) -> None:
        """Gematchte Patterns werden im Result aufgelistet."""
        result = router.classify("```python\nprint('hello')\n```")
        assert "```" in result.matched_patterns

    def test_matched_keywords_populated(self, router: TaskRouter) -> None:
        """Gematchte Keywords werden im Result aufgelistet."""
        result = router.classify("Bitte debug und refactor den Code")
        assert "debug" in result.matched_keywords
        assert "refactor" in result.matched_keywords


# ──────────────────────────────────────────────────────────────
# Modell-Resolution Tests
# ──────────────────────────────────────────────────────────────


class TestModelResolution:
    """Tests fuer resolve_model (ohne ModelService)."""

    def test_slot_default_code(self, router: TaskRouter) -> None:
        """CODE-Slot Default ist 'opus'."""
        model = router.resolve_model(user_id=1, slot=TaskSlot.CODE)
        assert model == "opus"

    def test_slot_default_chat(self, router: TaskRouter) -> None:
        """CHAT-Slot Default ist 'sonnet'."""
        model = router.resolve_model(user_id=1, slot=TaskSlot.CHAT)
        assert model == "sonnet"

    def test_slot_default_quick(self, router: TaskRouter) -> None:
        """QUICK-Slot Default ist 'haiku'."""
        model = router.resolve_model(user_id=1, slot=TaskSlot.QUICK)
        assert model == "haiku"

    def test_get_slot_defaults(self, router: TaskRouter) -> None:
        """get_slot_defaults gibt alle 6 Slot-Defaults."""
        defaults = router.get_slot_defaults()
        assert len(defaults) == 6
        assert defaults[TaskSlot.CODE] == "opus"
        assert defaults[TaskSlot.CHAT] == "sonnet"
        assert defaults[TaskSlot.QUICK] == "haiku"


# ──────────────────────────────────────────────────────────────
# YAML-Loading Tests
# ──────────────────────────────────────────────────────────────


class TestYamlLoading:
    """Tests fuer load_slot_configs YAML-Loader."""

    def test_production_yaml_loads(self) -> None:
        """Produktions-YAML laedt erfolgreich."""
        configs = load_slot_configs()
        assert len(configs) == 6
        slots = {c.slot for c in configs}
        assert slots == set(TaskSlot)

    def test_missing_yaml_fallback(self, tmp_path: Path) -> None:
        """Fehlende YAML faellt auf CHAT-only Default zurueck."""
        configs = load_slot_configs(tmp_path / "nonexistent.yaml")
        assert len(configs) == 1
        assert configs[0].slot == TaskSlot.CHAT
        assert configs[0].fallback is True

    def test_corrupt_yaml_fallback(self, tmp_path: Path) -> None:
        """Korrupte YAML faellt auf CHAT-only Default zurueck."""
        path = tmp_path / "bad.yaml"
        path.write_text(": : : [invalid yaml{{{", encoding="utf-8")
        configs = load_slot_configs(path)
        assert len(configs) == 1
        assert configs[0].slot == TaskSlot.CHAT

    def test_custom_yaml(self, tmp_path: Path) -> None:
        """Eigene YAML wird korrekt geladen."""
        yaml_content = dedent("""\
            slots:
              code:
                default_model: opus
                patterns:
                  - "```"
                keywords:
                  - debug
                min_keyword_matches: 1
              chat:
                default_model: sonnet
                fallback: true
        """)
        path = tmp_path / "custom.yaml"
        path.write_text(yaml_content, encoding="utf-8")
        configs = load_slot_configs(path)
        assert len(configs) == 2
        code_cfg = next(c for c in configs if c.slot == TaskSlot.CODE)
        assert code_cfg.default_model == "opus"
        assert "```" in code_cfg.patterns
        assert code_cfg.min_keyword_matches == 1

    def test_yaml_slot_configs_immutable(self) -> None:
        """Geladene SlotConfigs sind frozen (immutable)."""
        configs = load_slot_configs()
        with pytest.raises(AttributeError):
            configs[0].default_model = "changed"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────
# Modell-Resolution mit ModelService Tests
# ──────────────────────────────────────────────────────────────


class TestModelResolutionWithService:
    """Tests fuer resolve_model mit echtem ModelService + SQLite."""

    @pytest.fixture
    def model_service(self, tmp_path: Path):
        """ModelService mit SQLite-Backend."""
        from application.model_service import ModelService
        from infrastructure.sqlite_storage import SqliteConnection, SqliteModelStorage

        conn = SqliteConnection(tmp_path / "test.db")
        storage = SqliteModelStorage(conn)
        svc = ModelService(storage=storage)
        yield svc
        conn.close()

    @pytest.fixture
    def router_with_service(
        self, full_configs: list[SlotConfig], model_service
    ) -> TaskRouter:
        """TaskRouter mit ModelService."""
        return TaskRouter(slot_configs=full_configs, model_service=model_service)

    def test_slot_override_wins(
        self, router_with_service: TaskRouter, model_service
    ) -> None:
        """Slot-spezifischer Override hat Prioritaet ueber Default."""
        model_service.set_user_model(user_id=1, alias_or_id="haiku", slot="code")
        model = router_with_service.resolve_model(user_id=1, slot=TaskSlot.CODE)
        assert model == "claude-haiku-4-5-20251001"

    def test_global_override_fallback(
        self, router_with_service: TaskRouter, model_service
    ) -> None:
        """Globaler Override greift wenn kein Slot-Override gesetzt."""
        model_service.set_user_model(user_id=1, alias_or_id="opus", slot="global")
        model = router_with_service.resolve_model(user_id=1, slot=TaskSlot.CHAT)
        assert model == "claude-opus-4-7"

    def test_slot_override_beats_global(
        self, router_with_service: TaskRouter, model_service
    ) -> None:
        """Slot-Override hat Prioritaet ueber Global-Override."""
        model_service.set_user_model(user_id=1, alias_or_id="sonnet", slot="global")
        model_service.set_user_model(user_id=1, alias_or_id="haiku", slot="code")
        model = router_with_service.resolve_model(user_id=1, slot=TaskSlot.CODE)
        assert model == "claude-haiku-4-5-20251001"

    def test_no_override_uses_default(self, router_with_service: TaskRouter) -> None:
        """Ohne Override wird Slot-Default verwendet."""
        model = router_with_service.resolve_model(user_id=999, slot=TaskSlot.CODE)
        assert model == "opus"
