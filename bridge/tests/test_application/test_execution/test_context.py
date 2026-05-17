"""Tests for ExecutionContext, PartialExecutionContext, TimeContext, ChannelCapabilities."""

from __future__ import annotations

import pytest

from application.execution.context import (
    ChannelCapabilities,
    ExecutionContext,
    PartialExecutionContext,
    TimeContext,
)
from application.execution.envelope import RequestEnvelope
from application.language_resolver import LanguageContext


class TestTimeContext:
    """Test TimeContext defaults and immutability."""

    def test_default_values(self) -> None:
        """TimeContext has sensible defaults."""
        tc = TimeContext()
        assert tc.weekday >= 0
        assert tc.weekday <= 6
        assert tc.timezone_name == "UTC"
        assert tc.time_of_day in ("morning", "afternoon", "evening", "night")

    def test_frozen(self) -> None:
        """TimeContext is immutable."""
        tc = TimeContext()
        with pytest.raises(Exception):
            tc.weekday = 5  # type: ignore[misc]


class TestChannelCapabilities:
    """Test ChannelCapabilities defaults."""

    def test_telegram_defaults(self) -> None:
        """Default capabilities match Telegram."""
        caps = ChannelCapabilities()
        assert caps.streaming_supported is True
        assert caps.max_message_length == 4096
        assert caps.markdown_supported is True

    def test_frozen(self) -> None:
        """ChannelCapabilities is immutable."""
        caps = ChannelCapabilities()
        with pytest.raises(Exception):
            caps.max_message_length = 1000  # type: ignore[misc]


class TestExecutionContext:
    """Test ExecutionContext construction and immutability."""

    def test_frozen(self) -> None:
        """ExecutionContext is immutable."""
        ctx = ExecutionContext(request_id="abc123")
        with pytest.raises(Exception):
            ctx.request_id = "changed"  # type: ignore[misc]

    def test_default_language_is_de(self) -> None:
        """Default language context uses 'de'."""
        ctx = ExecutionContext()
        assert ctx.language.code == "de"

    def test_audit_tags_default_empty(self) -> None:
        """audit_tags default to empty tuple."""
        ctx = ExecutionContext()
        assert ctx.audit_tags == ()


class TestPartialExecutionContext:
    """Test PartialExecutionContext builder."""

    def test_from_envelope(self) -> None:
        """from_envelope transfers all fields correctly."""
        env = RequestEnvelope.from_telegram(
            user_id=42,
            chat_id=99,
            text="Hallo Welt",
            username="tester",
        )
        partial = PartialExecutionContext.from_envelope(env)
        assert partial.request_id == env.request_id
        assert partial.user_id == 42
        assert partial.chat_id == 99
        assert partial.raw_text == "Hallo Welt"
        assert partial.channel == "telegram"

    def test_from_envelope_with_override(self) -> None:
        """Language override is stored in partial."""
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="hi")
        partial = PartialExecutionContext.from_envelope(env, language_override="fr")
        assert partial.language_override == "fr"

    def test_freeze_with_defaults(self) -> None:
        """freeze() fills missing contexts with defaults."""
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        partial = PartialExecutionContext.from_envelope(env)
        ctx = partial.freeze()

        assert ctx.request_id == env.request_id
        assert ctx.language.code == "de"  # default
        assert ctx.time.timezone_name == "UTC"
        assert ctx.channel_capabilities.streaming_supported is True

    def test_freeze_preserves_resolved_language(self) -> None:
        """freeze() uses the resolved language when set."""
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="test")
        partial = PartialExecutionContext.from_envelope(env)
        partial.language = LanguageContext(
            code="fr",
            source="detected",
            confidence=0.95,
            switched_from=None,
            request_id=env.request_id,
        )
        ctx = partial.freeze()
        assert ctx.language.code == "fr"
        assert ctx.language.source == "detected"

    def test_freeze_produces_immutable_context(self) -> None:
        """Frozen context is truly immutable."""
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="x")
        partial = PartialExecutionContext.from_envelope(env)
        ctx = partial.freeze()
        with pytest.raises(Exception):
            ctx.user_id = 999  # type: ignore[misc]


class TestAuditTagsImmutability:
    """EK-06: audit_tags on frozen ExecutionContext must be immutable."""

    def test_audit_tags_is_tuple(self) -> None:
        """audit_tags is a tuple-of-tuples, not a mutable dict."""
        ctx = ExecutionContext(
            request_id="imm-1",
            audit_tags=(("key1", "val1"), ("key2", 42)),
        )
        assert isinstance(ctx.audit_tags, tuple)
        assert ctx.audit_tags == (("key1", "val1"), ("key2", 42))

    def test_audit_tags_immutable_no_append(self) -> None:
        """Cannot mutate audit_tags on a frozen ExecutionContext."""
        ctx = ExecutionContext(request_id="imm-2", audit_tags=(("a", 1),))
        with pytest.raises(Exception):
            ctx.audit_tags = (("b", 2),)  # type: ignore[misc]

    def test_audit_tags_mutation_via_subscript_fails(self) -> None:
        """Tuple does not support item assignment."""
        ctx = ExecutionContext(request_id="imm-3", audit_tags=(("x", "y"),))
        with pytest.raises(TypeError):
            ctx.audit_tags[0] = ("z", "w")  # type: ignore[index]

    def test_get_audit_tag_helper(self) -> None:
        """get_audit_tag looks up keys correctly."""
        ctx = ExecutionContext(
            request_id="imm-4",
            audit_tags=(("lang", "de"), ("mode", "debate")),
        )
        assert ctx.get_audit_tag("lang") == "de"
        assert ctx.get_audit_tag("mode") == "debate"
        assert ctx.get_audit_tag("missing") is None
        assert ctx.get_audit_tag("missing", "fallback") == "fallback"

    def test_as_audit_dict_helper(self) -> None:
        """as_audit_dict converts to plain dict for serialization."""
        ctx = ExecutionContext(
            request_id="imm-5",
            audit_tags=(("k1", "v1"), ("k2", 2)),
        )
        d = ctx.as_audit_dict()
        assert d == {"k1": "v1", "k2": 2}
        # Mutating returned dict does not affect original
        d["k3"] = "injected"
        assert ctx.get_audit_tag("k3") is None

    def test_freeze_converts_dict_to_tuple(self) -> None:
        """PartialExecutionContext.freeze() converts audit_tags dict to tuple."""
        env = RequestEnvelope.from_telegram(user_id=1, chat_id=2, text="t")
        partial = PartialExecutionContext.from_envelope(env)
        partial.audit_tags["source"] = "test"
        partial.audit_tags["count"] = 5
        ctx = partial.freeze()
        assert ("source", "test") in ctx.audit_tags
        assert ("count", 5) in ctx.audit_tags
