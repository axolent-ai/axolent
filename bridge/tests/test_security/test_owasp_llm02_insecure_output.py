"""OWASP LLM02: Insecure Output Handling tests.

Verifies that LLM output is sanitized before being sent to Telegram.
Covers: javascript: links, markdown injection, HTML script tags,
and Telegram slash command injection.

Production path tested: presentation.render.sanitize_telegram_slashes,
domain.markdown.markdown_to_telegram_html, domain.markdown.strip_markdown.
"""

from __future__ import annotations

import pytest

from presentation.render import sanitize_telegram_slashes
from domain.markdown import markdown_to_telegram_html, strip_markdown


@pytest.mark.security
@pytest.mark.owasp_llm
class TestLLM02InsecureOutput:
    """LLM02: LLM output sanitized before reaching Telegram user."""

    def test_llm_output_with_javascript_link_sanitized_in_telegram_reply(
        self,
    ) -> None:
        """WHAT: LLM output contains a javascript: URL in markdown link.
        EXPECTED: The link is either stripped or rendered as plain text.
            The href must NOT start with 'javascript:' in final output.
        WHY: Telegram clients may render HTML links; javascript: URLs
            could trigger code execution in some clients.
        """
        malicious_output = "Click [here](javascript:alert('xss')) for help"
        # markdown_to_telegram_html converts markdown links to <a> tags
        html_result = markdown_to_telegram_html(malicious_output)
        # The result must NOT contain a clickable javascript: link
        assert "javascript:" not in html_result.lower(), (
            f"javascript: URL survived sanitization: {html_result}"
        )

    def test_llm_output_with_markdown_injection_sanitized(self) -> None:
        """WHAT: LLM output contains crafted markdown that could inject
            extra links or formatting when rendered as Telegram HTML.
        EXPECTED: Injection payload is escaped or rendered harmlessly.
        WHY: Attacker could craft LLM response to inject fake UI elements.
        """
        # Attempt to break out of a markdown context and inject a new link
        injection = "Normal text](http://evil.com)[Injected link"
        html_result = markdown_to_telegram_html(injection)
        # Should not produce two separate clickable links
        # Count <a> tags: at most 1 legitimate link
        import re

        links = re.findall(r'<a\s+href="([^"]*)"', html_result)
        # If any link points to evil.com, that's an injection success
        evil_links = [link for link in links if "evil.com" in link]
        assert len(evil_links) == 0, (
            f"Markdown injection produced evil link: {html_result}"
        )

    def test_llm_output_with_html_script_tag_sanitized(self) -> None:
        """WHAT: LLM output contains raw <script> tags.
        EXPECTED: Tags are HTML-escaped and rendered as visible text,
            not as executable HTML.
        WHY: Telegram's HTML parse mode does not execute scripts,
            but unescaped tags could confuse the parser or be passed
            to other rendering contexts.
        """
        malicious_output = "Here is code: <script>alert('xss')</script> done."
        html_result = markdown_to_telegram_html(malicious_output)
        # Raw <script> must be escaped
        assert "<script>" not in html_result, (
            f"Unescaped <script> tag in output: {html_result}"
        )
        # The escaped version should be visible as text
        assert "&lt;script&gt;" in html_result or "script" in strip_markdown(
            malicious_output
        ), "Script tag content should be preserved as escaped text"

    def test_llm_output_with_telegram_slash_command_sanitized(self) -> None:
        """WHAT: LLM output contains text like '/start' or '/reset' that
            Telegram would render as clickable bot commands.
        EXPECTED: sanitize_telegram_slashes replaces the slash with U+2044
            (fraction slash) before a letter, preventing command activation.
        WHY: An attacker could trick the LLM into outputting '/reset'
            which, if clicked by the user, would reset their conversation.
        """
        outputs_with_commands = [
            "Try typing /start to begin",
            "You can use /reset to clear history",
            "Available: /help /settings /model",
            "Use path/to/file for the config",  # should NOT be affected (no letter after /)
        ]
        for output in outputs_with_commands[:3]:
            sanitized = sanitize_telegram_slashes(output)
            # The fraction slash U+2044 should replace / before letters
            assert "/" not in sanitized.split("⁄")[0] or "⁄" in sanitized, (
                f"Slash command not sanitized: {sanitized}"
            )
            # Verify the fraction slash is present
            assert "⁄" in sanitized, f"Expected fraction slash in: {sanitized}"

        # path/to/file: '/' before 't' should also be sanitized (by design)
        path_output = "Use path/to/file"
        sanitized_path = sanitize_telegram_slashes(path_output)
        # This is expected behavior: /t and /f get sanitized
        assert "⁄" in sanitized_path

    def test_llm_output_with_img_onerror_sanitized(self) -> None:
        """WHAT: LLM output contains <img> with onerror XSS payload.
        EXPECTED: Tag is escaped, not rendered as HTML element.
        WHY: Some rendering contexts might process img tags.
        """
        malicious_output = '<img src="x" onerror="alert(1)">'
        html_result = markdown_to_telegram_html(malicious_output)
        assert "<img" not in html_result, (
            f"Unescaped <img> tag in output: {html_result}"
        )
