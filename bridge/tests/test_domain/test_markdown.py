"""Tests for domain.markdown: Markdown-to-Telegram-HTML conversion.

Tests conversion, URL scheme whitelist, and plain-text fallback.
The URL scheme tests are security-critical (XSS prevention).
"""

from domain.markdown import (
    ALLOWED_URL_SCHEMES,
    markdown_to_telegram_html,
    strip_markdown,
)


class TestMarkdownToTelegramHtml:
    """Conversion from Markdown to Telegram-compatible HTML."""

    def test_headlines_to_bold(self) -> None:
        """Markdown headlines are converted to <b>...</b>."""
        assert "<b>Titel</b>" in markdown_to_telegram_html("## Titel")
        assert "<b>H1</b>" in markdown_to_telegram_html("# H1")
        assert "<b>H6</b>" in markdown_to_telegram_html("###### H6")

    def test_bold_conversion(self) -> None:
        """**text** is converted to <b>text</b>."""
        result = markdown_to_telegram_html("Das ist **wichtig** hier")
        assert "<b>wichtig</b>" in result

    def test_italic_conversion(self) -> None:
        """*text* is converted to <i>text</i>."""
        result = markdown_to_telegram_html("Das ist *kursiv* hier")
        assert "<i>kursiv</i>" in result

    def test_underscore_italic(self) -> None:
        """_text_ is converted to <i>text</i>."""
        result = markdown_to_telegram_html("Das ist _kursiv_ hier")
        assert "<i>kursiv</i>" in result

    def test_inline_code(self) -> None:
        """`code` is converted to <code>code</code>."""
        result = markdown_to_telegram_html("Nutze `pip install` hier")
        assert "<code>pip install</code>" in result

    def test_code_block(self) -> None:
        """Fenced code blocks are converted to <pre>...</pre>."""
        md = "```python\nprint('hello')\n```"
        result = markdown_to_telegram_html(md)
        assert "<pre>" in result
        assert "print(&#x27;hello&#x27;)" in result or "print('hello')" in result

    def test_html_escape_special_chars(self) -> None:
        """HTML special characters (<, >, &) are escaped."""
        result = markdown_to_telegram_html("a < b && c > d")
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_url_scheme_whitelist_allowed(self) -> None:
        """Allowed schemes (http, https, tg, mailto) produce <a> tags."""
        for scheme in ("http", "https", "tg", "mailto"):
            md = f"[Link]({scheme}://example.com)"
            result = markdown_to_telegram_html(md)
            assert "<a href=" in result, f"Scheme '{scheme}' should be allowed"

    def test_url_scheme_whitelist_blocked(self) -> None:
        """Dangerous schemes (javascript:, data:, file:) are blocked.

        Security-critical: prevents XSS via malicious links.
        """
        for scheme in ("javascript", "data", "file"):
            md = f"[Click]({scheme}:alert(1))"
            result = markdown_to_telegram_html(md)
            assert "<a " not in result, f"Scheme '{scheme}:' MUST be blocked"
            # The link text should still be displayed
            assert "Click" in result

    def test_url_scheme_whitelist_set(self) -> None:
        """Verify that the whitelist contains exactly the expected schemes."""
        assert ALLOWED_URL_SCHEMES == {"http", "https", "tg", "mailto"}

    def test_link_conversion(self) -> None:
        """[text](url) is converted to <a href="url">text</a>."""
        md = "[Google](https://google.com)"
        result = markdown_to_telegram_html(md)
        assert '<a href="https://google.com">Google</a>' in result

    def test_link_no_double_escape(self) -> None:
        """Links with & in text and URL are escaped only once, not double-escaped.

        Regression test: global html.escape ran BEFORE link conversion,
        then link text and URL were escaped again -> &amp;amp;
        """
        md = "[Tom & Jerry](https://example.com?a=1&b=2)"
        result = markdown_to_telegram_html(md)
        assert '<a href="https://example.com?a=1&amp;b=2">Tom &amp; Jerry</a>' in result
        # Must NOT be double-escaped
        assert "&amp;amp;" not in result

    def test_link_with_special_chars_in_text(self) -> None:
        """HTML special characters in link text are correctly escaped."""
        md = "[a < b](https://example.com)"
        result = markdown_to_telegram_html(md)
        assert "a &lt; b" in result
        assert "<a href=" in result

    def test_nested_bold_in_headline_stripped(self) -> None:
        """Bold markers inside headlines are removed (no nested <b>)."""
        result = markdown_to_telegram_html("## **Fette Headline**")
        assert "**" not in result
        assert "<b>Fette Headline</b>" in result

    def test_orphaned_bold_markers_cleaned(self) -> None:
        """Unpaired ** are removed instead of appearing in the output."""
        result = markdown_to_telegram_html("Text ** mit ** ungepaarten Markern")
        assert "**" not in result


class TestStripMarkdown:
    """strip_markdown removes syntax for plain-text fallback."""

    def test_strip_markdown_for_fallback(self) -> None:
        """All Markdown syntax is removed, only plain content remains."""
        md = "## Headline\n**bold** and *italic*\n`code`"
        result = strip_markdown(md)
        assert "##" not in result
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
        assert "Headline" in result
        assert "bold" in result
        assert "italic" in result
        assert "code" in result

    def test_strip_code_block(self) -> None:
        """Code block fences are removed, content remains."""
        md = "```python\nprint('hi')\n```"
        result = strip_markdown(md)
        assert "```" not in result
        assert "print('hi')" in result

    def test_strip_links_keep_text_and_url(self) -> None:
        """Links are converted to 'text (url)' format."""
        md = "[Google](https://google.com)"
        result = strip_markdown(md)
        assert "Google" in result
        assert "https://google.com" in result
