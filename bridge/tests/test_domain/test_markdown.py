"""Tests für domain.markdown: Markdown-zu-Telegram-HTML-Konvertierung.

Testet Konvertierung, URL-Scheme-Whitelist und Plain-Text-Fallback.
Die URL-Scheme-Tests sind sicherheitskritisch (XSS-Praevention).
"""

from domain.markdown import (
    ALLOWED_URL_SCHEMES,
    markdown_to_telegram_html,
    strip_markdown,
)


class TestMarkdownToTelegramHtml:
    """Konvertierung von Markdown zu Telegram-kompatiblem HTML."""

    def test_headlines_to_bold(self) -> None:
        """Markdown-Headlines werden zu <b>...</b>."""
        assert "<b>Titel</b>" in markdown_to_telegram_html("## Titel")
        assert "<b>H1</b>" in markdown_to_telegram_html("# H1")
        assert "<b>H6</b>" in markdown_to_telegram_html("###### H6")

    def test_bold_conversion(self) -> None:
        """**text** wird zu <b>text</b>."""
        result = markdown_to_telegram_html("Das ist **wichtig** hier")
        assert "<b>wichtig</b>" in result

    def test_italic_conversion(self) -> None:
        """*text* wird zu <i>text</i>."""
        result = markdown_to_telegram_html("Das ist *kursiv* hier")
        assert "<i>kursiv</i>" in result

    def test_underscore_italic(self) -> None:
        """_text_ wird zu <i>text</i>."""
        result = markdown_to_telegram_html("Das ist _kursiv_ hier")
        assert "<i>kursiv</i>" in result

    def test_inline_code(self) -> None:
        """`code` wird zu <code>code</code>."""
        result = markdown_to_telegram_html("Nutze `pip install` hier")
        assert "<code>pip install</code>" in result

    def test_code_block(self) -> None:
        """Fenced code blocks werden zu <pre>...</pre>."""
        md = "```python\nprint('hello')\n```"
        result = markdown_to_telegram_html(md)
        assert "<pre>" in result
        assert "print(&#x27;hello&#x27;)" in result or "print('hello')" in result

    def test_html_escape_special_chars(self) -> None:
        """HTML-Sonderzeichen (<, >, &) werden escaped."""
        result = markdown_to_telegram_html("a < b && c > d")
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_url_scheme_whitelist_allowed(self) -> None:
        """Erlaubte Schemes (http, https, tg, mailto) erzeugen <a>-Tags."""
        for scheme in ("http", "https", "tg", "mailto"):
            md = f"[Link]({scheme}://example.com)"
            result = markdown_to_telegram_html(md)
            assert "<a href=" in result, f"Scheme '{scheme}' sollte erlaubt sein"

    def test_url_scheme_whitelist_blocked(self) -> None:
        """Gefaehrliche Schemes (javascript:, data:, file:) werden blockiert.

        Sicherheitskritisch: verhindert XSS via boeswillige Links.
        """
        for scheme in ("javascript", "data", "file"):
            md = f"[Click]({scheme}:alert(1))"
            result = markdown_to_telegram_html(md)
            assert "<a " not in result, f"Scheme '{scheme}:' MUSS blockiert werden"
            # Der Link-Text soll trotzdem angezeigt werden
            assert "Click" in result

    def test_url_scheme_whitelist_set(self) -> None:
        """Prueft dass die Whitelist genau die erwarteten Schemes enthaelt."""
        assert ALLOWED_URL_SCHEMES == {"http", "https", "tg", "mailto"}

    def test_link_conversion(self) -> None:
        """[text](url) wird zu <a href="url">text</a>."""
        md = "[Google](https://google.com)"
        result = markdown_to_telegram_html(md)
        assert '<a href="https://google.com">Google</a>' in result

    def test_link_no_double_escape(self) -> None:
        """Links mit & im Text und URL werden nur einfach escaped, nicht doppelt.

        Regression-Test: globales html.escape lief VOR Link-Konvertierung,
        dann wurde Link-Text und URL nochmal escaped -> &amp;amp;
        """
        md = "[Tom & Jerry](https://example.com?a=1&b=2)"
        result = markdown_to_telegram_html(md)
        assert '<a href="https://example.com?a=1&amp;b=2">Tom &amp; Jerry</a>' in result
        # Darf NICHT doppelt escaped sein
        assert "&amp;amp;" not in result

    def test_link_with_special_chars_in_text(self) -> None:
        """HTML-Sonderzeichen im Link-Text werden korrekt escaped."""
        md = "[a < b](https://example.com)"
        result = markdown_to_telegram_html(md)
        assert "a &lt; b" in result
        assert "<a href=" in result

    def test_nested_bold_in_headline_stripped(self) -> None:
        """Bold-Marker innerhalb von Headlines werden entfernt (kein verschachteltes <b>)."""
        result = markdown_to_telegram_html("## **Fette Headline**")
        assert "**" not in result
        assert "<b>Fette Headline</b>" in result

    def test_orphaned_bold_markers_cleaned(self) -> None:
        """Ungepaarte ** werden entfernt statt im Output zu landen."""
        result = markdown_to_telegram_html("Text ** mit ** ungepaarten Markern")
        assert "**" not in result


class TestStripMarkdown:
    """strip_markdown entfernt Syntax für Plain-Text-Fallback."""

    def test_strip_markdown_for_fallback(self) -> None:
        """Alle Markdown-Syntax wird entfernt, reiner Inhalt bleibt."""
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
        """Code-Block-Fences werden entfernt, Inhalt bleibt."""
        md = "```python\nprint('hi')\n```"
        result = strip_markdown(md)
        assert "```" not in result
        assert "print('hi')" in result

    def test_strip_links_keep_text_and_url(self) -> None:
        """Links werden zu 'text (url)' Format."""
        md = "[Google](https://google.com)"
        result = strip_markdown(md)
        assert "Google" in result
        assert "https://google.com" in result
