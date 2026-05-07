"""Konvertiert Markdown zu Telegram-kompatiblem HTML.

Telegram unterstützt nur einen Subset von HTML-Tags:
<b>, <strong>, <i>, <em>, <u>, <s>, <code>, <pre>, <a>.

Dieser Konverter wandelt gängige Markdown-Elemente in diesen Subset um.
Zusätzlich: strip_markdown() entfernt Markdown-Syntax für sauberen Plain-Text-Fallback.
"""

import html
import logging
import re
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Nur sichere URL-Schemes erlauben (kein javascript:, data: etc.)
ALLOWED_URL_SCHEMES: set[str] = {"http", "https", "tg", "mailto"}


def markdown_to_telegram_html(text: str) -> str:
    """Konvertiert Standard-Markdown zu Telegram-HTML.

    Unterstützt:
        * ## Headlines -> <b>Headlines</b>
        * **bold** -> <b>bold</b>
        * *italic* -> <i>italic</i>
        * _italic_ -> <i>italic</i>
        * `code` -> <code>code</code>
        * ```code block``` -> <pre>code block</pre>
        * [text](url) -> <a href="url">text</a>

    Escaped HTML-Sonderzeichen (<, >, &) im normalen Text.
    Code-Blöcke und Inline-Code werden separat escaped.

    Args:
        text: Markdown-formatierter Text.

    Returns:
        Telegram-HTML-String.
    """
    # 1. Fenced Code-Blöcke extrahieren (vor allem anderen)
    code_blocks: list[str] = []

    def _extract_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    text = re.sub(
        r"```[a-zA-Z]*\n?(.*?)```",
        _extract_code_block,
        text,
        flags=re.DOTALL,
    )

    # 2. Inline-Code extrahieren (damit er nicht HTML-escaped wird)
    inline_codes: list[str] = []

    def _extract_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x01INLINE{len(inline_codes) - 1}\x01"

    text = re.sub(r"`([^`\n]+)`", _extract_inline_code, text)

    # 3. HTML-Sonderzeichen im normalen Text escapen
    text = html.escape(text, quote=False)

    # 4. Markdown-Links: [text](url) -> <a href="url">text</a> (mit Scheme-Validierung)
    def _convert_link(match: re.Match) -> str:
        link_text = html.escape(match.group(1), quote=True)
        url = match.group(2).strip()
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
            return link_text  # Unsicheres Scheme: nur Text, kein Link
        url_safe = html.escape(url, quote=True)
        return f'<a href="{url_safe}">{link_text}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _convert_link, text)

    # 5. Headlines (# bis ######) -> <b>...</b> + Newline
    #    Alle ** innerhalb der Headline entfernen, sonst verschachtelte <b>-Tags in Telegram
    def _headline_replace(m: re.Match) -> str:
        content = m.group(1).strip()
        # Alle ** (Bold-Marker) entfernen um verschachtelte <b> zu verhindern
        content = content.replace("**", "")
        return f"<b>{content}</b>"

    text = re.sub(r"^#{1,6}\s+(.+)$", _headline_replace, text, flags=re.MULTILINE)

    # 6. Bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # 7. Verwaiste ** aufräumen (ungematchte Bold-Marker aus Claude-Output)
    text = re.sub(r"\*\*", "", text)

    # 8. Italic: *text* -> <i>text</i> (nur alleinstehende *, kein ** drumherum)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    # 9. Italic: _text_ -> <i>text</i> (nur alleinstehende _, kein __ drumherum)
    text = re.sub(r"(?<!_)_([^_\n]+?)_(?!_)", r"<i>\1</i>", text)

    # 10. Code-Blöcke wieder einsetzen (HTML-escaped)
    for i, block in enumerate(code_blocks):
        escaped_block = html.escape(block, quote=False)
        text = text.replace(f"\x00CODEBLOCK{i}\x00", f"<pre>{escaped_block}</pre>")

    # 11. Inline-Codes wieder einsetzen (HTML-escaped)
    for i, code in enumerate(inline_codes):
        escaped_code = html.escape(code, quote=False)
        text = text.replace(f"\x01INLINE{i}\x01", f"<code>{escaped_code}</code>")

    return text


def strip_markdown(text: str) -> str:
    """Entfernt Markdown-Syntax für sauberen Plain-Text-Fallback.

    Wird verwendet wenn Telegram die HTML-Version ablehnt.
    Statt rohe ## und ** anzuzeigen, wird die Syntax entfernt
    und nur der Inhalt behalten.

    Args:
        text: Markdown-formatierter Text.

    Returns:
        Bereinigter Plain-Text ohne Markdown-Artefakte.
    """
    # Fenced Code-Blöcke: Inhalt behalten, Fences entfernen
    text = re.sub(r"```[a-zA-Z]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)

    # Headlines: #-Prefix entfernen, Text behalten
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # Bold: **text** -> text (Marker entfernen)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    # Verwaiste ** aufräumen
    text = re.sub(r"\*\*", "", text)

    # Italic: *text* -> text (Marker entfernen)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", text)

    # Italic: _text_ -> text (vorsichtig, snake_case nicht zerstören)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_\n]+?)_(?![a-zA-Z0-9])", r"\1", text)

    # Inline-Code: `code` -> code
    text = re.sub(r"`([^`\n]+)`", r"\1", text)

    # Links: [text](url) -> text (url) als Plain-Text
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    return text
