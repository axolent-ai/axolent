"""Converts markdown to Telegram-compatible HTML.

Telegram supports only a subset of HTML tags:
<b>, <strong>, <i>, <em>, <u>, <s>, <code>, <pre>, <a>.

This converter transforms common markdown elements into this subset.
Additionally: strip_markdown() removes markdown syntax for a clean plain-text fallback.
"""

import html
import logging
import re
import uuid
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Only allow safe URL schemes (no javascript:, data:, etc.)
ALLOWED_URL_SCHEMES: set[str] = {"http", "https", "tg", "mailto"}


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard markdown to Telegram HTML.

    Supports:
        * ## Headlines -> <b>Headlines</b>
        * **bold** -> <b>bold</b>
        * *italic* -> <i>italic</i>
        * _italic_ -> <i>italic</i>
        * `code` -> <code>code</code>
        * ```code block``` -> <pre>code block</pre>
        * [text](url) -> <a href="url">text</a>

    Escapes HTML special characters (<, >, &) in normal text.
    Code blocks and inline code are escaped separately.

    Args:
        text: Markdown-formatted text.

    Returns:
        Telegram HTML string.
    """
    sentinel = uuid.uuid4().hex[:8]

    # 1. Extract links first (before HTML escape, prevents double-escape)
    links: list[tuple[str, str]] = []

    def _extract_link(m: re.Match) -> str:
        text_part = m.group(1)
        url = m.group(2).strip()
        try:
            parsed = urlparse(url)
            if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
                return text_part
        except Exception:
            return text_part
        links.append((text_part, url))
        return f"\x02LINK-{sentinel}-{len(links) - 1}\x02"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _extract_link, text)

    # 2. Extract fenced code blocks (before everything else)
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

    # 3. Extract inline code (so it does not get HTML-escaped)
    inline_codes: list[str] = []

    def _extract_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x01INLINE{len(inline_codes) - 1}\x01"

    text = re.sub(r"`([^`\n]+)`", _extract_inline_code, text)

    # 4. Escape HTML special characters in normal text
    text = html.escape(text, quote=False)

    # 5. Headlines (# through ######) -> <b>...</b> + newline
    #    Remove all ** inside the headline, otherwise nested <b> tags in Telegram
    def _headline_replace(m: re.Match) -> str:
        content = m.group(1).strip()
        content = content.replace("**", "")
        return f"<b>{content}</b>"

    text = re.sub(r"^#{1,6}\s+(.+)$", _headline_replace, text, flags=re.MULTILINE)

    # 6. Bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # 7. Clean up orphaned ** (unmatched bold markers from Claude output)
    text = re.sub(r"\*\*", "", text)

    # 8. Italic: *text* -> <i>text</i> (only standalone *, no ** around it)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    # 9. Italic: _text_ -> <i>text</i> (only standalone _, no __ around it)
    text = re.sub(r"(?<!_)_([^_\n]+?)_(?!_)", r"<i>\1</i>", text)

    # 10. Reinsert code blocks (HTML-escaped)
    for i, block in enumerate(code_blocks):
        escaped_block = html.escape(block, quote=False)
        text = text.replace(f"\x00CODEBLOCK{i}\x00", f"<pre>{escaped_block}</pre>")

    # 11. Reinsert inline codes (HTML-escaped)
    for i, code in enumerate(inline_codes):
        escaped_code = html.escape(code, quote=False)
        text = text.replace(f"\x01INLINE{i}\x01", f"<code>{escaped_code}</code>")

    # 12. Reinsert links with independent escaping
    for i, (text_part, url) in enumerate(links):
        escaped_text = html.escape(text_part, quote=True)
        escaped_url = html.escape(url, quote=True)
        text = text.replace(
            f"\x02LINK-{sentinel}-{i}\x02",
            f'<a href="{escaped_url}">{escaped_text}</a>',
        )

    return text


def strip_markdown(text: str) -> str:
    """Remove markdown syntax for a clean plain-text fallback.

    Used when Telegram rejects the HTML version.
    Instead of displaying raw ## and **, the syntax is removed
    and only the content is kept.

    Args:
        text: Markdown-formatted text.

    Returns:
        Cleaned plain text without markdown artifacts.
    """
    # Fenced code blocks: keep content, remove fences
    text = re.sub(r"```[a-zA-Z]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)

    # Headlines: remove # prefix, keep text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # Bold: **text** -> text (remove markers)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    # Clean up orphaned **
    text = re.sub(r"\*\*", "", text)

    # Italic: *text* -> text (remove markers)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", text)

    # Italic: _text_ -> text (carefully, do not break snake_case)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_\n]+?)_(?![a-zA-Z0-9])", r"\1", text)

    # Inline code: `code` -> code
    text = re.sub(r"`([^`\n]+)`", r"\1", text)

    # Links: [text](url) -> text (url) as plain text
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    return text
