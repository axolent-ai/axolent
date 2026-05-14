"""Text adapter: applies Text Guard to plain text content.

Supports: .md, .txt, .json, .yaml, .html and any UTF-8 text file.
Pure domain logic, no file I/O (caller provides content as string).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from domain.text_guard.guard import TextGuard
from domain.text_guard.models import Issue

log = logging.getLogger(__name__)


def check_text(text: str, guard: TextGuard) -> list[Issue]:
    """Check plain text for diacritic issues.

    Args:
        text: The text content to check.
        guard: Configured TextGuard instance.

    Returns:
        List of detected issues.
    """
    return guard.check(text)


def fix_text(text: str, guard: TextGuard) -> str:
    """Fix diacritic issues in plain text.

    Args:
        text: The text content to fix.
        guard: Configured TextGuard instance.

    Returns:
        Corrected text.
    """
    return guard.fix(text)


def fix_json_values(data: Any, guard: TextGuard) -> Any:
    """Recursively fix string values in a JSON-compatible structure.

    Traverses dicts and lists, applying text guard to all string values.
    Dict keys are left unchanged to avoid breaking references.

    Args:
        data: Parsed JSON data (dict, list, str, int, float, bool, None).
        guard: Configured TextGuard instance.

    Returns:
        Data with corrected string values.
    """
    if isinstance(data, str):
        return guard.fix(data)
    if isinstance(data, dict):
        return {k: fix_json_values(v, guard) for k, v in data.items()}
    if isinstance(data, list):
        return [fix_json_values(item, guard) for item in data]
    return data


def fix_json_string(json_string: str, guard: TextGuard) -> str:
    """Fix diacritic issues in a JSON string.

    Parses JSON, fixes all string values, serializes back.

    Args:
        json_string: Raw JSON string.
        guard: Configured TextGuard instance.

    Returns:
        JSON string with corrected values.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        log.warning("JSON parse failed, treating as plain text")
        return guard.fix(json_string)

    fixed = fix_json_values(data, guard)
    return json.dumps(fixed, ensure_ascii=False, indent=2)
