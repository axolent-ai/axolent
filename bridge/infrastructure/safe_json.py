"""Safe JSON loading with size and depth limits.

Prevents DoS via oversized or deeply nested JSON payloads from
user-supplied content (conversation imports, skill uploads, etc.).

Usage:
    from infrastructure.safe_json import safe_json_load

    data = safe_json_load(raw_bytes, max_bytes=10*1024*1024, max_depth=64)
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any


class JsonTooLargeError(ValueError):
    """Raised when JSON input exceeds the maximum allowed size."""

    def __init__(self, size: int, max_bytes: int) -> None:
        self.size = size
        self.max_bytes = max_bytes
        super().__init__(
            f"JSON input too large: {size} bytes (limit: {max_bytes} bytes)"
        )


class JsonTooDeepError(ValueError):
    """Raised when parsed JSON exceeds the maximum allowed nesting depth."""

    def __init__(self, depth: int, max_depth: int) -> None:
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"JSON nesting too deep: {depth} levels (limit: {max_depth})")


# Default limits
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_DEPTH: int = 64


def _check_depth(obj: Any, max_depth: int) -> None:
    """Check nesting depth iteratively (no recursion, no stack overflow).

    Uses BFS with explicit depth tracking via a deque.

    Args:
        obj: Parsed JSON object.
        max_depth: Maximum allowed nesting depth.

    Raises:
        JsonTooDeepError: If nesting exceeds max_depth.
    """
    # Queue entries: (value, current_depth)
    queue: deque[tuple[Any, int]] = deque()
    queue.append((obj, 0))

    while queue:
        current, depth = queue.popleft()

        if depth > max_depth:
            raise JsonTooDeepError(depth, max_depth)

        if isinstance(current, dict):
            for v in current.values():
                if isinstance(v, (dict, list)):
                    queue.append((v, depth + 1))
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    queue.append((item, depth + 1))


def safe_json_load(
    text: str | bytes,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Any:
    """Parse JSON with size and depth limits.

    Steps:
      1. Check byte size BEFORE parsing (prevents memory exhaustion).
      2. Parse with stdlib json.loads.
      3. Check nesting depth iteratively AFTER parsing (prevents
         deeply nested structures from causing issues downstream).

    Args:
        text: JSON string or bytes.
        max_bytes: Maximum allowed input size in bytes.
        max_depth: Maximum allowed nesting depth.

    Returns:
        Parsed JSON value.

    Raises:
        JsonTooLargeError: Input exceeds max_bytes.
        JsonTooDeepError: Parsed structure exceeds max_depth.
            Also raised when json.loads itself hits RecursionError
            on deeply nested input (CPython C-parser limit).
        json.JSONDecodeError: Invalid JSON.
    """
    # Step 1: Size check before parse
    if isinstance(text, str):
        size = len(text.encode("utf-8", errors="replace"))
    else:
        size = len(text)

    if size > max_bytes:
        raise JsonTooLargeError(size, max_bytes)

    # Step 2: Parse (catch RecursionError from CPython's C-level JSON parser
    # which can recurse ~5000 deep on nested arrays/objects before our
    # iterative depth checker runs)
    try:
        if isinstance(text, bytes):
            result = json.loads(text.decode("utf-8", errors="replace"))
        else:
            result = json.loads(text)
    except RecursionError:
        raise JsonTooDeepError(
            depth=max_depth + 1,
            max_depth=max_depth,
        )

    # Step 3: Depth check after parse (iterative, no recursion)
    _check_depth(result, max_depth)

    return result
