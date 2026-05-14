"""Encoding helpers for Axolent.

Standard patterns for all file, subprocess, and JSON operations.
Enforces UTF-8 explicitly to guarantee multi-user multi-language
robustness from day one.

Never use raw open() or subprocess.run() in Axolent code;
use these helpers instead.
"""

from __future__ import annotations

import json
import logging
import subprocess  # nosec B404 - intentional central wrapper, all calls controlled here
from pathlib import Path
from typing import Any, TextIO

log = logging.getLogger(__name__)


def open_utf8(path: str | Path, mode: str = "r", **kwargs) -> TextIO:
    """Open a file with UTF-8 encoding and replace errors.

    Args:
        path: File path (str or Path).
        mode: File mode (r, w, a, etc.).
        **kwargs: Additional open() arguments.

    Returns:
        File handle with UTF-8 / errors=replace set.
    """
    return open(path, mode, encoding="utf-8", errors="replace", **kwargs)


def run_subprocess_utf8(
    cmd: list[str],
    timeout: float | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run an external program with guaranteed UTF-8 encoding.

    Args:
        cmd: Command as list (e.g. ["claude", "-p", "..."]).
        timeout: Timeout in seconds, None for unlimited.
        **kwargs: Additional subprocess.run() arguments.

    Returns:
        subprocess.CompletedProcess with stdout/stderr as str (UTF-8 decoded).
    """
    return subprocess.run(  # nosec B603 - shell=False, no injection risk
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **kwargs,
    )


def write_json_utf8(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write a Python object as a JSON file in UTF-8.

    Uses ensure_ascii=False so Unicode characters appear directly
    in the JSON, not escaped (\\u00e4 etc.).

    Args:
        data: Serializable Python object.
        path: Target path.
        indent: JSON indentation (default 2).
    """
    with open_utf8(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_json_utf8(path: str | Path) -> Any:
    """Read a JSON file in UTF-8.

    Args:
        path: Source path.

    Returns:
        Deserialized Python object.
    """
    with open_utf8(path, "r") as f:
        return json.load(f)


def append_jsonl_utf8(entry: dict, path: str | Path) -> None:
    """Append an entry as a JSON line to a file.

    Uses UTF-8 without BOM, ensure_ascii=False for Unicode readability.

    Args:
        entry: Dict to write.
        path: Target file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open_utf8(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
