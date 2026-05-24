"""Public/Private Boundary Scanner.

Scans all git-tracked files for:
1. Forbidden paths (e.g. .env, *.db, credentials files)
2. Forbidden content patterns (tokens, secrets, brand-internal terms)
3. Files not covered by public_allowed_paths (warnings)

Exit code 0 = clean, 1 = blocked items found.

Usage:
    python scripts/check_public_boundary.py
    python scripts/check_public_boundary.py --config path/to/config.yaml
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

import yaml

# Dummy-value indicators: if a token value contains any of these,
# it is treated as a placeholder (not a real secret).
DUMMY_INDICATORS = [
    "EXAMPLE",
    "REPLACE_ME",
    "your-token-here",
    "your_token_here",
    "your_bot_token",
    "your_telegram",
    "your_",
    "YOUR_",
    "xxx",
    "...",
    "placeholder",
    "PLACEHOLDER",
    "changeme",
    "CHANGEME",
    "TODO",
    "INSERT_",
    "DUMMY",
    "dummy",
    "test",
    "TEST",
    "fake",
    "FAKE",
    "sample",
    "SAMPLE",
    "_here",
    "_HERE",
]

# Env var patterns that need dummy-value detection.
# These match KEY=VALUE where VALUE looks real (not a placeholder).
ENV_VAR_SECRET_PATTERNS = [
    r"TELEGRAM_BOT_TOKEN\s*=\s*(.+)",
    r"SENTRY_DSN\s*=\s*(.+)",
    r"ANTHROPIC_API_KEY\s*=\s*(.+)",
    r"OPENAI_API_KEY\s*=\s*(.+)",
    r"GROQ_API_KEY\s*=\s*(.+)",
    r"AWS_SECRET_ACCESS_KEY\s*=\s*(.+)",
    r"GITHUB_TOKEN\s*=\s*(.+)",
]


def find_repo_root() -> Path:
    """Find the git repository root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: walk up from this script
        return Path(__file__).resolve().parent.parent


def get_tracked_files(repo_root: Path) -> list[str]:
    """Get all git-tracked files as posix-style relative paths."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(repo_root),
    )
    files = [f for f in result.stdout.strip().split("\n") if f]
    return files


def load_config(config_path: Path) -> dict:
    """Load the boundary scanner YAML config."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def matches_glob_pattern(filepath: str, pattern: str) -> bool:
    """Check if a filepath matches a glob pattern.

    Supports ** for recursive matching and * for single-level.
    """
    # Normalize to forward slashes
    filepath = filepath.replace("\\", "/")
    pattern = pattern.replace("\\", "/")

    # Use fnmatch for simple patterns
    if "**" in pattern:
        # For ** patterns, we need recursive matching
        # Convert ** to work with fnmatch
        # **/*.db -> should match any/path/file.db
        parts = pattern.split("**/")
        if len(parts) == 2 and parts[0] == "":
            # Pattern like **/*.db or **/dir/**
            sub_pattern = parts[1]
            if "**" in sub_pattern:
                # Pattern like **/private/**
                dir_name = sub_pattern.rstrip("/**")
                # Check if dir_name appears as a path component
                path_parts = filepath.split("/")
                return dir_name in path_parts
            # Check if any suffix of the path matches sub_pattern
            path_parts = filepath.split("/")
            for i in range(len(path_parts)):
                suffix = "/".join(path_parts[i:])
                if fnmatch.fnmatch(suffix, sub_pattern):
                    return True
            return False
        elif len(parts) == 2:
            # Pattern like prefix/**/suffix
            prefix = parts[0].rstrip("/")
            suffix = parts[1]
            if not filepath.startswith(prefix + "/") and filepath != prefix:
                return False
            remainder = filepath[len(prefix) :].lstrip("/")
            return fnmatch.fnmatch(remainder, suffix) or fnmatch.fnmatch(
                remainder, "**/" + suffix
            )
    # Simple glob (e.g. bridge/**)
    if pattern.endswith("/**"):
        dir_prefix = pattern[:-3]
        return filepath.startswith(dir_prefix + "/") or filepath == dir_prefix
    return fnmatch.fnmatch(filepath, pattern)


def is_binary_file(filepath: Path) -> bool:
    """Heuristic check if a file is binary."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            # If there are null bytes, it's likely binary
            if b"\x00" in chunk:
                return True
            return False
    except (OSError, IOError):
        return True


def is_dummy_value(value: str) -> bool:
    """Check if a token/secret value is a placeholder (not real)."""
    value = value.strip().strip("'\"")
    if not value:
        return True
    for indicator in DUMMY_INDICATORS:
        if indicator in value:
            return True
    return False


def check_forbidden_paths(
    tracked_files: list[str], forbidden_patterns: list[str]
) -> list[tuple[str, str]]:
    """Check tracked files against forbidden path patterns.

    Returns list of (filepath, matching_pattern) tuples.
    """
    blocked = []
    for filepath in tracked_files:
        for pattern in forbidden_patterns:
            if matches_glob_pattern(filepath, pattern):
                blocked.append((filepath, pattern))
                break  # One match is enough
    return blocked


def check_forbidden_content(
    tracked_files: list[str],
    repo_root: Path,
    content_patterns: list[str],
    whitelist: list[str],
) -> list[tuple[str, int, str]]:
    """Check file contents against forbidden content patterns.

    Returns list of (filepath, line_number, matching_pattern) tuples.
    """
    blocked = []
    compiled_patterns = [(p, re.compile(p, re.IGNORECASE)) for p in content_patterns]

    for filepath in tracked_files:
        # Skip whitelisted files
        normalized = filepath.replace("\\", "/")
        if normalized in whitelist:
            continue

        full_path = repo_root / filepath
        if not full_path.exists():
            continue
        if is_binary_file(full_path):
            continue

        try:
            with open(full_path, encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    for pattern_str, pattern_re in compiled_patterns:
                        if pattern_re.search(line):
                            blocked.append((filepath, line_num, pattern_str))
                            break  # One pattern match per line is enough
        except (OSError, IOError):
            continue

    return blocked


def check_env_var_secrets(
    tracked_files: list[str],
    repo_root: Path,
    whitelist: list[str],
) -> list[tuple[str, int, str]]:
    """Special check: env vars with real (non-dummy) values.

    Returns list of (filepath, line_number, description) tuples.
    """
    blocked = []
    compiled = [(p, re.compile(p)) for p in ENV_VAR_SECRET_PATTERNS]

    for filepath in tracked_files:
        normalized = filepath.replace("\\", "/")
        if normalized in whitelist:
            continue

        full_path = repo_root / filepath
        if not full_path.exists():
            continue
        if is_binary_file(full_path):
            continue

        try:
            with open(full_path, encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    for pattern_str, pattern_re in compiled:
                        match = pattern_re.search(line)
                        if match:
                            value = match.group(1).strip()
                            if value and not is_dummy_value(value):
                                env_name = pattern_str.split(r"\s*=")[0]
                                blocked.append(
                                    (filepath, line_num, f"{env_name} (real value)")
                                )
        except (OSError, IOError):
            continue

    return blocked


def check_allowed_paths(
    tracked_files: list[str], allowed_patterns: list[str]
) -> list[str]:
    """Find files not covered by any allowed path pattern.

    Returns list of filepaths that are not in allowed paths.
    """
    warnings = []
    for filepath in tracked_files:
        matched = False
        for pattern in allowed_patterns:
            if matches_glob_pattern(filepath, pattern):
                matched = True
                break
        if not matched:
            warnings.append(filepath)
    return warnings


def main(config_path: str | None = None) -> int:
    """Run the boundary scanner. Returns exit code (0=clean, 1=blocked)."""
    repo_root = find_repo_root()

    if config_path is None:
        config_path_resolved = repo_root / "scripts" / "public_boundary.yaml"
    else:
        config_path_resolved = Path(config_path)

    if not config_path_resolved.exists():
        print(f"ERROR: Config not found: {config_path_resolved}")
        return 1

    config = load_config(config_path_resolved)
    tracked_files = get_tracked_files(repo_root)

    print("=== Public Boundary Scanner ===")
    print(f"Scanned {len(tracked_files)} tracked files.")
    print()

    # Normalize whitelist paths
    whitelist = [
        p.replace("\\", "/") for p in config.get("content_pattern_whitelist", [])
    ]

    # 1. Check forbidden paths
    blocked_paths = check_forbidden_paths(
        tracked_files, config.get("private_forbidden_paths", [])
    )

    # 2. Check forbidden content
    blocked_content = check_forbidden_content(
        tracked_files,
        repo_root,
        config.get("private_forbidden_content_patterns", []),
        whitelist,
    )

    # 3. Check env var secrets with real values
    blocked_env = check_env_var_secrets(tracked_files, repo_root, whitelist)

    # 4. Check allowed paths (warnings only)
    warnings = check_allowed_paths(
        tracked_files, config.get("public_allowed_paths", [])
    )

    # Output results
    has_blocks = False

    if blocked_paths:
        has_blocks = True
        print("BLOCKED PATHS (must be removed from repo):")
        for filepath, pattern in blocked_paths:
            print(f"  {filepath} (matches: {pattern})")
        print()

    if blocked_content:
        has_blocks = True
        print("BLOCKED CONTENT (must be cleaned):")
        for filepath, line_num, pattern in blocked_content:
            print(f"  {filepath}:{line_num} (matches: {pattern})")
        print()

    if blocked_env:
        has_blocks = True
        print("BLOCKED ENV SECRETS (real values detected):")
        for filepath, line_num, desc in blocked_env:
            print(f"  {filepath}:{line_num} ({desc})")
        print()

    if warnings:
        print(
            "WARNINGS (not in allowed paths, consider adding to "
            ".gitignore or public_allowed_paths):"
        )
        for filepath in warnings:
            print(f"  {filepath}")
        print()

    # Summary
    print(
        f"SUMMARY: {len(blocked_paths)} blocked paths, "
        f"{len(blocked_content) + len(blocked_env)} blocked content, "
        f"{len(warnings)} warnings"
    )

    if has_blocks:
        print("Exit code: 1 (blocked)")
        return 1
    else:
        print("Exit code: 0 (clean)")
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Public/Private Boundary Scanner")
    parser.add_argument("--config", help="Path to config YAML file")
    args = parser.parse_args()
    sys.exit(main(config_path=args.config))
