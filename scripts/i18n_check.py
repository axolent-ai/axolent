#!/usr/bin/env python3
"""Pre-commit hook: validate i18n locale files.

Checks:
  1. All locale files have exactly the same keys as en.json (no missing, no extra)
  2. All source_hash fields are not "PENDING" (must run bootstrap first)
  3. source_hash matches current EN text (detects stale translations)
  4. All placeholders in translated texts match the EN source
  5. No empty text fields

Exit code 0 = pass, 1 = fail (blocks commit).

Usage:
    python scripts/i18n_check.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

LOCALES_DIR = Path(__file__).parent.parent / "bridge" / "i18n" / "locales"
EN_FILE = LOCALES_DIR / "en.json"
META_FILE = LOCALES_DIR / "_meta.json"

PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def get_placeholders(text: str) -> set[str]:
    """Extract {placeholder} names from text."""
    return set(PLACEHOLDER_RE.findall(text))


def compute_source_hash(text: str) -> str:
    """Compute sha256 of EN source text, return first 16 hex chars.

    Must match the algorithm in i18n_bootstrap_hashes.py.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not EN_FILE.exists():
        print("ERROR: en.json not found")
        return 1

    with open(EN_FILE, "r", encoding="utf-8") as f:
        en_data = json.load(f)
    en_keys = set(en_data.get("keys", {}).keys())

    if not META_FILE.exists():
        print("ERROR: _meta.json not found")
        return 1

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_locales = set(meta.keys())

    # Check each expected locale has a file
    for lang_code in expected_locales:
        locale_file = LOCALES_DIR / f"{lang_code}.json"
        if not locale_file.exists():
            errors.append(f"MISSING FILE: {lang_code}.json")
            continue

        with open(locale_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_keys = set(data.get("keys", {}).keys())

        # Check missing keys
        missing = en_keys - file_keys
        if missing:
            errors.append(
                f"{lang_code}: MISSING {len(missing)} keys: {sorted(missing)[:5]}..."
            )

        # Check extra keys
        extra = file_keys - en_keys
        if extra:
            errors.append(
                f"{lang_code}: EXTRA {len(extra)} keys: {sorted(extra)[:5]}..."
            )

        # Check each key
        for key, entry in data.get("keys", {}).items():
            text = entry.get("text", "")

            # Empty text
            if not text.strip():
                errors.append(f"{lang_code}/{key}: empty text")

            # PENDING hash (only error if not auto_translated with reviewed=false)
            source_hash = entry.get("source_hash", "")
            if source_hash == "PENDING":
                errors.append(
                    f"{lang_code}/{key}: source_hash is PENDING (run bootstrap)"
                )

            # Staleness check: verify hash matches current EN text
            # (skip for en.json itself and for PENDING hashes already reported)
            # Staleness is a warning by default, error with --strict
            if (
                lang_code != "en"
                and source_hash
                and source_hash != "PENDING"
                and key in en_keys
            ):
                en_text = en_data["keys"][key].get("text", "")
                expected_hash = compute_source_hash(en_text)
                if source_hash != expected_hash:
                    stale_msg = (
                        f"{lang_code}/{key}: STALE translation "
                        f"(source_hash={source_hash[:8]}... != current EN hash={expected_hash[:8]}...)"
                    )
                    if "--strict" in sys.argv:
                        errors.append(stale_msg)
                    else:
                        warnings.append(stale_msg)

            # Placeholder mismatch (only for keys that exist in EN)
            if key in en_keys:
                en_text = en_data["keys"][key].get("text", "")
                en_placeholders = get_placeholders(en_text)
                locale_placeholders = get_placeholders(text)

                if en_placeholders != locale_placeholders:
                    errors.append(
                        f"{lang_code}/{key}: placeholder mismatch "
                        f"EN={en_placeholders} LOCALE={locale_placeholders}"
                    )

    if warnings:
        print(f"i18n check WARNINGS ({len(warnings)} stale translations):")
        for w in warnings[:10]:
            print(f"  [WARN] {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
        print("  Run 'python scripts/i18n_bootstrap_hashes.py' to fix.\n")

    if errors:
        print(f"i18n check FAILED ({len(errors)} errors):\n")
        for err in errors[:30]:  # Show first 30
            print(f"  {err}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        return 1

    print(
        f"i18n check PASSED: {len(expected_locales)} locales, {len(en_keys)} keys each."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
