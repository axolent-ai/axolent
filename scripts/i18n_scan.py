#!/usr/bin/env python3
"""Scan tool: detect stale translations (source_hash mismatch).

Compares each locale's source_hash with the actual hash of en.json text.
If they differ, the EN source has changed and the translation is stale.

Usage:
    python scripts/i18n_scan.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

LOCALES_DIR = Path(__file__).parent.parent / "bridge" / "i18n" / "locales"
EN_FILE = LOCALES_DIR / "en.json"


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    if not EN_FILE.exists():
        print("ERROR: en.json not found", file=sys.stderr)
        return 1

    with open(EN_FILE, "r", encoding="utf-8") as f:
        en_data = json.load(f)
    en_keys = en_data.get("keys", {})

    # Compute current EN hashes
    en_hashes: dict[str, str] = {}
    for key, entry in en_keys.items():
        en_hashes[key] = compute_hash(entry.get("text", ""))

    stale_count = 0
    locale_files = sorted(LOCALES_DIR.glob("*.json"))

    for locale_file in locale_files:
        if locale_file.name.startswith("_") or locale_file.name == "en.json":
            continue

        with open(locale_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        lang = data.get("_language", locale_file.stem)
        keys = data.get("keys", {})
        lang_stale: list[str] = []

        for key, entry in keys.items():
            if key not in en_hashes:
                continue
            stored_hash = entry.get("source_hash", "")
            if (
                stored_hash
                and stored_hash != "PENDING"
                and stored_hash != en_hashes[key]
            ):
                lang_stale.append(key)

        if lang_stale:
            stale_count += len(lang_stale)
            print(f"\n{lang} ({len(lang_stale)} stale):")
            for key in lang_stale[:10]:
                print(f"  {key}")
            if len(lang_stale) > 10:
                print(f"  ... and {len(lang_stale) - 10} more")

    if stale_count == 0:
        print("No stale translations found. All hashes match.")
    else:
        print(f"\nTotal: {stale_count} stale translations across all locales.")
        print("These need re-translation (EN source text has changed).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
