#!/usr/bin/env python3
"""Bootstrap script: compute and write source_hash for all locale files.

For each locale file (xx.json):
  - Reads en.json as the master source
  - For each key in xx.json, computes sha256(en_text)[:16]
  - Writes the computed hash into source_hash field
  - Preserves all other fields (text, auto_translated, reviewed)

Usage:
    python scripts/i18n_bootstrap_hashes.py

Run once after locale files are created or when en.json text changes.
The i18n_check.py pre-commit hook validates staleness based on these hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

LOCALES_DIR = Path(__file__).parent.parent / "bridge" / "i18n" / "locales"
EN_FILE = LOCALES_DIR / "en.json"


def compute_hash(text: str) -> str:
    """Compute sha256 of text, return first 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    if not EN_FILE.exists():
        print(f"ERROR: en.json not found at {EN_FILE}", file=sys.stderr)
        return 1

    with open(EN_FILE, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    en_keys = en_data.get("keys", {})

    # Compute hashes for all EN keys
    en_hashes: dict[str, str] = {}
    for key, entry in en_keys.items():
        text = entry.get("text", "")
        en_hashes[key] = compute_hash(text)

    # Process all locale files (including en.json itself)
    locale_files = sorted(LOCALES_DIR.glob("*.json"))
    updated_count = 0
    total_keys_updated = 0

    for locale_file in locale_files:
        if locale_file.name.startswith("_"):
            continue  # skip _meta.json

        with open(locale_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys = data.get("keys", {})
        file_changed = False

        for key, entry in keys.items():
            if key not in en_hashes:
                # Key exists in locale but not in EN (orphan)
                continue

            expected_hash = en_hashes[key]
            current_hash = entry.get("source_hash", "")

            if current_hash != expected_hash:
                entry["source_hash"] = expected_hash
                file_changed = True
                total_keys_updated += 1

        if file_changed:
            with open(locale_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            updated_count += 1
            print(f"  Updated: {locale_file.name}")

    print(
        f"\nDone. {updated_count} files updated, {total_keys_updated} hashes written."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
