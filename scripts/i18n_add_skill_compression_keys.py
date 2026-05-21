#!/usr/bin/env python3
"""One-shot script: Add all skill.* i18n keys to non-EN locale files.

For non-EN locales that are MISSING the skill.* keys, this copies
the EN text as fallback with auto_translated=true, reviewed=false.
EN itself is skipped (already has the keys).
DE is skipped (already has hand-written translations).

After running this script, run `python scripts/i18n_bootstrap_hashes.py`
to fix source_hash values across all locales.

Run once, then delete this script.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

LOCALES_DIR = Path(__file__).parent.parent / "bridge" / "i18n" / "locales"
EN_FILE = LOCALES_DIR / "en.json"

# Locales that already have the keys (skip them)
SKIP_LOCALES = frozenset({"en", "de"})


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    if not EN_FILE.exists():
        print("ERROR: en.json not found", file=sys.stderr)
        return 1

    with open(EN_FILE, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    en_keys = en_data.get("keys", {})

    # Collect all skill.* keys from EN
    skill_keys: dict[str, str] = {}
    for key, entry in en_keys.items():
        if key.startswith("skill."):
            skill_keys[key] = entry.get("text", "")

    if not skill_keys:
        print("ERROR: No skill.* keys found in en.json", file=sys.stderr)
        return 1

    print(f"Found {len(skill_keys)} skill.* keys in EN.")

    # Compute EN hashes
    en_hashes = {key: compute_hash(text) for key, text in skill_keys.items()}

    # Process all locale files
    locale_files = sorted(LOCALES_DIR.glob("*.json"))
    total_locales_updated = 0

    for locale_file in locale_files:
        if locale_file.name.startswith("_"):
            continue

        lang_code = locale_file.stem
        if lang_code in SKIP_LOCALES:
            continue

        with open(locale_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys_section = data.setdefault("keys", {})
        added = 0

        for key, en_text in skill_keys.items():
            if key in keys_section:
                continue  # already exists

            keys_section[key] = {
                "text": en_text,
                "source_hash": en_hashes[key],
                "auto_translated": True,
                "reviewed": False,
            }
            added += 1

        if added > 0:
            with open(locale_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"  {lang_code}: added {added} keys")
            total_locales_updated += 1
        else:
            print(f"  {lang_code}: already complete")

    print(f"\nDone. {total_locales_updated} locales updated with EN fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
