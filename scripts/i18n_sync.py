#!/usr/bin/env python3
"""Sync tool: add missing keys from en.json to all locale files.

For each locale file:
  - If EN has a key that the locale doesn't: adds it with EN text,
    marks as auto_translated=true, reviewed=false, source_hash from EN.
  - If locale has a key that EN doesn't: removes it (orphan cleanup).
  - Does NOT overwrite existing translations.

Usage:
    python scripts/i18n_sync.py [--dry-run]
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
    dry_run = "--dry-run" in sys.argv

    if not EN_FILE.exists():
        print("ERROR: en.json not found", file=sys.stderr)
        return 1

    with open(EN_FILE, "r", encoding="utf-8") as f:
        en_data = json.load(f)
    en_keys = en_data.get("keys", {})

    locale_files = sorted(LOCALES_DIR.glob("*.json"))
    total_added = 0
    total_removed = 0

    for locale_file in locale_files:
        if locale_file.name.startswith("_") or locale_file.name == "en.json":
            continue

        with open(locale_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys = data.get("keys", {})
        file_changed = False

        # Add missing keys
        for key, en_entry in en_keys.items():
            if key not in keys:
                en_text = en_entry.get("text", "")
                keys[key] = {
                    "text": en_text,
                    "source_hash": compute_hash(en_text),
                    "auto_translated": True,
                    "reviewed": False,
                }
                file_changed = True
                total_added += 1
                if dry_run:
                    print(f"  [ADD] {locale_file.stem}/{key}")

        # Remove orphan keys
        orphans = [k for k in keys if k not in en_keys]
        for key in orphans:
            del keys[key]
            file_changed = True
            total_removed += 1
            if dry_run:
                print(f"  [DEL] {locale_file.stem}/{key}")

        if file_changed and not dry_run:
            data["keys"] = keys
            with open(locale_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"  Synced: {locale_file.name}")

    action = "Would sync" if dry_run else "Synced"
    print(f"\n{action}: +{total_added} added, -{total_removed} removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
