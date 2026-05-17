#!/usr/bin/env python3
"""One-shot script: Add rate-limit and usage i18n keys to all locale files.

Adds the keys needed for Blocker 7 (handlers.py rate-limit/usage on t()).
For non-EN locales, copies EN text with auto_translated=true, reviewed=false.

Run once, then delete this script.
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


# New keys with EN text
NEW_KEYS: dict[str, str] = {
    "rate_limit.reset_minute": "Reset in {seconds}s",
    "rate_limit.reset_hour": "Reset in {minutes} minutes",
    "rate_limit.reset_day": "Reset in {hours}h",
    "rate_limit.period_minute": "minute",
    "rate_limit.period_hour": "hour",
    "rate_limit.period_day": "day",
    "rate_limit.window_minute": "this minute",
    "rate_limit.window_hour": "this hour",
    "rate_limit.window_day": "today",
    "rate_limit.options_light": (
        "You can change your limit anytime for free:\n"
        "• /usage — current overview\n"
        "• /setlimit normal — more headroom (350/h, 1,500/day)"
    ),
    "rate_limit.options_normal": (
        "You can change your limit anytime for free:\n"
        "• /usage — current overview\n"
        "• /setlimit power — much more headroom (900/h, 10,000/day)"
    ),
    "rate_limit.options_power": (
        "• /usage — current overview\n• /setlimit unlimited — disable all limits"
    ),
    "rate_limit.upgrade_light": "Want to do more? /setlimit normal raises the limit to 350/h.",
    "rate_limit.upgrade_normal": "Want to do more? /setlimit power raises the limit to 900/h.",
    "rate_limit.upgrade_power": "Change profile: /setlimit",
    "usage.body": (
        "Profile: {profile}\n\n"
        "This minute: {min_used}/{min_limit} {min_bar} (Reset in {min_reset})\n"
        "This hour: {hour_used}/{hour_limit} {hour_bar} (Reset in {hour_reset})\n"
        "Today: {day_used}/{day_limit} {day_bar} (Reset at {day_reset})\n\n"
        "Change profile: /setlimit <light|normal|power|unlimited>"
    ),
}


def main() -> int:
    if not EN_FILE.exists():
        print("ERROR: en.json not found", file=sys.stderr)
        return 1

    # Compute EN hashes for new keys
    en_hashes = {key: compute_hash(text) for key, text in NEW_KEYS.items()}

    # Process all locale files
    locale_files = sorted(LOCALES_DIR.glob("*.json"))
    for locale_file in locale_files:
        if locale_file.name.startswith("_"):
            continue

        lang_code = locale_file.stem
        is_en = lang_code == "en"

        with open(locale_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys_section = data.setdefault("keys", {})
        added = 0

        for key, en_text in NEW_KEYS.items():
            if key in keys_section:
                continue  # already exists

            entry = {
                "text": en_text,
                "source_hash": en_hashes[key],
            }
            if not is_en:
                entry["auto_translated"] = True
                entry["reviewed"] = False
            else:
                entry["auto_translated"] = False
                entry["reviewed"] = True

            keys_section[key] = entry
            added += 1

        if added > 0:
            with open(locale_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"{lang_code}: added {added} keys")

    print(f"\nDone. Added {len(NEW_KEYS)} new keys across all locales.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
