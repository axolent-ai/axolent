#!/usr/bin/env python3
"""One-shot script: Add /settings v2 i18n keys to all locale files.

Adds the 25 keys needed for the new hierarchical /settings menu
(6 top-level categories: Language, Model, Debate, Rate-Limit, Personality, Timezone).

For non-EN/DE locales, copies EN text with auto_translated=true, reviewed=false.
DE translations are provided explicitly.

Run once after implementation, then commit.
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


# EN text (source of truth for hashes and non-DE non-EN locales)
NEW_KEYS_EN: dict[str, str] = {
    "settings.title": "Settings",
    "settings.btn_language": "Language",
    "settings.btn_model": "Model",
    "settings.btn_debate": "Debate",
    "settings.btn_rate_limit": "Rate Limit",
    "settings.btn_personality": "Personality",
    "settings.btn_timezone": "Timezone",
    "settings.btn_close": "Close",
    "settings.btn_back": "Back",
    "settings.btn_save": "Save",
    "settings.model.title": "Model Selection (current: {current})",
    "settings.debate.title": "Debate Providers (multi-select)",
    "settings.debate.help_multi": "Select one or more providers for /debate",
    "settings.debate.planned_toast": "Coming in Phase 1",
    "settings.personality.title": "Personality Features",
    "settings.timezone.title": "Timezone (current: {current})",
    "settings.timezone.search": "Search...",
    "settings.timezone.other": "Other...",
    "settings.toast_saved": "Saved",
    "settings.toast_provider_planned": "This provider is coming in Phase 1",
    "settings.rate_limit.title": "Rate Limit Profile (current: {current})",
    "settings.personality.p1_label": "Proactive Memory",
    "settings.personality.p2_label": "Less AI-Talk",
    "settings.personality.p3_label": "Style Adaption",
    "settings.personality.p4_label": "Confidence Signal",
    "settings.personality.p5_label": "Time Awareness",
    "settings.personality.p6_label": "Show Weakness",
}

# DE translations (reviewed, not auto-translated)
NEW_KEYS_DE: dict[str, str] = {
    "settings.title": "Einstellungen",
    "settings.btn_language": "Sprache",
    "settings.btn_model": "Modell",
    "settings.btn_debate": "Debate",
    "settings.btn_rate_limit": "Rate-Limit",
    "settings.btn_personality": "Persönlichkeit",
    "settings.btn_timezone": "Zeitzone",
    "settings.btn_close": "Schliessen",
    "settings.btn_back": "Zurück",
    "settings.btn_save": "Speichern",
    "settings.model.title": "Modell-Auswahl (aktuell: {current})",
    "settings.debate.title": "Debate-Provider (Mehrfachauswahl)",
    "settings.debate.help_multi": "Wähle einen oder mehrere Provider für /debate",
    "settings.debate.planned_toast": "Kommt in Phase 1",
    "settings.personality.title": "Persönlichkeits-Features",
    "settings.timezone.title": "Zeitzone (aktuell: {current})",
    "settings.timezone.search": "Suchen...",
    "settings.timezone.other": "Andere...",
    "settings.toast_saved": "Gespeichert",
    "settings.toast_provider_planned": "Dieser Provider kommt in Phase 1",
    "settings.rate_limit.title": "Rate-Limit-Profil (aktuell: {current})",
    "settings.personality.p1_label": "Proaktive Memory",
    "settings.personality.p2_label": "Wenig AI-Talk",
    "settings.personality.p3_label": "Stil-Adaption",
    "settings.personality.p4_label": "Confidence-Signal",
    "settings.personality.p5_label": "Time-Awareness",
    "settings.personality.p6_label": "Schwäche-Zeigung",
}


def main() -> int:
    if not EN_FILE.exists():
        print("ERROR: en.json not found", file=sys.stderr)
        return 1

    # Compute EN hashes for new keys
    en_hashes = {key: compute_hash(text) for key, text in NEW_KEYS_EN.items()}

    # Process all locale files
    locale_files = sorted(LOCALES_DIR.glob("*.json"))
    total_added = 0

    for locale_file in locale_files:
        if locale_file.name.startswith("_"):
            continue

        lang_code = locale_file.stem
        is_en = lang_code == "en"
        is_de = lang_code == "de"

        with open(locale_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys_section = data.setdefault("keys", {})
        added = 0

        for key, en_text in NEW_KEYS_EN.items():
            if key in keys_section:
                continue  # already exists, skip

            if is_en:
                entry = {
                    "text": en_text,
                    "source_hash": en_hashes[key],
                    "auto_translated": False,
                    "reviewed": True,
                }
            elif is_de:
                de_text = NEW_KEYS_DE[key]
                entry = {
                    "text": de_text,
                    "source_hash": en_hashes[key],
                    "auto_translated": False,
                    "reviewed": True,
                }
            else:
                # Non-EN, non-DE: use EN text as fallback
                entry = {
                    "text": en_text,
                    "source_hash": en_hashes[key],
                    "auto_translated": True,
                    "reviewed": False,
                }

            keys_section[key] = entry
            added += 1

        if added > 0:
            with open(locale_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"{lang_code}: added {added} keys")
            total_added += added

    print(
        f"\nDone. Total entries added: {total_added} ({len(NEW_KEYS_EN)} keys x locales)."
    )
    print(
        "Note: 18 non-EN/DE locales use EN text as placeholder (auto_translated=true)."
    )
    print(
        "Cosmo should review and translate those 18 locales before production release."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
