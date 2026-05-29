"""Pre-commit hook: enforce English-only in production code.

Blocks commits that introduce German text in production source files.
This is the structural counterpart to AXOLENT AI's EN-only policy
defined in CLAUDE.md.

Scope (German blocked):
    bridge/application/**/*.py
    bridge/domain/**/*.py
    bridge/infrastructure/**/*.py
    bridge/presentation/**/*.py
    scripts/**/*.py (except check_no_fake_umlauts.py)
    bridge/config/**.md (system_prompt.example, user_constitution.example)
    bridge/.env.example
    .pre-commit-config.yaml
    start-bot.bat
    bridge/pyproject.toml

Whitelist (German allowed):
    bridge/config/task_slots.yaml      (German keywords are a feature)
    scripts/check_no_fake_umlauts.py   (wordlist with German stems)
    bridge/tests/**                    (test docstrings, future hardening)
    docs/**                            (historical context, removed via filter-repo)
    README.md, bridge/README.md        (already EN, this hook double-checks)
    CLAUDE.md                          (already EN)

Detection (Tier 1, strict):
    - German umlauts (ae/oe/ue/AE/OE/UE/ss already covered by no-fake-umlauts;
      THIS hook checks for real umlauts in strings/comments: ä ö ü ß Ä Ö Ü)
    - German marker words with word boundaries: die, der, das, und, oder,
      nicht, eine, einen, eines, einer, mit, fuer, aus, auf, ist, sind,
      werden, wird, dem, des, von, im, am, zu, kann, koennen, soll, sollen,
      muss, muessen, hat, haben, war, waren

Exit codes:
    0 = clean (no German detected in production files)
    1 = German detected (commit blocked)
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from typing import Iterable

# Force UTF-8 stdout on Windows so umlauts and emojis don't crash the hook.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# Scope: files to check
# ---------------------------------------------------------------

PRODUCTION_GLOBS = [
    "bridge/application/**/*.py",
    "bridge/domain/**/*.py",
    "bridge/infrastructure/**/*.py",
    "bridge/presentation/**/*.py",
    "bridge/main.py",
    "scripts/*.py",
    "bridge/config/system_prompt.example.md",
    "bridge/config/user_constitution.example.md",
    "bridge/.env.example",
    ".pre-commit-config.yaml",
    "start-bot.bat",
    "bridge/pyproject.toml",
]

WHITELIST_PATHS = {
    # German keywords for TaskRouter slot classification (feature, not a bug).
    "bridge/config/task_slots.yaml",
    # Contains German word stems as regex patterns (the detection target IS German).
    "scripts/check_no_fake_umlauts.py",
    # Self-exclude: this script contains German detection patterns by definition.
    "scripts/check_en_only_production.py",
    # Commit-msg hook: German stop-words and umlaut patterns as detection data.
    "scripts/check_commit_message_english_only.py",
    # Contains German umlaut patterns and ASCII fallback mappings (detection target IS German).
    "scripts/check_umlauts_in_obsidian.py",
    # Public documentation: already EN, hook double-checks for regressions.
    "README.md",
    "bridge/README.md",
    "CLAUDE.md",
    # i18n: wizard texts, welcome messages, button labels in 20 languages.
    "bridge/domain/onboarding.py",
    # i18n: centralized i18n strings for all 20 languages (commands, status, etc.).
    "bridge/domain/i18n.py",
    # i18n: German word markers for language detection heuristic (feature data).
    "bridge/domain/language.py",
    # i18n: LanguageRegistry marker_words per language (feature data, not prose).
    "bridge/application/language/registry.py",
    # i18n: self-awareness block has DE + EN branches (lang="de" / lang="en").
    "bridge/domain/personality.py",
    # i18n: HELP_TEXT_DE, _RESET_TEXTS, _MODEL_STRINGS, /lang names (DE + EN).
    "bridge/presentation/handlers.py",
    # i18n: _SETTINGS_STRINGS dict with DE + EN translations for /settings menu.
    "bridge/presentation/settings_callbacks.py",
    # i18n: calls domain.onboarding i18n functions; no own DE strings.
    "bridge/presentation/onboarding_callbacks.py",
    # Bookmark callbacks: no own DE strings, but whitelisted as presentation layer.
    "bridge/presentation/callbacks.py",
    # Render logic: no DE strings, whitelisted for safety (Markdown conversion).
    "bridge/presentation/render.py",
    # Feature data: _STOP_WORDS_DE for keyword extraction (the words ARE the feature).
    "bridge/application/chat_service.py",
    # Feature data: German nudge texts for proactive triggers (user-facing reminder strings).
    "bridge/application/proactive_trigger_service.py",
    # Feature data: German formality detection words (du/Sie) and English code-switching markers.
    "bridge/application/style_adaption_service.py",
    # Feature data: _UMLAUT_REPLACEMENTS, _SS_TO_ESZETT_WORDS for German input
    # normalization. Umlauts in code ARE the feature.
    "bridge/application/task_router.py",
    # Feature data: German keyword patterns for intent/domain/correction classification.
    # The German words in regex patterns ARE the feature (multilingual NLU heuristics).
    "bridge/application/skill_compression/event_normalizer.py",
    # Feature data: German healthcare/clinical keywords for privacy filter (HC-SC-14).
    # The German keywords ARE the detection target (bilingual keyword matching).
    "bridge/application/skill_compression/privacy/healthcare_filter.py",
    # Feature data: German PII/secret descriptions for user feedback (HC-SC-13).
    "bridge/application/skill_compression/privacy/secret_scanner.py",
    # Feature data: German nudge-policy descriptions and regex patterns (HC-SC-15).
    "bridge/application/skill_compression/privacy/nudge_filter.py",
    # i18n: Step 10 complete. Remaining: _QUESTION_TYPE_ALIASES contains German
    # keywords as feature data (dict keys like "nötig", "warum"). Not user-facing prose.
    "bridge/presentation/skill_commands.py",
    # i18n: Step 10 complete. Remaining: render_skill_detail_text field labels
    # ("Beschreibung:", "Belege:", "Geltungsbereich:") contain umlauts. TODO: Phase D full label i18n.
    "bridge/presentation/skill_profile_view.py",
    # i18n: German user-facing explainer texts (8 question types). Full i18n migration in Step 10.
    "bridge/application/skill_compression/skill_explainer.py",
    # i18n: German install-hint text for optional BERTopic dependency.
    "bridge/application/skill_compression/topic_classifier.py",
    # i18n: _STATUS_TEXTS dict with DE + EN status messages (memory, thinking).
    "bridge/application/status_manager.py",
    # one-shot i18n add-script: contains DE translation strings as data literals.
    "scripts/i18n_add_settings_v2_keys.py",
    # one-shot i18n translate-script: contains 18 non-EN languages as data literals.
    "scripts/i18n_translate_settings_v2.py",
    # Smoke test: contains German user messages as test input data.
    "scripts/smoke_test.py",
    # License compliance: SPDX identifiers (e.g. "MIT") trigger false-positives
    # on German marker word "mit". Content is purely license names + logic.
    "scripts/check_license_compliance.py",
    # License compliance YAML: same issue (SPDX identifiers + comments).
    "scripts/license_compliance.yaml",
    # Feature data: German PII/secret descriptions and i18n label keys (HC-SC-13).
    # SecretScanner is the canonical location in application/security/.
    "bridge/application/security/secret_scanner.py",
    # Feature data: German trigger-extraction regex patterns and stoplist for
    # skill alias matching (DE "wenn ich X sage" pattern). Words ARE the feature.
    "bridge/application/skill_compression/skill_learning_service.py",
    # Feature data: German trigger-extraction regex patterns for ContractBuilder
    # (DE "wenn ich X sage/schreibe" pattern). The German words ARE the feature.
    "bridge/application/skill_compression/contract_builder.py",
    # Feature data: Memory-conflict detection with German regex patterns
    # (DE "Mein/Meine X ist Y" patterns). The German words ARE the feature.
    "bridge/application/memory_conflict_detector.py",
    # Feature data: Sharp-s / double-s normalization for German alias matching.
    # The sharp-s character in REPLACE() SQL and Python code IS the feature.
    "bridge/application/skill_compression/skill_matcher.py",
}

WHITELIST_DIR_PREFIXES = (
    "bridge/tests/",
    "docs/",
    "bridge/data/",
    "bridge/logs/",
    "bridge/.venv/",
    # Text Guard core engine: docstring examples, word-pair data, CLI help texts,
    # and YAML rule files all contain intentional German (and other language)
    # text as detection targets. The module is self-referential by design.
    "bridge/domain/text_guard/",
)

# ---------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------

# Real umlauts in any context (strings, comments, docstrings).
# These cannot appear in identifiers (Python doesn't allow them in usual
# code style), so any hit is German text.
UMLAUT_PATTERN = re.compile(r"[äöüßÄÖÜ]")

# German marker words with word boundaries.
# Each entry is a regex pattern that matches the word as a standalone token.
GERMAN_MARKERS = [
    r"\bdie\b",
    r"\bder\b",
    r"\bdas\b",
    r"\bund\b",
    r"\boder\b",
    r"\bnicht\b",
    r"\beine\b",
    r"\beinen\b",
    r"\beines\b",
    r"\beiner\b",
    r"\bein\b",
    r"\bmit\b",
    r"\baus\b",
    r"\bauf\b",
    r"\bist\b",
    r"\bsind\b",
    r"\bwerden\b",
    r"\bwird\b",
    r"\bdem\b",
    r"\bdes\b",
    r"\bvon\b",
    r"\bkann\b",
    r"\bkoennen\b",
    r"\bsoll\b",
    r"\bsollen\b",
    r"\bmuss\b",
    r"\bmuessen\b",
    r"\bhat\b",
    r"\bhaben\b",
    r"\bwar\b",
    r"\bwaren\b",
    r"\bauch\b",
    r"\bwenn\b",
    r"\bdann\b",
    r"\bdoch\b",
    r"\bnoch\b",
    r"\bnur\b",
    r"\bsehr\b",
    r"\bsich\b",
    r"\bbei\b",
    r"\bnach\b",
    r"\bvor\b",
    r"\bueber\b",
    r"\bzwischen\b",
    r"\bohne\b",
    r"\bgegen\b",
    r"\bdurch\b",
]

GERMAN_MARKER_PATTERN = re.compile("|".join(GERMAN_MARKERS), re.IGNORECASE)

# Case-sensitive token whitelist: tokens that match a German marker pattern
# (case-insensitive) but are NOT German. Each entry is the exact matched text.
# Example: "MIT" is the SPDX license identifier, not the German word "mit".
FALSE_POSITIVE_TOKENS = frozenset({"MIT"})


# ---------------------------------------------------------------
# Whitelist helpers
# ---------------------------------------------------------------


def _is_whitelisted(path: Path) -> bool:
    """Return True if the given path is exempt from the EN check."""
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Path is outside the repo root (e.g. tmp file passed explicitly).
        # Not whitelisted by default.
        return False

    if rel in WHITELIST_PATHS:
        return True

    for prefix in WHITELIST_DIR_PREFIXES:
        if rel.startswith(prefix):
            return True

    return False


# ---------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------


def _collect_files(explicit_paths: Iterable[str]) -> list[Path]:
    """Resolve the list of files to scan.

    If explicit paths are given (from pre-commit), use those.
    Otherwise glob the full production scope.
    """
    files: list[Path] = []

    if explicit_paths:
        for raw in explicit_paths:
            p = Path(raw)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.is_file():
                files.append(p)
        return files

    for pattern in PRODUCTION_GLOBS:
        files.extend(REPO_ROOT.glob(pattern))

    return sorted(set(files))


# ---------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Scan a single file for German markers.

    Returns:
        List of (line_number, matched_token, line_excerpt) tuples.
        Empty list means file is clean.
    """
    hits: list[tuple[int, str, str]] = []

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        # Skip lines that are clearly URLs or imports (low false-positive risk).
        if line.startswith(("http://", "https://", "git@")):
            continue

        # Inline suppress marker allows intentional German terms in English
        # comments (e.g. quoting a concept). Marker: noqa + colon + en-only.
        if "noqa: en-only" in line:
            continue

        # Check umlauts
        m_umlaut = UMLAUT_PATTERN.search(line)
        if m_umlaut:
            hits.append((line_no, m_umlaut.group(0), line[:120]))
            continue  # one hit per line is enough to flag

        # Check German marker words
        m_marker = GERMAN_MARKER_PATTERN.search(line)
        if m_marker and m_marker.group(0) not in FALSE_POSITIVE_TOKENS:
            hits.append((line_no, m_marker.group(0), line[:120]))

    return hits


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main(argv: list[str]) -> int:
    explicit = argv[1:] if len(argv) > 1 else []
    files = _collect_files(explicit)

    if not files:
        print("check_en_only_production: no files to scan")
        return 0

    failures: dict[Path, list[tuple[int, str, str]]] = {}

    for path in files:
        if _is_whitelisted(path):
            continue
        hits = _scan_file(path)
        if hits:
            failures[path] = hits

    if not failures:
        return 0

    print("check_en_only_production: German text found in production files")
    print("(AXOLENT AI is English-only per CLAUDE.md)")
    print()

    total_hits = 0
    for path, hits in sorted(failures.items()):
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = str(path)
        print(f"{rel}")
        for line_no, token, excerpt in hits:
            print(f"  {line_no}: '{token}'  ->  {excerpt}")
            total_hits += 1
        print()

    print(f"Total: {total_hits} German hits across {len(failures)} files")
    print()
    print("Fix: translate the lines above to English.")
    print("If a German term is intentional (e.g. proper noun, quote, glossary),")
    print("move it into the whitelist in scripts/check_en_only_production.py.")

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
