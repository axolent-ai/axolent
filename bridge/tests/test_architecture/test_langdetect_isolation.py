"""Architecture guard: langdetect must stay an implementation detail
of the ResponseLanguageVerifier backend, never imported elsewhere.

Codex blocker rule (2026-05-20): direct langdetect imports outside
the verifier backend re-create the _CHAR_HINTS distributed-state
anti-pattern that caused multi-day language-bug hunts.
"""

from pathlib import Path
import re


def test_langdetect_only_imported_in_backends_module():
    """langdetect must only appear in application/language/backends.py."""
    repo_root = Path(__file__).resolve().parents[2]  # bridge/

    pattern = re.compile(r"^(from langdetect|import langdetect)", re.MULTILINE)
    forbidden_hits: list[str] = []
    allowed_path = "application/language/backends.py"

    for py_file in repo_root.rglob("*.py"):
        # Skip caches and venv
        rel = str(py_file.relative_to(repo_root))
        if any(skip in rel for skip in [".venv", "__pycache__", ".pytest_cache"]):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        if pattern.search(content):
            normalized = rel.replace("\\", "/")
            if normalized != allowed_path:
                forbidden_hits.append(normalized)

    assert not forbidden_hits, (
        f"langdetect imported outside {allowed_path}: {forbidden_hits}. "
        f"Per Codex blocker rule from 2026-05-20, langdetect is an "
        f"implementation detail of the LangdetectBackend only. All "
        f"other code must speak to the LanguageDetectorBackend Protocol, "
        f"never to langdetect directly."
    )


def test_domain_language_not_used_in_output_verifier():
    """domain.language is for short user-input detection only.

    The ResponseLanguageVerifier and StreamGuard must NOT import
    domain.language directly -- they must go through the backend abstraction.
    """
    repo_root = Path(__file__).resolve().parents[2]

    forbidden_files = [
        repo_root / "application" / "language" / "verifier.py",
        repo_root / "application" / "language" / "stream_guard.py",
    ]

    pattern = re.compile(r"from domain\.language|import domain\.language")
    leaks = [
        str(f.relative_to(repo_root))
        for f in forbidden_files
        if f.exists() and pattern.search(f.read_text(encoding="utf-8"))
    ]

    assert not leaks, (
        f"domain.language imported in output-verification code: {leaks}. "
        f"That detector is calibrated for short user inputs, not long "
        f"LLM outputs. Use the LanguageDetectorBackend abstraction instead."
    )
