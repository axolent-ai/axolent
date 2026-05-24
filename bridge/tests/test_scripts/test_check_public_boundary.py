"""Tests for scripts/check_public_boundary.py (Public/Private Boundary Scanner)."""

from __future__ import annotations

from pathlib import Path

import pytest

# We need to import from the scripts directory
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_public_boundary import (  # noqa: E402
    check_env_var_secrets,
    check_forbidden_content,
    check_forbidden_paths,
    is_dummy_value,
    main,
    matches_glob_pattern,
)


class TestMatchesGlobPattern:
    """Tests for the glob matching logic."""

    def test_simple_file_match(self):
        assert matches_glob_pattern("README.md", "README.md")

    def test_directory_wildcard(self):
        assert matches_glob_pattern("bridge/main.py", "bridge/**")
        assert matches_glob_pattern("bridge/sub/deep.py", "bridge/**")

    def test_double_star_prefix(self):
        assert matches_glob_pattern("bridge/.env", "**/.env")
        assert matches_glob_pattern("deep/nested/.env", "**/.env")
        assert matches_glob_pattern(".env", "**/.env")

    def test_double_star_extension(self):
        assert matches_glob_pattern("data/test.db", "**/*.db")
        assert matches_glob_pattern("bridge/data/jarvis.db", "**/*.db")

    def test_double_star_directory(self):
        assert matches_glob_pattern("src/private/secret.py", "**/private/**")
        assert matches_glob_pattern("private/file.txt", "**/private/**")

    def test_no_match(self):
        assert not matches_glob_pattern("bridge/main.py", "desktop/**")
        assert not matches_glob_pattern("main.py", "**/.env")


class TestIsDummyValue:
    """Tests for dummy value detection."""

    def test_empty_is_dummy(self):
        assert is_dummy_value("")
        assert is_dummy_value("   ")

    def test_placeholder_values(self):
        assert is_dummy_value("your_bot_token_here")
        assert is_dummy_value("YOUR_TELEGRAM_BOT_TOKEN_HERE")
        assert is_dummy_value("REPLACE_ME")
        assert is_dummy_value("xxx")
        assert is_dummy_value("...")
        assert is_dummy_value("placeholder_value")

    def test_real_values_not_dummy(self):
        # A real Telegram bot token pattern
        assert not is_dummy_value("123456789:ABCdefGHIjklMNOpqrSTUvwxYZ")
        # A real-looking API key
        assert not is_dummy_value("sk-ant-api03-realkey1234567890abcdef")


class TestCheckForbiddenPaths:
    """Tests for forbidden path detection."""

    def test_env_file_blocked(self):
        tracked = ["bridge/.env", "bridge/main.py"]
        patterns = ["**/.env"]
        result = check_forbidden_paths(tracked, patterns)
        assert len(result) == 1
        assert result[0][0] == "bridge/.env"
        assert result[0][1] == "**/.env"

    def test_db_file_blocked(self):
        tracked = ["bridge/data/jarvis.db", "bridge/main.py"]
        patterns = ["**/*.db"]
        result = check_forbidden_paths(tracked, patterns)
        assert len(result) == 1
        assert result[0][0] == "bridge/data/jarvis.db"

    def test_clean_files_pass(self):
        tracked = ["bridge/main.py", "docs/README.md"]
        patterns = ["**/.env", "**/*.db"]
        result = check_forbidden_paths(tracked, patterns)
        assert len(result) == 0


class TestCheckForbiddenContent:
    """Tests for forbidden content pattern detection."""

    def test_token_content_blocked(self, tmp_path):
        """A file containing a real Telegram token pattern is blocked."""
        test_file = tmp_path / "bad.py"
        test_file.write_text(
            'TOKEN = "123"\nTELEGRAM_BOT_TOKEN = 123456789:ABCdefGHI_jklMNO\n# end\n'
        )
        patterns = [r"TELEGRAM_BOT_TOKEN\s*=\s*\d+:[A-Za-z0-9_-]+"]
        # Use relative path simulation
        tracked = ["bad.py"]
        result = check_forbidden_content(tracked, tmp_path, patterns, whitelist=[])
        assert len(result) == 1
        assert result[0][0] == "bad.py"
        assert result[0][1] == 2  # line number

    def test_whitelisted_file_passes(self, tmp_path):
        """A whitelisted file is not scanned for content patterns."""
        test_file = tmp_path / "scanner.py"
        test_file.write_text(
            "# This file contains patterns for checking:\n"
            "# TELEGRAM_BOT_TOKEN = 123456789:ABCdefGHI_jklMNO\n"
        )
        patterns = [r"TELEGRAM_BOT_TOKEN\s*=\s*\d+:[A-Za-z0-9_-]+"]
        tracked = ["scanner.py"]
        result = check_forbidden_content(
            tracked, tmp_path, patterns, whitelist=["scanner.py"]
        )
        assert len(result) == 0

    def test_brand_internal_term_blocked(self, tmp_path):
        """Brand-internal terms are blocked."""
        test_file = tmp_path / "readme.md"
        test_file.write_text("This uses Semantic Bridge internally.\n")
        patterns = [r"Semantic[\s-]?Bridge"]
        tracked = ["readme.md"]
        result = check_forbidden_content(tracked, tmp_path, patterns, whitelist=[])
        assert len(result) == 1


class TestCheckEnvVarSecrets:
    """Tests for env var secret detection with dummy value handling."""

    def test_real_token_blocked(self, tmp_path):
        """A real-looking token value is blocked."""
        test_file = tmp_path / ".env"
        test_file.write_text("TELEGRAM_BOT_TOKEN=123456789:ABCdef_GHI-jkl\n")
        tracked = [".env"]
        result = check_env_var_secrets(tracked, tmp_path, whitelist=[])
        assert len(result) == 1

    def test_dummy_token_passes(self, tmp_path):
        """A dummy/placeholder token value passes."""
        test_file = tmp_path / ".env.example"
        test_file.write_text("TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE\n")
        tracked = [".env.example"]
        result = check_env_var_secrets(tracked, tmp_path, whitelist=[])
        assert len(result) == 0

    def test_empty_value_passes(self, tmp_path):
        """An empty env var value passes."""
        test_file = tmp_path / ".env.example"
        test_file.write_text("SENTRY_DSN=\n")
        tracked = [".env.example"]
        result = check_env_var_secrets(tracked, tmp_path, whitelist=[])
        assert len(result) == 0


class TestMainIntegration:
    """Integration test running the scanner against the real repo."""

    def test_real_repo_passes(self):
        """The actual AXOLENT repo passes the boundary scanner."""
        config_path = REPO_ROOT / "scripts" / "public_boundary.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found (not in repo root)")
        exit_code = main(config_path=str(config_path))
        assert exit_code == 0, "Real repo should pass boundary scanner"
