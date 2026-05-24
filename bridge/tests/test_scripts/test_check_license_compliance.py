"""Tests for the AGPL-3.0 license compliance scanner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts directory to path for import
scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(scripts_dir))

from check_license_compliance import check_license, run_compliance_check  # noqa: E402


@pytest.fixture
def base_config() -> dict:
    """Minimal config for testing."""
    return {
        "allowed_licenses": [
            "MIT",
            "MIT License",
            "BSD-3-Clause",
            "BSD License",
            "Apache-2.0",
            "Apache Software License",
            "ISC",
            "AGPL-3.0-only",
            "GPL-3.0-only",
            "LGPL-3.0-only",
            "LGPL-2.1-only",
            "MPL-2.0",
            "Python Software Foundation License",
            "PSF-2.0",
        ],
        "forbidden_licenses": [
            "GPL-2.0-only",
            "GPL-2.0",
            "Proprietary",
            "Commercial",
            "EPL-1.0",
        ],
        "own_packages": ["jarvis-bridge"],
        "manual_exceptions": {
            "special-pkg": {
                "license": "Custom",
                "actual_license": "MIT",
                "reason": "Reviewed 2026-05-24, equivalent to MIT",
            }
        },
        "allowed_compound_keywords": [
            "MIT",
            "BSD",
            "Apache",
            "ISC",
            "AGPL-3",
            "GPL-3",
            "LGPL",
            "Zlib",
            "CC0",
            "MPL",
        ],
    }


class TestCheckLicense:
    """Unit tests for the check_license function."""

    def test_allowed_license_mit(self, base_config: dict) -> None:
        """MIT license is recognized as compatible."""
        status, _ = check_license("requests", "MIT License", base_config)
        assert status == "OK"

    def test_allowed_license_apache(self, base_config: dict) -> None:
        """Apache-2.0 is recognized as compatible."""
        status, _ = check_license("httpx", "Apache-2.0", base_config)
        assert status == "OK"

    def test_forbidden_license_gpl2(self, base_config: dict) -> None:
        """GPL-2.0-only is blocked as incompatible with AGPL-3.0."""
        status, reason = check_license("bad-pkg", "GPL-2.0-only", base_config)
        assert status == "FAIL"
        assert "Forbidden" in reason

    def test_forbidden_license_proprietary(self, base_config: dict) -> None:
        """Proprietary license is blocked."""
        status, reason = check_license("closed-pkg", "Proprietary", base_config)
        assert status == "FAIL"
        assert "Forbidden" in reason

    def test_own_package_skipped(self, base_config: dict) -> None:
        """Own packages are skipped."""
        status, _ = check_license("jarvis-bridge", "UNKNOWN", base_config)
        assert status == "OWN"

    def test_manual_exception_applies(self, base_config: dict) -> None:
        """Manual exception overrides unknown license."""
        status, reason = check_license("special-pkg", "Custom", base_config)
        assert status == "EXCEPTION"
        assert "Reviewed" in reason

    def test_unknown_license_warns(self, base_config: dict) -> None:
        """UNKNOWN license generates a warning."""
        status, _ = check_license("mystery-pkg", "UNKNOWN", base_config)
        assert status == "WARN"

    def test_compound_license_all_compatible(self, base_config: dict) -> None:
        """Compound license with all compatible components passes."""
        status, _ = check_license("multi-pkg", "MIT AND BSD-3-Clause", base_config)
        assert status == "OK"


class TestRunComplianceCheck:
    """Integration tests for run_compliance_check."""

    def test_clean_repo_passes(self, base_config: dict) -> None:
        """All-compatible packages produce exit code 0."""
        packages = [
            {"Name": "requests", "Version": "2.31.0", "License": "Apache-2.0"},
            {"Name": "click", "Version": "8.1.7", "License": "BSD-3-Clause"},
            {"Name": "pydantic", "Version": "2.5.0", "License": "MIT License"},
        ]
        exit_code = run_compliance_check(packages=packages, config=base_config)
        assert exit_code == 0

    def test_blocked_gpl2_fails(self, base_config: dict) -> None:
        """GPL-2.0-only dependency causes exit code 1."""
        packages = [
            {"Name": "requests", "Version": "2.31.0", "License": "Apache-2.0"},
            {"Name": "gpl2-lib", "Version": "1.0.0", "License": "GPL-2.0-only"},
        ]
        exit_code = run_compliance_check(packages=packages, config=base_config)
        assert exit_code == 1

    def test_manual_exception_overrides_unknown(self, base_config: dict) -> None:
        """Package with manual exception passes despite non-standard license."""
        packages = [
            {"Name": "special-pkg", "Version": "3.0.0", "License": "Custom"},
            {"Name": "click", "Version": "8.1.7", "License": "BSD-3-Clause"},
        ]
        exit_code = run_compliance_check(packages=packages, config=base_config)
        assert exit_code == 0

    def test_unknown_license_warns_but_passes(self, base_config: dict) -> None:
        """UNKNOWN license warns but does not block (exit 0)."""
        packages = [
            {"Name": "mystery-pkg", "Version": "0.1.0", "License": "UNKNOWN"},
            {"Name": "click", "Version": "8.1.7", "License": "BSD-3-Clause"},
        ]
        exit_code = run_compliance_check(packages=packages, config=base_config)
        # UNKNOWN is a warning, not a blocker
        assert exit_code == 0

    def test_unknown_in_forbidden_list_fails(self, base_config: dict) -> None:
        """If UNKNOWN is in forbidden_licenses, it blocks."""
        config = {**base_config}
        config["forbidden_licenses"] = [*base_config["forbidden_licenses"], "Unknown"]
        packages = [
            {"Name": "shady-pkg", "Version": "0.1.0", "License": "Unknown"},
        ]
        exit_code = run_compliance_check(packages=packages, config=config)
        assert exit_code == 1
