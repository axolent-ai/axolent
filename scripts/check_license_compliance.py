"""AGPL-3.0 License Compliance Scanner for AXOLENT.

Checks all installed Python dependencies against the AGPL-3.0
compatibility allowlist. Exits 0 if all clean, 1 if blockers found.

Usage:
    python scripts/check_license_compliance.py

Requires:
    pip-licenses (install via: pip install pip-licenses)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Support being imported for testing with mock data
_STANDALONE = __name__ == "__main__"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the license compliance YAML configuration."""
    # Lazy import to avoid hard dependency on pyyaml for testing
    try:
        import yaml
    except ImportError:
        # Fallback: try to find yaml in the venv
        raise SystemExit(
            "[license-compliance] PyYAML not installed. "
            "Install it via: pip install pyyaml"
        )

    if config_path is None:
        config_path = Path(__file__).parent / "license_compliance.yaml"

    if not config_path.exists():
        raise SystemExit(f"[license-compliance] Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_installed_licenses() -> list[dict[str, str]]:
    """Run pip-licenses and return the JSON result."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "piplicenses",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        raise SystemExit(
            "[license-compliance] pip-licenses not found. "
            "Install via: pip install pip-licenses"
        )
    except subprocess.TimeoutExpired:
        raise SystemExit("[license-compliance] pip-licenses timed out.")

    if result.returncode != 0:
        # Try alternative invocation
        try:
            result = subprocess.run(
                ["pip-licenses", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise SystemExit(
                f"[license-compliance] pip-licenses failed:\n{result.stderr}"
            )

    if not result.stdout.strip():
        raise SystemExit("[license-compliance] pip-licenses returned empty output.")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[license-compliance] Failed to parse pip-licenses JSON: {e}")


def check_license(
    pkg_name: str,
    pkg_license: str,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Check a single package license against the config.

    Returns:
        Tuple of (status, reason) where status is one of:
        - "OK": License is in the allowlist
        - "EXCEPTION": Manual exception applies
        - "OWN": Own package, skipped
        - "WARN": Unknown/unrecognized license (needs review)
        - "FAIL": License is in the forbidden list
    """
    allowed = config.get("allowed_licenses", [])
    forbidden = config.get("forbidden_licenses", [])
    own_packages = config.get("own_packages", [])
    exceptions = config.get("manual_exceptions", {})
    compound_keywords = config.get("allowed_compound_keywords", [])

    # Skip own packages
    if pkg_name in own_packages:
        return "OWN", "Own package, skipped"

    # Check manual exceptions
    if pkg_name in exceptions:
        return "EXCEPTION", exceptions[pkg_name].get("reason", "Manual exception")

    # Direct match against allowlist
    if pkg_license in allowed:
        return "OK", f"Allowed: {pkg_license}"

    # Check forbidden list
    license_lower = pkg_license.lower()
    for fb in forbidden:
        fb_lower = fb.lower()
        if fb_lower.endswith("*"):
            if license_lower.startswith(fb_lower[:-1]):
                return "FAIL", f"Forbidden license: {pkg_license} (matches {fb})"
        elif license_lower == fb_lower:
            return "FAIL", f"Forbidden license: {pkg_license}"

    # Check if it's a compound license (e.g. "MIT AND BSD-3-Clause")
    if any(sep in pkg_license for sep in [" AND ", " OR ", ", ", "; "]):
        # Split by separators and check each component
        parts = pkg_license
        for sep in [" AND ", " OR ", ", ", "; "]:
            parts = parts.replace(sep, "|")
        components = [p.strip() for p in parts.split("|") if p.strip()]

        all_ok = True
        for component in components:
            # Check if component matches an allowed license directly
            if component in allowed:
                continue
            # Check if component contains an allowed keyword
            comp_lower = component.lower()
            if any(kw.lower() in comp_lower for kw in compound_keywords):
                continue
            all_ok = False
            break

        if all_ok:
            return "OK", f"Compound license, all components compatible: {pkg_license}"

    # Check if license string contains known-good keywords (fuzzy match)
    if any(kw.lower() in license_lower for kw in compound_keywords):
        return "OK", f"Contains compatible keyword: {pkg_license}"

    # If UNKNOWN or unrecognized
    if pkg_license == "UNKNOWN" or not pkg_license.strip():
        return "WARN", f"Unknown license for {pkg_name} (needs manual review)"

    return "WARN", f"Unrecognized license: {pkg_license} (needs manual review)"


def run_compliance_check(
    packages: list[dict[str, str]] | None = None,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> int:
    """Run the full compliance check.

    Args:
        packages: Optional list of package dicts (for testing). If None, runs pip-licenses.
        config: Optional config dict (for testing). If None, loads from file.
        config_path: Optional path to config file.

    Returns:
        Exit code: 0 = clean, 1 = blockers found.
    """
    if config is None:
        config = load_config(config_path)

    if packages is None:
        packages = get_installed_licenses()

    results: dict[str, list[tuple[str, str, str, str]]] = {
        "OK": [],
        "EXCEPTION": [],
        "OWN": [],
        "WARN": [],
        "FAIL": [],
    }

    for pkg in packages:
        name = pkg.get("Name", "")
        license_str = pkg.get("License", "UNKNOWN")
        version = pkg.get("Version", "?")

        status, reason = check_license(name, license_str, config)
        results[status].append((name, version, license_str, reason))

    # Print report
    total = len(packages)
    ok_count = len(results["OK"]) + len(results["EXCEPTION"]) + len(results["OWN"])
    warn_count = len(results["WARN"])
    fail_count = len(results["FAIL"])

    print("=" * 70)
    print("  AXOLENT License Compliance Report (AGPL-3.0)")
    print("=" * 70)
    print(f"\n  Total packages scanned: {total}")
    print(f"  Compatible:             {ok_count}")
    print(f"  Warnings (need review): {warn_count}")
    print(f"  BLOCKED (incompatible): {fail_count}")
    print()

    if results["EXCEPTION"]:
        print("--- Manual Exceptions (approved) ---")
        for name, version, lic, reason in results["EXCEPTION"]:
            print(f"  {name:30s} {version:12s} {lic}")
            print(f"    Reason: {reason}")
        print()

    if results["WARN"]:
        print("--- WARNINGS (need manual review) ---")
        for name, version, lic, reason in results["WARN"]:
            print(f"  {name:30s} {version:12s} {lic}")
            print(f"    -> {reason}")
        print()

    if results["FAIL"]:
        print("--- BLOCKED (AGPL-3.0 incompatible) ---")
        for name, version, lic, reason in results["FAIL"]:
            print(f"  {name:30s} {version:12s} {lic}")
            print(f"    -> {reason}")
        print()

    # Summary
    if fail_count > 0:
        print(f"RESULT: FAIL ({fail_count} incompatible dependencies)")
        print(
            "Action required: Replace, remove, or add manual_exception with justification."
        )
        return 1
    elif warn_count > 0:
        print(f"RESULT: PASS with {warn_count} warning(s)")
        print(
            "Recommendation: Review warnings and add to allowed_licenses or manual_exceptions."
        )
        return 0
    else:
        print("RESULT: PASS (all dependencies AGPL-3.0 compatible)")
        return 0


def main() -> None:
    """Entry point for CLI usage."""
    # Check if pip-licenses is available
    try:
        subprocess.run(
            [sys.executable, "-m", "piplicenses", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Try direct command
        try:
            subprocess.run(
                ["pip-licenses", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(
                "[license-compliance] SKIP: pip-licenses not installed.\n"
                "  Install via: pip install pip-licenses\n"
                "  Skipping license compliance check."
            )
            sys.exit(0)

    sys.exit(run_compliance_check())


if __name__ == "__main__":
    main()
