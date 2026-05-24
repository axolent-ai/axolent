# License Compliance (AGPL-3.0)

AXOLENT is licensed under **AGPL-3.0-only** (GNU Affero General Public License v3).
This means all dependencies must be license-compatible with AGPL-3.0.

## What AGPL-3.0 means for dependencies

AGPL-3.0 is strong copyleft. Any code linked into the project (statically or
dynamically) must be under a compatible license:

**Compatible:**
- Same family: AGPL-3.0+, GPL-3.0+, LGPL-3.0+, LGPL-2.1+
- Permissive: MIT, BSD (2/3-clause), Apache-2.0, ISC, PSF-2.0, MPL-2.0
- Public domain: CC0, Unlicense, 0BSD, Zlib, Boost

**Incompatible:**
- GPL-2.0-only (without "or later" clause)
- Apache-1.0 (patent clause conflict)
- BSD-4-Clause (advertising clause)
- CC-BY-NC (non-commercial restriction)
- Proprietary / Commercial / SSPL
- EPL-1.0, EPL-2.0, OSL-3.0

## How the scanner works

The automated scanner (`scripts/check_license_compliance.py`) runs:

1. Invokes `pip-licenses` to list all installed packages and their licenses
2. Loads the allowlist/blocklist from `scripts/license_compliance.yaml`
3. For each package, checks:
   - Is it our own package? (skip)
   - Is there a manual exception? (approved)
   - Is the license in the allowlist? (OK)
   - Is the license in the forbidden list? (FAIL)
   - Is it a compound license with all-compatible components? (OK)
   - Otherwise: WARN (needs manual review)
4. Exits 0 if no blockers, exits 1 if any forbidden license found

## Adding a new dependency

Before adding any new dependency to `pyproject.toml`:

1. Check the package's license on PyPI or GitHub
2. Verify it appears in `scripts/license_compliance.yaml` under `allowed_licenses`
3. If the license is unusual or compound, add it to `allowed_licenses` or
   `manual_exceptions` with justification
4. Run: `python scripts/check_license_compliance.py`
5. The pre-commit hook will also catch violations on commit

## Adding a manual exception

For packages with non-standard license strings that are actually compatible:

Edit `scripts/license_compliance.yaml` and add under `manual_exceptions`:

```yaml
manual_exceptions:
  package-name:
    license: "What pip-licenses reports"
    actual_license: "The real SPDX license"
    reason: "Reviewed YYYY-MM-DD, verified on GitHub/PyPI, equivalent to X"
```

This requires a human decision. Document the reasoning clearly.

## Integration points

- **Pre-commit hook:** Runs on every commit (skips if pip-licenses not installed)
- **GitHub Action:** `.github/workflows/license-check.yml` runs on PR and push to main
- **SBOM:** Generated in `docs/SBOM_YYYY-MM-DD.json` and `.md` (NIST SSDF format)

## Running manually

```bash
# Full compliance check
python scripts/check_license_compliance.py

# Generate fresh SBOM
cd bridge && pip-licenses --format json > ../docs/SBOM_$(date +%F).json
cd bridge && pip-licenses --format markdown --order=license > ../docs/SBOM_$(date +%F).md
```
