# CI/CD Security: zizmor + OpenSSF Scorecard

## Overview

AXOLENT uses two complementary tools to secure its GitHub Actions CI/CD pipeline:

1. **zizmor** - Static analyzer for GitHub Actions workflow files
2. **OpenSSF Scorecard** - Holistic repository security posture assessment

## zizmor (GitHub Actions SAST)

### What it checks

| Audit | Severity | Description |
|-------|----------|-------------|
| `dangerous-triggers` | High | `pull_request_target` misuse enabling code injection |
| `template-injection` | High | Untrusted input in `run:` steps via `${{ github.event.X }}` |
| `unpinned-uses` | High | Actions referenced by tag (`@v4`) instead of SHA |
| `bot-conditions` | High | Spoofable `github.actor` checks for bot verification |
| `excessive-permissions` | Medium | Overly broad or missing `permissions:` blocks |
| `artipacked` | Medium | Credential persistence via `persist-credentials: true` |
| `secrets-inherit` | Medium | Uncontrolled secrets propagation to called workflows |

### Where it runs

- **Pre-commit hook**: Blocks commits that modify `.github/workflows/*.yml` if findings exist
- **GitHub Actions**: `.github/workflows/zizmor.yml` runs on PR + push to main + weekly Monday 05:30 UTC
- **Local**: `zizmor --config .zizmor.yml .github/workflows/`

### Configuration

`.zizmor.yml` in repo root contains suppression rules for intentional patterns:
- `dangerous-triggers` suppressed for `dependabot-auto-merge.yml` (legitimate use case)

### Triaging findings

1. **HIGH**: Fix immediately. Template injection = RCE vector. Unpinned actions = supply chain risk.
2. **MEDIUM**: Fix in same PR. Excessive permissions and credential persistence are defense-in-depth.
3. **LOW**: Document reasoning in `.zizmor.yml` if suppressed.

### Fixing common findings

**Unpinned actions** - Replace tag with full SHA:
```yaml
# Before (vulnerable to tag mutation)
uses: actions/checkout@v4

# After (immutable reference)
uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5  # v4.3.1
```

**Excessive permissions** - Add explicit permissions block:
```yaml
permissions:
  contents: read  # Minimum required
```

**Artipacked (credential persistence)** - Add to checkout step:
```yaml
- uses: actions/checkout@...
  with:
    persist-credentials: false
```

**Bot conditions** - Use event payload instead of actor context:
```yaml
# Before (spoofable)
if: github.actor == 'dependabot[bot]'

# After (verified from event)
if: github.event.pull_request.user.login == 'dependabot[bot]'
```

## OpenSSF Scorecard

### What it measures

Scorecard assigns a 0-10 score across multiple checks:

| Check | What it evaluates |
|-------|-------------------|
| Branch-Protection | Required reviews, status checks, admin enforcement |
| Code-Review | PRs reviewed before merge |
| CII-Best-Practices | OpenSSF Best Practices badge |
| Vulnerabilities | Known CVEs in dependencies |
| License | OSI-approved license present |
| Maintained | Recent commits, issue response time |
| SAST | Static analysis tools configured (CodeQL, semgrep) |
| Token-Permissions | Workflow permissions follow least-privilege |
| Pinned-Dependencies | Actions and container images pinned by hash |
| Dangerous-Workflow | Risky patterns in CI configuration |
| Dependency-Update-Tool | Dependabot or Renovate configured |
| Fuzzing | Fuzzing infrastructure present |
| Security-Policy | SECURITY.md present |
| Signed-Releases | Release artifacts cryptographically signed |

### Where it runs

- `.github/workflows/scorecard.yml`: Weekly Monday 04:15 UTC + on push to main + branch protection changes
- Results published to GitHub Security tab (SARIF format)
- Results published to OpenSSF Best Practices dashboard

### Score interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 8-10 | Excellent security posture | Maintain |
| 6-7 | Good, minor improvements possible | Address low-hanging fruit |
| 4-5 | Moderate risk | Prioritize fixes within 2 weeks |
| 0-3 | Significant gaps | Immediate remediation required |

### Improving your score

Common improvements ordered by impact:

1. **Enable Branch Protection** (Settings > Branches > Branch protection rules)
   - Require PR reviews (min 1 reviewer)
   - Require status checks to pass
   - Require signed commits (optional, high effort)

2. **Pin all dependencies by SHA** (already done for Actions)

3. **Add SECURITY.md** with vulnerability disclosure policy

4. **Maintain Dependabot** (already active)

5. **Run SAST** (CodeQL, semgrep, bandit already configured)

6. **Add CII Best Practices badge** (self-assessment questionnaire at bestpractices.coreinfrastructure.org)

## Quick reference

```bash
# Run zizmor locally
zizmor --config .zizmor.yml .github/workflows/

# Run with verbose output
zizmor --config .zizmor.yml --format=json .github/workflows/

# Check specific file
zizmor .github/workflows/pr-check.yml
```
