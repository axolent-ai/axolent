# Dependabot Configuration

## What Dependabot Does

Dependabot is configured to automatically scan and update dependencies for this repository:

1. **Python dependencies** (`bridge/pyproject.toml`): Scanned weekly on Monday at 06:00 Europe/Berlin
2. **GitHub Actions** (all workflow files): Scanned weekly on Monday at 06:00 Europe/Berlin
3. **Security vulnerabilities**: Detected immediately, PRs opened individually regardless of schedule

## PR Grouping Strategy

To reduce noise, Dependabot groups updates:

| Update Type | Grouping | Auto-Merge | Review Required |
|-------------|----------|------------|-----------------|
| Python patch + minor | Grouped into one PR | Patch only | Minor: yes |
| Python major | Individual PRs | No | Yes (breaking changes) |
| GitHub Actions | All grouped | No | Yes |
| Security fixes | Individual PRs | No | Yes (verify fix) |

## How to Handle Dependabot PRs

### Patch Updates (auto-merged)
- The `dependabot-auto-merge.yml` workflow automatically approves and squash-merges patch-only updates
- These still pass through the full PR Quality Gate (lint, tests, security scans)
- If CI fails, auto-merge is blocked and manual intervention is needed

### Minor Updates
- Review the changelog for behavioral changes
- Run tests locally if the dependency is critical (e.g., `anthropic`, `openai`, `cryptography`)
- Merge via squash after CI passes

### Major Updates
- Always review the changelog and migration guide
- Test locally with `cd bridge && pip install -e ".[dev,test]" && pytest`
- Consider if API changes affect our code
- May require code changes before merging

## How to Pin a Dependency (Prevent Updates)

Add an `ignore` block to `.github/dependabot.yml` under the relevant ecosystem:

```yaml
updates:
  - package-ecosystem: "pip"
    directory: "/bridge"
    # ... existing config ...
    ignore:
      - dependency-name: "some-package"
        # Ignore all updates for this package:
        update-types: ["version-update:semver-major", "version-update:semver-minor", "version-update:semver-patch"]
      - dependency-name: "another-package"
        # Ignore only major updates:
        update-types: ["version-update:semver-major"]
```

## How to Disable Patch Auto-Merge

To disable auto-merge for patch updates, either:

1. **Delete** `.github/workflows/dependabot-auto-merge.yml`, or
2. **Comment out** the workflow content and commit

Auto-merge requires that branch protection rules and required status checks are configured.
Without passing CI, no PR is auto-merged regardless of this workflow.

## GitHub Vulnerability Alerts (Separate Feature)

GitHub Security Alerts are a **separate** feature from Dependabot version updates:

- They run continuously (not just weekly) regardless of `dependabot.yml`
- Visible in the repository's **Security** tab > **Dependabot alerts**
- They detect known CVEs in your dependency tree
- They can also create automated security PRs (Dependabot security updates)

Both features complement each other:
- `dependabot.yml` = proactive version freshness (weekly)
- Security Alerts = reactive vulnerability detection (continuous)

## PR Limits

| Ecosystem | Max Open PRs |
|-----------|--------------|
| Python (pip) | 5 |
| GitHub Actions | 3 |

If the limit is reached, Dependabot queues remaining updates until existing PRs are merged or closed.

## Commit Message Convention

Dependabot PRs follow the project's conventional commit format:
- Python: `chore(deps): bump <package> from X to Y`
- Actions: `chore(ci): bump <action> from X to Y`
