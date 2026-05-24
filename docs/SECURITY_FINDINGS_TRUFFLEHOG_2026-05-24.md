# TruffleHog Full-History Scan Report

**Date:** 2026-05-24
**Tool:** TruffleHog v3.95.3
**Repo:** axolent at HEAD ea63134 (branch: main)
**Commits scanned:** 156
**Scan duration:** ~1.4s (no-verification), ~1.5s (with verification)

## Summary

| Status | Count |
|--------|-------|
| Verified (active) | 0 |
| Unverified | 1 |
| Total | 1 |

## Verified Findings

None. No active secrets detected in git history.

## Unverified Findings

### 1. URI with embedded credentials (FALSE POSITIVE)

| Field | Value |
|-------|-------|
| Detector | URI (Type 17) |
| File | `bridge/tests/test_application/test_skill_compression/test_privacy_secrets.py` |
| Commit | `ca3c88ab461d76eaa5c7906cd8a71789d75e2fdf` |
| Author | Jessica (jessica@adagencee.com) |
| Date | 2026-05-21 |
| Verified | No (verification attempted, failed) |

**Assessment:** This is an intentional test fixture (`user:password@example.com`) used in a
unit test that validates the SecretScanner privacy filter (HC-SC-13). The domain is `example.com`
(IANA reserved, not a real service). The credentials are dummy values designed to trigger the
scanner during testing. This is a textbook false positive.

**Action required:** None. No suppression file needed since TruffleHog already marks it unverified.

## Conclusion

**Public-switch readiness from secret-scanning perspective: GREEN LIGHT.**

No real secrets (API keys, tokens, passwords, private keys) were found anywhere in the
156-commit git history. The single unverified finding is an intentional test fixture with
dummy credentials against a reserved domain.

The repository is clean for public release from a credential-leak standpoint.

## Tool Installation

- **Path:** `D:\tools\trufflehog-v3\trufflehog.exe`
- **Version:** 3.95.3
- **Source:** GitHub release (direct download)
- **Scan command (no-verify):** `trufflehog git file://. --json --no-update --no-verification`
- **Scan command (verified):** `trufflehog git file://. --json --no-update`
