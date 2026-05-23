## Description
<!-- What changed? Why? -->

## Bug class addressed
<!-- Check all that apply, leave unchecked if N/A -->
- [ ] Wiring bug (code exists but not active in production path)
- [ ] API mismatch (signature/kwarg names)
- [ ] Phase 0 stub (not-yet-implemented code path)
- [ ] UI rendering (Telegram-side display)
- [ ] State edge case (sticky language, memory)
- [ ] Performance / scaling
- [ ] Security / privacy
- [ ] Other: ...

## Tests added
<!-- Required for every feature -->
- [ ] Unit tests for new components
- [ ] **Production-path test** (uses real wrappers, not mocks)
- [ ] Architecture guard (if new main.py-wired component)
- [ ] Edge cases covered

## Architecture compliance
- [ ] import-linter: 0 contracts broken
- [ ] Semgrep: no new ERROR-severity findings
- [ ] beartype: no new untyped function signatures in critical paths
- [ ] icontract: pre/post-conditions added for new pipeline-level methods

## Live behavior verification
- [ ] Bot started locally and tested manually
- [ ] Smoke-test passes: `python scripts/smoke_test.py`
- [ ] No new Phase 0 / TODO / FIXME markers in committed code

## Documentation
- [ ] CHANGELOG entry (if user-visible change)
- [ ] README updated (if public API change)
- [ ] CONTRIBUTING.md updated (if process change)

## Breaking changes
<!-- Anything that breaks existing users or callers? -->
