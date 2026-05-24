# CodeQL Static Analysis (SAST)

## What is CodeQL?

CodeQL is GitHub's native Static Application Security Testing (SAST) engine.
It performs cross-file data-flow analysis (taint tracking) to find vulnerabilities
that simpler pattern-based tools (like Bandit or Semgrep) cannot detect.

CodeQL is free for public repositories.

## What it finds

CodeQL's Python analysis detects:

* **SQL Injection:** Data flow from user input to `cursor.execute()` sinks
* **Command Injection:** Subprocess calls with user-influenced arguments
* **Path Traversal:** `open()` and `Path()` with untrusted input
* **Insecure Deserialization:** `pickle.loads()` on untrusted data
* **Hardcoded Credentials:** Secrets embedded in source code
* **Cryptographic Misuse:** Weak algorithms, insufficient key lengths
* **XSS:** Reflected/stored cross-site scripting (less relevant for CLI bots)
* **Race Conditions:** Async code with shared mutable state

## When it runs

| Trigger | Scope |
|---------|-------|
| Push to `main` | Full analysis of changed files |
| Pull request to `main` | Diff-aware analysis, findings shown inline |
| Weekly (Monday 04:23 UTC) | Full repository scan |

## Viewing findings

1. Go to the repository on GitHub
2. Navigate to **Security** tab > **Code scanning alerts**
3. Filter by tool: "CodeQL"
4. Each finding shows the data-flow path from source to sink

## Triage workflow

### Severity levels

| Severity | Action | Merge blocked? |
|----------|--------|----------------|
| Critical | Fix immediately | Yes |
| High | Fix before merge | Yes |
| Medium | Fix within 1 sprint | No (but tracked) |
| Low | Evaluate, may accept risk | No |

### Handling findings on a PR

1. Click the finding in the PR's "Security" checks
2. Review the data-flow path CodeQL shows
3. Decide: **Fix**, **Dismiss as false positive**, or **Suppress**

### Suppressing false positives

**Option A: Inline suppression (preferred for obvious false positives)**

```python
result = cursor.execute(query)  # codeql[py/sql-injection] false positive: query is a constant
```

**Option B: CodeQL alert dismissal on GitHub**

1. Open the alert in Security > Code scanning
2. Click "Dismiss alert"
3. Select reason: "False positive", "Won't fix", or "Used in tests"
4. Add a comment explaining why

### When to suppress vs. fix

* **Suppress** if: The data flow CodeQL traces is impossible at runtime
  (e.g., the "user input" is actually a hardcoded enum value)
* **Fix** if: There is any realistic scenario where untrusted data reaches the sink

## Scoped paths

CodeQL only analyzes production code:

**Included:**
* `bridge/domain/**`
* `bridge/application/**`
* `bridge/infrastructure/**`
* `bridge/presentation/**`
* `bridge/main.py`
* `scripts/**`

**Excluded:**
* `bridge/tests/**`
* `bridge/.venv/**`
* `bridge/data/**`
* `bridge/pytest_tmp_v9/**`
* `docs/**`

## Adding custom queries (Phase 2)

For project-specific sinks (e.g., ensuring all subprocess calls go through a
sanitization wrapper), custom CodeQL queries can be added:

1. Create `.github/codeql/custom-queries/` directory
2. Write a `.ql` file with the query
3. Add a `qlpack.yml` manifest
4. Reference in `codeql-config.yml`:

```yaml
queries:
  - uses: security-extended
  - uses: security-and-quality
  - uses: ./.github/codeql/custom-queries
```

## Relationship to other security tools

| Tool | Type | Scope |
|------|------|-------|
| CodeQL | Cross-file taint tracking | Data-flow vulnerabilities |
| Bandit | Pattern-based SAST | Python-specific anti-patterns |
| Semgrep | Rule-based SAST | 2000+ generic rules |
| TruffleHog | Secret scanner | Leaked credentials in git history |
| pip-audit | Dependency audit | Known CVEs in packages |

CodeQL complements (not replaces) the existing Bandit + Semgrep setup by adding
cross-file data-flow analysis that pattern matchers cannot perform.
