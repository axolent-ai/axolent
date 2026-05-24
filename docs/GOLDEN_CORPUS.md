# Golden Corpus

The Golden Corpus is a YAML-based regression test suite containing critical
prompts with expected behaviour. It serves as a safety net against regressions
in AXOLENT's core features: language detection, streaming, commands, debate,
memory, skills, privacy, and edge-case handling.

## Why

LLM-backed systems are prone to subtle regressions when refactoring infrastructure,
updating prompts, or changing provider routing. The Golden Corpus provides
concrete, deterministic test cases that catch these regressions early.

## Location

```
bridge/tests/corpus/golden_prompts.yaml   # The corpus (source of truth)
bridge/tests/corpus/golden_runner.py      # Validation engine
bridge/tests/test_corpus/conftest.py      # Fake chat service fixture
bridge/tests/test_corpus/test_golden_corpus.py  # Pytest parametrized runner
```

## Running

```bash
# Run all golden corpus tests
cd bridge
pytest tests/test_corpus/ -v -m golden_corpus

# Run specific category
pytest tests/test_corpus/ -v -k "lang_"
pytest tests/test_corpus/ -v -k "privacy"
pytest tests/test_corpus/ -v -k "edge"

# Run against real provider (requires API keys, non-deterministic)
AXOLENT_GOLDEN_REAL=1 pytest tests/test_corpus/ -v -m golden_corpus
```

## Adding New Entries

1. Open `bridge/tests/corpus/golden_prompts.yaml`
2. Pick the correct category section
3. Add an entry with this schema:

```yaml
- id: category_descriptive_name    # unique, snake_case
  category: language               # one of 8 categories
  input: "The user message"        # what the user sends
  setup:                           # optional pre-conditions
    sticky_language: de
    history_messages: 5
  action_after_seconds: 0.5        # optional timed action
  action: /stop                    # the action to take
  input_multiply: 5000             # optional: repeat input N times
  expected:                        # assertions (see schema below)
    language: de
    min_length: 50
```

4. Run tests to verify: `pytest tests/test_corpus/ -v -k "your_new_id"`

## Expected Schema Reference

### Language Assertions

| Key | Type | Description |
|-----|------|-------------|
| `language` | str | Expected detected language code (ISO 639-1) |
| `sticky_after` | str | Expected sticky language after processing |
| `no_german` | bool | Response must not contain German indicator words |
| `no_english_only` | bool | Response must not be purely English |
| `no_critical_switch` | bool | No unexpected language switch occurred |

### Length and Content Assertions

| Key | Type | Description |
|-----|------|-------------|
| `min_length` | int | Minimum response text length in characters |
| `contains_one_of` | list[str] | At least one must appear in response |
| `response_contains` | str | Must appear in response (case-insensitive) |
| `response_includes_one_of` | list[str] | At least one must appear |
| `response_excludes` | list[str] | None of these may appear in response |

### Streaming Assertions

| Key | Type | Description |
|-----|------|-------------|
| `streaming_aborted` | bool | Whether streaming was aborted |
| `streaming_completes` | bool | Whether streaming completed normally |
| `no_messages_after_cancel` | bool | No messages sent after cancel |
| `max_duration_seconds` | float | Maximum allowed response duration |

### Command Assertions

| Key | Type | Description |
|-----|------|-------------|
| `memory_count_delta` | int | Change in memory count (0, +1, -1) |
| `history_count` | int | Expected history length after command |
| `streaming_active_after` | bool | Streaming state after command |

### Debate Assertions

| Key | Type | Description |
|-----|------|-------------|
| `providers_called_min` | int | Minimum providers consulted |
| `synthesis_present` | bool | Whether synthesis was generated |
| `no_raw_provider_output` | bool | Raw output must not leak through |
| `uses_previous_debate_context` | bool | Followup uses debate context |

### Skill Assertions

| Key | Type | Description |
|-----|------|-------------|
| `pending_skill_created` | bool | Whether a pending skill was created |
| `privacy_pipeline_ran` | bool | Whether privacy check ran on skill |
| `skill_count_delta` | int | Change in skill count |
| `no_duplicate_created` | bool | No duplicate skill created |

### Privacy Assertions

| Key | Type | Description |
|-----|------|-------------|
| `privacy_rejection` | str/null | Rejection category or null if allowed |

### Edge Case Assertions

| Key | Type | Description |
|-----|------|-------------|
| `no_crash` | bool | Service did not crash |
| `response_present` | bool | A non-empty response was returned |
| `preserves_unicode` | bool | Unicode characters preserved correctly |

## Categories

| Category | Count | Tests |
|----------|-------|-------|
| language | 10 | Language detection and sticky persistence |
| streaming | 3 | Stream start/stop/cancel |
| commands | 5 | Slash command parsing and execution |
| debate | 3 | Multi-provider synthesis |
| memory | 3 | Memory scoping and retrieval |
| skills | 3 | Skill learn/forget lifecycle |
| privacy | 4 | Privacy guard (healthcare, secrets) |
| edge | 6 | Empty, long, unicode, injection |

## Real Provider Testing

Set `AXOLENT_GOLDEN_REAL=1` to run against actual LLM providers.
This mode is:
- Non-deterministic (LLM responses vary)
- Slow (network calls)
- Requires API keys in environment
- Useful for pre-release validation

The fake service (default) is deterministic and runs in CI without credentials.

## CI Integration

The Golden Corpus runs as a dedicated step in `.github/workflows/pr-check.yml`
after the main test suite and before the smoke test. Any failure blocks the PR.
