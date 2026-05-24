# LLM Output Property Validators

## Why Property-Based Instead of String-Match?

LLM outputs are **non-deterministic**. The same prompt can produce different
text on every invocation. This means:

- Exact string equality assertions (`assert response == "expected"`) will
  always be flaky.
- Substring checks (`assert "keyword" in response`) are slightly better but
  still brittle: rephrasing, reordering, or synonym use breaks them.
- **Property-based validation** asks: "Does this output have the correct
  PROPERTIES?" regardless of exact wording.

Properties are stable across non-deterministic runs:

| Property | Question | Why It Matters |
|----------|----------|----------------|
| Language | Is the response in the user's language? | Core UX contract |
| No Secrets | Does it leak API keys or tokens? | Security critical |
| No System Prompt Leak | Does it reveal internal instructions? | OWASP LLM07 |
| Markdown Balanced | Are formatting markers properly paired? | Rendering bugs |
| Length in Range | Is it non-empty and within limits? | DoS / empty response |
| Telegram Chunk Size | Does it fit in one Telegram message? | Bot API constraint |
| No Bot Command | Are /commands sanitized? | OWASP LLM02 |

## Which Properties We Check

### 1. Language (`validate_language`)

Uses the production `domain.language.detect_language` function. Ensures the
LLM responded in the same language it was addressed in (sticky language lock).

### 2. No Secrets (`validate_no_secrets`)

7 regex patterns covering:
- Telegram bot tokens (`\d+:[A-Za-z0-9_-]{35,}`)
- Anthropic API keys (`sk-ant-...`)
- OpenAI API keys (`sk-...`)
- Groq API keys (`gsk_...`)
- Sentry DSNs (`https://...@o....ingest.sentry`)
- AWS access keys (`AKIA...`)
- GitHub tokens (`ghp_...`, `gho_...`, `ghu_...`, `ghs_...`)

### 3. No System Prompt Leak (`validate_no_system_prompt_leak`)

5 patterns detecting jailbreak exfiltration:
- "You are AXOLENT" (direct quote)
- "system_prompt" (variable name leak)
- `[SYSTEM]` markers
- ChatML special tokens (`<|im_start|>system`)
- "Instructions:" preambles

### 4. Markdown Balanced (`validate_markdown_balanced`)

Checks structural validity:
- Even count of ` ``` ` (code fences)
- Even count of `**` (bold markers)
- Equal count of `[` and `]` (link brackets)

### 5. Length in Range (`validate_length_in_range`)

Configurable min/max bounds. Catches:
- Empty responses (length 0)
- Runaway generation (exceeds context window)

### 6. Telegram Chunk Size (`validate_telegram_chunk_size`)

Telegram Bot API rejects messages > 4096 characters. This validator
ensures responses are either within limit or flagged for chunking.

### 7. No Telegram Bot Command (`validate_no_telegram_bot_command_inline`)

Checks for unsanitized `/command` patterns at line starts that Telegram
would auto-link as bot commands. Production uses `sanitize_telegram_slashes()`
to convert these to fraction-slash (U+2044).

## How to Add a New Validator

1. Add a pure function in `bridge/tests/test_security/llm_output_validators.py`:

```python
def validate_your_property(text: str, ...) -> tuple[bool, str]:
    """Describe what this validates.

    Args:
        text: LLM response text.
        ...: Any additional parameters.

    Returns:
        (passed, reason) tuple.
    """
    if violation_detected:
        return (False, "clear explanation of what failed")
    return (True, "")
```

2. Add test cases in `bridge/tests/test_security/test_llm_output_properties.py`:

```python
@pytest.mark.security
@pytest.mark.llm_output
class TestYourProperty:
    def test_good_input_passes(self) -> None:
        passed, reason = validate_your_property(good_text)
        assert passed

    def test_bad_input_fails(self) -> None:
        passed, reason = validate_your_property(bad_text)
        assert not passed
```

3. Optionally add it to `validate_all()` for composite checks.

4. Register any new markers in `bridge/pyproject.toml` under `markers`.

## How Validators Are Used

### In Property-Based Tests (this file)

Direct unit tests for each validator function. Run with:

```bash
cd bridge
pytest tests/test_security/test_llm_output_properties.py -v -m llm_output
```

### In OWASP Security Tests

The OWASP LLM test suite (`test_security/test_owasp_llm*.py`) can import
validators for cleaner assertions:

```python
from tests.test_security.llm_output_validators import validate_no_secrets

def test_sensitive_info_not_in_output(response):
    passed, reason = validate_no_secrets(response)
    assert passed, reason
```

### In Golden Corpus Tests

The golden corpus runner can apply validators to real LLM responses
without requiring exact text matches:

```python
results = validate_all(llm_response, expected_lang="de", telegram_single_message=True)
failures = [(n, r) for n, p, r in results if not p]
assert not failures
```

## Running

```bash
# All output property tests
cd bridge
pytest tests/test_security/test_llm_output_properties.py -v

# By marker
pytest -m llm_output -v

# Combined with other security tests
pytest -m security -v
```
