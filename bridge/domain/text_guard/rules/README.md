# Text Guard: Adding New Languages

## How to add a new language

1. **Create a YAML rule file** in this directory: `<lang_code>.yml`
   (e.g. `tr.yml` for Turkish, `sv.yml` for Swedish).

2. **Add built-in Python rules** in `domain/text_guard/rules_registry.py`:
   * Define `_XX_WORD_PAIRS` tuple of `WordPair` objects
   * Define `_XX_WHITELIST` frozenset of loan words
   * Add entry to `_BUILTIN_RULES` dict

3. **Add language markers** in `domain/language.py`:
   * Add word markers to `_MARKERS` dict
   * Add character hints to `_CHAR_HINTS` dict (if the language has unique chars)

4. **Write tests** in `tests/test_domain/test_text_guard/rules/test_<lang>.py`:
   * At least 10 positive cases (replacement happens)
   * At least 5 negative cases (loan words stay unchanged)

## YAML Format

```yaml
language: xx  # ISO 639-1 code

word_pairs:
  - ascii: "incorrect_form"
    correct: "correct_form"
  - ascii: "another_wrong"
    correct: "another_right"

loan_word_whitelist:
  - "english_word_that_looks_similar"
  - "another_false_positive"
```

## Guidelines

* **Word-list based, NOT blind substitution.**
  Never add a rule like "ae -> ae_with_umlaut" as a global regex.
  Always use specific whole words.

* **Loan words are critical.** English words like "queue", "blue",
  "user", "module" must be whitelisted to prevent false positives.

* **Case handling** is automatic: the engine preserves the original
  case pattern (lowercase, UPPERCASE, Title Case).

* **Code blocks** are automatically skipped (both fenced ``` and inline `).

## Currently supported languages

| Code | Language   | Word pairs | Whitelist entries |
|------|-----------|------------|-------------------|
| de   | German    | ~230       | ~170              |
| fr   | French    | ~75        | ~30               |
| es   | Spanish   | ~85        | ~35               |
| it   | Italian   | ~45        | ~17               |
| pt   | Portuguese| ~65        | ~20               |
| en   | English   | 0 (no-op)  | 0                 |

## Planned languages (Phase 2)

* `tr` Turkish (i/I dotting issue)
* `sv` Swedish (a-ring, o-with-dots)
* `nb`/`nn` Norwegian (o-slash, a-ring)
* `da` Danish (o-slash, a-ring, ae-ligature)
* `pl` Polish (multiple diacritics)
* `cs` Czech (haceks and carons)
