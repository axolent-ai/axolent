# ADR-0004: Language Resolution Contract

**Status:** Accepted
**Date:** 2026-05-12
**Decision makers:** Jessica (AXOLENT AI)
**Fix commit:** 4a66252

## Context

AXOLENT AI supports 20 languages and detects the user's language automatically
from message text. For Latin-script languages, detection relies on marker-word
frequency scoring.

An edge case was discovered where shared diacritical characters caused
misclassification between Swedish and German:

* Characters like `a` and `o` appear in both Swedish and German text.
* A short German message containing words with these characters could be
  scored as Swedish if the Swedish marker set happened to match more
  character-level features.

The root cause: the detection algorithm gave equal weight to character-level
hints (diacritics) and word-level markers (known words per language). For
languages with overlapping character sets, character hints alone are ambiguous.

## Decision

Establish a **marker-precedence rule** for the language detection contract:

**When shared diacritical characters create ambiguity between two Latin-script
languages, the language with the highest word-marker score wins.**

Formally:

1. Compute word-marker scores for all candidate languages.
2. If two or more languages have overlapping character hints (shared
   diacritical characters), resolve the tie by word-marker score.
3. The word-marker score is the count of recognized marker words in the
   input text, divided by total word count.

### Implementation

In `domain/language.py`, the `detect_language_with_confidence()` function:

1. First checks non-Latin scripts (Arabic, CJK, Cyrillic, etc.) via
   Unicode ranges. These are deterministic and unambiguous.
2. For Latin-script text, scores all candidate languages by marker-word
   frequency.
3. Applies the marker-precedence rule: the language with the highest
   word-marker score wins, regardless of character-level hints.
4. Returns a `DetectionResult` with the language code and a confidence
   score (0.0 to 1.0).

### Smart-Switch Threshold

The `LanguageResolver` (application layer) applies an additional gate:
auto-detection must exceed **0.7 confidence** to change the sticky language.
This prevents low-confidence detections from disrupting the user's
established language preference.

```
Detection confidence >= 0.7 AND different from sticky -> switch
Detection confidence < 0.7                            -> keep sticky
No sticky language set                                -> use detection result
```

## Consequences

### Positive

* **Correct classification:** German text is no longer misclassified as
  Swedish (or vice versa) due to shared diacritical characters.
* **Predictable behavior:** Word-level markers are more reliable than
  character-level hints for closely related Latin-script languages.
* **Smart-switch prevents flapping:** Low-confidence detections do not
  disrupt the user's established language.

### Negative

* **Short messages remain ambiguous:** A single word (e.g., `"Hallo"`)
  may not contain enough marker words for reliable detection. The
  smart-switch threshold mitigates this by preserving the sticky language.
* **New language additions require marker curation:** Adding a new
  Latin-script language requires building a marker-word set and verifying
  it does not conflict with existing languages.

## Edge Cases Handled

| Input | Before Fix | After Fix |
|-------|-----------|-----------|
| `"Jag har en fraga"` (Swedish) | Correctly detected as `sv` | Correctly detected as `sv` |
| `"Ich habe eine Frage"` (German) | Sometimes misdetected as `sv` | Correctly detected as `de` |
| `"Hallo"` (ambiguous, 1 word) | Unpredictable | Falls back to sticky or default |
| `"Bonjour, comment allez-vous?"` (French) | Correct | Correct (no shared ambiguity) |

## References

* `bridge/domain/language.py`: Detection implementation
* `bridge/application/language_resolver.py`: Smart-switch logic
* [docs/I18N.md](../I18N.md): i18n system overview
* [docs/adr/0003-execution-kernel-architecture.md](0003-execution-kernel-architecture.md):
  LanguageResolver in the kernel pipeline
