# Dogfood Day Practice

Once a week (suggested: Sundays), spend 30 minutes using AXOLENT as a normal
user without any test mindset. Report any unexpected behavior.

## What to Do

1. Open Telegram, chat with the bot like any other AI assistant
2. Ask questions you would actually want answered
3. Use commands you do not normally use (`/skills`, `/import`, `/debate`)
4. Try edge cases that come naturally:
   * Very short messages ("ja", "hi")
   * Very long messages (paste a Wikipedia article)
   * Code blocks, emojis, multilingual content
   * Switch languages mid-conversation
   * Use commands in unusual sequences
   * Rapid-fire messages (rate limiting behavior)
   * Send only punctuation or special characters

## What to Report

Anything that surprises you. Be specific:

* "I expected X but got Y"
* "The bot took 12 seconds to respond, too slow"
* "The output had `**bold**` markdown not rendering as bold"
* "After /reset, the language was still Russian"
* "I sent a code block and the response lost indentation"

## Report Format

File: `12_concept-papers/dogfood-findings.md`

Use this template for each finding:

```
## [DATE] Finding: [short title]

**Observed:** What happened?
**Expected:** What should have happened?
**Severity:** Critical / Major / Minor / Cosmetic
**Steps to reproduce:**
1. ...
2. ...
**Screenshot/log:** (if available)
```

## Why Dogfooding Matters

Automated tests verify correctness. Dogfooding finds:

* UX friction that tests cannot express
* Response quality issues (tone, formatting, helpfulness)
* Performance problems that only surface in real usage
* Edge cases nobody thought to write a test for
* Regression in features that "just work" but stopped working
