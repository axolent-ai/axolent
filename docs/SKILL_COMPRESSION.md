# Skill Compression Engine

The Skill Compression Engine learns user preferences from conversations and applies them automatically to future interactions. It transforms repeated behavioral patterns into reusable "skills" that personalize the assistant without explicit configuration.

## Architecture: 7-Layer Model

```
Layer 7: Privacy Safeguards
         HealthcareFilter + SecretScanner + NudgeFilter + PrivacyPipeline
         Blocks PII, secrets, healthcare data, and sensitive content (HC-SC-13/14/15).

Layer 6: UI / Explainer
         Telegram commands (/skills, /skill, /forget, /learn, /explain, /import).
         Profile view, skill indicator, inline keyboards.

Layer 5: SkillMatcher
         Matches incoming messages to stored skills.
         Collision detection, skill versioning, ask-before-applying.

Layer 4: Pattern Judge
         Lifecycle management: suggested -> confirmed -> active -> paused -> retired.
         FSRS-based decay, Elo rating, Bayesian Knowledge Tracing.

Layer 3: Evidence Ledger
         Proof tracking for each hypothesis.
         Support/contradiction signals, evidence history.

Layer 2: Algorithmic Candidate Layer
         N-Gram extraction, Markov chain analysis, Elo rating.
         Detects repeating patterns in user behavior.

Layer 1: Event Normalizer
         Converts raw chat events into structured SkillEvent objects.
         Fingerprint similarity, intent/domain/correction classification.
```

## Implementation Steps (Completed)

| Step | Component | Status |
|------|-----------|--------|
| 1 | Event Normalizer + Fingerprint Similarity + Storage Schema | Complete |
| 2 | N-Gram Extractor + Markov Chain + Elo Rating | Complete |
| 3 | Evidence Ledger + BKT + Pattern Judge + FSRS Decay | Complete |
| 4 | SkillMatcher + Collision Detection + Skill Versioning + Ask Before Applying | Complete |
| 5 | UI Profile View + 4 Chat Shortcuts + Skill Indicator + ChatService Integration | Complete |
| 6 | Explainer (8 Question Types) + Local Evaluation Set + BERTopic Background Classification | Complete |
| 7 | Conversation Import (4 Parsers + Orchestrator + /import Command) | Complete |
| 8 | Privacy & Safety Layer (HealthcareFilter + SecretScanner + NudgeFilter + PrivacyPipeline) | Complete |
| 9 | Skill Formatting Refactor + Pre-commit EN-only Compliance | Complete |
| 10 | i18n Migration (all user-facing strings to locale files) | Complete |

## User-Facing Features

### Skill Profile (`/skills`)
Shows the top 10 active skills as a compact bullet-point list. Each skill has inline buttons for details, pausing, or forgetting.

### Skill Detail (`/skill <name>`)
Shows full details for a skill: version, description, status, evidence counts, type, scope, source, dates, decay immunity, and version history. Fuzzy name matching supported.

### Manual Skill Creation (`/learn`)
Saves a user-described preference as a permanent, decay-immune skill. Checks for secrets (HC-SC-13) and enforces a 50-skill cap (HC-SC-8).

### Skill Forgetting (`/forget <name>`)
Deletes a skill with a 30-day tombstone (default) or permanently (`--permanent`). Tombstoned patterns can be re-learned after expiration.

### Skill Explanation (`/explain <name> [question]`)
Eight question types for understanding skill decisions:
1. What was recognized?
2. Why not promoted to skill?
3. Why was it promoted?
4. When was drift detected?
5. What evidence is needed?
6. What lessons were learned?
7. Where does the scope NOT apply?
8. What counter-evidence exists?

### Conversation Import (`/import <path>`)
Imports conversations from local folders (ChatGPT, Claude, generic Markdown/JSON). Always dry-run first. All imported hypotheses start as "suggested". Supports undo.

### Skill Indicator
When a skill is applied to a response, an inline indicator shows which skill was used, with [/undo] and [Why?] buttons.

## Privacy Guarantees (3 Filters)

### 1. SecretScanner (HC-SC-13)
Blocks storage of API tokens, passwords, prices, phone numbers, email addresses, and other PII. Multi-layered regex detection.

### 2. HealthcareFilter (HC-SC-14)
Prevents storage of healthcare/clinical data. Bilingual keyword matching (DE + EN) for medical terms, diagnoses, medications.

### 3. NudgeFilter (HC-SC-15)
Ensures skills do not nudge users toward unwanted behaviors. Pattern-based detection with German nudge-policy descriptions.

All three filters run in the PrivacyPipeline before any hypothesis is stored or promoted.

## Configuration

### `auto_apply_enabled`
Controls whether confirmed/active skills are automatically applied to responses. Default: `True`. When disabled, skills are tracked but not applied.

### Storage
Skills are stored in a local SQLite database per user. No data leaves the local machine. The schema includes tables for hypotheses, evidence, tombstones, version history, and import metadata.

### Decay
Skills decay over time via FSRS (Free Spaced Repetition Scheduler). Decay-immune skills (created via `/learn`) are exempt. The Pattern Judge manages lifecycle transitions based on evidence strength.

## Troubleshooting

### "Skill system not yet initialized"
The HypothesisStorage was not injected into bot_data at startup. Check `main.py` initialization.

### Skills not being applied
1. Check that `auto_apply_enabled` is `True` in the configuration.
2. Verify the skill status is "active" or "confirmed" via `/skill <name>`.
3. Check if the skill's scope matches the current context.

### Import finds no files
Supported formats: `.md`, `.json`, `.jsonl`, `.txt`. The folder must contain files in a recognized conversation format (ChatGPT export, Claude export, or generic chat format).

### Secret filter blocking `/learn`
The No-Model-Secret Rule (HC-SC-13) prevents storage of text containing patterns that look like passwords, API keys, prices, or personal identifiers. Rephrase the skill to describe the preference without including the sensitive data itself.

### Skill limit reached (50 skills)
Use `/forget <name>` to remove unused skills. Consider using `--permanent` for patterns you definitely do not want re-learned.
