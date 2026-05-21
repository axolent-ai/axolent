"""Skill-Compression Engine: Pattern recognition from user interactions.

Architecture follows the 7-Layer model:
  Layer 1: Event Normalizer (structured events from chat)
  Layer 2: Algorithmic Candidate Layer (pattern detection)
  Layer 3: Evidence Ledger (proof tracking)
  Layer 4: Pattern Judge (lifecycle management)
  Layer 5: SkillMatcher (application)
  Layer 6: UI / Explainer
  Layer 7: Privacy Safeguards

Implementation history (all 10 steps complete):
  Step 1: Event Normalizer + Fingerprint Similarity + Storage Schema.
  Step 2: N-Gram Extractor + Markov Chain + Elo Rating.
  Step 3: Evidence Ledger + BKT + Pattern Judge + FSRS Decay.
  Step 4: SkillMatcher + Collision Detection + Skill Versioning + Ask Before Applying.
  Step 5: UI Profile View + 4 Chat-Shortcuts + Skill Indicator + ChatService Integration.
  Step 6: Explainer (8 Question Types) + Local Evaluation Set + BERTopic Background Classification.
  Step 7: Conversation Import (4 parsers + orchestrator + /import command).
  Step 8: Privacy & Safety Layer (HealthcareFilter + SecretScanner + NudgeFilter + PrivacyPipeline).
  Step 9: Skill Formatting Refactor + Pre-commit EN-only Compliance.
  Step 10: i18n Migration (all user-facing strings from Python to locale files de.json/en.json).

Final state: All 7 layers implemented across 10 steps.
  User-facing commands: /skills, /skill, /forget, /learn, /explain, /import.
  Privacy: 3-filter pipeline (SecretScanner + HealthcareFilter + NudgeFilter).
  i18n: All UI strings in bridge/i18n/locales/{de,en}.json via t() function.
  See docs/SKILL_COMPRESSION.md for full architecture documentation.
"""
