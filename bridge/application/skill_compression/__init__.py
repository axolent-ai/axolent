"""Skill-Compression Engine: Pattern recognition from user interactions.

Architecture follows the 7-Layer model:
  Layer 1: Event Normalizer (structured events from chat)
  Layer 2: Algorithmic Candidate Layer (pattern detection)
  Layer 3: Evidence Ledger (proof tracking)
  Layer 4: Pattern Judge (lifecycle management)
  Layer 5: SkillMatcher (application)
  Layer 6: UI / Explainer
  Layer 7: Privacy Safeguards

Step 1 implements: Event Normalizer + Fingerprint Similarity + Storage Schema.
Step 2 implements: N-Gram Extractor + Markov Chain + Elo Rating.
Step 3 implements: Evidence Ledger + BKT + Pattern Judge + FSRS Decay.
Step 4 implements: SkillMatcher + Collision Detection + Skill Versioning + Ask Before Applying.
Step 5 implements: UI Profile View + 4 Chat-Shortcuts + Skill Indicator + ChatService Integration.
Step 6 implements: Explainer (8 Question Types) + Local Evaluation Set + BERTopic Background Classification.
Step 7 implements: Conversation Import (4 parsers + orchestrator + /import command).
"""
