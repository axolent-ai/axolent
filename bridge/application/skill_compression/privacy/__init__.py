"""Privacy Safeguards for Skill-Compression (Layer 7).

Three filters that form the pre-hypothesis-creation privacy pipeline:

  1. HealthcareFilter (HC-SC-14): blocks behavioral-clinical phenotyping
  2. SecretScanner (HC-SC-13): blocks secrets/PII in hypothesis claims
  3. NudgeFilter (HC-SC-15): enforces the nudge negative list

All filters are hard gates: if any filter rejects, the hypothesis
is NOT materialized. No performance-based bypass allowed.

Architecture guard: Privacy filters are called by PatternJudge
BEFORE any promotion. Filter bypass is not possible.
"""
