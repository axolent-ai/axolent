#!/usr/bin/env bash
# Mutation testing: application/skill_compression/privacy/ (3-filter pipeline)
# Run from repo root: bash scripts/mutation_test_privacy.sh

set -euo pipefail
cd "$(dirname "$0")/../bridge"

echo "=== Mutation Testing: skill_compression/privacy/ ==="

mutmut run \
  --paths-to-mutate "application/skill_compression/privacy/" \
  --tests-dir "tests/" \
  --runner "python -m pytest tests/test_application/test_skill_compression/test_privacy_pipeline.py tests/test_application/test_skill_compression/test_privacy_healthcare.py tests/test_application/test_skill_compression/test_privacy_nudge.py tests/test_application/test_skill_compression/test_privacy_secrets.py tests/test_adversarial/test_K6_privacy_bypass.py -x -q --tb=no" \
  || true

echo ""
echo "=== Results ==="
mutmut results
echo ""
echo "=== Survived Mutants ==="
mutmut show survived || true
