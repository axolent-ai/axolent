#!/usr/bin/env bash
# Mutation testing: main.py _sentry_before_send (privacy filter)
# Run from repo root: bash scripts/mutation_test_sentry.sh

set -euo pipefail
cd "$(dirname "$0")/../bridge"

echo "=== Mutation Testing: main.py (_sentry_before_send) ==="

mutmut run \
  --paths-to-mutate "main.py" \
  --tests-dir "tests/" \
  --runner "python -m pytest tests/test_infrastructure/test_sentry_integration.py tests/test_adversarial/test_K8_sentry_pii.py -x -q --tb=no" \
  || true

echo ""
echo "=== Results ==="
mutmut results
echo ""
echo "=== Survived Mutants ==="
mutmut show survived || true
