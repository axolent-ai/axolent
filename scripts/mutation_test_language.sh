#!/usr/bin/env bash
# Mutation testing: application/language/ module (excluding stream_guard.py)
# Run from repo root: bash scripts/mutation_test_language.sh

set -euo pipefail
cd "$(dirname "$0")/../bridge"

echo "=== Mutation Testing: application/language/ ==="
echo "Excluding stream_guard.py (has its own runner)"

mutmut run \
  --paths-to-mutate "application/language/" \
  --tests-dir "tests/" \
  --runner "python -m pytest tests/test_application/test_language/ tests/test_adversarial/ -x -q --tb=no" \
  || true

echo ""
echo "=== Results ==="
mutmut results
echo ""
echo "=== Survived Mutants ==="
mutmut show survived || true
