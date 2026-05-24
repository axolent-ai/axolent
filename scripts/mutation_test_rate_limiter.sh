#!/usr/bin/env bash
# Mutation testing: application/rate_limiter.py
# Run from repo root: bash scripts/mutation_test_rate_limiter.sh

set -euo pipefail
cd "$(dirname "$0")/../bridge"

echo "=== Mutation Testing: rate_limiter.py ==="

mutmut run \
  --paths-to-mutate "application/rate_limiter.py" \
  --tests-dir "tests/" \
  --runner "python -m pytest tests/test_application/test_rate_limiter.py tests/test_application/test_rate_limiter_benchmark.py -x -q --tb=no" \
  || true

echo ""
echo "=== Results ==="
mutmut results
echo ""
echo "=== Survived Mutants ==="
mutmut show survived || true
