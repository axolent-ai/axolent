#!/usr/bin/env bash
# Mutation testing: application/language/stream_guard.py
# Run from repo root: bash scripts/mutation_test_stream_guard.sh

set -euo pipefail
cd "$(dirname "$0")/../bridge"

echo "=== Mutation Testing: stream_guard.py ==="

mutmut run \
  --paths-to-mutate "application/language/stream_guard.py" \
  --tests-dir "tests/" \
  --runner "python -m pytest tests/test_application/test_language/test_stream_guard.py tests/test_application/test_language/test_stream_guard_hypothesis.py -x -q --tb=no" \
  || true

echo ""
echo "=== Results ==="
mutmut results
echo ""
echo "=== Survived Mutants ==="
mutmut show survived || true
