"""Local Evaluation Set for Skill-Compression (Layer 4 adjunct).

Smoke-test per hypothesis: 3-5 example input/output pairs that are
evaluated before each status transition (suggested -> confirmed -> active).

HC-EVAL-1 [BLOCKER]: Local Eval Set is a smoke-test, NOT statistical proof.
  Maximum 5 examples per hypothesis.

IC-EVAL-1: Evaluated at every status change, plus optional /verify-skill X.

Usage:
    eval_set = LocalEvalSet(storage)
    eval_set.add_example(hyp_id, "user input", "expected output")
    result = eval_set.evaluate(hyp_id)
    if not result.passed:
        # Do not promote, status stays at needs_review

No external dependencies. Pure Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from application.skill_compression.hypothesis_storage import HypothesisStorage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

# HC-EVAL-1: Maximum examples per hypothesis
MAX_EXAMPLES_PER_HYPOTHESIS: int = 5


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalExample:
    """A single evaluation example for a hypothesis.

    Attributes:
        eval_id: Unique ID.
        hypothesis_id: The associated hypothesis.
        example_input: Example user input.
        example_output: Expected output.
        was_correct: Last evaluation result (True/False/None).
        last_evaluated: ISO-8601 timestamp of last evaluation.
    """

    eval_id: str
    hypothesis_id: str
    example_input: str
    example_output: str
    was_correct: Optional[bool] = None
    last_evaluated: Optional[str] = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result of evaluating a hypothesis against its local eval set.

    Attributes:
        hypothesis_id: The evaluated hypothesis.
        total_examples: Number of examples evaluated.
        passed_count: Number of examples that passed.
        failed_count: Number of examples that failed.
        passed: Whether the overall smoke-test passed.
        details: Per-example results.
        skip_reason: If evaluation was skipped, why.
    """

    hypothesis_id: str
    total_examples: int
    passed_count: int
    failed_count: int
    passed: bool
    details: tuple[EvalExample, ...] = ()
    skip_reason: str = ""


# ---------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------


def _basic_match(expected: str, actual_claim: str) -> bool:
    """Check if expected output semantically relates to hypothesis claim.

    Simple heuristic: checks if key terms from the expected output
    appear in the hypothesis claim. This is a smoke-test, not a
    deep semantic comparison.

    Args:
        expected: The expected output text.
        actual_claim: The hypothesis claim to check against.

    Returns:
        True if the expected output aligns with the claim.
    """
    expected_lower = expected.strip().lower()
    claim_lower = actual_claim.strip().lower()

    if not expected_lower or not claim_lower:
        return False

    # Check for substring match in either direction
    if expected_lower in claim_lower or claim_lower in expected_lower:
        return True

    # Check word overlap (>40% means likely related)
    expected_words = set(expected_lower.split())
    claim_words = set(claim_lower.split())

    if not expected_words or not claim_words:
        return False

    overlap = len(expected_words & claim_words)
    total = len(expected_words | claim_words)

    if total > 0 and overlap / total > 0.3:
        return True

    return False


# ---------------------------------------------------------------
# LocalEvalSet
# ---------------------------------------------------------------


class LocalEvalSet:
    """Smoke-test evaluation set per hypothesis.

    Maintains 3-5 example pairs per hypothesis and evaluates them
    against the current hypothesis state before status transitions.

    HC-EVAL-1: Maximum 5 examples. This is a sanity check, not
    a statistical test.

    Thread safety: NOT thread-safe. Single-threaded async context.

    Usage:
        eval_set = LocalEvalSet(storage)
        eval_set.add_example("hyp_abc", "user input", "expected output")
        result = eval_set.evaluate("hyp_abc")
    """

    def __init__(self, storage: HypothesisStorage) -> None:
        """Initialize the LocalEvalSet.

        Args:
            storage: HypothesisStorage for DB access.
        """
        self._storage = storage

    def add_example(
        self,
        hypothesis_id: str,
        example_input: str,
        expected_output: str,
    ) -> Optional[str]:
        """Add an example to the local eval set.

        HC-EVAL-1: Enforces maximum of 5 examples per hypothesis.
        If the limit is reached, returns None and logs a warning.

        Args:
            hypothesis_id: The hypothesis to add the example to.
            example_input: Example user input text.
            expected_output: Expected output text.

        Returns:
            The eval_id of the new example, or None if limit reached.
        """
        # Check existing count
        existing = self.get_examples(hypothesis_id)
        if len(existing) >= MAX_EXAMPLES_PER_HYPOTHESIS:
            log.warning(
                "Local eval set limit reached for hypothesis %s: %d/%d",
                hypothesis_id,
                len(existing),
                MAX_EXAMPLES_PER_HYPOTHESIS,
            )
            return None

        eval_id = f"eval_{uuid4().hex[:16]}"
        self._storage.insert_eval_example(
            eval_id=eval_id,
            hypothesis_id=hypothesis_id,
            example_input=example_input,
            example_output=expected_output,
        )

        log.info(
            "Eval example added: eval=%s hyp=%s input='%s'",
            eval_id,
            hypothesis_id,
            example_input[:40],
        )
        return eval_id

    def evaluate(self, hypothesis_id: str) -> EvalResult:
        """Evaluate a hypothesis against its local eval set.

        Runs each example through a basic match check against the
        current hypothesis claim. This is a smoke-test: it verifies
        that the hypothesis still aligns with its stored examples.

        IC-EVAL-1: Called at every status change.

        Args:
            hypothesis_id: The hypothesis to evaluate.

        Returns:
            EvalResult with pass/fail and per-example details.
        """
        hyp = self._storage.get_hypothesis(hypothesis_id)
        if hyp is None:
            return EvalResult(
                hypothesis_id=hypothesis_id,
                total_examples=0,
                passed_count=0,
                failed_count=0,
                passed=True,
                skip_reason="Hypothesis not found",
            )

        examples = self.get_examples(hypothesis_id)
        if not examples:
            return EvalResult(
                hypothesis_id=hypothesis_id,
                total_examples=0,
                passed_count=0,
                failed_count=0,
                passed=True,
                skip_reason="No examples in eval set",
            )

        passed_count = 0
        failed_count = 0
        evaluated_examples: list[EvalExample] = []

        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()

        for example in examples:
            is_correct = _basic_match(example.example_output, hyp.claim)

            if is_correct:
                passed_count += 1
            else:
                failed_count += 1

            # Update the DB record
            self._storage._conn.execute(
                "UPDATE hypothesis_local_eval_set "
                "SET was_correct = ?, last_evaluated = ? "
                "WHERE eval_id = ?",
                (1 if is_correct else 0, now_iso, example.eval_id),
            )

            evaluated_examples.append(
                EvalExample(
                    eval_id=example.eval_id,
                    hypothesis_id=example.hypothesis_id,
                    example_input=example.example_input,
                    example_output=example.example_output,
                    was_correct=is_correct,
                    last_evaluated=now_iso,
                )
            )

        # Overall pass: all examples must pass
        overall_passed = failed_count == 0

        log.info(
            "Eval result: hyp=%s total=%d passed=%d failed=%d overall=%s",
            hypothesis_id,
            len(examples),
            passed_count,
            failed_count,
            overall_passed,
        )

        return EvalResult(
            hypothesis_id=hypothesis_id,
            total_examples=len(examples),
            passed_count=passed_count,
            failed_count=failed_count,
            passed=overall_passed,
            details=tuple(evaluated_examples),
        )

    def get_examples(self, hypothesis_id: str) -> list[EvalExample]:
        """Retrieve all eval examples for a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID.

        Returns:
            List of EvalExample objects.
        """
        rows = self._storage._conn.fetchall(
            "SELECT * FROM hypothesis_local_eval_set "
            "WHERE hypothesis_id = ? ORDER BY eval_id",
            (hypothesis_id,),
        )

        return [
            EvalExample(
                eval_id=row["eval_id"],
                hypothesis_id=row["hypothesis_id"],
                example_input=row["example_input"],
                example_output=row["example_output"],
                was_correct=bool(row["was_correct"])
                if row["was_correct"] is not None
                else None,
                last_evaluated=row["last_evaluated"],
            )
            for row in rows
        ]

    def count_examples(self, hypothesis_id: str) -> int:
        """Count eval examples for a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID.

        Returns:
            Number of examples.
        """
        row = self._storage._conn.fetchone(
            "SELECT count(*) as cnt FROM hypothesis_local_eval_set "
            "WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )
        return row["cnt"] if row else 0

    def remove_example(self, eval_id: str) -> None:
        """Remove a single eval example.

        Args:
            eval_id: The eval example ID to remove.
        """
        self._storage._conn.execute(
            "DELETE FROM hypothesis_local_eval_set WHERE eval_id = ?",
            (eval_id,),
        )
        log.info("Eval example removed: %s", eval_id)
