"""Production-path test for W2: claim text not in logs.

Verifies that hypothesis_storage and collision_detector do not
log raw claim text.
"""

from __future__ import annotations


class TestLogDoesNotContainClaimText:
    """W2: Log statements must NOT contain raw claim text."""

    def test_hypothesis_storage_version_log_redacted(self) -> None:
        """create_new_version log must use hash, not raw claim."""
        from pathlib import Path

        source_path = (
            Path(__file__).resolve().parents[3]
            / "application"
            / "skill_compression"
            / "hypothesis_storage.py"
        )
        source = source_path.read_text(encoding="utf-8")

        # The old pattern logged raw claims: current.claim, new_claim
        # The new pattern logs hashes: old_hash, new_hash, len
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if "log.info(" in line and "new version" in line.lower():
                # Check surrounding lines for raw claim logging
                context = "\n".join(lines[i : i + 8])
                assert "current.claim" not in context, (
                    f"Line {i + 1}: log.info still contains 'current.claim' "
                    "(raw claim text). Must use hash instead."
                )
                assert "new_claim," not in context or "len(new_claim)" in context, (
                    f"Line {i + 1}: log.info still contains raw 'new_claim' "
                    "in log parameters."
                )

    def test_collision_detector_log_redacted(self) -> None:
        """Collision detector log must use hyp ID, not claim text."""
        from pathlib import Path

        source_path = (
            Path(__file__).resolve().parents[3]
            / "application"
            / "skill_compression"
            / "collision_detector.py"
        )
        source = source_path.read_text(encoding="utf-8")

        # Check that log.info doesn't contain winner.claim
        for i, line in enumerate(source.splitlines()):
            if "log.info(" in line and "collision" in line.lower():
                context = "\n".join(source.splitlines()[i : i + 5])
                assert "winner.claim" not in context, (
                    f"Line {i + 1}: log.info contains 'winner.claim'. "
                    "Must use hypothesis_id instead."
                )
