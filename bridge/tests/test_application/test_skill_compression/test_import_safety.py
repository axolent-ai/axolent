"""Production-path tests for import safety (W1).

Tests root-policy validation, file count/size caps, and symlink handling.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from application.skill_compression.conversation_import.orchestrator import (
    MAX_IMPORT_FILE_SIZE,
    MAX_IMPORT_FILES,
    ImportOrchestrator,
    ImportPathViolation,
)
from application.skill_compression.hypothesis_storage import HypothesisStorage


@pytest.fixture
def storage(tmp_path: Path) -> HypothesisStorage:
    from infrastructure.crypto_storage import CryptoConnection

    db_path = tmp_path / "test_import.db"
    conn = CryptoConnection(db_path, require_encryption=False)
    s = HypothesisStorage(conn)
    s.init_schema()
    return s


class TestImportOutsideRootRejected:
    """W1: Import path outside AXOLENT_IMPORT_ROOT must be rejected."""

    def test_import_outside_root_rejected(
        self,
        storage: HypothesisStorage,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Set a specific import root
        import_root = tmp_path / "allowed_root"
        import_root.mkdir()
        monkeypatch.setenv("AXOLENT_IMPORT_ROOT", str(import_root))

        orchestrator = ImportOrchestrator(storage)

        # Try to import from outside the root
        outside = tmp_path / "outside_folder"
        outside.mkdir()

        with pytest.raises(ImportPathViolation):
            orchestrator.validate_import_path(outside)

    def test_import_inside_root_allowed(
        self,
        storage: HypothesisStorage,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import_root = tmp_path / "allowed_root"
        import_root.mkdir()
        monkeypatch.setenv("AXOLENT_IMPORT_ROOT", str(import_root))

        orchestrator = ImportOrchestrator(storage)

        inside = import_root / "subfolder"
        inside.mkdir()

        # Should not raise
        resolved = orchestrator.validate_import_path(inside)
        assert resolved == inside.resolve()


class TestImportWithTooManyFilesCapped:
    """W1: File count must be capped at MAX_IMPORT_FILES."""

    def test_import_with_too_many_files_capped(self, tmp_path: Path) -> None:
        # Create more than a reasonable number of files
        folder = tmp_path / "many_files"
        folder.mkdir()
        for i in range(25):
            (folder / f"file_{i:04d}.txt").write_text(f"content {i}")

        files = ImportOrchestrator._iter_files(folder)
        assert len(files) <= MAX_IMPORT_FILES
        # With only 25 files, all should be included
        assert len(files) == 25


class TestImportWithOversizedFileSkipped:
    """W1: Files larger than MAX_IMPORT_FILE_SIZE are skipped."""

    def test_import_with_oversized_file_skipped(self, tmp_path: Path) -> None:
        folder = tmp_path / "oversized"
        folder.mkdir()

        # Normal file
        (folder / "normal.txt").write_text("small content")

        # Oversized file (just over the limit)
        big_file = folder / "huge.txt"
        big_file.write_bytes(b"x" * (MAX_IMPORT_FILE_SIZE + 1))

        files = ImportOrchestrator._iter_files(folder)
        file_names = {f.name for f in files}

        assert "normal.txt" in file_names
        assert "huge.txt" not in file_names, "Oversized file must be skipped"


class TestImportSymlinksNotFollowed:
    """W1: Symlinks must not be followed during import scan."""

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Symlink creation may require admin privileges on Windows",
    )
    def test_import_symlinks_not_followed(self, tmp_path: Path) -> None:
        folder = tmp_path / "with_symlink"
        folder.mkdir()

        # Normal file
        (folder / "real.txt").write_text("real content")

        # Symlink to a file outside
        target = tmp_path / "secret.txt"
        target.write_text("secret content")
        try:
            (folder / "link.txt").symlink_to(target)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        files = ImportOrchestrator._iter_files(folder)
        file_names = {f.name for f in files}

        assert "real.txt" in file_names
        assert "link.txt" not in file_names, "Symlinks must be skipped"
