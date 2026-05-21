"""Import Orchestrator for Skill-Compression conversation import.

Coordinates the import workflow:
  1. dry_run()      - Preview what would be imported (HC-SC-16)
  2. import_folder() - Execute the actual import with progress
  3. delete_from_source() - Cascade-delete imported data (HC-IMPORT-3)

HC-SC-16 [BLOCKER]: Strictly opt-in, dry-run first, progress display.
HC-IMPORT-1 [BLOCKER]: All imported hypotheses start as 'suggested'.
HC-IMPORT-2 [BLOCKER]: Raw input text never becomes hypothesis claim.
  Only structured patterns (intent, format, etc.) are stored.
HC-IMPORT-3 [BLOCKER]: Source deletable via cascade delete.

Architecture:
  - Lives in application layer (uses domain types + infra storage)
  - No cloud API calls (Mode B compliant)
  - All file I/O is local and read-only on source files
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from application.skill_compression.event_normalizer import (
    NormalizedEvent,
    compute_fingerprint,
    normalize_event,
)
from application.skill_compression.hypothesis_storage import (
    Hypothesis,
    HypothesisScope,
    HypothesisStorage,
)
from application.skill_compression.conversation_import.chatgpt_importer import (
    ChatGPTImporter,
)
from application.skill_compression.conversation_import.claude_importer import (
    ClaudeImporter,
)
from application.skill_compression.conversation_import.conversation_source import (
    ConversationSource,
    ParsedConversation,
)
from application.skill_compression.conversation_import.markdown_importer import (
    MarkdownImporter,
)
from application.skill_compression.conversation_import.plaintext_importer import (
    PlaintextImporter,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Schema DDL for import tracking (HC-IMPORT-3)
# ---------------------------------------------------------------

IMPORT_TRACKING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS import_sources (
    import_id       TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    file_count      INTEGER NOT NULL DEFAULT 0,
    conversation_count INTEGER NOT NULL DEFAULT 0,
    hypothesis_count INTEGER NOT NULL DEFAULT 0,
    imported_at     TEXT NOT NULL,
    user_id         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_import_sources_user
    ON import_sources(user_id);
CREATE INDEX IF NOT EXISTS idx_import_sources_path
    ON import_sources(source_path);

CREATE TABLE IF NOT EXISTS import_hypothesis_map (
    map_id          TEXT PRIMARY KEY,
    import_id       TEXT NOT NULL,
    hypothesis_id   TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    FOREIGN KEY (import_id) REFERENCES import_sources(import_id) ON DELETE CASCADE,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_import_map_import
    ON import_hypothesis_map(import_id);
CREATE INDEX IF NOT EXISTS idx_import_map_hypothesis
    ON import_hypothesis_map(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_import_map_source
    ON import_hypothesis_map(source_path);
"""


# ---------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilePreview:
    """Preview info for a single file in dry-run.

    Attributes:
        path: File path.
        source_type: Detected parser type.
        size_bytes: File size in bytes.
        conversation_count: Number of conversations detected.
    """

    path: str
    source_type: str
    size_bytes: int
    conversation_count: int


@dataclass(frozen=True, slots=True)
class DryRunResult:
    """Result of a dry-run import preview.

    Attributes:
        folder_path: The scanned folder path.
        files: Preview info for each parseable file.
        total_files_scanned: Total files examined.
        total_conversations: Total conversations detected.
        estimated_duration_seconds: Rough estimate of import duration.
    """

    folder_path: str
    files: tuple[FilePreview, ...]
    total_files_scanned: int
    total_conversations: int
    estimated_duration_seconds: float


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Result of an actual import operation.

    Attributes:
        import_id: Unique ID for this import (for later deletion).
        folder_path: The imported folder path.
        files_processed: Number of files processed.
        conversations_parsed: Number of conversations extracted.
        hypotheses_created: Number of new hypotheses created.
        hypotheses_skipped: Number skipped (duplicates or empty).
        duration_seconds: Actual duration.
        errors: List of error descriptions.
    """

    import_id: str
    folder_path: str
    files_processed: int
    conversations_parsed: int
    hypotheses_created: int
    hypotheses_skipped: int
    duration_seconds: float
    errors: tuple[str, ...] = ()


# ---------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------


class ImportOrchestrator:
    """Orchestrates conversation import from external sources.

    Thread safety: NOT thread-safe. Designed for single-threaded
    async context (Telegram bot handler).

    Usage:
        orchestrator = ImportOrchestrator(storage)
        result = orchestrator.dry_run(Path("/export/folder"))
        if user_confirms:
            import_result = orchestrator.import_folder(
                Path("/export/folder"),
                user_id=42,
            )
    """

    def __init__(
        self,
        storage: HypothesisStorage,
        *,
        sources: list[ConversationSource] | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            storage: Hypothesis storage for persisting results.
            sources: Custom list of parsers. Defaults to all built-in parsers.
        """
        self._storage = storage
        self._sources: list[ConversationSource] = sources or [
            ChatGPTImporter(),
            ClaudeImporter(),
            MarkdownImporter(),
            PlaintextImporter(),
        ]

    def init_schema(self) -> None:
        """Create import tracking tables (idempotent).

        Must be called before first import. Safe to call multiple times.
        """
        self._storage._conn.executescript(IMPORT_TRACKING_SCHEMA_SQL)
        log.info("Import tracking schema initialized")

    def dry_run(self, folder_path: Path) -> DryRunResult:
        """Preview what would be imported from a folder (HC-SC-16).

        Scans the folder recursively, detects parseable files, and
        counts conversations without creating any database records.

        Args:
            folder_path: Path to the folder to scan.

        Returns:
            DryRunResult with file and conversation counts.

        Raises:
            FileNotFoundError: If folder does not exist.
            NotADirectoryError: If path is not a directory.
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        files_scanned = 0
        file_previews: list[FilePreview] = []

        for file_path in self._iter_files(folder_path):
            files_scanned += 1

            source = self._find_source(file_path)
            if source is None:
                continue

            # Count conversations without storing
            conv_count = 0
            try:
                for _ in source.parse(file_path):
                    conv_count += 1
            except Exception:
                log.debug("Error during dry-run parse of %s", file_path, exc_info=True)
                continue

            if conv_count > 0:
                try:
                    size = file_path.stat().st_size
                except OSError:
                    size = 0

                source_type = self._detect_source_type(source)
                file_previews.append(
                    FilePreview(
                        path=str(file_path),
                        source_type=source_type,
                        size_bytes=size,
                        conversation_count=conv_count,
                    )
                )

        total_conversations = sum(fp.conversation_count for fp in file_previews)

        # Rough estimate: ~0.5s per conversation for pattern extraction
        estimated_duration = total_conversations * 0.5

        return DryRunResult(
            folder_path=str(folder_path),
            files=tuple(file_previews),
            total_files_scanned=files_scanned,
            total_conversations=total_conversations,
            estimated_duration_seconds=estimated_duration,
        )

    def import_folder(
        self,
        folder_path: Path,
        user_id: int,
        *,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> ImportResult:
        """Execute the actual import from a folder.

        Parses all files, extracts patterns, creates hypotheses as
        'suggested' status (HC-IMPORT-1).

        Args:
            folder_path: Path to the folder to import.
            user_id: Telegram user ID for the imported hypotheses.
            on_progress: Optional callback(files_done, total_files, hypotheses_found).

        Returns:
            ImportResult with counts and the import_id for later deletion.

        Raises:
            FileNotFoundError: If folder does not exist.
            NotADirectoryError: If path is not a directory.
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        start_time = datetime.now(timezone.utc)
        import_id = f"imp_{uuid4().hex[:16]}"
        now_iso = start_time.isoformat()

        # Insert import_sources record FIRST so FK constraints work
        # for import_hypothesis_map entries created during processing.
        # Counts will be updated at the end.
        self._storage._conn.execute(
            """INSERT INTO import_sources (
                import_id, source_path, source_type, file_count,
                conversation_count, hypothesis_count, imported_at, user_id
            ) VALUES (?, ?, ?, 0, 0, 0, ?, ?)""",
            (import_id, str(folder_path), "mixed", now_iso, user_id),
        )

        # Collect parseable files first for progress tracking
        parseable: list[tuple[Path, ConversationSource]] = []
        for file_path in self._iter_files(folder_path):
            source = self._find_source(file_path)
            if source is not None:
                parseable.append((file_path, source))

        total_files = len(parseable)
        files_processed = 0
        conversations_parsed = 0
        hypotheses_created = 0
        hypotheses_skipped = 0
        errors: list[str] = []

        for file_path, source in parseable:
            try:
                for conversation in source.parse(file_path):
                    conversations_parsed += 1

                    # Extract patterns from conversation (HC-IMPORT-2)
                    created = self._extract_and_store_patterns(
                        conversation=conversation,
                        user_id=user_id,
                        import_id=import_id,
                    )

                    if created:
                        hypotheses_created += created
                    else:
                        hypotheses_skipped += 1

            except Exception as exc:
                error_msg = f"{file_path}: {exc}"
                errors.append(error_msg)
                log.warning("Import error for %s: %s", file_path, exc)

            files_processed += 1

            if on_progress is not None:
                on_progress(files_processed, total_files, hypotheses_created)

        # Update import source record with final counts (HC-IMPORT-3)
        self._storage._conn.execute(
            """UPDATE import_sources SET
                file_count = ?, conversation_count = ?, hypothesis_count = ?
            WHERE import_id = ?""",
            (files_processed, conversations_parsed, hypotheses_created, import_id),
        )

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        log.info(
            "Import complete: id=%s files=%d conversations=%d "
            "hypotheses=%d skipped=%d duration=%.1fs",
            import_id,
            files_processed,
            conversations_parsed,
            hypotheses_created,
            hypotheses_skipped,
            duration,
        )

        return ImportResult(
            import_id=import_id,
            folder_path=str(folder_path),
            files_processed=files_processed,
            conversations_parsed=conversations_parsed,
            hypotheses_created=hypotheses_created,
            hypotheses_skipped=hypotheses_skipped,
            duration_seconds=duration,
            errors=tuple(errors),
        )

    def delete_from_source(self, import_id: str) -> int:
        """Delete all hypotheses imported from a specific source (HC-IMPORT-3).

        Cascade-deletes:
          1. All hypotheses linked to this import
          2. All evidence linked to those hypotheses
          3. The import tracking record itself

        Args:
            import_id: The import ID returned by import_folder.

        Returns:
            Number of hypotheses deleted.
        """
        # Find all hypothesis IDs from this import
        rows = self._storage._conn.fetchall(
            "SELECT hypothesis_id FROM import_hypothesis_map WHERE import_id = ?",
            (import_id,),
        )

        deleted_count = 0
        for row in rows:
            hyp_id = row["hypothesis_id"]

            # Delete evidence for this hypothesis
            self._storage._conn.execute(
                "DELETE FROM hypothesis_evidence WHERE hypothesis_id = ?",
                (hyp_id,),
            )

            # Delete the hypothesis itself
            self._storage._conn.execute(
                "DELETE FROM hypotheses WHERE hypothesis_id = ?",
                (hyp_id,),
            )

            deleted_count += 1

        # Delete the import mapping records
        self._storage._conn.execute(
            "DELETE FROM import_hypothesis_map WHERE import_id = ?",
            (import_id,),
        )

        # Delete the import source record
        self._storage._conn.execute(
            "DELETE FROM import_sources WHERE import_id = ?",
            (import_id,),
        )

        log.info(
            "Deleted import source %s: %d hypotheses removed",
            import_id,
            deleted_count,
        )

        return deleted_count

    def get_import_sources(self, user_id: int) -> list[dict]:
        """List all import sources for a user.

        Args:
            user_id: Telegram user ID.

        Returns:
            List of import source dicts.
        """
        rows = self._storage._conn.fetchall(
            "SELECT * FROM import_sources WHERE user_id = ? ORDER BY imported_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------

    def _extract_and_store_patterns(
        self,
        conversation: ParsedConversation,
        user_id: int,
        import_id: str,
    ) -> int:
        """Extract patterns from a conversation and store as hypotheses.

        HC-IMPORT-1: All hypotheses start as 'suggested'.
        HC-IMPORT-2: Only structured patterns, no raw input text in claim.

        Args:
            conversation: Parsed conversation data.
            user_id: Telegram user ID.
            import_id: Import session ID for tracking.

        Returns:
            Number of hypotheses created.
        """
        created = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        # Process each user message through the event normalizer
        for user_msg in conversation.user_messages:
            if not user_msg.strip():
                continue

            # Normalize the event to extract structured fields
            event = normalize_event(
                text=user_msg,
                user_id=user_id,
            )

            if event is None:
                continue

            # Skip trivial events (no clear intent detected)
            if event.intent == "unknown" and event.domain == "unknown":
                continue

            # Build a structured claim from extracted fields (HC-IMPORT-2)
            claim = self._build_structured_claim(event)
            if not claim:
                continue

            # Compute fingerprint for deduplication
            fp = compute_fingerprint(
                intent=event.intent,
                domain=event.domain,
                format_type=event.format_type,
                constraints={},
                scope={},
                language=event.language,
            )

            # Check for existing hypothesis with same fingerprint
            existing = self._storage._conn.fetchone(
                "SELECT hypothesis_id FROM hypotheses "
                "WHERE user_id = ? AND pattern_hash = ?",
                (user_id, fp),
            )

            if existing is not None:
                # Update existing hypothesis: add evidence count
                self._storage.update_hypothesis_support(
                    hypothesis_id=existing["hypothesis_id"],
                    increment_support=1,
                    last_seen=now_iso,
                )
                continue

            # Create new hypothesis (HC-IMPORT-1: status = suggested)
            hyp_id = f"hyp_{uuid4().hex[:16]}"
            hypothesis = Hypothesis(
                hypothesis_id=hyp_id,
                user_id=user_id,
                type="preference",
                scope=HypothesisScope(),
                claim=claim,
                status="suggested",  # HC-IMPORT-1: NEVER 'active'
                version=1,
                elo_rating=1500.0,
                source_type="import",
                decay_immune=False,
                created_at=now_iso,
                last_seen=now_iso,
                pattern_hash=fp,
            )

            self._storage.insert_hypothesis(hypothesis)

            # Record import mapping (HC-IMPORT-3)
            map_id = f"imap_{uuid4().hex[:16]}"
            self._storage._conn.execute(
                """INSERT INTO import_hypothesis_map (
                    map_id, import_id, hypothesis_id, source_path
                ) VALUES (?, ?, ?, ?)""",
                (map_id, import_id, hyp_id, conversation.source_path),
            )

            # Add evidence record with import source tracking
            evidence_id = f"ev_{uuid4().hex[:16]}"
            self._storage.insert_evidence(
                evidence_id=evidence_id,
                hypothesis_id=hyp_id,
                hypothesis_version=1,
                signal_type="no_correction",
                signal_strength=0.5,  # Lower confidence for imports
                created_at=now_iso,
            )

            created += 1

        return created

    @staticmethod
    def _build_structured_claim(event: NormalizedEvent) -> str:
        """Build a structured claim from normalized event fields.

        HC-IMPORT-2: Only structured patterns, no raw input text.
        The claim describes WHAT pattern was detected, not the
        original user message.

        Args:
            event: Normalized event from the event normalizer.

        Returns:
            Structured claim string, or empty string if trivial.
        """
        parts: list[str] = []

        if event.intent and event.intent != "unknown":
            parts.append(f"intent={event.intent}")

        if event.domain and event.domain != "unknown":
            parts.append(f"domain={event.domain}")

        if event.format_type and event.format_type != "unknown":
            parts.append(f"format={event.format_type}")

        if event.language and event.language != "unknown":
            parts.append(f"language={event.language}")

        if not parts:
            return ""

        return f"Imported pattern: {', '.join(parts)}"

    def _find_source(self, path: Path) -> ConversationSource | None:
        """Find the first source that can handle a file.

        Tries each registered source in order. Returns the first
        match, or None if no source can handle the file.

        Args:
            path: File path to check.

        Returns:
            ConversationSource or None.
        """
        for source in self._sources:
            try:
                if source.can_handle(path):
                    return source
            except Exception:
                log.debug(
                    "Source %s raised exception for %s",
                    type(source).__name__,
                    path,
                    exc_info=True,
                )
        return None

    @staticmethod
    def _detect_source_type(source: ConversationSource) -> str:
        """Detect the source type string from a source instance.

        Args:
            source: ConversationSource instance.

        Returns:
            Source type identifier string.
        """
        type_map = {
            "ChatGPTImporter": "chatgpt",
            "ClaudeImporter": "claude",
            "MarkdownImporter": "markdown",
            "PlaintextImporter": "plaintext",
        }
        return type_map.get(type(source).__name__, "unknown")

    @staticmethod
    def _iter_files(folder_path: Path) -> list[Path]:
        """Iterate over all files in a folder recursively.

        Returns files sorted by name for deterministic ordering.
        Skips hidden files and directories (starting with '.').

        Args:
            folder_path: Root folder to scan.

        Returns:
            Sorted list of file paths.
        """
        files: list[Path] = []
        try:
            for item in sorted(folder_path.rglob("*")):
                if item.is_file() and not any(
                    part.startswith(".") for part in item.parts
                ):
                    files.append(item)
        except PermissionError:
            log.warning("Permission denied scanning: %s", folder_path)
        return files
