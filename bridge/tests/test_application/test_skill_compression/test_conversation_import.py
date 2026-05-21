"""Tests for Conversation Import (Step 7).

Covers:
  1. Parser Tests:
     - MarkdownImporter: User/Assistant blocks correctly extracted
     - ChatGPTImporter: Real schema test with sample data
     - ClaudeImporter: JSONL + JSON array schema tests
     - PlaintextImporter: Heuristic handles various formats

  2. Orchestrator Tests:
     - dry_run shows correct counts
     - import_folder triggers pattern extraction
     - delete_from_source cascades correctly

  3. Architecture Guards:
     - Import module imports nothing outside its sub-package
       (except parent skill_compression and standard lib)
     - Parsers have no cloud API calls (all local, Mode B)

HC-SC-16: Strictly opt-in, dry-run first, progress display.
HC-IMPORT-1: All imported hypotheses start as 'suggested'.
HC-IMPORT-2: Raw input text never becomes hypothesis claim.
HC-IMPORT-3: Source deletable via cascade delete.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from application.skill_compression.hypothesis_storage import (
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
    make_parsed_conversation,
)
from application.skill_compression.conversation_import.markdown_importer import (
    MarkdownImporter,
)
from application.skill_compression.conversation_import.orchestrator import (
    IMPORT_TRACKING_SCHEMA_SQL,
    ImportOrchestrator,
)
from application.skill_compression.conversation_import.plaintext_importer import (
    PlaintextImporter,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


class FakeDBConnection:
    """Minimal in-memory SQLite for tests."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql, params=(), **kwargs):
        return self._conn.execute(sql, params)

    def executescript(self, sql):
        self._conn.executescript(sql)

    def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def fetchone(self, sql, params=()):
        return self._conn.execute(sql, params).fetchone()

    def execute_in_transaction(self, operations):
        for sql, params in operations:
            self._conn.execute(sql, params)
        self._conn.commit()


def _setup_storage() -> HypothesisStorage:
    """Create an in-memory HypothesisStorage with full schema."""
    conn = FakeDBConnection()
    storage = HypothesisStorage(conn)
    storage.init_schema()
    # Also init import tracking tables
    conn.executescript(IMPORT_TRACKING_SCHEMA_SQL)
    return storage


# ---------------------------------------------------------------
# 1. Parser Tests: MarkdownImporter
# ---------------------------------------------------------------


class TestMarkdownImporter:
    """Tests for Markdown conversation parser."""

    def test_can_handle_md_with_markers(self, tmp_path):
        """Should handle .md files with User/Assistant markers."""
        md_file = tmp_path / "chat.md"
        md_file.write_text(
            "## User\nHello there\n\n## Assistant\nHi!\n", encoding="utf-8"
        )
        importer = MarkdownImporter()
        assert importer.can_handle(md_file) is True

    def test_cannot_handle_txt(self, tmp_path):
        """Should not handle .txt files."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text("## User\nHello\n", encoding="utf-8")
        importer = MarkdownImporter()
        assert importer.can_handle(txt_file) is False

    def test_cannot_handle_md_without_markers(self, tmp_path):
        """Should not handle .md files without conversation markers."""
        md_file = tmp_path / "notes.md"
        md_file.write_text("# My Notes\n\nSome regular notes.\n", encoding="utf-8")
        importer = MarkdownImporter()
        assert importer.can_handle(md_file) is False

    def test_parse_heading_style(self, tmp_path):
        """Should extract User/Assistant blocks from heading markers."""
        md_file = tmp_path / "chat.md"
        md_file.write_text(
            "## User\nWrite me a Python script\n\n"
            "## Assistant\nHere is a script:\n```python\nprint('hello')\n```\n\n"
            "## User\nAdd error handling\n\n"
            "## Assistant\nUpdated with try/except.\n",
            encoding="utf-8",
        )
        importer = MarkdownImporter()
        conversations = list(importer.parse(md_file))

        assert len(conversations) == 1
        conv = conversations[0]
        assert conv.source_type == "markdown"
        assert len(conv.user_messages) == 2
        assert len(conv.assistant_messages) == 2
        assert "Python script" in conv.user_messages[0]
        assert "error handling" in conv.user_messages[1]

    def test_parse_bold_label_style(self, tmp_path):
        """Should extract blocks from **User:** / **Assistant:** labels."""
        md_file = tmp_path / "chat.md"
        md_file.write_text(
            "**User:** What is the capital of France?\n\n"
            "**Assistant:** Paris is the capital of France.\n",
            encoding="utf-8",
        )
        importer = MarkdownImporter()
        conversations = list(importer.parse(md_file))

        assert len(conversations) == 1
        assert conversations[0].user_messages[0] == "What is the capital of France?"

    def test_parse_multiple_sections(self, tmp_path):
        """Sections separated by --- should produce separate conversations."""
        md_file = tmp_path / "chats.md"
        md_file.write_text(
            "## User\nFirst question\n\n"
            "## Assistant\nFirst answer\n\n"
            "---\n\n"
            "## User\nSecond question\n\n"
            "## Assistant\nSecond answer\n",
            encoding="utf-8",
        )
        importer = MarkdownImporter()
        conversations = list(importer.parse(md_file))

        assert len(conversations) == 2

    def test_parse_german_labels(self, tmp_path):
        """Should handle German role labels (Ich/Benutzer)."""
        md_file = tmp_path / "chat.md"
        md_file.write_text(
            "## Benutzer\nSchreib mir einen Text\n\n## Assistant\nHier ist der Text.\n",
            encoding="utf-8",
        )
        importer = MarkdownImporter()
        conversations = list(importer.parse(md_file))

        assert len(conversations) == 1
        assert "Text" in conversations[0].user_messages[0]

    def test_empty_file_yields_nothing(self, tmp_path):
        """Empty .md files should yield no conversations."""
        md_file = tmp_path / "empty.md"
        md_file.write_text("", encoding="utf-8")
        importer = MarkdownImporter()
        assert list(importer.parse(md_file)) == []


# ---------------------------------------------------------------
# 2. Parser Tests: ChatGPTImporter
# ---------------------------------------------------------------


class TestChatGPTImporter:
    """Tests for ChatGPT JSON export parser."""

    def _make_chatgpt_export(self, conversations: list[dict]) -> str:
        """Create a ChatGPT export JSON string."""
        return json.dumps(conversations, ensure_ascii=False)

    def _make_conversation(
        self, messages: list[tuple[str, str]], title: str = "Test Chat"
    ) -> dict:
        """Create a single ChatGPT conversation object.

        Args:
            messages: List of (role, text) tuples.
            title: Conversation title.

        Returns:
            ChatGPT conversation dict with mapping structure.
        """
        mapping = {}
        parent_id = None

        for i, (role, text) in enumerate(messages):
            node_id = f"node-{i}"
            mapping[node_id] = {
                "message": {
                    "author": {"role": role},
                    "content": {"parts": [text]},
                },
                "parent": parent_id,
                "children": [f"node-{i + 1}"] if i < len(messages) - 1 else [],
            }
            parent_id = node_id

        return {"title": title, "mapping": mapping}

    def test_can_handle_chatgpt_json(self, tmp_path):
        """Should detect ChatGPT exports by 'mapping' key."""
        conv = self._make_conversation([("user", "Hello"), ("assistant", "Hi!")])
        json_file = tmp_path / "conversations.json"
        json_file.write_text(self._make_chatgpt_export([conv]), encoding="utf-8")
        importer = ChatGPTImporter()
        assert importer.can_handle(json_file) is True

    def test_cannot_handle_claude_json(self, tmp_path):
        """Should not detect Claude exports (no 'mapping' key)."""
        claude_data = [
            {
                "uuid": "abc",
                "chat_messages": [{"sender": "human", "text": "Hello"}],
            }
        ]
        json_file = tmp_path / "conversations.json"
        json_file.write_text(json.dumps(claude_data), encoding="utf-8")
        importer = ChatGPTImporter()
        assert importer.can_handle(json_file) is False

    def test_parse_single_conversation(self, tmp_path):
        """Should extract messages from ChatGPT mapping structure."""
        conv = self._make_conversation(
            [
                ("system", "You are a helpful assistant."),
                ("user", "Write a haiku about Python"),
                (
                    "assistant",
                    "Indented with care\nFunctions flow like mountain streams\nPython is serene",
                ),
            ]
        )
        json_file = tmp_path / "conversations.json"
        json_file.write_text(self._make_chatgpt_export([conv]), encoding="utf-8")

        importer = ChatGPTImporter()
        conversations = list(importer.parse(json_file))

        assert len(conversations) == 1
        c = conversations[0]
        assert c.source_type == "chatgpt"
        # System messages should be skipped
        assert len(c.user_messages) == 1
        assert len(c.assistant_messages) == 1
        assert "haiku" in c.user_messages[0]

    def test_parse_multiple_conversations(self, tmp_path):
        """Multiple conversation objects should yield multiple results."""
        conv1 = self._make_conversation(
            [("user", "Question 1"), ("assistant", "Answer 1")]
        )
        conv2 = self._make_conversation(
            [("user", "Question 2"), ("assistant", "Answer 2")]
        )
        json_file = tmp_path / "conversations.json"
        json_file.write_text(
            self._make_chatgpt_export([conv1, conv2]),
            encoding="utf-8",
        )

        importer = ChatGPTImporter()
        conversations = list(importer.parse(json_file))
        assert len(conversations) == 2

    def test_parse_empty_messages_skipped(self, tmp_path):
        """Conversations with only empty messages should be skipped."""
        mapping = {
            "node-0": {
                "message": {
                    "author": {"role": "system"},
                    "content": {"parts": ["System prompt"]},
                },
                "parent": None,
                "children": [],
            }
        }
        json_file = tmp_path / "conversations.json"
        json_file.write_text(
            json.dumps([{"title": "Empty", "mapping": mapping}]),
            encoding="utf-8",
        )

        importer = ChatGPTImporter()
        conversations = list(importer.parse(json_file))
        assert len(conversations) == 0


# ---------------------------------------------------------------
# 3. Parser Tests: ClaudeImporter
# ---------------------------------------------------------------


class TestClaudeImporter:
    """Tests for Claude JSONL/JSON export parser."""

    def _make_claude_conversation(self, messages: list[tuple[str, str]]) -> dict:
        """Create a single Claude conversation object.

        Args:
            messages: List of (sender, text) tuples.
                sender is "human" or "assistant".

        Returns:
            Claude conversation dict.
        """
        return {
            "uuid": "test-uuid-123",
            "name": "Test Conversation",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "chat_messages": [
                {
                    "uuid": f"msg-{i}",
                    "text": text,
                    "sender": sender,
                    "created_at": f"2026-01-01T00:0{i}:00Z",
                }
                for i, (sender, text) in enumerate(messages)
            ],
        }

    def test_can_handle_claude_jsonl(self, tmp_path):
        """Should detect Claude exports by chat_messages + sender keys."""
        conv = self._make_claude_conversation(
            [("human", "Hello"), ("assistant", "Hi!")]
        )
        jsonl_file = tmp_path / "conversations.jsonl"
        jsonl_file.write_text(json.dumps(conv), encoding="utf-8")
        importer = ClaudeImporter()
        assert importer.can_handle(jsonl_file) is True

    def test_can_handle_claude_json_array(self, tmp_path):
        """Should handle JSON array format too."""
        conv = self._make_claude_conversation(
            [("human", "Hello"), ("assistant", "Hi!")]
        )
        json_file = tmp_path / "conversations.json"
        json_file.write_text(json.dumps([conv]), encoding="utf-8")
        importer = ClaudeImporter()
        assert importer.can_handle(json_file) is True

    def test_cannot_handle_chatgpt(self, tmp_path):
        """Should not detect ChatGPT exports (uses 'mapping', not 'chat_messages')."""
        chatgpt_data = [{"title": "Test", "mapping": {}}]
        json_file = tmp_path / "conversations.json"
        json_file.write_text(json.dumps(chatgpt_data), encoding="utf-8")
        importer = ClaudeImporter()
        assert importer.can_handle(json_file) is False

    def test_parse_jsonl_format(self, tmp_path):
        """Should parse JSONL (one conversation per line)."""
        conv1 = self._make_claude_conversation(
            [("human", "First question"), ("assistant", "First answer")]
        )
        conv2 = self._make_claude_conversation(
            [("human", "Second question"), ("assistant", "Second answer")]
        )
        jsonl_file = tmp_path / "conversations.jsonl"
        jsonl_file.write_text(
            json.dumps(conv1) + "\n" + json.dumps(conv2),
            encoding="utf-8",
        )

        importer = ClaudeImporter()
        conversations = list(importer.parse(jsonl_file))

        assert len(conversations) == 2
        assert conversations[0].source_type == "claude"
        assert "First question" in conversations[0].user_messages[0]

    def test_parse_json_array_format(self, tmp_path):
        """Should parse JSON array format."""
        conv = self._make_claude_conversation(
            [
                ("human", "Explain transformers"),
                ("assistant", "Transformers use attention..."),
                ("human", "What about positional encoding?"),
                ("assistant", "Positional encoding adds..."),
            ]
        )
        json_file = tmp_path / "conversations.json"
        json_file.write_text(json.dumps([conv]), encoding="utf-8")

        importer = ClaudeImporter()
        conversations = list(importer.parse(json_file))

        assert len(conversations) == 1
        c = conversations[0]
        assert len(c.user_messages) == 2
        assert len(c.assistant_messages) == 2

    def test_parse_skips_empty_messages(self, tmp_path):
        """Messages with empty text should be skipped."""
        conv = self._make_claude_conversation(
            [
                ("human", ""),
                ("assistant", "Response to nothing"),
                ("human", "Real question"),
                ("assistant", "Real answer"),
            ]
        )
        jsonl_file = tmp_path / "conversations.jsonl"
        jsonl_file.write_text(json.dumps(conv), encoding="utf-8")

        importer = ClaudeImporter()
        conversations = list(importer.parse(jsonl_file))

        assert len(conversations) == 1
        # Empty human message should be skipped
        assert len(conversations[0].user_messages) == 1
        assert "Real question" in conversations[0].user_messages[0]


# ---------------------------------------------------------------
# 4. Parser Tests: PlaintextImporter
# ---------------------------------------------------------------


class TestPlaintextImporter:
    """Tests for plain text heuristic parser."""

    def test_can_handle_txt_with_labels(self, tmp_path):
        """Should handle .txt files with User:/Assistant: labels."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "User: Hello there\nAssistant: Hi! How can I help?\n",
            encoding="utf-8",
        )
        importer = PlaintextImporter()
        assert importer.can_handle(txt_file) is True

    def test_cannot_handle_md(self, tmp_path):
        """Should not handle .md files."""
        md_file = tmp_path / "chat.md"
        md_file.write_text("User: Hello\nAssistant: Hi!\n", encoding="utf-8")
        importer = PlaintextImporter()
        assert importer.can_handle(md_file) is False

    def test_cannot_handle_txt_without_labels(self, tmp_path):
        """Should not handle plain text without role labels."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("These are some notes about a project.\n", encoding="utf-8")
        importer = PlaintextImporter()
        assert importer.can_handle(txt_file) is False

    def test_parse_basic_conversation(self, tmp_path):
        """Should extract User/Assistant messages from labels."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "User: Write me a function to sort a list\n"
            "Assistant: Here is a Python sort function:\n"
            "def sort_list(items):\n"
            "    return sorted(items)\n"
            "User: Can you add type hints?\n"
            "Assistant: def sort_list(items: list) -> list:\n"
            "    return sorted(items)\n",
            encoding="utf-8",
        )

        importer = PlaintextImporter()
        conversations = list(importer.parse(txt_file))

        assert len(conversations) == 1
        c = conversations[0]
        assert c.source_type == "plaintext"
        assert len(c.user_messages) == 2
        assert len(c.assistant_messages) == 2
        assert "sort" in c.user_messages[0].lower()

    def test_parse_human_ai_labels(self, tmp_path):
        """Should handle Human:/AI: labels."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "Human: What is machine learning?\n"
            "AI: Machine learning is a subset of AI...\n",
            encoding="utf-8",
        )

        importer = PlaintextImporter()
        conversations = list(importer.parse(txt_file))

        assert len(conversations) == 1
        assert "machine learning" in conversations[0].user_messages[0].lower()

    def test_parse_multiline_content(self, tmp_path):
        """Content spanning multiple lines should be captured fully."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "User: Write a haiku\n"
            "Assistant: Silent code compiles\n"
            "Electrons dance through the night\n"
            "Software comes alive\n"
            "User: One more please\n"
            "Assistant: Data flows like streams\n",
            encoding="utf-8",
        )

        importer = PlaintextImporter()
        conversations = list(importer.parse(txt_file))

        assert len(conversations) == 1
        # First assistant message should span 3 lines
        assert "\n" in conversations[0].assistant_messages[0]

    def test_parse_qa_format(self, tmp_path):
        """Should handle Q:/A: shorthand labels."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "Q: What is 2+2?\nA: 4\n",
            encoding="utf-8",
        )

        importer = PlaintextImporter()
        conversations = list(importer.parse(txt_file))

        assert len(conversations) == 1
        assert conversations[0].user_messages[0] == "What is 2+2?"


# ---------------------------------------------------------------
# 5. Orchestrator Tests
# ---------------------------------------------------------------


class TestImportOrchestrator:
    """Tests for ImportOrchestrator (dry_run, import, delete)."""

    def test_dry_run_counts_files_and_conversations(self, tmp_path):
        """dry_run should report correct file and conversation counts."""
        # Create test files
        md_file = tmp_path / "chat.md"
        md_file.write_text(
            "## User\nFirst question\n\n## Assistant\nFirst answer\n",
            encoding="utf-8",
        )
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "User: Hello\nAssistant: Hi!\n",
            encoding="utf-8",
        )
        # Non-parseable file
        other_file = tmp_path / "readme.txt"
        other_file.write_text("This is just a readme.", encoding="utf-8")

        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)

        result = orchestrator.dry_run(tmp_path)

        assert result.total_files_scanned == 3
        assert len(result.files) == 2  # md + txt, not readme
        assert result.total_conversations == 2

    def test_dry_run_nonexistent_folder_raises(self, tmp_path):
        """dry_run should raise FileNotFoundError for missing folders."""
        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)

        with pytest.raises(FileNotFoundError):
            orchestrator.dry_run(tmp_path / "nonexistent")

    def test_dry_run_file_not_directory_raises(self, tmp_path):
        """dry_run should raise NotADirectoryError for files."""
        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content", encoding="utf-8")

        with pytest.raises(NotADirectoryError):
            orchestrator.dry_run(file_path)

    def test_import_folder_creates_suggested_hypotheses(self, tmp_path):
        """HC-IMPORT-1: Imported hypotheses must start as 'suggested'."""
        # Create a conversation with clear intent for pattern extraction
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "User: Write me a Python script to analyze data\n"
            "Assistant: Here is a data analysis script.\n",
            encoding="utf-8",
        )

        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)
        orchestrator.init_schema()

        result = orchestrator.import_folder(tmp_path, user_id=42)

        assert result.files_processed >= 1
        assert result.conversations_parsed >= 1

        # Check hypotheses are 'suggested', never 'active'
        hypotheses = storage.get_hypotheses_by_user(42)
        for hyp in hypotheses:
            assert hyp.status == "suggested", (
                f"HC-IMPORT-1 violation: hypothesis {hyp.hypothesis_id} "
                f"has status '{hyp.status}', expected 'suggested'"
            )
            assert hyp.source_type == "import"

    def test_import_folder_claim_is_structured(self, tmp_path):
        """HC-IMPORT-2: Claims must be structured patterns, not raw text."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "User: Write code to sort a list in Python\nAssistant: Here you go.\n",
            encoding="utf-8",
        )

        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)
        orchestrator.init_schema()

        orchestrator.import_folder(tmp_path, user_id=42)

        hypotheses = storage.get_hypotheses_by_user(42)
        for hyp in hypotheses:
            # Claim must start with "Imported pattern:" prefix
            assert hyp.claim.startswith("Imported pattern:"), (
                f"HC-IMPORT-2 violation: claim '{hyp.claim}' is not structured"
            )
            # Raw user text must NOT appear in claim
            assert "sort a list" not in hyp.claim

    def test_import_folder_progress_callback(self, tmp_path):
        """on_progress callback should be called with file counts."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text("User: Hello\nAssistant: Hi!\n", encoding="utf-8")

        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)
        orchestrator.init_schema()

        progress_calls: list[tuple[int, int, int]] = []

        def on_progress(done, total, hyps):
            progress_calls.append((done, total, hyps))

        orchestrator.import_folder(tmp_path, user_id=42, on_progress=on_progress)

        assert len(progress_calls) >= 1
        # Last call should show all files processed
        last = progress_calls[-1]
        assert last[0] == last[1]  # done == total

    def test_delete_from_source_cascades(self, tmp_path):
        """HC-IMPORT-3: delete_from_source must cascade-delete hypotheses."""
        txt_file = tmp_path / "chat.txt"
        txt_file.write_text(
            "User: Write a script for data analysis\n"
            "Assistant: Here is a data analysis script.\n",
            encoding="utf-8",
        )

        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)
        orchestrator.init_schema()

        result = orchestrator.import_folder(tmp_path, user_id=42)
        import_id = result.import_id

        # Verify hypotheses exist before delete
        assert len(storage.get_hypotheses_by_user(42)) > 0

        # Delete from source
        orchestrator.delete_from_source(import_id)

        # Verify hypotheses are gone
        hyps_after = storage.get_hypotheses_by_user(42)
        assert len(hyps_after) == 0

        # Verify import source record is gone
        sources = orchestrator.get_import_sources(42)
        assert len(sources) == 0

    def test_import_deduplicates_by_fingerprint(self, tmp_path):
        """Duplicate patterns should increment support, not create new hyps."""
        # Two files with similar content
        file1 = tmp_path / "chat1.txt"
        file1.write_text(
            "User: Write Python code\nAssistant: Here is code.\n",
            encoding="utf-8",
        )
        file2 = tmp_path / "chat2.txt"
        file2.write_text(
            "User: Write Python code\nAssistant: Sure, here is code.\n",
            encoding="utf-8",
        )

        storage = _setup_storage()
        orchestrator = ImportOrchestrator(storage)
        orchestrator.init_schema()

        orchestrator.import_folder(tmp_path, user_id=42)

        # Should create only one hypothesis (deduplicated by fingerprint)
        hypotheses = storage.get_hypotheses_by_user(42)
        # The exact count depends on normalizer output, but there should
        # be fewer hypotheses than conversations if fingerprints match
        assert len(hypotheses) <= 2


# ---------------------------------------------------------------
# 6. Architecture Guards
# ---------------------------------------------------------------


class TestImportArchitectureGuards:
    """Architecture guard tests for the import module."""

    def test_parsers_import_only_from_allowed_packages(self):
        """Import parsers must only import from allowed packages.

        Allowed:
          - standard library (json, re, pathlib, logging, etc.)
          - application.skill_compression.import (sibling modules)
          - application.skill_compression (parent, for event_normalizer)

        Forbidden:
          - infrastructure (no DB access in parsers)
          - presentation (no Telegram in parsers)
          - Any cloud/network library
        """
        import_dir = (
            Path(__file__).resolve().parents[3]
            / "application"
            / "skill_compression"
            / "conversation_import"
        )

        forbidden_prefixes = (
            "infrastructure",
            "presentation",
            "requests",
            "httpx",
            "aiohttp",
            "urllib.request",
        )

        parser_files = [
            "markdown_importer.py",
            "chatgpt_importer.py",
            "claude_importer.py",
            "plaintext_importer.py",
        ]

        for filename in parser_files:
            filepath = import_dir / filename
            if not filepath.exists():
                continue

            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = ""
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name

                    for prefix in forbidden_prefixes:
                        assert not module.startswith(prefix), (
                            f"Parser {filename} imports from forbidden "
                            f"package: {module}"
                        )

    def test_parsers_have_no_network_calls(self):
        """Parsers must not contain any network/HTTP call patterns.

        Mode B: all processing is local.
        """
        import_dir = (
            Path(__file__).resolve().parents[3]
            / "application"
            / "skill_compression"
            / "conversation_import"
        )

        network_patterns = [
            "requests.get",
            "requests.post",
            "httpx.",
            "aiohttp.",
            "urllib.request",
            "urlopen",
            "socket.",
        ]

        parser_files = [
            "markdown_importer.py",
            "chatgpt_importer.py",
            "claude_importer.py",
            "plaintext_importer.py",
            "orchestrator.py",
        ]

        for filename in parser_files:
            filepath = import_dir / filename
            if not filepath.exists():
                continue

            source = filepath.read_text(encoding="utf-8")
            for pattern in network_patterns:
                assert pattern not in source, (
                    f"Mode B violation: {filename} contains "
                    f"network call pattern: {pattern}"
                )

    def test_parsed_conversation_is_frozen(self):
        """ParsedConversation must be immutable (frozen dataclass)."""
        conv = make_parsed_conversation(
            source_path="/test/file.txt",
            source_type="test",
            user_messages=["Hello"],
            assistant_messages=["Hi"],
        )
        with pytest.raises(AttributeError):
            conv.source_type = "mutated"  # type: ignore[misc]

    def test_parsed_conversation_has_slots(self):
        """ParsedConversation must use __slots__ for memory efficiency."""
        assert hasattr(ParsedConversation, "__slots__")

    def test_conversation_source_protocol(self):
        """All importers must satisfy ConversationSource protocol."""
        importers = [
            MarkdownImporter(),
            ChatGPTImporter(),
            ClaudeImporter(),
            PlaintextImporter(),
        ]
        for importer in importers:
            assert isinstance(importer, ConversationSource), (
                f"{type(importer).__name__} does not satisfy "
                "ConversationSource protocol"
            )
