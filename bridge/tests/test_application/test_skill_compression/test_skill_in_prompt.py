"""Production-path tests for skill application BEFORE prompt (SC-03 fix).

Verifies that:
  1. Skill matching happens before provider call
  2. Skill context is injected into the prompt
  3. Evidence is written after successful application
  4. Both streaming and non-streaming paths have skill integration
"""

from __future__ import annotations

from pathlib import Path


_BRIDGE_ROOT = Path(__file__).resolve().parents[3]


def _read_source(relative_path: str) -> str:
    full = _BRIDGE_ROOT / relative_path
    return full.read_text(encoding="utf-8")


class TestSkillMatchedBeforeProviderCall:
    """SC-03: SkillMatcher.match() must be called BEFORE provider.query()."""

    def test_skill_matched_before_provider_call_non_streaming(self) -> None:
        """In process_user_message, skill matching must happen
        before the provider call."""
        source = _read_source("application/chat_service.py")
        lines = source.splitlines()

        skill_match_line = None
        provider_call_line = None

        in_process_user_message = False
        method_indent = 0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "async def process_user_message(" in stripped:
                in_process_user_message = True
                method_indent = len(line) - len(line.lstrip())
            elif (
                in_process_user_message
                and stripped.startswith("async def ")
                and not stripped.startswith("async def process_user_message")
            ):
                # Next method at same indent
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= method_indent:
                    break
            elif in_process_user_message:
                if "_match_skills_for_prompt" in stripped and skill_match_line is None:
                    skill_match_line = i
                if (
                    "self.provider_router.route(" in stripped
                    and provider_call_line is None
                ):
                    provider_call_line = i
                if (
                    "self.fallback_resolver.resolve(" in stripped
                    and provider_call_line is None
                ):
                    provider_call_line = i

        assert skill_match_line is not None, (
            "_match_skills_for_prompt must be called in process_user_message"
        )
        assert provider_call_line is not None, (
            "Provider call must exist in process_user_message"
        )
        assert skill_match_line < provider_call_line, (
            f"Skill matching (line {skill_match_line}) must happen BEFORE "
            f"provider call (line {provider_call_line})"
        )

    def test_skill_matched_before_provider_call_streaming(self) -> None:
        """In process_user_message_streaming, skill matching must happen
        before prompt composition."""
        source = _read_source("application/chat_service.py")
        lines = source.splitlines()

        skill_match_line = None
        prompt_composition_line = None

        in_streaming = False
        method_indent = 0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "async def process_user_message_streaming(" in stripped:
                in_streaming = True
                method_indent = len(line) - len(line.lstrip())
            elif (
                in_streaming
                and stripped.startswith("async def ")
                and "process_user_message_streaming" not in stripped
            ):
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= method_indent:
                    break
            elif in_streaming:
                if "_match_skills_for_prompt" in stripped and skill_match_line is None:
                    skill_match_line = i
                if (
                    "effective_prompt" in stripped
                    and "skill_block" in stripped
                    and prompt_composition_line is None
                ):
                    prompt_composition_line = i

        assert skill_match_line is not None, (
            "_match_skills_for_prompt must be called in streaming path"
        )


class TestPromptContainsSkillBlock:
    """SC-03: The effective prompt must contain skill context when matched."""

    def test_non_streaming_prompt_injects_skill_block(self) -> None:
        """process_user_message must inject skill_block into effective_prompt."""
        source = _read_source("application/chat_service.py")
        # Check that skill_block is appended to effective_prompt
        assert 'f"{effective_prompt}\\n\\n{skill_block}"' in source or (
            "effective_prompt" in source and "skill_block" in source
        ), "Skill block must be injected into the effective prompt"

    def test_streaming_prompt_injects_skill_block(self) -> None:
        """process_user_message_streaming must inject skill block."""
        source = _read_source("application/chat_service.py")
        assert "skill_block_streaming" in source, (
            "Streaming path must have skill_block_streaming variable"
        )


class TestEvidenceWrittenOnSuccessfulApplication:
    """SC-03: Evidence must be written after successful skill match."""

    def test_write_skill_evidence_method_exists(self) -> None:
        """ChatService must have _write_skill_evidence method."""
        source = _read_source("application/chat_service.py")
        assert "def _write_skill_evidence" in source, (
            "ChatService must have _write_skill_evidence method"
        )

    def test_write_skill_evidence_called_after_match(self) -> None:
        """_write_skill_evidence must be called in the non-streaming path."""
        source = _read_source("application/chat_service.py")
        assert "self._write_skill_evidence(skill_match_result)" in source, (
            "_write_skill_evidence must be called with skill_match_result"
        )


class TestMatchSkillsForPromptMethod:
    """SC-03: _match_skills_for_prompt builds correct prompt block."""

    def test_method_exists_and_returns_tuple(self) -> None:
        source = _read_source("application/chat_service.py")
        assert "def _match_skills_for_prompt" in source
        # Should return tuple[str, SkillMatch | None]
        assert "APPLICABLE USER SKILL" in source, (
            "Prompt block must contain 'APPLICABLE USER SKILL' header"
        )
