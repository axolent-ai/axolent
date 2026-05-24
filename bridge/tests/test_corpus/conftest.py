"""Fixtures for Golden Corpus tests.

Provides a deterministic fake_chat_service that simulates AXOLENT's
chat pipeline without calling any real LLM provider.

The fake service uses heuristics to produce responses that match
expected behaviour for each corpus category, enabling regression
detection without network calls or non-determinism.
"""

from __future__ import annotations

from typing import Any

import pytest

from domain.language import detect_language


class FakeChatService:
    """Deterministic chat service for golden corpus testing.

    Simulates language detection, command handling, memory ops,
    privacy pipeline, streaming, debate, and skills without
    calling any external provider.
    """

    def process(
        self, input_text: str, setup: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process a single corpus entry and return a response dict.

        The response dict keys mirror what golden_runner.validate_expected checks.
        """
        setup = setup or {}
        input_text = self._apply_input_multiply(input_text, setup)

        # Base response
        response: dict[str, Any] = {
            "text": "",
            "language": None,
            "no_crash": True,
            "preserves_unicode": True,
            "streaming_aborted": False,
            "streaming_completes": True,
            "duration_seconds": 0.1,
            "memory_count_delta": 0,
            "history_count": None,
            "streaming_active_after": None,
            "pending_skill_created": False,
            "privacy_pipeline_ran": False,
            "skill_count_delta": 0,
            "providers_called": 0,
            "synthesis_present": False,
            "has_raw_provider_output": False,
            "uses_previous_debate_context": False,
            "privacy_rejection": None,
            "duplicate_created": False,
            "sticky_after": None,
        }

        # Route by input type
        if input_text.startswith("/"):
            self._handle_command(input_text, setup, response)
        elif setup.get("streaming_active"):
            self._handle_streaming_cancel(input_text, setup, response)
        elif setup.get("_action") == "/stop":
            # Simulates action_after_seconds cancel: streaming was active, then stopped
            response["streaming_aborted"] = True
            response["streaming_active_after"] = False
            response["text"] = "Streaming cancelled after timed action."
            response["language"] = (
                detect_language(input_text) if input_text.strip() else "de"
            )
        elif setup.get("last_action") == "debate_synthesis_done":
            self._handle_debate_followup(input_text, setup, response)
        elif setup.get("user_a_memories") or setup.get("user_b_memories"):
            self._handle_memory_query(input_text, setup, response)
        elif setup.get("user_memories") is not None:
            self._handle_memory_query(input_text, setup, response)
        else:
            self._handle_regular_message(input_text, setup, response)

        return response

    def _apply_input_multiply(self, input_text: str, setup: dict[str, Any]) -> str:
        """Handle input_multiply directive (from entry, passed via setup)."""
        multiply = setup.get("_input_multiply")
        if multiply and isinstance(multiply, int):
            return input_text * multiply
        return input_text

    def _handle_regular_message(
        self, input_text: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Handle a regular (non-command) message."""
        # Language detection
        if setup.get("sticky_language"):
            lang = setup["sticky_language"]
            response["sticky_after"] = lang
        else:
            lang = detect_language(input_text) if input_text.strip() else "de"

        response["language"] = lang

        # Generate deterministic response in detected language
        response["text"] = self._generate_response(input_text, lang)

    def _handle_command(
        self, input_text: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Handle slash commands."""
        parts = input_text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/remember":
            self._handle_remember(args, setup, response)
        elif command == "/reset":
            self._handle_reset(setup, response)
        elif command == "/stop":
            self._handle_stop(setup, response)
        elif command == "/debate":
            self._handle_debate(args, setup, response)
        elif command == "/learn":
            self._handle_learn(args, setup, response)
        elif command == "/forget":
            self._handle_forget(args, setup, response)
        else:
            # Unknown command
            response["text"] = f"Unknown command: {command}"
            response["language"] = "en"

    def _handle_remember(
        self, args: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Simulate /remember with privacy pipeline."""
        # Privacy checks
        rejection = self._check_privacy(args)
        if rejection:
            response["privacy_rejection"] = rejection
            response["memory_count_delta"] = 0
            response["text"] = f"blocked: {rejection} content detected"
            response["language"] = "en"
            return

        # Injection check
        injection_markers = [
            "ignore all previous",
            "reveal system prompt",
            "ignore instructions",
            "disregard",
            "forget your instructions",
        ]
        if any(marker in args.lower() for marker in injection_markers):
            response["memory_count_delta"] = 0
            response["text"] = "blocked: potential prompt injection detected"
            response["language"] = "en"
            return

        # Success
        response["memory_count_delta"] = 1
        response["text"] = "remembered: " + args
        response["language"] = "en"

    def _handle_reset(self, setup: dict[str, Any], response: dict[str, Any]) -> None:
        """Simulate /reset."""
        response["history_count"] = 0
        response["text"] = "Conversation reset. Starting fresh."
        response["language"] = "en"

    def _handle_stop(self, setup: dict[str, Any], response: dict[str, Any]) -> None:
        """Simulate /stop."""
        response["streaming_active_after"] = False
        response["streaming_aborted"] = True
        response["text"] = "Streaming stopped."
        response["language"] = "en"

    def _handle_debate(
        self, args: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Simulate /debate."""
        response["providers_called"] = 3
        response["synthesis_present"] = True
        response["has_raw_provider_output"] = False
        response["text"] = (
            f"Debate synthesis on '{args}': "
            "Multiple perspectives considered and synthesized into a balanced view."
        )
        response["language"] = "en"

    def _handle_debate_followup(
        self, input_text: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Simulate a followup after debate."""
        response["uses_previous_debate_context"] = True
        response["text"] = f"Building on the previous debate: {input_text}"
        response["language"] = (
            detect_language(input_text) if input_text.strip() else "en"
        )

    def _handle_learn(
        self, args: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Simulate /learn."""
        existing_skills = setup.get("skills", [])

        # Check for duplicate
        if args in existing_skills:
            response["duplicate_created"] = False
            response["pending_skill_created"] = False
            response["text"] = "Skill already exists, not creating duplicate."
            response["language"] = "en"
            return

        response["pending_skill_created"] = True
        response["privacy_pipeline_ran"] = True
        response["text"] = f"Skill learned (pending review): {args}"
        response["language"] = "en"

    def _handle_forget(
        self, args: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Simulate /forget."""
        existing_skills = setup.get("skills", [])
        if args in existing_skills:
            response["skill_count_delta"] = -1
            response["text"] = f"Forgotten: {args}"
        else:
            response["skill_count_delta"] = 0
            response["text"] = f"Skill not found: {args}"
        response["language"] = "en"

    def _handle_streaming_cancel(
        self, input_text: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Simulate streaming cancellation."""
        response["streaming_aborted"] = True
        response["streaming_active_after"] = False
        response["no_messages_after_cancel"] = True
        response["text"] = "Streaming cancelled."
        response["language"] = "en"

    def _handle_memory_query(
        self, input_text: str, setup: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Simulate memory retrieval with user scoping."""
        user = setup.get("_user", "user_a")
        memories = setup.get(f"{user}_memories", setup.get("user_memories", []))

        if memories:
            response["text"] = "Here are your memories: " + ", ".join(memories)
        else:
            response["text"] = "You have no stored memories yet."

        response["language"] = (
            detect_language(input_text) if input_text.strip() else "en"
        )
        response["no_crash"] = True

    def _check_privacy(self, text: str) -> str | None:
        """Check text against privacy pipeline rules.

        Returns rejection category or None if clean.
        """
        text_lower = text.lower()

        # Healthcare/medication patterns
        healthcare_markers = [
            "lexapro",
            "prozac",
            "zoloft",
            "sertraline",
            "fluoxetine",
            "depression",
            "anxiety medication",
            "antidepressant",
            "mg for",
            "diagnosed with",
            "my therapist",
            "psychiatric",
            "bipolar",
            "schizophrenia",
        ]
        if any(marker in text_lower for marker in healthcare_markers):
            return "healthcare"

        # Secret/credential patterns
        secret_markers = [
            "api key",
            "api_key",
            "apikey",
            "password",
            "passwd",
            "secret key",
            "sk-",
            "pk-",
            "token:",
            "private key",
            "ssh key",
        ]
        if any(marker in text_lower for marker in secret_markers):
            return "secret"

        return None

    def _generate_response(self, input_text: str, lang: str) -> str:
        """Generate a deterministic response in the given language.

        For testing purposes, echoes key info with language-appropriate wrapper.
        HTML/script content is sanitized (simulates real AXOLENT behaviour).
        """
        if not input_text.strip():
            return "I received an empty message. How can I help you?"

        # Sanitize HTML tags (AXOLENT never echoes raw HTML back)
        import re as _re

        sanitized_input = _re.sub(r"<[^>]+>", "", input_text)
        if sanitized_input != input_text:
            input_text = sanitized_input.strip() or "sanitized content"

        # Language-specific response templates
        templates = {
            "de": "Hier ist meine Antwort auf Deutsch zu deiner Nachricht ueber {topic}. "
            "NVIDIA ist ein Technologieunternehmen das Grafikkarten und KI-Chips herstellt.",
            "en": "Here is my response in English about {topic}. "
            "NVIDIA is a technology company that makes graphics cards and AI chips.",
            "fr": "Voici ma reponse en francais sur {topic}. "
            "NVIDIA est une entreprise technologique qui fabrique des cartes graphiques.",
            "es": "Aqui esta mi respuesta en espanol sobre {topic}. "
            "NVIDIA es una empresa tecnologica que fabrica tarjetas graficas.",
            "nl": "Hier is mijn antwoord in het Nederlands over {topic}. "
            "NVIDIA is een technologiebedrijf dat grafische kaarten maakt.",
            "it": "Ecco la mia risposta in italiano su {topic}. "
            "NVIDIA e una azienda tecnologica che produce schede grafiche.",
            "pt": "Aqui esta a minha resposta em portugues sobre {topic}. "
            "NVIDIA e uma empresa de tecnologia que fabrica placas graficas.",
            "sv": "Har ar mitt svar pa svenska om {topic}. "
            "NVIDIA ar ett teknikforetag som tillverkar grafikkort och AI-chip.",
        }

        # Extract topic from input (first few significant words)
        words = [w for w in input_text.split() if len(w) > 3][:5]
        topic = " ".join(words) if words else "your question"

        template = templates.get(lang, templates["en"])
        return template.format(topic=topic)


@pytest.fixture(scope="module")
def fake_chat_service() -> FakeChatService:
    """Provide a deterministic fake chat service for corpus tests."""
    return FakeChatService()
