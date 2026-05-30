"""Tests for OllamaProvider.

Verifies:
    - is_available() checks Ollama via HTTP with mock
    - is_available() returns False on connection error
    - query() extracts response field correctly (mocked)
    - query() handles timeout correctly
    - query() handles HTTP errors correctly
    - Capabilities: privacy_class == "local", cost_class == "free"
    - user_id/chat_id are accepted without crash
    - Provider-Name ist "ollama_local"
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from infrastructure.providers.ollama_local import OllamaProvider


class TestOllamaProviderAvailability:
    """Tests for is_available() (async, uses httpx)."""

    @pytest.mark.asyncio
    async def test_available_when_ollama_responds_200(self) -> None:
        """is_available() returns True on HTTP 200 from /api/tags."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_not_available_when_connection_refused(self) -> None:
        """is_available() returns False on connection error."""
        import httpx

        provider = OllamaProvider()

        with patch(
            "httpx.AsyncClient.get",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_not_available_when_timeout(self) -> None:
        """is_available() returns False on timeout."""
        import httpx

        provider = OllamaProvider()

        with patch(
            "httpx.AsyncClient.get",
            side_effect=httpx.TimeoutException("timed out"),
        ):
            assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_not_available_when_oserror(self) -> None:
        """is_available() returns False on OSError (network unreachable)."""
        provider = OllamaProvider()

        with patch(
            "httpx.AsyncClient.get",
            side_effect=OSError("Network unreachable"),
        ):
            assert await provider.is_available() is False

    def test_name_is_ollama_local(self) -> None:
        provider = OllamaProvider()
        assert provider.name == "ollama_local"

    @pytest.mark.asyncio
    async def test_respects_ollama_host_env(self) -> None:
        """is_available() uses OLLAMA_HOST env-var."""
        provider = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.dict("os.environ", {"OLLAMA_HOST": "http://custom-host:9999"}):
            with patch("httpx.AsyncClient.get", return_value=mock_response) as mock_get:
                await provider.is_available()
                # Verify custom host was used
                call_args = mock_get.call_args
                assert "custom-host:9999" in call_args[0][0]


class TestOllamaProviderCapabilities:
    """Tests für get_capabilities()."""

    def test_privacy_class_is_local(self) -> None:
        provider = OllamaProvider()
        caps = provider.get_capabilities()
        assert caps.privacy_class == "local"

    def test_cost_class_is_free(self) -> None:
        provider = OllamaProvider()
        caps = provider.get_capabilities()
        assert caps.cost_class == "free"

    def test_no_streaming_support(self) -> None:
        """MVP: non-streaming only."""
        provider = OllamaProvider()
        caps = provider.get_capabilities()
        assert caps.supports_streaming is False

    def test_no_tool_use(self) -> None:
        provider = OllamaProvider()
        caps = provider.get_capabilities()
        assert caps.supports_tool_use is False

    def test_available_models_listed(self) -> None:
        provider = OllamaProvider()
        caps = provider.get_capabilities()
        assert "llama3.2:3b" in caps.available_models


class TestOllamaProviderQuery:
    """Tests für query() mit gemocktem HTTP."""

    @pytest.mark.asyncio
    async def test_successful_query(self) -> None:
        """query() extrahiert response-Feld korrekt."""
        provider = OllamaProvider()

        response_body = json.dumps({"response": "Hallo von Llama!"}).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = MagicMock(return_value=response_body)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await provider.query("Was ist Bitcoin?")

        assert result.text == "Hallo von Llama!"
        assert result.error is None
        assert result.provider_name == "ollama_local"
        assert result.model == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_query_with_system_prompt(self) -> None:
        """System-Prompt wird im Payload als 'system' gesendet."""
        provider = OllamaProvider()

        response_body = json.dumps({"response": "OK"}).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = MagicMock(return_value=response_body)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            await provider.query("Hi", system_prompt="Sei nett")
            # Verify payload contains system prompt
            call_args = mock_open.call_args
            req = call_args[0][0]
            payload = json.loads(req.data)
            assert payload["system"] == "Sei nett"

    @pytest.mark.asyncio
    async def test_query_accepts_user_id_and_chat_id(self) -> None:
        """user_id/chat_id werden akzeptiert ohne Crash."""
        provider = OllamaProvider()

        response_body = json.dumps({"response": "Fine"}).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = MagicMock(return_value=response_body)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await provider.query("Test", user_id=12345, chat_id=67890)

        assert result.text == "Fine"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_query_custom_model(self) -> None:
        """Explizites Modell wird im Payload verwendet."""
        provider = OllamaProvider()

        response_body = json.dumps({"response": "Mistral here"}).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = MagicMock(return_value=response_body)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = await provider.query("Hi", model="mistral:7b")
            call_args = mock_open.call_args
            req = call_args[0][0]
            payload = json.loads(req.data)
            assert payload["model"] == "mistral:7b"
            assert result.model == "mistral:7b"

    @pytest.mark.asyncio
    async def test_query_http_error(self) -> None:
        """HTTP-Fehler von Ollama wird als error im Response zurückgegeben."""
        provider = OllamaProvider()

        import urllib.error

        error_resp = MagicMock()
        error_resp.read = MagicMock(return_value=b"model not found")

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/generate",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=error_resp,
            ),
        ):
            result = await provider.query("Test")

        assert result.error is not None
        assert "404" in result.error
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_query_connection_refused(self) -> None:
        """Connection-Error wird als error im Response zurückgegeben."""
        provider = OllamaProvider()

        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = await provider.query("Test")

        assert result.error is not None
        assert "unreachable" in result.error
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_query_invalid_json_response(self) -> None:
        """Ungültige JSON-Antwort wird als error gehandelt."""
        provider = OllamaProvider()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = MagicMock(return_value=b"not json at all")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await provider.query("Test")

        assert result.error is not None
        assert "JSON" in result.error

    @pytest.mark.asyncio
    async def test_query_missing_response_field(self) -> None:
        """JSON ohne response-Feld wird als error gehandelt."""
        provider = OllamaProvider()

        response_body = json.dumps({"model": "llama3.2:3b", "done": True}).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read = MagicMock(return_value=response_body)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await provider.query("Test")

        assert result.error is not None
        assert "response field missing" in result.error
