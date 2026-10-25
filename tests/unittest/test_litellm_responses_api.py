"""Unit tests for the selective Responses API path (config.responses_models).

Covers model matching, Responses output extraction, finish-reason mapping,
request translation, transport branching, and invalid-config handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.ai_handlers.litellm_helpers import EmptyTruncatedResponseError


def create_settings(responses_models=None, reasoning_effort="minimal"):
    """Fake settings resolving responses_models plus the handler defaults."""
    return type('', (), {
        'config': type('', (), {
            'reasoning_effort': reasoning_effort,
            'ai_timeout': 120,
            'custom_reasoning_model': False,
            'max_model_tokens': 32000,
            'verbosity_level': 0,
            'get': lambda self, key, default=None: {
                "responses_models": responses_models,
            }.get(key, default)
        })(),
        'litellm': type('', (), {
            'get': lambda self, key, default=None: default
        })(),
        'get': lambda self, key, default=None: default
    })()


def make_responses_result(text="hello", status="completed", reason=None):
    """Fake litellm.aresponses result with output_text and a status."""
    result = MagicMock()
    result.output_text = text
    result.status = status
    result.incomplete_details = {"reason": reason} if reason else None
    result.usage = MagicMock()
    result.usage.prompt_tokens = 10
    result.usage.completion_tokens = 5
    result.usage.total_tokens = 15
    result.dict.return_value = {"status": status, "output": text}
    return result


@pytest.fixture
def mock_logger():
    with patch('pr_agent.algo.ai_handlers.litellm_ai_handler.get_logger') as mock_log:
        yield mock_log.return_value


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    for name in ("AWS_USE_IMDS", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "AWS_SESSION_TOKEN", "AWS_REGION_NAME", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def make_handler(monkeypatch, responses_models):
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: create_settings(responses_models))
    return LiteLLMAIHandler()


class TestIsResponsesModel:
    @pytest.mark.parametrize("model", [
        "muse-spark-1.3-contributor",
        "openai/muse-spark-1.3-contributor",
        "openrouter/openai/muse-spark-1.3-contributor:nitro",
    ])
    def test_matches_exact_and_prefixed(self, monkeypatch, mock_logger, model):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        assert handler._is_responses_model(model) is True

    def test_non_listed_model_is_chat(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        assert handler._is_responses_model("openai/gpt-4o") is False
        assert handler._is_responses_model("kimi-k2.7-code") is False

    def test_empty_list_matches_nothing(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, [])
        assert handler._is_responses_model("openai/muse-spark-1.3-contributor") is False

    def test_default_config_matches_nothing(self, monkeypatch, mock_logger):
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: create_settings(None))
        handler = LiteLLMAIHandler()
        assert handler.responses_models == []
        assert handler._is_responses_model("openai/muse-spark-1.3-contributor") is False

    @pytest.mark.parametrize("invalid", [{"model": "x"}, ["ok", 123], [""]])
    def test_invalid_config_ignored_with_warning(self, monkeypatch, mock_logger, invalid):
        handler = make_handler(monkeypatch, invalid)
        assert handler.responses_models == []
        assert handler._is_responses_model("openai/muse-spark-1.3-contributor") is False
        assert any("responses_models" in call.args[0] for call in mock_logger.warning.call_args_list)

    def test_env_string_accepted(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, "muse-spark-1.3-contributor")
        assert handler.responses_models == ["muse-spark-1.3-contributor"]
        assert handler._is_responses_model("openai/muse-spark-1.3-contributor") is True


class TestExtractResponsesText:
    def test_prefers_output_text_attr(self):
        assert LiteLLMAIHandler._extract_responses_text(make_responses_result(text="pong")) == "pong"

    def test_parses_output_items_skipping_reasoning(self):
        response = {
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "he"},
                    {"type": "output_text", "text": "llo"},
                ]},
            ]
        }
        assert LiteLLMAIHandler._extract_responses_text(response) == "hello"

    def test_empty_when_no_message(self):
        assert LiteLLMAIHandler._extract_responses_text({"output": [{"type": "reasoning"}]}) == ""


class TestResponsesFinishReason:
    @pytest.mark.parametrize("status,reason,expected", [
        ("completed", None, "stop"),
        ("incomplete", "max_output_tokens", "length"),
        ("incomplete", "content_filter", "stop"),
        (None, None, "stop"),
    ])
    def test_mapping(self, status, reason, expected):
        response = make_responses_result(status=status, reason=reason)
        assert LiteLLMAIHandler._responses_finish_reason(response) == expected


class TestBuildResponsesKwargs:
    def test_translates_prompts_headers_and_budgets(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        kwargs = {
            "model": "openai/muse-spark-1.3-contributor",
            "temperature": 0.2,
            "timeout": 120,
            "api_key": "k",
            "api_base": "https://example/v1",
            "headers": {"x-opencode-session": "s"},
            "extra_headers": {"X-Custom": "c"},
            "max_completion_tokens": 4096,
        }
        built = handler._build_responses_kwargs("openai/muse-spark-1.3-contributor", "sys", "usr", kwargs)
        assert built["model"] == "openai/muse-spark-1.3-contributor"
        assert built["instructions"] == "sys"
        assert built["input"] == "usr"
        assert built["temperature"] == 0.2
        assert built["reasoning"] == {"effort": "minimal"}
        assert built["max_output_tokens"] == 4096
        assert built["extra_headers"] == {"X-Custom": "c", "x-opencode-session": "s"}
        assert built["api_base"] == "https://example/v1"

    def test_defaults_output_cap_when_unset(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        built = handler._build_responses_kwargs("openai/muse-spark-1.3-contributor", "s", "u", {})
        assert built["max_output_tokens"] == 8192

    def test_temperature_dropped_for_blocked_model(self, monkeypatch, mock_logger):
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: create_settings(["m1"]))
        handler = LiteLLMAIHandler()
        handler.no_temperature_models = ["m1"]
        built = handler._build_responses_kwargs("openai/m1", "s", "u", {})
        assert "temperature" not in built

    def test_sdk_header_markers_stripped(self, monkeypatch, mock_logger):
        """Internal omission sentinels must not reach the wire (live-bug regression)."""
        from pr_agent.algo.ai_handlers.cloud_auth import _CapturedSDKHeader
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        built = handler._build_responses_kwargs(
            "openai/muse-spark-1.3-contributor", "s", "u",
            {"headers": {
                "x-opencode-session": "sess",
                "OpenAI-Organization": _CapturedSDKHeader(),
                "OpenAI-Project": _CapturedSDKHeader(),
            }},
        )
        assert built["extra_headers"] == {"x-opencode-session": "sess"}

    def test_configured_org_forwarded_as_header(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        handler._sdk_header_defaults = {
            "organization": "org-123", "project": None, "custom_headers": {"X-Custom": "c"},
        }
        built = handler._build_responses_kwargs("openai/muse-spark-1.3-contributor", "s", "u", {})
        assert built["extra_headers"]["OpenAI-Organization"] == "org-123"
        assert built["extra_headers"]["X-Custom"] == "c"
        assert "OpenAI-Project" not in built["extra_headers"]


class TestResponsesTransport:
    @pytest.mark.asyncio
    async def test_responses_model_uses_aresponses(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        with patch.object(litellm_handler.litellm, "aresponses", new_callable=AsyncMock) as aresponses, \
                patch.object(litellm_handler, "acompletion", new_callable=AsyncMock) as acompletion:
            aresponses.return_value = make_responses_result(text="review ok")
            resp, finish = await handler.chat_completion(
                model="openai/muse-spark-1.3-contributor", system="sys", user="usr")
        assert resp == "review ok"
        assert finish == "stop"
        acompletion.assert_not_called()
        sent = aresponses.call_args.kwargs
        assert sent["instructions"] == "sys"
        assert sent["input"] == "usr"
        assert sent["reasoning"] == {"effort": "minimal"}

    @pytest.mark.asyncio
    async def test_chat_model_still_uses_acompletion(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, key: {
            "choices": [{"message": {"content": "chat ok"}, "finish_reason": "stop"}]
        }[key]
        mock_response.dict.return_value = {"choices": []}
        with patch.object(litellm_handler.litellm, "aresponses", new_callable=AsyncMock) as aresponses, \
                patch.object(litellm_handler, "acompletion", new_callable=AsyncMock) as acompletion:
            acompletion.return_value = mock_response
            resp, _ = await handler.chat_completion(model="openai/gpt-4o", system="s", user="u")
        assert resp == "chat ok"
        aresponses.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_length_response_raises_truncation(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        with patch.object(litellm_handler.litellm, "aresponses", new_callable=AsyncMock) as aresponses:
            aresponses.return_value = make_responses_result(
                text="", status="incomplete", reason="max_output_tokens")
            with pytest.raises(EmptyTruncatedResponseError):
                await handler.chat_completion(
                    model="openai/muse-spark-1.3-contributor", system="s", user="u")

    @pytest.mark.asyncio
    async def test_empty_response_raises_api_error(self, monkeypatch, mock_logger):
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        with patch.object(litellm_handler.litellm, "aresponses", new_callable=AsyncMock) as aresponses:
            aresponses.return_value = make_responses_result(text="   ", status="completed")
            with pytest.raises(openai.APIError):
                await handler.chat_completion(
                    model="openai/muse-spark-1.3-contributor", system="s", user="u")

    @pytest.mark.asyncio
    async def test_httpx_request_uses_model_url(self, monkeypatch, mock_logger):
        """The truncation error request must not leak prompts (regression guard)."""
        handler = make_handler(monkeypatch, ["muse-spark-1.3-contributor"])
        with patch.object(litellm_handler.litellm, "aresponses", new_callable=AsyncMock) as aresponses:
            aresponses.return_value = make_responses_result(text="", status="completed")
            with pytest.raises(openai.APIError) as exc_info:
                await handler.chat_completion(
                    model="openai/muse-spark-1.3-contributor", system="secret-sys", user="secret-user")
        assert isinstance(exc_info.value.request, httpx.Request)
        assert "secret-user" not in str(exc_info.value.request.url)
