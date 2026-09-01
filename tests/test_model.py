from __future__ import annotations

from types import SimpleNamespace

import pytest

from bare_agent.model import ConfigurationError, ModelError, OpenAICompatibleModel
from bare_agent.types import ModelRequest


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def fake_client(completions: FakeCompletions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_openai_adapter_normalizes_text_and_tool_calls() -> None:
    message = SimpleNamespace(
        content="inspect",
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="read_file", arguments='{"path":"a.py"}'),
            )
        ],
    )
    completions = FakeCompletions(SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    model = OpenAICompatibleModel(fake_client(completions), "model-name")
    request = ModelRequest(
        messages=({"role": "user", "content": "task"},),
        tools=({"type": "function", "function": {"name": "read_file"}},),
    )

    reply = model.complete(request)

    assert reply.text == "inspect"
    assert reply.tool_calls[0].id == "call-1"
    assert reply.tool_calls[0].name == "read_file"
    assert completions.kwargs["model"] == "model-name"
    assert completions.kwargs["tool_choice"] == "auto"


@pytest.mark.parametrize(
    ("error_name", "retryable"),
    [("RateLimitError", True), ("AuthenticationError", False)],
)
def test_openai_adapter_hides_sdk_error_details(error_name: str, retryable: bool) -> None:
    error_type = type(error_name, (Exception,), {})
    completions = FakeCompletions(error=error_type("secret-token-in-error"))
    model = OpenAICompatibleModel(fake_client(completions), "model")

    with pytest.raises(ModelError) as raised:
        model.complete(ModelRequest(messages=(), tools=()))

    assert raised.value.retryable is retryable
    assert "secret-token-in-error" not in str(raised.value)


def test_model_configuration_requires_environment(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAICompatibleModel.from_env()
