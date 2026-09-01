from __future__ import annotations

import os
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


def test_model_configuration_requires_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAICompatibleModel.from_env()


def test_model_configuration_loads_project_env_without_overriding_process_env(
    tmp_path, monkeypatch
) -> None:
    env_names = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL")
    original = {name: os.environ.get(name) for name in env_names}
    for name in env_names:
        os.environ.pop(name, None)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY='file-credential'\n"
        "OPENAI_MODEL='deepseek-v4-flash'\n"
        "OPENAI_BASE_URL='https://api.deepseek.com'\n"
    )
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_openai(*, api_key: str, base_url: str):
        captured.update(api_key=api_key, base_url=base_url)
        message = SimpleNamespace(content="configured", tool_calls=[])
        completions = FakeCompletions(SimpleNamespace(choices=[SimpleNamespace(message=message)]))
        captured["completions"] = completions
        return fake_client(completions)

    monkeypatch.setattr("openai.OpenAI", fake_openai)
    try:
        model = OpenAICompatibleModel.from_env()
        assert captured["api_key"] == "file-credential"
        assert captured["base_url"] == "https://api.deepseek.com"
        model.complete(ModelRequest(messages=(), tools=()))
        completions = captured["completions"]
        assert isinstance(completions, FakeCompletions)
        assert completions.kwargs["model"] == "deepseek-v4-flash"

        os.environ["OPENAI_API_KEY"] = "process-credential"
        os.environ["OPENAI_MODEL"] = "process-model"
        os.environ["OPENAI_BASE_URL"] = "https://process.example/v1"
        model = OpenAICompatibleModel.from_env()
        assert captured["api_key"] == "process-credential"
        assert captured["base_url"] == "https://process.example/v1"
        model.complete(ModelRequest(messages=(), tools=()))
        completions = captured["completions"]
        assert isinstance(completions, FakeCompletions)
        assert completions.kwargs["model"] == "process-model"
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
