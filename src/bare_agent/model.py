from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable
from typing import Any, Protocol

from bare_agent.types import ModelReply, ModelRequest, ToolCall


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelReply: ...


class ModelError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class ConfigurationError(ValueError):
    pass


class ScriptedModel:
    """Deterministic model boundary for tests and offline demonstrations."""

    def __init__(self, replies: Iterable[ModelReply | BaseException]) -> None:
        self._replies = deque(replies)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request)
        if not self._replies:
            raise ModelError("scripted model has no reply", retryable=False)
        reply = self._replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        return reply


class OpenAICompatibleModel:
    """Normalize OpenAI-compatible Chat Completions into BareAgent types."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls) -> OpenAICompatibleModel:
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is not configured")
        if not model:
            raise ConfigurationError("OPENAI_MODEL is not configured")
        from openai import OpenAI

        return cls(OpenAI(api_key=api_key, base_url=base_url), model)

    def complete(self, request: ModelRequest) -> ModelReply:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(request.messages),
                tools=list(request.tools),
                tool_choice="auto",
            )
            message = response.choices[0].message
            calls = tuple(
                ToolCall(item.id or "", item.function.name, item.function.arguments)
                for item in (message.tool_calls or [])
            )
            return ModelReply(text=message.content or "", tool_calls=calls)
        except ModelError:
            raise
        except Exception as error:  # noqa: BLE001 - normalize SDK-specific failures
            raise ModelError(
                "model request failed",
                retryable=_is_retryable_openai_error(error),
            ) from None


def _is_retryable_openai_error(error: Exception) -> bool:
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
    return type(error).__name__ in retryable_names
