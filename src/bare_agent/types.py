from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

JsonObject: TypeAlias = dict[str, Any]
RunStatus: TypeAlias = Literal["completed", "stopped", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    content: str
    is_error: bool = False
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ModelReply:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[JsonObject, ...]
    tools: tuple[JsonObject, ...]


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_steps: int = 20
    max_tool_calls: int = 60
    context_char_budget: int = 60_000
    command_timeout_seconds: float = 30.0
    tool_output_chars: int = 16_000
    max_verification_retries: int = 2

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tool_calls < 1:
            raise ValueError("step and tool-call limits must be positive")
        if self.context_char_budget < 1 or self.command_timeout_seconds <= 0:
            raise ValueError("context and timeout limits must be positive")
        if self.tool_output_chars < 256:
            raise ValueError("tool output limit must be at least 256 characters")
        if self.max_verification_retries < 0:
            raise ValueError("verification retry limit must not be negative")


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    reason: str
    final_text: str
    steps: int
    tool_calls: int
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class ModelText:
    text: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolFinished:
    call: ToolCall
    outcome: ToolOutcome


@dataclass(frozen=True, slots=True)
class VerificationRequired:
    attempt: int
    pending_files: int


@dataclass(frozen=True, slots=True)
class RunFinished:
    result: RunResult


AgentEvent: TypeAlias = ModelText | ToolStarted | ToolFinished | VerificationRequired | RunFinished
