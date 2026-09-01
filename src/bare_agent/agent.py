from __future__ import annotations

import json
import posixpath
import time
from collections.abc import Callable
from typing import Protocol

from bare_agent.context import ContextExhaustedError, build_context
from bare_agent.model import ModelClient, ModelError
from bare_agent.session import AgentSession, CompletedRound, SessionTurn
from bare_agent.types import (
    AgentEvent,
    ModelReply,
    ModelRequest,
    ModelText,
    RunFinished,
    RunLimits,
    RunResult,
    RunStatus,
    ToolCall,
    ToolFinished,
    ToolOutcome,
    ToolStarted,
    VerificationRequired,
)

DEFAULT_SYSTEM_PROMPT = """You are a coding agent operating in a local workspace.
Inspect the project before changing it. Use the provided tools to read, edit, and test code.
Continue until the user's programming task is complete or a tool reports a blocker.
Use workspace-relative paths. Never invent tool results. Prefer focused, reversible changes.
Finish with a concise summary of changes and verification performed."""

VERIFICATION_PROMPT = """VERIFICATION REQUIRED: Files were modified in this task.
Before giving a final answer, inspect every changed file with read_file, or run a relevant
verification command that exits with code 0. Compare the observed result with the original
user request. If anything is incomplete or incorrect, fix it and verify again."""

EventSink = Callable[[AgentEvent], None]


class ToolSet(Protocol):
    @property
    def definitions(self) -> tuple[dict[str, object], ...]: ...

    def invoke(self, call: ToolCall) -> ToolOutcome: ...


class BareAgent:
    """The harness: model decision, local tool execution, and loop control."""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolSet,
        *,
        limits: RunLimits | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model_retries: int = 2,
    ) -> None:
        self.model = model
        self.tools = tools
        self.limits = limits or RunLimits()
        self.system_prompt = system_prompt
        if model_retries < 0:
            raise ValueError("model_retries must not be negative")
        self.model_retries = model_retries
        self._session_owner = object()

    def new_session(self) -> AgentSession:
        return AgentSession(self._session_owner)

    def run(self, task: str, on_event: EventSink | None = None) -> RunResult:
        return self.submit(self.new_session(), task, on_event=on_event)

    def submit(
        self,
        session: AgentSession,
        task: str,
        on_event: EventSink | None = None,
    ) -> RunResult:
        if not session._belongs_to(self._session_owner):
            raise ValueError("session belongs to a different BareAgent")
        if not task.strip():
            raise ValueError("task must not be empty")

        started = time.monotonic()
        active = SessionTurn(task)
        steps = 0
        tool_count = 0
        used_ids = session._tool_call_ids()
        previous_batch: tuple[tuple[str, str], ...] | None = None
        repeated_batches = 0
        pending_verification: set[str] = set()
        verification_rejections = 0

        def finish(status: RunStatus, reason: str, final_text: str = "") -> RunResult:
            active.final_text = final_text if status == "completed" else None
            session._append(active)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            result = RunResult(status, reason, final_text, steps, tool_count, elapsed_ms)
            self._emit(on_event, RunFinished(result))
            return result

        try:
            while steps < self.limits.max_steps:
                try:
                    active_system_prompt = self.system_prompt
                    if pending_verification:
                        active_system_prompt = f"{self.system_prompt}\n\n{VERIFICATION_PROMPT}"
                    messages = build_context(
                        active_system_prompt,
                        session._completed_turns(),
                        active,
                        self.limits.context_char_budget,
                    )
                except ContextExhaustedError:
                    return finish("stopped", "context_exhausted")

                request = ModelRequest(messages=messages, tools=self.tools.definitions)
                try:
                    reply = self._complete_with_retry(request)
                except ModelError:
                    return finish("failed", "model_error")
                steps += 1
                if not reply.tool_calls:
                    if pending_verification:
                        verification_rejections += 1
                        self._emit(
                            on_event,
                            VerificationRequired(
                                attempt=verification_rejections,
                                pending_files=len(pending_verification),
                            ),
                        )
                        if verification_rejections > self.limits.max_verification_retries:
                            return finish("stopped", "verification_required")
                        continue
                    if reply.text:
                        self._emit(on_event, ModelText(reply.text))
                    return finish("completed", "final_answer", reply.text)
                if reply.text:
                    self._emit(on_event, ModelText(reply.text))
                if not self._valid_call_ids(reply, used_ids):
                    return finish("failed", "protocol_error")
                if tool_count + len(reply.tool_calls) > self.limits.max_tool_calls:
                    return finish("stopped", "max_tool_calls")

                batch = tuple((call.name, call.arguments_json) for call in reply.tool_calls)
                if batch == previous_batch:
                    repeated_batches += 1
                else:
                    previous_batch = batch
                    repeated_batches = 1
                if repeated_batches >= 3:
                    return finish("stopped", "loop_detected")

                outcomes: list[ToolOutcome] = []
                for call in reply.tool_calls:
                    self._emit(on_event, ToolStarted(call))
                    try:
                        outcome = self.tools.invoke(call)
                    except Exception as error:  # noqa: BLE001 - preserve model protocol pairing
                        outcome = ToolOutcome(
                            f"tool failed safely: {type(error).__name__}",
                            is_error=True,
                        )
                    outcomes.append(outcome)
                    tool_count += 1
                    used_ids.add(call.id)
                    self._update_verification_state(call, outcome, pending_verification)
                    self._emit(on_event, ToolFinished(call, outcome))
                active.rounds.append(CompletedRound(reply.text, reply.tool_calls, tuple(outcomes)))
            return finish("stopped", "max_steps")
        except KeyboardInterrupt:
            return finish("cancelled", "keyboard_interrupt")

    def _complete_with_retry(self, request: ModelRequest) -> ModelReply:
        for attempt in range(self.model_retries + 1):
            try:
                return self.model.complete(request)
            except ModelError as error:
                if not error.retryable or attempt >= self.model_retries:
                    raise
                time.sleep(0.1 * (2**attempt))
        raise AssertionError("retry loop did not return or raise")

    @staticmethod
    def _valid_call_ids(reply: ModelReply, used_ids: set[str]) -> bool:
        ids = [call.id for call in reply.tool_calls]
        return all(ids) and len(ids) == len(set(ids)) and not used_ids.intersection(ids)

    @staticmethod
    def _update_verification_state(
        call: ToolCall,
        outcome: ToolOutcome,
        pending_files: set[str],
    ) -> None:
        if outcome.is_error:
            return
        arguments = _tool_arguments(call)
        if call.name in {"write_file", "edit_file"}:
            path = _normalized_path(arguments.get("path"))
            if path is not None:
                pending_files.add(path)
            return
        if call.name == "read_file":
            path = _normalized_path(arguments.get("path"))
            if path is not None:
                pending_files.discard(path)
            return
        if call.name == "run_command" and outcome.content.startswith("exit_code: 0\n"):
            pending_files.clear()

    @staticmethod
    def _emit(sink: EventSink | None, event: AgentEvent) -> None:
        if sink is not None:
            sink(event)


def _tool_arguments(call: ToolCall) -> dict[str, object]:
    try:
        arguments = json.loads(call.arguments_json)
    except json.JSONDecodeError:
        return {}
    return arguments if isinstance(arguments, dict) else {}


def _normalized_path(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return posixpath.normpath(value.replace("\\", "/"))
