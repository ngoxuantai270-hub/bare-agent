from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from bare_agent.agent import BareAgent
from bare_agent.model import ModelError, ScriptedModel
from bare_agent.tools import LocalToolSet
from bare_agent.types import ModelReply, RunLimits, ToolCall, VerificationRequired


def call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(call_id, name, json.dumps(arguments))


def agent_for(
    tmp_path: Path,
    replies: Iterable[ModelReply | Exception],
    *,
    limits: RunLimits | None = None,
) -> tuple[BareAgent, ScriptedModel]:
    model = ScriptedModel(replies)
    tools = LocalToolSet(tmp_path, command_timeout_seconds=1)
    return BareAgent(model, tools, limits=limits, system_prompt="system"), model


def test_agent_returns_direct_model_answer(tmp_path: Path) -> None:
    agent, model = agent_for(tmp_path, [ModelReply("done")])

    result = agent.run("answer the task")

    assert result.status == "completed"
    assert result.final_text == "done"
    assert result.steps == 1
    assert result.tool_calls == 0
    assert [message["role"] for message in model.requests[0].messages] == ["system", "user"]


def test_agent_pairs_every_tool_call_before_next_model_request(tmp_path: Path) -> None:
    first = ModelReply(
        "working",
        (
            call("write-1", "write_file", {"path": "a.txt", "content": "hello"}),
            call("read-1", "read_file", {"path": "a.txt"}),
        ),
    )
    agent, model = agent_for(tmp_path, [first, ModelReply("finished")])

    result = agent.run("create and inspect a file")

    assert result.status == "completed"
    assert result.tool_calls == 2
    messages = model.requests[1].messages
    assistant = messages[-3]
    tool_messages = messages[-2:]
    assert assistant["role"] == "assistant"
    assert [item["id"] for item in assistant["tool_calls"]] == ["write-1", "read-1"]
    assert [item["tool_call_id"] for item in tool_messages] == ["write-1", "read-1"]
    assert all(item["role"] == "tool" for item in tool_messages)


def test_agent_rejects_unverified_final_and_recovers_wrong_file_write(tmp_path: Path) -> None:
    replies = [
        ModelReply(
            tool_calls=(call("wrong", "write_file", {"path": "README.md", "content": "H.\n"}),)
        ),
        ModelReply('Created README.md containing "H.".'),
        ModelReply(tool_calls=(call("inspect-wrong", "read_file", {"path": "README.md"}),)),
        ModelReply(
            tool_calls=(
                call(
                    "correct",
                    "write_file",
                    {"path": "README.md", "content": "Hello World\n"},
                ),
            )
        ),
        ModelReply(tool_calls=(call("inspect-correct", "read_file", {"path": "README.md"}),)),
        ModelReply('Created README.md containing "Hello World".'),
    ]
    agent, model = agent_for(tmp_path, replies)
    events = []

    result = agent.run(
        'Create README.md whose complete content is "Hello World".',
        on_event=events.append,
    )

    assert result.status == "completed"
    assert result.final_text == 'Created README.md containing "Hello World".'
    assert (tmp_path / "README.md").read_text() == "Hello World\n"
    assert result.steps == 6
    assert result.tool_calls == 4
    assert any(isinstance(event, VerificationRequired) for event in events)
    assert "VERIFICATION REQUIRED" in model.requests[1].messages[0]["content"]
    assert 'containing "H."' not in str(model.requests[2].messages)


def test_agent_stops_after_repeated_unverified_final_answers(tmp_path: Path) -> None:
    agent, _ = agent_for(
        tmp_path,
        [
            ModelReply(
                tool_calls=(call("write", "write_file", {"path": "a.txt", "content": "x"}),)
            ),
            ModelReply("done without checking"),
            ModelReply("still done without checking"),
            ModelReply("again done without checking"),
        ],
    )

    result = agent.run("write the requested file")

    assert result.status == "stopped"
    assert result.reason == "verification_required"
    assert result.steps == 4
    assert result.tool_calls == 1


def test_reading_a_different_file_does_not_verify_a_write(tmp_path: Path) -> None:
    (tmp_path / "other.txt").write_text("other\n")
    agent, _ = agent_for(
        tmp_path,
        [
            ModelReply(
                tool_calls=(
                    call("write", "write_file", {"path": "target.txt", "content": "target"}),
                    call("wrong-read", "read_file", {"path": "other.txt"}),
                )
            ),
            ModelReply("done"),
            ModelReply("still done"),
            ModelReply("again done"),
        ],
    )

    result = agent.run("write and verify target.txt")

    assert result.status == "stopped"
    assert result.reason == "verification_required"


@pytest.mark.parametrize(
    "argv",
    [
        ["ls"],
        ["uv", "run", "--with", "pytest", "python", "script.py"],
    ],
)
def test_successful_non_test_command_does_not_verify_a_write(
    tmp_path: Path, argv: list[str]
) -> None:
    agent, _ = agent_for(
        tmp_path,
        [
            ModelReply(
                tool_calls=(
                    call("write", "write_file", {"path": "target.txt", "content": "target"}),
                    call("inspect", "run_command", {"argv": argv}),
                )
            ),
            ModelReply("done without a real verification"),
            ModelReply("still done"),
            ModelReply("again done"),
        ],
    )

    result = agent.run("write and verify target.txt")

    assert result.status == "stopped"
    assert result.reason == "verification_required"


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
        ["python3", "-m", "unittest"],
        ["uv", "run", "pytest", "-q"],
        ["uv", "run", "--extra", "dev", "pytest", "-q"],
        ["uv", "run", "python", "-m", "pytest"],
        ["npm", "test"],
        ["npm", "run", "test"],
        ["pnpm", "test"],
        ["yarn", "test"],
        ["cargo", "test"],
        ["go", "test", "./..."],
    ],
)
def test_successful_test_commands_verify_file_changes(tmp_path: Path, argv: list[str]) -> None:
    class SuccessfulTools:
        definitions: tuple[dict[str, object], ...] = ()

        def invoke(self, tool_call: ToolCall):
            from bare_agent.types import ToolOutcome

            if tool_call.name == "run_command":
                return ToolOutcome("exit_code: 0\ntimed_out: false", exit_code=0)
            return ToolOutcome("wrote file")

    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    call("write", "write_file", {"path": "target.txt", "content": "target"}),
                    call("verify", "run_command", {"argv": argv}),
                )
            ),
            ModelReply("verified"),
        ]
    )
    agent = BareAgent(model, SuccessfulTools())

    result = agent.run("write and verify target.txt")

    assert result.status == "completed"
    assert result.final_text == "verified"


def test_failing_test_command_does_not_verify_file_changes(tmp_path: Path) -> None:
    class FailingTestTools:
        definitions: tuple[dict[str, object], ...] = ()

        def invoke(self, tool_call: ToolCall):
            from bare_agent.types import ToolOutcome

            if tool_call.name == "run_command":
                return ToolOutcome("exit_code: 1\ntimed_out: false", exit_code=1)
            return ToolOutcome("wrote file")

    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    call("write", "write_file", {"path": "target.txt", "content": "target"}),
                    call("verify", "run_command", {"argv": ["pytest", "-q"]}),
                )
            ),
            ModelReply("done despite failing tests"),
            ModelReply("still done"),
            ModelReply("again done"),
        ]
    )
    agent = BareAgent(model, FailingTestTools())

    result = agent.run("write and verify target.txt")

    assert result.status == "stopped"
    assert result.reason == "verification_required"


def test_tool_errors_are_observations_and_keep_protocol_valid(tmp_path: Path) -> None:
    first = ModelReply(
        tool_calls=(
            call("bad", "unknown", {}),
            call("missing", "read_file", {"path": "missing.txt"}),
        )
    )
    agent, model = agent_for(tmp_path, [first, ModelReply("recovered")])

    result = agent.run("try tools")

    assert result.status == "completed"
    assert result.final_text == "recovered"
    outcomes = model.requests[1].messages[-2:]
    assert all(item["role"] == "tool" for item in outcomes)
    assert "unknown tool" in outcomes[0]["content"]
    assert "does not exist" in outcomes[1]["content"]


def test_unexpected_tool_exception_is_converted_to_paired_error(tmp_path: Path) -> None:
    class ExplodingTools:
        definitions: tuple[dict[str, object], ...] = ()

        def invoke(self, tool_call: ToolCall):
            raise RuntimeError("private detail")

    model = ScriptedModel(
        [ModelReply(tool_calls=(call("boom", "explode", {}),)), ModelReply("recovered")]
    )
    agent = BareAgent(model, ExplodingTools(), system_prompt="system")

    result = agent.run("handle failure")

    assert result.status == "completed"
    tool_result = model.requests[1].messages[-1]["content"]
    assert tool_result == "tool failed safely: RuntimeError"
    assert "private detail" not in tool_result


def test_invalid_tool_call_ids_fail_before_any_tool_executes(tmp_path: Path) -> None:
    duplicate_ids = ModelReply(
        tool_calls=(
            call("same", "write_file", {"path": "a.txt", "content": "a"}),
            call("same", "write_file", {"path": "b.txt", "content": "b"}),
        )
    )
    agent, _ = agent_for(tmp_path, [duplicate_ids])

    result = agent.run("write files")

    assert result.status == "failed"
    assert result.reason == "protocol_error"
    assert not (tmp_path / "a.txt").exists()
    assert not (tmp_path / "b.txt").exists()


def test_session_reuses_complete_history_and_reset_clears_it(tmp_path: Path) -> None:
    agent, model = agent_for(
        tmp_path,
        [
            ModelReply(tool_calls=(call("w", "write_file", {"path": "a", "content": "x"}),)),
            ModelReply(tool_calls=(call("r", "read_file", {"path": "a"}),)),
            ModelReply("first done"),
            ModelReply("second done"),
            ModelReply("after reset"),
        ],
    )
    session = agent.new_session()

    first = agent.submit(session, "first task")
    second = agent.submit(session, "explain the previous task")

    assert first.tool_calls == 2
    assert second.steps == 1 and second.tool_calls == 0
    roles = [message["role"] for message in model.requests[3].messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert model.requests[3].messages[-1]["content"] == "explain the previous task"
    assert session.turn_count == 2

    session.reset()
    agent.submit(session, "new start")

    assert session.turn_count == 1
    assert [message["role"] for message in model.requests[4].messages] == ["system", "user"]


def test_one_shot_run_uses_a_fresh_temporary_session(tmp_path: Path) -> None:
    agent, model = agent_for(tmp_path, [ModelReply("one"), ModelReply("two")])

    first = agent.run("first")
    second = agent.run("second")

    assert first.steps == second.steps == 1
    assert [message["role"] for message in model.requests[1].messages] == ["system", "user"]


def test_context_window_drops_oldest_whole_session_turn(tmp_path: Path) -> None:
    limits = RunLimits(context_char_budget=310, tool_output_chars=256)
    agent, model = agent_for(
        tmp_path,
        [ModelReply("A" * 90), ModelReply("B" * 90), ModelReply("last")],
        limits=limits,
    )
    session = agent.new_session()
    agent.submit(session, "old-user-" + "x" * 50)
    agent.submit(session, "new-user-" + "y" * 50)

    result = agent.submit(session, "current")

    assert result.status == "completed"
    serialized = json.dumps(model.requests[2].messages)
    assert "old-user" not in serialized
    assert "new-user" in serialized
    assert "current" in serialized


def test_context_exhaustion_stops_without_calling_model(tmp_path: Path) -> None:
    limits = RunLimits(context_char_budget=10, tool_output_chars=256)
    agent, model = agent_for(tmp_path, [ModelReply("unused")], limits=limits)

    result = agent.run("too large")

    assert result.status == "stopped"
    assert result.reason == "context_exhausted"
    assert model.requests == []


def test_repeated_identical_tool_batches_stop_the_loop(tmp_path: Path) -> None:
    repeated = ModelReply(tool_calls=(call("id", "glob_files", {"pattern": "*.py"}),))
    replies = [
        repeated,
        ModelReply(tool_calls=(call("id-2", "glob_files", {"pattern": "*.py"}),)),
        ModelReply(tool_calls=(call("id-3", "glob_files", {"pattern": "*.py"}),)),
    ]
    agent, _ = agent_for(tmp_path, replies)

    result = agent.run("loop")

    assert result.status == "stopped"
    assert result.reason == "loop_detected"
    assert result.steps == 3
    assert result.tool_calls == 2


def test_step_and_tool_call_limits_stop_cleanly(tmp_path: Path) -> None:
    tool_reply = ModelReply(tool_calls=(call("1", "glob_files", {"pattern": "*"}),))
    step_agent, _ = agent_for(
        tmp_path,
        [tool_reply],
        limits=RunLimits(max_steps=1, tool_output_chars=256),
    )
    tool_agent, _ = agent_for(
        tmp_path,
        [
            ModelReply(
                tool_calls=(
                    call("1", "glob_files", {"pattern": "*"}),
                    call("2", "glob_files", {"pattern": "*"}),
                )
            )
        ],
        limits=RunLimits(max_tool_calls=1, tool_output_chars=256),
    )

    step_result = step_agent.run("step limit")
    tool_result = tool_agent.run("tool limit")

    assert step_result.status == "stopped" and step_result.reason == "max_steps"
    assert tool_result.status == "stopped" and tool_result.reason == "max_tool_calls"
    assert tool_result.tool_calls == 0


def test_retryable_model_error_retries_but_auth_error_does_not(tmp_path: Path) -> None:
    retry_agent, retry_model = agent_for(
        tmp_path,
        [ModelError("temporary", retryable=True), ModelReply("ok")],
    )
    auth_agent, auth_model = agent_for(
        tmp_path,
        [ModelError("authentication failed", retryable=False), ModelReply("unused")],
    )

    retried = retry_agent.run("retry")
    rejected = auth_agent.run("auth")

    assert retried.status == "completed" and len(retry_model.requests) == 2
    assert rejected.status == "failed" and rejected.reason == "model_error"
    assert len(auth_model.requests) == 1
