from __future__ import annotations

import io
import json
from pathlib import Path

from bare_agent.cli import main
from bare_agent.model import ScriptedModel
from bare_agent.types import ModelReply, ToolCall


def test_one_shot_cli_returns_zero_and_prints_answer(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    model = ScriptedModel([ModelReply("completed answer")])

    code = main(
        ["--workspace", str(tmp_path), "fix", "the", "task"],
        model=model,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert "completed answer" in stdout.getvalue()
    assert stderr.getvalue() == ""
    assert model.requests[0].messages[-1]["content"] == "fix the task"


def test_cli_shows_safe_write_summary_and_verification_progress(tmp_path: Path) -> None:
    stdout = io.StringIO()
    trace_path = tmp_path / "verification.jsonl"
    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "write",
                        "write_file",
                        json.dumps({"path": "README.md", "content": "Hello World"}),
                    ),
                )
            ),
            ModelReply("premature final"),
            ModelReply(
                tool_calls=(ToolCall("read", "read_file", json.dumps({"path": "README.md"})),)
            ),
            ModelReply("verified final"),
        ]
    )

    code = main(
        [
            "--workspace",
            str(tmp_path),
            "--trace-jsonl",
            str(trace_path),
            "create README",
        ],
        model=model,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert "wrote 11 characters to README.md" in stdout.getvalue()
    assert "verification required for 1 changed file" in stdout.getvalue()
    assert "premature final" not in stdout.getvalue()
    assert "verified final" in stdout.getvalue()
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    verification = [record for record in records if record["event"] == "verification_required"]
    assert len(verification) == 1
    assert verification[0]["attempt"] == 1
    assert verification[0]["pending_files"] == 1


def test_cli_distinguishes_nonzero_command_exit_from_tool_failure(tmp_path: Path) -> None:
    stdout = io.StringIO()
    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "command",
                        "run_command",
                        json.dumps({"argv": ["python", "-c", "import sys; sys.exit(3)"]}),
                    ),
                )
            ),
            ModelReply("command inspected"),
        ]
    )

    code = main(
        ["--workspace", str(tmp_path), "inspect command"],
        model=model,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert "! run_command — exit_code: 3" in stdout.getvalue()
    assert "✓ run_command" not in stdout.getvalue()


def test_repl_reuses_session_but_reset_and_slash_commands_are_local(tmp_path: Path) -> None:
    stdin = io.StringIO("first\n/unknown\n/reset\nsecond\n/exit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()
    model = ScriptedModel([ModelReply("one"), ModelReply("two")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert len(model.requests) == 2
    assert [message["role"] for message in model.requests[1].messages] == ["system", "user"]
    assert "Unknown command" in stderr.getvalue()
    assert "Session reset" in stdout.getvalue()


def test_repl_multiline_submits_one_task_with_blank_lines_and_indentation(tmp_path: Path) -> None:
    stdin = io.StringIO("/multi\nfirst line\n\n  indented code\n/send\n/exit\n")
    stdout = io.StringIO()
    model = ScriptedModel([ModelReply("multiline received")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert len(model.requests) == 1
    assert model.requests[0].messages[-1]["content"] == "first line\n\n  indented code"
    assert "Multiline mode" in stdout.getvalue()
    assert "...> " in stdout.getvalue()


def test_repl_multiline_cancel_discards_buffer_and_returns_to_single_line(tmp_path: Path) -> None:
    stdin = io.StringIO("/multi\ndiscard me\n/cancel\nsingle task\n/exit\n")
    stdout = io.StringIO()
    model = ScriptedModel([ModelReply("single received")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert len(model.requests) == 1
    assert model.requests[0].messages[-1]["content"] == "single task"
    assert "Multiline input cancelled" in stdout.getvalue()


def test_repl_multiline_treats_slash_commands_as_content(tmp_path: Path) -> None:
    stdin = io.StringIO("/multi\n/help\n/status\n/send\n/exit\n")
    model = ScriptedModel([ModelReply("received")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    assert model.requests[0].messages[-1]["content"] == "/help\n/status"


def test_repl_multiline_empty_send_stays_in_mode(tmp_path: Path) -> None:
    stdin = io.StringIO("/multi\n/send\nactual task\n/send\n/exit\n")
    stdout = io.StringIO()
    model = ScriptedModel([ModelReply("received")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert len(model.requests) == 1
    assert model.requests[0].messages[-1]["content"] == "actual task"
    assert "Multiline input is empty" in stdout.getvalue()


def test_repl_multiline_eof_discards_unsubmitted_buffer(tmp_path: Path) -> None:
    stdin = io.StringIO("/multi\npartial task\n")
    stdout = io.StringIO()
    model = ScriptedModel([])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert model.requests == []
    assert "Multiline input discarded" in stdout.getvalue()


def test_repl_multiline_keyboard_interrupt_cancels_only_the_buffer(tmp_path: Path) -> None:
    class InterruptingInput(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.lines = iter(["/multi\n", KeyboardInterrupt(), "after interrupt\n", "/exit\n"])

        def readline(self, size: int = -1) -> str:
            item = next(self.lines, "")
            if isinstance(item, BaseException):
                raise item
            return item

    stdout = io.StringIO()
    model = ScriptedModel([ModelReply("continued")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=InterruptingInput(),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert model.requests[0].messages[-1]["content"] == "after interrupt"
    assert "Multiline input cancelled" in stdout.getvalue()


def test_one_shot_cli_preserves_newlines_inside_task_argument(tmp_path: Path) -> None:
    model = ScriptedModel([ModelReply("received")])

    code = main(
        ["--workspace", str(tmp_path), "line one\nline two"],
        model=model,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    assert model.requests[0].messages[-1]["content"] == "line one\nline two"


def test_repl_keyboard_interrupt_cancels_run_and_accepts_next_task(tmp_path: Path) -> None:
    stdin = io.StringIO("interrupt me\ncontinue\n/exit\n")
    stdout = io.StringIO()
    model = ScriptedModel([KeyboardInterrupt(), ModelReply("continued")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert "cancelled" in stdout.getvalue()
    assert "continued" in stdout.getvalue()


def test_cli_reports_missing_configuration_without_traceback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    stderr = io.StringIO()

    code = main(
        ["--workspace", str(tmp_path), "task"],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert "OPENAI_API_KEY is not configured" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def test_cli_rejects_non_directory_workspace(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_text("x")
    stderr = io.StringIO()

    code = main(
        ["--workspace", str(path), "task"],
        model=ScriptedModel([ModelReply("unused")]),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert "workspace is not a directory" in stderr.getvalue()


def test_repl_status_reports_turn_count_without_calling_model(tmp_path: Path) -> None:
    stdin = io.StringIO("/status\nfirst\n/status\n/exit\n")
    stdout = io.StringIO()
    model = ScriptedModel([ModelReply("done")])

    code = main(
        ["--workspace", str(tmp_path)],
        model=model,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert "turns=0" in stdout.getvalue()
    assert "turns=1" in stdout.getvalue()
    assert len(model.requests) == 1


def test_jsonl_trace_records_metadata_without_prompts_arguments_or_outputs(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.jsonl"
    sensitive = "do-not-record-this-value"
    model = ScriptedModel(
        [
            ModelReply(
                "private reasoning",
                (
                    ToolCall(
                        "call-1",
                        "write_file",
                        json.dumps({"path": "private-name.txt", "content": sensitive}),
                    ),
                    ToolCall("call-2", sensitive, "{}"),
                ),
            ),
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "call-3",
                        "read_file",
                        json.dumps({"path": "private-name.txt"}),
                    ),
                )
            ),
            ModelReply("private final answer"),
        ]
    )

    code = main(
        [
            "--workspace",
            str(tmp_path),
            "--trace-jsonl",
            str(trace_path),
            "private user task",
        ],
        model=model,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    serialized = trace_path.read_text()
    assert code == 0
    assert [record["event"] for record in records] == [
        "model_text",
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
        "model_text",
        "run_finished",
    ]
    assert records[2]["tool"] == "write_file"
    assert records[4]["tool"] == "unknown"
    assert records[-1]["status"] == "completed"
    for forbidden in [sensitive, "private-name.txt", "private user task", "private final answer"]:
        assert forbidden not in serialized


def test_jsonl_trace_records_command_exit_code(tmp_path: Path) -> None:
    trace_path = tmp_path / "command.jsonl"
    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "command",
                        "run_command",
                        json.dumps({"argv": ["python", "-c", "import sys; sys.exit(4)"]}),
                    ),
                )
            ),
            ModelReply("observed"),
        ]
    )

    code = main(
        [
            "--workspace",
            str(tmp_path),
            "--trace-jsonl",
            str(trace_path),
            "inspect command",
        ],
        model=model,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    finished = [record for record in records if record["event"] == "tool_finished"]
    assert code == 0
    assert finished == [
        {
            "timestamp": finished[0]["timestamp"],
            "event": "tool_finished",
            "tool": "run_command",
            "is_error": False,
            "truncated": False,
            "output_characters": len("exit_code: 4\ntimed_out: false"),
            "exit_code": 4,
        }
    ]
