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
        "model_text",
        "run_finished",
    ]
    assert records[2]["tool"] == "write_file"
    assert records[4]["tool"] == "unknown"
    assert records[-1]["status"] == "completed"
    for forbidden in [sensitive, "private-name.txt", "private user task", "private final answer"]:
        assert forbidden not in serialized
