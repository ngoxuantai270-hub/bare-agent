from __future__ import annotations

import io
from pathlib import Path

from bare_agent.cli import main
from bare_agent.model import ScriptedModel
from bare_agent.types import ModelReply


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
