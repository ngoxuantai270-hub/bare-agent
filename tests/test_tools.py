from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from bare_agent.tools import LocalToolSet
from bare_agent.types import ToolCall


def invoke(tools: LocalToolSet, name: str, arguments: dict[str, object]):
    return tools.invoke(ToolCall("call-1", name, json.dumps(arguments)))


def test_file_tools_support_read_write_and_unique_edit(tmp_path: Path) -> None:
    tools = LocalToolSet(tmp_path)

    written = invoke(tools, "write_file", {"path": "src/app.py", "content": "one\ntwo\n"})
    read = invoke(tools, "read_file", {"path": "src/app.py", "offset": 2, "limit": 1})
    edited = invoke(
        tools,
        "edit_file",
        {"path": "src/app.py", "old_text": "two", "new_text": "three"},
    )

    assert not written.is_error
    assert read.content == "2: two"
    assert not edited.is_error
    assert (tmp_path / "src/app.py").read_text() == "one\nthree\n"
    assert stat.S_IMODE((tmp_path / "src/app.py").stat().st_mode) == 0o644


def test_edit_rejects_missing_or_non_unique_text(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("same\nsame\n")
    tools = LocalToolSet(tmp_path)

    missing = invoke(
        tools,
        "edit_file",
        {"path": "data.txt", "old_text": "absent", "new_text": "x"},
    )
    duplicate = invoke(
        tools,
        "edit_file",
        {"path": "data.txt", "old_text": "same", "new_text": "x"},
    )

    assert missing.is_error and "not found" in missing.content
    assert duplicate.is_error and "2 matches" in duplicate.content


def test_paths_cannot_escape_workspace(tmp_path: Path) -> None:
    tools = LocalToolSet(tmp_path)

    traversal = invoke(tools, "read_file", {"path": "../secret.txt"})
    absolute = invoke(tools, "write_file", {"path": "/tmp/x", "content": "x"})

    assert traversal.is_error and "workspace" in traversal.content
    assert absolute.is_error and "relative" in absolute.content


def test_symlink_cannot_escape_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret")
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    tools = LocalToolSet(tmp_path)

    result = invoke(tools, "read_file", {"path": "link/secret.txt"})
    searched = invoke(tools, "search_text", {"query": "secret", "path": "."})

    assert result.is_error and "workspace" in result.content
    assert searched.content == "(no matches)"


def test_glob_and_literal_search_are_bounded_and_ignore_git(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("needle here\n")
    (tmp_path / "src/b.py").write_text("nothing\nneedle again\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/hidden.py").write_text("needle\n")
    tools = LocalToolSet(tmp_path)

    globbed = invoke(tools, "glob_files", {"pattern": "**/*.py"})
    searched = invoke(tools, "search_text", {"query": "needle", "path": "."})

    assert globbed.content.splitlines() == ["src/a.py", "src/b.py"]
    assert searched.content.splitlines() == [
        "src/a.py:1:needle here",
        "src/b.py:2:needle again",
    ]


def test_command_reports_nonzero_exit_without_infrastructure_error(tmp_path: Path) -> None:
    tools = LocalToolSet(tmp_path)
    result = invoke(
        tools,
        "run_command",
        {"argv": ["python3", "-c", "import sys; print('out'); sys.exit(7)"], "cwd": "."},
    )

    assert not result.is_error
    assert result.exit_code == 7
    assert "exit_code: 7" in result.content
    assert "out" in result.content


def test_command_timeout_and_secret_environment_filtering(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BARE_AGENT_TEST_SECRET", "must-not-leak")
    monkeypatch.setenv("BARE_AGENT_MONKEY", "kept")
    tools = LocalToolSet(tmp_path, command_timeout_seconds=0.1)

    env_result = invoke(
        tools,
        "run_command",
        {
            "argv": [
                "python3",
                "-c",
                (
                    "import os; "
                    "print(os.getenv('BARE_AGENT_TEST_SECRET', 'filtered')); "
                    "print(os.getenv('BARE_AGENT_MONKEY', 'missing'))"
                ),
            ]
        },
    )
    timeout = invoke(
        tools,
        "run_command",
        {"argv": ["python3", "-c", "import time; time.sleep(2)"]},
    )

    assert "filtered" in env_result.content
    assert "kept" in env_result.content
    assert "must-not-leak" not in env_result.content
    assert timeout.is_error and "timed out" in timeout.content
    assert timeout.exit_code is not None


def test_bad_json_unknown_tool_and_invalid_arguments_become_results(tmp_path: Path) -> None:
    tools = LocalToolSet(tmp_path)

    bad_json = tools.invoke(ToolCall("1", "read_file", "{"))
    unknown = invoke(tools, "does_not_exist", {})
    bad_args = invoke(tools, "read_file", {"path": 12})

    assert bad_json.is_error and "JSON" in bad_json.content
    assert unknown.is_error and "unknown tool" in unknown.content
    assert bad_args.is_error and "path" in bad_args.content


def test_obvious_destructive_remote_and_inline_shell_commands_are_blocked(tmp_path: Path) -> None:
    tools = LocalToolSet(tmp_path)

    recursive_delete = invoke(tools, "run_command", {"argv": ["rm", "-rf", "build"]})
    remote_push = invoke(tools, "run_command", {"argv": ["git", "push"]})
    inline_shell = invoke(tools, "run_command", {"argv": ["sh", "-c", "echo bypass"]})

    assert recursive_delete.is_error and "deletion" in recursive_delete.content
    assert remote_push.is_error and "remote" in remote_push.content
    assert inline_shell.is_error and "structured argv" in inline_shell.content


def test_read_only_git_remote_inspection_is_allowed(tmp_path: Path) -> None:
    tools = LocalToolSet(tmp_path)
    initialized = invoke(tools, "run_command", {"argv": ["git", "init", "-q"]})
    inspected = invoke(tools, "run_command", {"argv": ["git", "remote", "-v"]})
    changed = invoke(
        tools,
        "run_command",
        {"argv": ["git", "remote", "add", "origin", "https://example.invalid/repo"]},
    )

    assert not initialized.is_error and "exit_code: 0" in initialized.content
    assert not inspected.is_error and "exit_code: 0" in inspected.content
    assert changed.is_error and "remote" in changed.content


def test_command_output_keeps_head_and_tail_when_truncated(tmp_path: Path) -> None:
    tools = LocalToolSet(tmp_path, output_chars=256)

    result = invoke(
        tools,
        "run_command",
        {"argv": ["python", "-c", "print('HEAD' + 'x' * 500 + 'TAIL')"]},
    )

    assert result.truncated
    assert "HEAD" in result.content
    assert "TAIL" in result.content
    assert "output truncated" in result.content


def test_keyboard_interrupt_kills_command_process_group(tmp_path: Path, monkeypatch) -> None:
    class InterruptingProcess:
        pid = 43210
        returncode = -2
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return b"", b""

    process = InterruptingProcess()
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    tools = LocalToolSet(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        invoke(tools, "run_command", {"argv": ["python", "-c", "pass"]})

    assert killed == [(process.pid, 9)]
    assert process.calls == 2
