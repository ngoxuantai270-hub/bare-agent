from __future__ import annotations

import json
from pathlib import Path

from bare_agent.agent import BareAgent
from bare_agent.model import ScriptedModel
from bare_agent.tools import LocalToolSet
from bare_agent.types import ModelReply, ToolCall


def call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(call_id, name, json.dumps(arguments))


def test_read_glob_search_and_command_results_reach_model_without_corruption(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("Hello World\n")
    (tmp_path / "app.py").write_text("print('ok')\n")
    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    call("read", "read_file", {"path": "README.md"}),
                    call("glob", "glob_files", {"pattern": "*.py"}),
                    call(
                        "search",
                        "search_text",
                        {"query": "World", "path": "README.md"},
                    ),
                    call(
                        "command",
                        "run_command",
                        {"argv": ["python", "-c", "print('command-output')"]},
                    ),
                )
            ),
            ModelReply("inspection complete"),
        ]
    )
    agent = BareAgent(model, LocalToolSet(tmp_path))

    result = agent.run("inspect the workspace")

    assert result.status == "completed"
    tool_messages = model.requests[1].messages[-4:]
    assert tool_messages[0]["content"] == "1: Hello World"
    assert tool_messages[1]["content"] == "app.py"
    assert tool_messages[2]["content"] == "README.md:1:Hello World"
    assert tool_messages[3]["content"] == (
        "exit_code: 0\ntimed_out: false\nstdout:\ncommand-output"
    )


def test_edit_requires_and_accepts_readback_of_the_same_file(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Hello World\n")
    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    call(
                        "edit",
                        "edit_file",
                        {
                            "path": "README.md",
                            "old_text": "Hello",
                            "new_text": "Goodbye",
                        },
                    ),
                    call("read", "read_file", {"path": "README.md"}),
                )
            ),
            ModelReply("edit verified"),
        ]
    )
    agent = BareAgent(model, LocalToolSet(tmp_path))

    result = agent.run("replace Hello with Goodbye and verify it")

    assert result.status == "completed"
    assert (tmp_path / "README.md").read_text() == "Goodbye World\n"
    tool_messages = model.requests[1].messages[-2:]
    assert tool_messages[0]["content"] == "edited README.md"
    assert tool_messages[1]["content"] == "1: Goodbye World"
