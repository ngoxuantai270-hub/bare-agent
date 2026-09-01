from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bare_agent.agent import BareAgent
from bare_agent.model import ScriptedModel
from bare_agent.tools import LocalToolSet
from bare_agent.types import ModelReply, ToolCall


def tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(call_id, name, json.dumps(arguments))


@pytest.mark.parametrize("run_number", range(3))
def test_scripted_agent_repairs_demo_project(tmp_path: Path, run_number: int) -> None:
    source = Path(__file__).parents[1] / "examples" / "bugfix_demo"
    workspace = tmp_path / f"demo-{run_number}"
    shutil.copytree(source, workspace)
    model = ScriptedModel(
        [
            ModelReply(tool_calls=(tool_call("read", "read_file", {"path": "calculator.py"}),)),
            ModelReply(
                tool_calls=(
                    tool_call(
                        "edit",
                        "edit_file",
                        {
                            "path": "calculator.py",
                            "old_text": "return sum(range(start, end))",
                            "new_text": "return sum(range(start, end + 1))",
                        },
                    ),
                )
            ),
            ModelReply(
                tool_calls=(
                    tool_call(
                        "test",
                        "run_command",
                        {"argv": ["python", "-m", "pytest", "-q"]},
                    ),
                )
            ),
            ModelReply("Fixed the inclusive endpoint bug; the test now passes."),
        ]
    )
    agent = BareAgent(model, LocalToolSet(workspace, command_timeout_seconds=5))

    result = agent.run("Fix the failing test and verify the change.")

    assert result.status == "completed"
    assert result.tool_calls == 3
    test_result = model.requests[3].messages[-1]["content"]
    assert "1 passed" in test_result
