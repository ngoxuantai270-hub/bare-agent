from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from bare_agent.types import (
    AgentEvent,
    ModelText,
    RunFinished,
    ToolFinished,
    ToolStarted,
)

_KNOWN_TOOL_NAMES = {
    "edit_file",
    "glob_files",
    "read_file",
    "run_command",
    "search_text",
    "write_file",
}


class JsonlEventWriter:
    """Write privacy-minimized run metadata without prompts or tool payloads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._stream: TextIO = self.path.open("a", encoding="utf-8")

    def __call__(self, event: AgentEvent) -> None:
        record: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": _event_name(event),
        }
        if isinstance(event, ModelText):
            record["characters"] = len(event.text)
        elif isinstance(event, ToolStarted):
            record["tool"] = _safe_tool_name(event.call.name)
        elif isinstance(event, ToolFinished):
            record.update(
                {
                    "tool": _safe_tool_name(event.call.name),
                    "is_error": event.outcome.is_error,
                    "truncated": event.outcome.truncated,
                    "output_characters": len(event.outcome.content),
                }
            )
        elif isinstance(event, RunFinished):
            record.update(
                {
                    "status": event.result.status,
                    "reason": event.result.reason,
                    "steps": event.result.steps,
                    "tool_calls": event.result.tool_calls,
                    "elapsed_ms": event.result.elapsed_ms,
                }
            )
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _event_name(event: AgentEvent) -> str:
    if isinstance(event, ModelText):
        return "model_text"
    if isinstance(event, ToolStarted):
        return "tool_started"
    if isinstance(event, ToolFinished):
        return "tool_finished"
    return "run_finished"


def _safe_tool_name(name: str) -> str:
    return name if name in _KNOWN_TOOL_NAMES else "unknown"
