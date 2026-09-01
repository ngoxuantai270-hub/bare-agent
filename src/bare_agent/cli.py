from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from bare_agent import __version__
from bare_agent.agent import BareAgent, EventSink
from bare_agent.model import ConfigurationError, ModelClient, OpenAICompatibleModel
from bare_agent.tools import LocalToolSet
from bare_agent.types import ModelText, RunFinished, RunLimits, ToolFinished, ToolStarted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bare-agent",
        description="Run a small from-scratch coding agent in one-shot or REPL mode.",
    )
    parser.add_argument("task", nargs="*", help="programming task; omit to enter the REPL")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="workspace directory")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-tool-calls", type=int, default=60)
    parser.add_argument("--context-chars", type=int, default=60_000)
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    model: ModelClient | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    try:
        limits = RunLimits(
            max_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls,
            context_char_budget=args.context_chars,
            command_timeout_seconds=args.command_timeout,
        )
        tools = LocalToolSet(
            args.workspace,
            command_timeout_seconds=limits.command_timeout_seconds,
            output_chars=limits.tool_output_chars,
        )
        active_model = model or OpenAICompatibleModel.from_env()
    except (ConfigurationError, ValueError) as error:
        error_stream.write(f"Configuration error: {error}\n")
        return 2

    agent = BareAgent(active_model, tools, limits=limits)
    renderer = _console_renderer(output_stream)
    task = " ".join(args.task).strip()
    if task:
        result = agent.run(task, on_event=renderer)
        return _result_exit_code(result.status)
    return _run_repl(agent, input_stream, output_stream, error_stream, renderer)


def _run_repl(
    agent: BareAgent,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    renderer: EventSink,
) -> int:
    session = agent.new_session()
    stdout.write("BareAgent REPL. Type /help for commands.\n")
    while True:
        try:
            stdout.write("bare> ")
            stdout.flush()
            line = stdin.readline()
        except KeyboardInterrupt:
            stdout.write("\n")
            continue
        if line == "":
            stdout.write("\n")
            return 0
        task = line.strip()
        if not task:
            continue
        if task.startswith("/"):
            if task == "/exit":
                return 0
            if task == "/reset":
                session.reset()
                stdout.write("Session reset.\n")
                continue
            if task == "/help":
                stdout.write("/help  show commands\n/reset clear in-memory history\n/exit  quit\n")
                continue
            stderr.write(f"Unknown command: {task}\n")
            continue
        agent.submit(session, task, on_event=renderer)


def _console_renderer(stdout: TextIO) -> EventSink:
    def render(event: object) -> None:
        if isinstance(event, ModelText):
            stdout.write(f"{event.text}\n")
        elif isinstance(event, ToolStarted):
            stdout.write(f"→ {event.call.name}\n")
        elif isinstance(event, ToolFinished):
            marker = "✗" if event.outcome.is_error else "✓"
            suffix = " (truncated)" if event.outcome.truncated else ""
            stdout.write(f"{marker} {event.call.name}{suffix}\n")
        elif isinstance(event, RunFinished) and event.result.status != "completed":
            stdout.write(f"[{event.result.status}: {event.result.reason}]\n")
        stdout.flush()

    return render


def _result_exit_code(status: str) -> int:
    if status == "completed":
        return 0
    if status == "cancelled":
        return 130
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
