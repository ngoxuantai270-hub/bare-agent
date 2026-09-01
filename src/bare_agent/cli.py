from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TextIO

from bare_agent import __version__
from bare_agent.agent import BareAgent, EventSink
from bare_agent.model import ConfigurationError, ModelClient, OpenAICompatibleModel
from bare_agent.tools import LocalToolSet
from bare_agent.trace import JsonlEventWriter
from bare_agent.types import (
    AgentEvent,
    ModelText,
    RunFinished,
    RunLimits,
    ToolFinished,
    ToolStarted,
    VerificationRequired,
)


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
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        help="append privacy-minimized event metadata to a local JSONL file",
    )
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

    trace_writer: JsonlEventWriter | None = None
    if args.trace_jsonl is not None:
        try:
            trace_writer = JsonlEventWriter(args.trace_jsonl)
        except OSError as error:
            error_stream.write(f"Configuration error: cannot open trace file ({error.strerror})\n")
            return 2

    agent = BareAgent(active_model, tools, limits=limits)
    renderer = _fanout(_console_renderer(output_stream), trace_writer)
    try:
        task = " ".join(args.task).strip()
        if task:
            result = agent.run(task, on_event=renderer)
            return _result_exit_code(result.status)
        return _run_repl(agent, input_stream, output_stream, error_stream, renderer)
    finally:
        if trace_writer is not None:
            trace_writer.close()


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
        if task == "/multi":
            action, multiline_task = _read_multiline_task(stdin, stdout)
            if action == "eof":
                return 0
            if action == "cancel":
                continue
            task = multiline_task
        elif task.startswith("/"):
            if task == "/exit":
                return 0
            if task == "/reset":
                session.reset()
                stdout.write("Session reset.\n")
                continue
            if task == "/help":
                stdout.write(
                    "/help   show commands\n"
                    "/status show in-memory session status\n"
                    "/multi  enter multiline input mode\n"
                    "/reset  clear in-memory history\n"
                    "/exit   quit\n"
                )
                continue
            if task == "/status":
                stdout.write(f"turns={session.turn_count}\n")
                continue
            stderr.write(f"Unknown command: {task}\n")
            continue
        agent.submit(session, task, on_event=renderer)


def _read_multiline_task(
    stdin: TextIO,
    stdout: TextIO,
) -> tuple[Literal["submit", "cancel", "eof"], str]:
    stdout.write("Multiline mode. Enter /send to submit or /cancel to discard.\n")
    lines: list[str] = []
    while True:
        try:
            stdout.write("...> ")
            stdout.flush()
            line = stdin.readline()
        except KeyboardInterrupt:
            stdout.write("\nMultiline input cancelled.\n")
            return "cancel", ""
        if line == "":
            stdout.write("\nMultiline input discarded.\n")
            return "eof", ""
        content = line.rstrip("\r\n")
        if content == "/cancel":
            stdout.write("Multiline input cancelled.\n")
            return "cancel", ""
        if content == "/send":
            task = "\n".join(lines)
            if not task.strip():
                stdout.write("Multiline input is empty; continue typing or /cancel.\n")
                continue
            return "submit", task
        lines.append(content)


def _console_renderer(stdout: TextIO) -> EventSink:
    def render(event: object) -> None:
        if isinstance(event, ModelText):
            stdout.write(f"{event.text}\n")
        elif isinstance(event, ToolStarted):
            stdout.write(f"→ {event.call.name}\n")
        elif isinstance(event, ToolFinished):
            if event.outcome.is_error:
                marker = "✗"
            elif (
                event.call.name == "run_command"
                and event.outcome.exit_code is not None
                and event.outcome.exit_code != 0
            ):
                marker = "!"
            else:
                marker = "✓"
            suffix = " (truncated)" if event.outcome.truncated else ""
            summary = ""
            if not event.outcome.is_error and event.call.name in {
                "edit_file",
                "run_command",
                "write_file",
            }:
                summary = f" — {event.outcome.content.splitlines()[0]}"
            stdout.write(f"{marker} {event.call.name}{suffix}{summary}\n")
        elif isinstance(event, VerificationRequired):
            stdout.write(f"↻ verification required for {event.pending_files} changed file(s)\n")
        elif isinstance(event, RunFinished) and event.result.status != "completed":
            stdout.write(f"[{event.result.status}: {event.result.reason}]\n")
        stdout.flush()

    return render


def _fanout(*sinks: EventSink | None) -> EventSink:
    active_sinks = tuple(sink for sink in sinks if sink is not None)

    def emit(event: AgentEvent) -> None:
        for sink in active_sinks:
            sink(event)

    return emit


def _result_exit_code(status: str) -> int:
    if status == "completed":
        return 0
    if status == "cancelled":
        return 130
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
