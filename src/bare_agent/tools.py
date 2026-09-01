from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from pathlib import Path

from bare_agent.types import JsonObject, ToolCall, ToolOutcome

_IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
_SENSITIVE_ENV_WORDS = {"CREDENTIAL", "CREDENTIALS", "KEY", "PASSWORD", "SECRET", "TOKEN"}
_SENSITIVE_ENV_NAMES = {"GIT_ASKPASS", "SSH_ASKPASS", "SSH_AUTH_SOCK"}


class ToolInputError(ValueError):
    """A safe, user-facing tool validation error."""


class LocalToolSet:
    """Workspace-confined file and subprocess tools implemented locally."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        command_timeout_seconds: float = 30.0,
        output_chars: int = 16_000,
        max_file_chars: int = 2_000_000,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")
        if command_timeout_seconds <= 0 or output_chars < 256 or max_file_chars < 1:
            raise ValueError("tool limits must be positive and output_chars must be at least 256")
        self.command_timeout_seconds = command_timeout_seconds
        self.output_chars = output_chars
        self.max_file_chars = max_file_chars
        self._handlers: Mapping[str, Callable[[JsonObject], ToolOutcome]] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "glob_files": self._glob_files,
            "search_text": self._search_text,
            "run_command": self._run_command,
        }

    @property
    def definitions(self) -> tuple[JsonObject, ...]:
        return _TOOL_DEFINITIONS

    def invoke(self, call: ToolCall) -> ToolOutcome:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolOutcome(f"unknown tool: {call.name}", is_error=True)
        try:
            arguments = json.loads(call.arguments_json)
        except json.JSONDecodeError:
            return ToolOutcome("tool arguments must be valid JSON", is_error=True)
        if not isinstance(arguments, dict):
            return ToolOutcome("tool arguments JSON must be an object", is_error=True)
        try:
            return handler(arguments)
        except (ToolInputError, OSError, UnicodeError) as error:
            return ToolOutcome(str(error), is_error=True)
        except Exception as error:  # noqa: BLE001 - tool failures must return a paired result
            return ToolOutcome(f"tool failed safely: {type(error).__name__}", is_error=True)

    def _read_file(self, arguments: JsonObject) -> ToolOutcome:
        self._validate_keys(arguments, required={"path"}, optional={"offset", "limit"})
        path = self._path(self._string(arguments, "path"), must_exist=True)
        if not path.is_file():
            raise ToolInputError("path is not a file")
        offset = self._integer(arguments, "offset", default=1, minimum=1)
        limit = self._integer(arguments, "limit", default=500, minimum=1, maximum=500)
        text = self._read_text(path)
        lines = text.splitlines()
        selected = lines[offset - 1 : offset - 1 + limit]
        content = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, offset))
        return self._bounded(content or "(no lines in requested range)")

    def _write_file(self, arguments: JsonObject) -> ToolOutcome:
        self._validate_keys(arguments, required={"path", "content"})
        path = self._path(self._string(arguments, "path"))
        content = self._string(arguments, "content")
        if len(content) > self.max_file_chars:
            raise ToolInputError("content exceeds file size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_inside(path.parent.resolve())
        self._atomic_write(path, content)
        return ToolOutcome(f"wrote {len(content)} characters to {path.relative_to(self.workspace)}")

    def _edit_file(self, arguments: JsonObject) -> ToolOutcome:
        self._validate_keys(arguments, required={"path", "old_text", "new_text"})
        path = self._path(self._string(arguments, "path"), must_exist=True)
        old_text = self._string(arguments, "old_text")
        new_text = self._string(arguments, "new_text")
        if old_text == "":
            raise ToolInputError("old_text must not be empty")
        original = self._read_text(path)
        matches = original.count(old_text)
        if matches == 0:
            raise ToolInputError("old_text not found")
        if matches != 1:
            raise ToolInputError(f"old_text must be unique; found {matches} matches")
        updated = original.replace(old_text, new_text, 1)
        if len(updated) > self.max_file_chars:
            raise ToolInputError("edited content exceeds file size limit")
        self._atomic_write(path, updated)
        return ToolOutcome(f"edited {path.relative_to(self.workspace)}")

    def _glob_files(self, arguments: JsonObject) -> ToolOutcome:
        self._validate_keys(arguments, required={"pattern"})
        pattern = self._string(arguments, "pattern")
        self._validate_relative_pattern(pattern)
        matches: list[str] = []
        for path in self.workspace.glob(pattern):
            if len(matches) >= 200:
                break
            if self._ignored(path) or not path.is_file():
                continue
            resolved = path.resolve()
            if resolved.is_relative_to(self.workspace):
                matches.append(resolved.relative_to(self.workspace).as_posix())
        return self._bounded("\n".join(sorted(set(matches))) or "(no matches)")

    def _search_text(self, arguments: JsonObject) -> ToolOutcome:
        self._validate_keys(arguments, required={"query"}, optional={"path"})
        query = self._string(arguments, "query")
        if not query:
            raise ToolInputError("query must not be empty")
        base = self._path(self._string(arguments, "path", default="."), must_exist=True)
        paths: Iterable[Path] = [base] if base.is_file() else base.rglob("*")
        results: list[str] = []
        for path in paths:
            if len(results) >= 200:
                break
            if not path.is_file() or self._ignored(path):
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.workspace):
                continue
            try:
                lines = self._read_text(resolved).splitlines()
            except (ToolInputError, UnicodeError, OSError):
                continue
            relative = resolved.relative_to(self.workspace).as_posix()
            for line_number, line in enumerate(lines, 1):
                if query in line:
                    results.append(f"{relative}:{line_number}:{line[:500]}")
                    if len(results) >= 200:
                        break
        return self._bounded("\n".join(results) or "(no matches)")

    def _run_command(self, arguments: JsonObject) -> ToolOutcome:
        self._validate_keys(arguments, required={"argv"}, optional={"cwd"})
        argv = arguments["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise ToolInputError("argv must be a non-empty list of strings")
        command = list(argv)
        self._validate_command(command)
        cwd = self._path(self._string(arguments, "cwd", default="."), must_exist=True)
        if not cwd.is_dir():
            raise ToolInputError("cwd is not a directory")
        process = subprocess.Popen(  # noqa: S603 - argv list and shell=False are deliberate
            command,
            cwd=cwd,
            env=self._sanitized_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=self.command_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_process_group(process.pid)
            stdout_bytes, stderr_bytes = process.communicate()
        except KeyboardInterrupt:
            self._kill_process_group(process.pid)
            process.communicate()
            raise
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        content = self._format_command_result(process.returncode, stdout, stderr, timed_out)
        bounded = self._bounded(content)
        if timed_out:
            return ToolOutcome(
                f"command timed out after {self.command_timeout_seconds:g}s\n{bounded.content}",
                is_error=True,
                truncated=bounded.truncated,
            )
        return bounded

    @staticmethod
    def _kill_process_group(pid: int) -> None:
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)

    def _path(self, raw: str, *, must_exist: bool = False) -> Path:
        relative = Path(raw)
        if relative.is_absolute():
            raise ToolInputError("path must be relative to the workspace")
        candidate = (self.workspace / relative).resolve(strict=False)
        self._ensure_inside(candidate)
        if must_exist and not candidate.exists():
            raise ToolInputError("path does not exist")
        return candidate

    def _ensure_inside(self, path: Path) -> None:
        if not path.is_relative_to(self.workspace):
            raise ToolInputError("path escapes the workspace")

    def _read_text(self, path: Path) -> str:
        if path.stat().st_size > self.max_file_chars * 4:
            raise ToolInputError("file exceeds size limit")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolInputError("file is not UTF-8 text") from error
        if len(text) > self.max_file_chars:
            raise ToolInputError("file exceeds size limit")
        return text

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.chmod(temporary_name, mode)
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _bounded(self, content: str) -> ToolOutcome:
        if len(content) <= self.output_chars:
            return ToolOutcome(content)
        marker = "\n... output truncated ...\n"
        available = self.output_chars - len(marker)
        head = available // 2
        return ToolOutcome(content[:head] + marker + content[-(available - head) :], truncated=True)

    @staticmethod
    def _format_command_result(exit_code: int, stdout: str, stderr: str, timed_out: bool) -> str:
        parts = [f"exit_code: {exit_code}", f"timed_out: {str(timed_out).lower()}"]
        if stdout:
            parts.append(f"stdout:\n{stdout.rstrip()}")
        if stderr:
            parts.append(f"stderr:\n{stderr.rstrip()}")
        return "\n".join(parts)

    @staticmethod
    def _sanitized_environment() -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not _is_sensitive_environment_name(key)
        }
        environment["GIT_TERMINAL_PROMPT"] = "0"
        return environment

    @staticmethod
    def _validate_command(argv: list[str]) -> None:
        executable = Path(argv[0]).name.lower()
        blocked = {"dd", "doas", "halt", "mkfs", "poweroff", "reboot", "shutdown", "su", "sudo"}
        if executable in blocked:
            raise ToolInputError(f"command is blocked by safety policy: {executable}")
        if executable == "git" and len(argv) > 1:
            if argv[1] == "push":
                raise ToolInputError("remote-changing git commands are blocked")
            if argv[1] == "remote" and any(
                item
                in {
                    "add",
                    "prune",
                    "remove",
                    "rename",
                    "set-branches",
                    "set-head",
                    "set-url",
                    "update",
                }
                for item in argv[2:]
            ):
                raise ToolInputError("remote-changing git commands are blocked")
        if executable == "rm" and any(
            item in {"-r", "-R", "-rf", "-fr", "--recursive", "--force"} for item in argv[1:]
        ):
            raise ToolInputError("recursive or forced deletion is blocked")
        shell_executables = {"bash", "csh", "dash", "fish", "ksh", "sh", "tcsh", "zsh"}
        if executable in shell_executables and "-c" in argv[1:]:
            raise ToolInputError(
                "inline shell commands are blocked; provide a structured argv instead"
            )

    @staticmethod
    def _validate_relative_pattern(pattern: str) -> None:
        path = Path(pattern)
        if path.is_absolute() or ".." in path.parts:
            raise ToolInputError("glob pattern must stay inside the workspace")

    def _ignored(self, path: Path) -> bool:
        try:
            parts = path.relative_to(self.workspace).parts
        except ValueError:
            return True
        return any(part in _IGNORED_PARTS for part in parts)

    @staticmethod
    def _validate_keys(
        arguments: JsonObject,
        *,
        required: set[str],
        optional: set[str] | None = None,
    ) -> None:
        optional = optional or set()
        missing = required - arguments.keys()
        unexpected = arguments.keys() - required - optional
        if missing:
            raise ToolInputError(f"missing arguments: {', '.join(sorted(missing))}")
        if unexpected:
            raise ToolInputError(f"unexpected arguments: {', '.join(sorted(unexpected))}")

    @staticmethod
    def _string(arguments: JsonObject, key: str, *, default: str | None = None) -> str:
        value = arguments.get(key, default)
        if not isinstance(value, str):
            raise ToolInputError(f"{key} must be a string")
        return value

    @staticmethod
    def _integer(
        arguments: JsonObject,
        key: str,
        *,
        default: int,
        minimum: int,
        maximum: int | None = None,
    ) -> int:
        value = arguments.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolInputError(f"{key} must be an integer")
        if value < minimum or (maximum is not None and value > maximum):
            raise ToolInputError(f"{key} is outside the allowed range")
        return value


def _is_sensitive_environment_name(name: str) -> bool:
    upper = name.upper()
    if upper in _SENSITIVE_ENV_NAMES or upper == "GIT_CONFIG_COUNT":
        return True
    if upper.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
        return True
    words = upper.replace("-", "_").split("_")
    return any(word in _SENSITIVE_ENV_WORDS for word in words)


def _function_tool(
    name: str,
    description: str,
    properties: JsonObject,
    required: list[str],
) -> JsonObject:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_PATH = {"type": "string", "description": "Workspace-relative path"}
_TOOL_DEFINITIONS: tuple[JsonObject, ...] = (
    _function_tool(
        "read_file",
        "Read numbered lines from a UTF-8 text file.",
        {
            "path": _PATH,
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        ["path"],
    ),
    _function_tool(
        "write_file",
        "Atomically create or replace a UTF-8 text file.",
        {"path": _PATH, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _function_tool(
        "edit_file",
        "Replace exactly one occurrence of old_text in a UTF-8 text file.",
        {"path": _PATH, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
        ["path", "old_text", "new_text"],
    ),
    _function_tool(
        "glob_files",
        "List files matching a workspace-relative glob.",
        {"pattern": {"type": "string"}},
        ["pattern"],
    ),
    _function_tool(
        "search_text",
        "Search UTF-8 files for a literal string.",
        {"query": {"type": "string"}, "path": _PATH},
        ["query"],
    ),
    _function_tool(
        "run_command",
        "Run an argv command without a shell in the workspace.",
        {
            "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "cwd": _PATH,
        },
        ["argv"],
    ),
)
