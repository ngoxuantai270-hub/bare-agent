# BareAgent

BareAgent is a small coding agent implemented from first principles. An
OpenAI-compatible model decides what to do; the local harness owns conversation
history, tool execution, protocol validation, context trimming, and termination.
It does not use an agent framework or remotely hosted file/code tools.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev

# Interactive local configuration (API key input is hidden)
./scripts/configure-env.sh

# One task, one temporary session
uv run bare-agent --workspace ./project "fix the failing tests"

# In-memory REPL session
uv run bare-agent --workspace ./project
```

The configuration script defaults to `deepseek-v4-flash` and
`https://api.deepseek.com`. It writes a project-local `.env` with file mode
`600`; BareAgent loads that file automatically. Existing process environment
variables take precedence. The script does not modify a shell profile or print
the API key.

The REPL supports `/help`, `/status`, `/reset`, and `/exit`. It preserves prior user turns,
assistant tool calls, matching tool results, and final answers only for the life
of the current process. One-shot and REPL modes share the same agent loop.

V1.1 adds an opt-in privacy-minimized event trace:

```bash
uv run bare-agent --workspace ./project --trace-jsonl ./run.jsonl "fix tests"
```

The JSONL trace records timestamps, event kinds, tool names, status, and counts.
It deliberately omits user tasks, model text, tool arguments, file contents, and
command output. Treat the trace as a local runtime artifact and do not commit it.

## Design

The loop exposes six fixed local tools:

- `read_file(path, offset, limit)`
- `write_file(path, content)`
- `edit_file(path, old_text, new_text)`
- `glob_files(pattern)`
- `search_text(query, path)`
- `run_command(argv, cwd)`

Every model tool call receives exactly one result, including invalid arguments
and tool failures. Context trimming removes whole session turns first, then
whole completed rounds, so tool-call/result pairs never become orphaned. Runs
stop on a final model answer, configured limits, repeated tool batches, context
exhaustion, model failure, or interruption.

## Security boundary

Credentials are read from process environment variables or the ignored local
`.env` file. Child commands receive an environment with
key/token/secret/password-like variables removed. Paths are resolved under the
workspace, writes are atomic, commands use an argv list with `shell=False`, and
process groups are killed on timeout. Obvious destructive, elevation, shutdown,
and remote-push commands are blocked.

These guardrails are not an OS sandbox. Run untrusted repositories inside a
container or VM.

## Verification

```bash
uv run --extra dev pytest --cov=bare_agent --cov-report=term-missing
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv build
```

`examples/bugfix_demo` is a deliberately broken tiny project suitable for a
short agent demonstration; copy it before running so the repository fixture
remains unchanged.
