# BareAgent

BareAgent is a small coding agent implemented from first principles. An
OpenAI-compatible model decides what to do; the local harness owns conversation
history, tool execution, protocol validation, context trimming, and termination.
It does not use an agent framework or remotely hosted file/code tools.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
export OPENAI_API_KEY="..."
export OPENAI_MODEL="your-model-name"
# Optional for compatible providers:
export OPENAI_BASE_URL="https://api.openai.com/v1"

# One task, one temporary session
uv run bare-agent --workspace ./project "fix the failing tests"

# In-memory REPL session
uv run bare-agent --workspace ./project
```

The REPL supports `/help`, `/reset`, and `/exit`. It preserves prior user turns,
assistant tool calls, matching tool results, and final answers only for the life
of the current process. One-shot and REPL modes share the same agent loop.

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

Credentials are read only from environment variables. `.env` files are ignored,
and child commands receive an environment with key/token/secret/password-like
variables removed. Paths are resolved under the workspace, writes are atomic,
commands use an argv list with `shell=False`, and process groups are killed on
timeout. Obvious destructive, elevation, shutdown, and remote-push commands are
blocked.

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
