from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_ROOT = PROJECT_ROOT / "evals" / "cases"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / ".eval-results"
_SENSITIVE_ENV_WORDS = {"CREDENTIAL", "CREDENTIALS", "KEY", "PASSWORD", "SECRET", "TOKEN"}
_SENSITIVE_ENV_NAMES = {"GIT_ASKPASS", "SSH_ASKPASS", "SSH_AUTH_SOCK"}


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    root: Path
    workspace: Path
    hidden_tests: Path
    task: str
    verify_argv: tuple[str, ...]
    protected_globs: tuple[str, ...]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass(frozen=True, slots=True)
class EvalResult:
    case_id: str
    run_number: int
    passed: bool
    agent_exit_code: int
    verifier_exit_code: int
    protected_files_unchanged: bool
    false_success: bool
    agent_timed_out: bool
    verifier_timed_out: bool
    status: str | None
    reason: str | None
    steps: int | None
    tool_calls: int | None
    elapsed_ms: int | None
    result_path: Path


def load_case(case_dir: Path) -> CaseSpec:
    root = case_dir.resolve()
    manifest_path = root / "case.toml"
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)
    task = _required_string(manifest, "task")
    verify_argv = _string_tuple(manifest, "verify_argv")
    protected_globs = _string_tuple(manifest, "protected_globs")
    timeout = manifest.get("timeout_seconds", 180)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(f"invalid timeout_seconds in {manifest_path}")
    workspace = root / "workspace"
    hidden_tests = root / "hidden_tests"
    if not workspace.is_dir() or not hidden_tests.is_dir():
        raise ValueError(f"case must contain workspace and hidden_tests directories: {root.name}")
    return CaseSpec(
        case_id=root.name,
        root=root,
        workspace=workspace,
        hidden_tests=hidden_tests,
        task=task,
        verify_argv=verify_argv,
        protected_globs=protected_globs,
        timeout_seconds=float(timeout),
    )


def run_case(
    spec: CaseSpec,
    *,
    run_number: int,
    results_dir: Path,
    agent_command: list[str] | None = None,
) -> EvalResult:
    run_dir = results_dir / spec.case_id / f"run-{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = agent_command or [sys.executable, "-m", "bare_agent"]

    with tempfile.TemporaryDirectory(prefix=f"bare-agent-eval-{spec.case_id}-") as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(spec.workspace, workspace)
        protected_before = _hash_globs(workspace, spec.protected_globs)
        if not protected_before:
            raise ValueError(f"protected_globs matched no files for case: {spec.case_id}")
        trace_path = run_dir / "trace.jsonl"
        agent = _run_process(
            [
                *command,
                "--workspace",
                str(workspace),
                "--trace-jsonl",
                str(trace_path),
                spec.task,
            ],
            cwd=PROJECT_ROOT,
            env=dict(os.environ),
            timeout=spec.timeout_seconds,
        )
        (run_dir / "agent.stdout.txt").write_text(agent.stdout, encoding="utf-8")
        (run_dir / "agent.stderr.txt").write_text(agent.stderr, encoding="utf-8")
        protected_after = _hash_globs(workspace, spec.protected_globs)
        protected_unchanged = protected_before == protected_after

        hidden_destination = workspace / "_hidden_tests"
        if hidden_destination.exists():
            raise RuntimeError("agent created the reserved hidden-test destination")
        shutil.copytree(spec.hidden_tests, hidden_destination)
        verifier_argv = list(spec.verify_argv)
        if verifier_argv and verifier_argv[0] == "python":
            verifier_argv[0] = sys.executable
        verifier = _run_process(
            verifier_argv,
            cwd=workspace,
            env=_sanitized_environment(),
            timeout=min(spec.timeout_seconds, 60.0),
        )
        (run_dir / "verifier.stdout.txt").write_text(verifier.stdout, encoding="utf-8")
        (run_dir / "verifier.stderr.txt").write_text(verifier.stderr, encoding="utf-8")

    trace = _run_finished_record(trace_path)
    passed = verifier.exit_code == 0 and protected_unchanged
    result_path = run_dir / "result.json"
    result = EvalResult(
        case_id=spec.case_id,
        run_number=run_number,
        passed=passed,
        agent_exit_code=agent.exit_code,
        verifier_exit_code=verifier.exit_code,
        protected_files_unchanged=protected_unchanged,
        false_success=agent.exit_code == 0 and not passed,
        agent_timed_out=agent.timed_out,
        verifier_timed_out=verifier.timed_out,
        status=_optional_string(trace.get("status")),
        reason=_optional_string(trace.get("reason")),
        steps=_optional_int(trace.get("steps")),
        tool_calls=_optional_int(trace.get("tool_calls")),
        elapsed_ms=_optional_int(trace.get("elapsed_ms")),
        result_path=result_path,
    )
    serialized = asdict(result)
    serialized["result_path"] = str(result_path.relative_to(results_dir))
    result_path.write_text(
        json.dumps(serialized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> ProcessResult:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return ProcessResult(process.returncode, stdout, stderr, False)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return ProcessResult(124, stdout, stderr, True)


def _hash_globs(workspace: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pattern in patterns:
        for path in workspace.glob(pattern):
            if path.is_file():
                relative = path.relative_to(workspace).as_posix()
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _run_finished_record(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        return {}
    result: dict[str, Any] = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("event") == "run_finished":
            result = record
    return result


def _sanitized_environment() -> dict[str, str]:
    return {
        key: value for key, value in os.environ.items() if not _is_sensitive_environment_name(key)
    }


def _environment_words(name: str) -> set[str]:
    return set(name.upper().replace("-", "_").split("_"))


def _is_sensitive_environment_name(name: str) -> bool:
    upper = name.upper()
    if upper in _SENSITIVE_ENV_NAMES or upper == "GIT_CONFIG_COUNT":
        return True
    if upper.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
        return True
    return any(word in _environment_words(name) for word in _SENSITIVE_ENV_WORDS)


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_tuple(values: dict[str, Any], key: str) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a non-empty string array")
    return tuple(value)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BareAgent end-to-end evaluation cases.")
    parser.add_argument("--case", default="all", help="case directory name or 'all'")
    parser.add_argument("--repeat", type=int, default=1, help="number of runs per case")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    case_dirs = sorted(path for path in DEFAULT_CASES_ROOT.iterdir() if path.is_dir())
    if args.case != "all":
        case_dirs = [DEFAULT_CASES_ROOT / args.case]
    if not case_dirs or any(not path.is_dir() for path in case_dirs):
        raise SystemExit(f"unknown evaluation case: {args.case}")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results_dir = args.results.expanduser().resolve() / timestamp
    all_results: list[EvalResult] = []
    for case_dir in case_dirs:
        spec = load_case(case_dir)
        for run_number in range(1, args.repeat + 1):
            result = run_case(spec, run_number=run_number, results_dir=results_dir)
            all_results.append(result)
            state = "PASS" if result.passed else "FAIL"
            print(
                f"{state} {result.case_id} run={run_number} "
                f"agent={result.agent_exit_code} verifier={result.verifier_exit_code} "
                f"steps={result.steps} tools={result.tool_calls}"
            )
    passed = sum(result.passed for result in all_results)
    print(f"summary: {passed}/{len(all_results)} passed; results={results_dir}")
    return 0 if passed == len(all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
