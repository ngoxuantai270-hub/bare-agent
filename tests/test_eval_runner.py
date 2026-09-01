from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bare_agent.evals import load_case, run_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CASES = [
    "config-errors",
    "discount-type",
    "inventory-idempotency",
    "mutable-default",
    "pagination-boundary",
    "path-traversal",
]


def make_case(tmp_path: Path) -> Path:
    case = tmp_path / "sample-case"
    workspace = case / "workspace"
    hidden = case / "hidden_tests"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    hidden.mkdir()
    (workspace / "src/__init__.py").write_text("")
    (workspace / "src/value.py").write_text("VALUE = 1\n")
    (workspace / "tests/test_public.py").write_text(
        "from src.value import VALUE\n\ndef test_value():\n    assert VALUE == 2\n"
    )
    (hidden / "test_hidden.py").write_text(
        "import os\n\n"
        "from src.value import VALUE\n\n"
        "def test_hidden_value():\n    assert VALUE == 2\n\n"
        "def test_verifier_environment_is_sanitized():\n"
        "    assert 'OPENAI_API_KEY' not in os.environ\n"
        "    assert 'SSH_AUTH_SOCK' not in os.environ\n"
    )
    (case / "case.toml").write_text(
        """task = "Fix the value without modifying tests."
timeout_seconds = 30
verify_argv = ["python", "-m", "pytest", "-q", "tests", "_hidden_tests"]
protected_globs = ["tests/**/*.py"]
"""
    )
    return case


def make_fake_agent(tmp_path: Path, *, modify_tests: bool = False) -> Path:
    script = tmp_path / ("bad_agent.py" if modify_tests else "good_agent.py")
    test_mutation = (
        "(workspace / 'tests/test_public.py').write_text("
        "'def test_cheated():\\n    assert True\\n')"
        if modify_tests
        else "pass"
    )
    script.write_text(
        f"""import json
import sys
from pathlib import Path

workspace = Path(sys.argv[sys.argv.index('--workspace') + 1])
trace = Path(sys.argv[sys.argv.index('--trace-jsonl') + 1])
(workspace / 'src/value.py').write_text('VALUE = 2\\n')
{test_mutation}
trace.write_text(json.dumps({{
    'event': 'run_finished',
    'status': 'completed',
    'reason': 'final_answer',
    'steps': 2,
    'tool_calls': 3,
    'elapsed_ms': 10,
}}) + '\\n')
"""
    )
    return script


def test_eval_runner_uses_hidden_verifier_and_records_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-verifier")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/private/agent.sock")
    spec = load_case(make_case(tmp_path))
    result = run_case(
        spec,
        run_number=1,
        results_dir=tmp_path / "results",
        agent_command=[sys.executable, str(make_fake_agent(tmp_path))],
    )

    assert result.passed
    assert result.agent_exit_code == 0
    assert result.verifier_exit_code == 0
    assert result.protected_files_unchanged
    assert not result.false_success
    assert result.steps == 2
    assert result.tool_calls == 3
    saved = json.loads(result.result_path.read_text())
    assert saved["case_id"] == "sample-case"
    assert "task" not in saved
    assert "model_text" not in saved


def test_eval_runner_rejects_test_tampering_even_when_hidden_tests_pass(tmp_path: Path) -> None:
    spec = load_case(make_case(tmp_path))
    result = run_case(
        spec,
        run_number=1,
        results_dir=tmp_path / "results",
        agent_command=[sys.executable, str(make_fake_agent(tmp_path, modify_tests=True))],
    )

    assert not result.passed
    assert result.verifier_exit_code == 0
    assert not result.protected_files_unchanged
    assert result.false_success


@pytest.mark.parametrize("case_id", COMMITTED_CASES)
def test_committed_eval_case_starts_with_a_real_failure(tmp_path: Path, case_id: str) -> None:
    spec = load_case(PROJECT_ROOT / "evals/cases" / case_id)
    workspace = tmp_path / case_id
    shutil.copytree(spec.workspace, workspace)
    shutil.copytree(spec.hidden_tests, workspace / "_hidden_tests")
    argv = list(spec.verify_argv)
    argv[0] = sys.executable

    result = subprocess.run(argv, cwd=workspace, capture_output=True, text=True, check=False)

    assert result.returncode != 0, f"{case_id} is not a failing evaluation fixture"
