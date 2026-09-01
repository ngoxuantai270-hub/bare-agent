from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def test_configure_script_writes_private_env_file_without_echoing_key(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "configure-env.sh"
    env_file = tmp_path / ".env"
    credential = "unit-test-credential-123"

    completed = subprocess.run(
        ["bash", str(script)],
        input=f"{credential}\n\n\n",
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "BARE_AGENT_ENV_FILE": str(env_file),
        },
        check=False,
    )

    assert completed.returncode == 0
    assert credential not in completed.stdout
    assert credential not in completed.stderr
    assert "API_KEY" in completed.stderr
    assert "MODEL" in completed.stderr
    assert "BASE_URL" in completed.stderr
    assert env_file.read_text() == (
        f"OPENAI_API_KEY='{credential}'\n"
        "OPENAI_MODEL='deepseek-v4-flash'\n"
        "OPENAI_BASE_URL='https://api.deepseek.com'\n"
    )
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_configure_script_rejects_empty_api_key(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "configure-env.sh"

    completed = subprocess.run(
        ["bash", str(script)],
        input="\n",
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "BARE_AGENT_ENV_FILE": str(tmp_path / ".env"),
        },
        check=False,
    )

    assert completed.returncode != 0
    assert "API key cannot be empty" in completed.stderr
    assert not (tmp_path / ".env").exists()


def test_configure_script_does_not_replace_existing_env_without_confirmation(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "configure-env.sh"
    env_file = tmp_path / ".env"
    env_file.write_text("existing configuration\n")

    completed = subprocess.run(
        ["bash", str(script)],
        input="n\n",
        text=True,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "BARE_AGENT_ENV_FILE": str(env_file),
        },
        check=False,
    )

    assert completed.returncode != 0
    assert env_file.read_text() == "existing configuration\n"
    assert "not changed" in completed.stderr
