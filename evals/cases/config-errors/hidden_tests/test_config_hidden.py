from pathlib import Path

import pytest

from src.config import load_config


def test_missing_file_preserves_file_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def test_invalid_json_keeps_original_cause(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("[broken", encoding="utf-8")
    with pytest.raises(ValueError) as captured:
        load_config(path)
    assert captured.value.__cause__ is not None
