import json
from pathlib import Path

import pytest

from src.config import load_config


def test_load_valid_config(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 8080}), encoding="utf-8")
    assert load_config(path) == {"port": 8080}


def test_invalid_json_has_clear_error(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid configuration JSON"):
        load_config(path)
