from pathlib import Path

import pytest

from src.storage import save_report


def test_save_nested_report(tmp_path: Path):
    root = tmp_path / "reports"
    result = save_report(root, "daily/report.txt", "ok")
    assert result.read_text(encoding="utf-8") == "ok"
    assert result.is_relative_to(root)


def test_reject_parent_directory_escape(tmp_path: Path):
    root = tmp_path / "reports"
    escaped = tmp_path / "escaped.txt"
    with pytest.raises(ValueError, match="outside storage root"):
        save_report(root, "../escaped.txt", "secret")
    assert not escaped.exists()
