from pathlib import Path

import pytest

from src.storage import save_report


def test_reject_absolute_path(tmp_path: Path):
    root = tmp_path / "reports"
    with pytest.raises(ValueError, match="outside storage root"):
        save_report(root, str(tmp_path / "absolute.txt"), "secret")


def test_sibling_prefix_is_not_inside_root(tmp_path: Path):
    root = tmp_path / "reports"
    sibling = tmp_path / "reports-other" / "file.txt"
    with pytest.raises(ValueError, match="outside storage root"):
        save_report(root, str(sibling), "secret")
