from pathlib import Path

import pytest
from PIL import Image

from core.paths import list_image_folders, resolve_under


def test_resolve_under_stays_in_root(tmp_path: Path) -> None:
    nested = tmp_path / "DS-1" / "all"
    nested.mkdir(parents=True)
    resolved = resolve_under(tmp_path, "DS-1/all")
    assert resolved == nested.resolve()


def test_resolve_under_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_under(tmp_path, "../outside")
    with pytest.raises(ValueError):
        resolve_under(tmp_path, "/tmp")


def test_list_image_folders_finds_nested(tmp_path: Path) -> None:
    folder = tmp_path / "DS-1" / "all"
    folder.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 10, 10)).save(folder / "tile.png")
    (tmp_path / "empty").mkdir()
    found = list_image_folders(tmp_path)
    assert found == [{"relative_path": "DS-1/all", "name": "DS-1/all"}]
