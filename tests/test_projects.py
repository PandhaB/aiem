from pathlib import Path
import json

from PIL import Image

from core.projects import ProjectStore


def test_create_project_writes_folder_layout(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("Loops TEM", ["Loop-A", "Loop-B"])
    assert record.id == "loops-tem"
    assert (record.root / "project.json").is_file()
    assert record.images_dir.is_dir()
    assert record.annotations_path.is_file()
    assert record.weights_dir.is_dir()
    assert record.runs_dir.is_dir()
    assert [item.name for item in record.classes] == ["Loop-A", "Loop-B"]
    assert record.engine == "detectron2"
    assert record.model == "mask_rcnn_r50_fpn"

    duplicate = store.create("Loops TEM", ["Loop-A"])
    assert duplicate.id == "loops-tem-2"


def test_add_image_updates_coco(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("Demo", ["Loop-A"])
    png = tmp_path / "tile.png"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(png)
    info = store.add_image(record.id, "tile.png", png.read_bytes())
    assert info["width"] == 16
    assert info["height"] == 12
    images = store.list_images(record.id)
    assert images[0]["file_name"] == "tile.png"


def test_delete_project_removes_folder(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("To Remove", ["Loop-A"])
    root = record.root
    assert root.is_dir()
    store.delete(record.id)
    assert not root.exists()
    listed = store.list_projects()
    assert listed == []


def test_legacy_project_json_gets_default_model(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("Legacy", ["Loop-A"], engine="stub")
    payload = json.loads((record.root / "project.json").read_text(encoding="utf-8"))
    del payload["model"]
    (record.root / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    loaded = store.get(record.id)
    assert loaded.engine == "stub"
    assert loaded.model == "mask_rcnn_r50_fpn"
