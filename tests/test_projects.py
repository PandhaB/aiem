from pathlib import Path
import json

import pytest
from PIL import Image

from core.jobs import JobRunner, _backend_options_for_checkpoint
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


def test_create_ultralytics_project_uses_yolo_default(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("YOLO Loops", ["Loop-A"], engine="ultralytics")
    assert record.engine == "ultralytics"
    assert record.model == "yolov8s-seg"


def test_update_model_rejects_other_engine(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("YOLO Cards", ["Loop-A"], engine="ultralytics")
    with pytest.raises(ValueError, match="does not belong"):
        store.update_model(record.id, "mask_rcnn_r50_fpn")


def test_list_checkpoints_skips_backend_sidecar(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("YOLO Weights", ["Loop-A"], engine="ultralytics")
    (record.weights_dir / "model_0001.pt").write_bytes(b"weights")
    (record.weights_dir / "model_0001.backend.json").write_text("{}", encoding="utf-8")
    names = [item["name"] for item in store.list_checkpoints(record.id)]
    assert names == ["model_0001.pt"]


def test_update_model_persists(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("Cards", ["Loop-A"], engine="detectron2")
    updated = store.update_model(record.id, "mask_rcnn_r101_fpn")
    assert updated.model == "mask_rcnn_r101_fpn"
    reloaded = store.get(record.id)
    assert reloaded.model == "mask_rcnn_r101_fpn"


def test_legacy_project_json_gets_default_model(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("Legacy", ["Loop-A"], engine="stub")
    payload = json.loads((record.root / "project.json").read_text(encoding="utf-8"))
    del payload["model"]
    (record.root / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    loaded = store.get(record.id)
    assert loaded.engine == "stub"
    assert loaded.model == "mask_rcnn_r50_fpn"


def test_publish_checkpoint_copies_vitdet_backend_sidecar(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    runner = JobRunner(store, tmp_path / "weights")
    run_output = tmp_path / "run-output"
    run_output.mkdir()
    checkpoint = run_output / "model_0003.pth"
    checkpoint.write_bytes(b"weights")
    (run_output / "backend.json").write_text(json.dumps({"input_size": 256, "square_pad": 256}), encoding="utf-8")
    dest_dir = tmp_path / "project-weights"
    published = runner._publish_checkpoint(dest_dir, checkpoint)
    assert published == dest_dir / "model_0003.pth"
    sidecar = dest_dir / "model_0003.backend.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["input_size"] == 256
    assert _backend_options_for_checkpoint(published) == {"min_size": 256}
