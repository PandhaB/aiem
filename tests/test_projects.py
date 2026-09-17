from pathlib import Path
import json

import pytest
from PIL import Image

from core.jobs import JobRunner, _backend_options_for_checkpoint, _model_for_checkpoint
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
    items = store.list_checkpoints(record.id)
    assert [item["name"] for item in items] == ["model_0001.pt"]
    assert items[0]["ref"] == "model_0001.pt"
    assert items[0]["label"] == "project weights / model_0001.pt"


def test_list_checkpoints_keeps_same_filename_from_separate_runs(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("YOLO Runs", ["Loop-A"], engine="ultralytics")
    (record.weights_dir / "model_0100.pt").write_bytes(b"published")

    def fake_run(run_id: str, model: str) -> None:
        run_dir = record.runs_dir / run_id
        (run_dir / "output").mkdir(parents=True)
        (run_dir / "output" / "model_0100.pt").write_bytes(b"run-weights")
        (run_dir / "output" / "metrics.json").write_text(
            json.dumps({"model": model}),
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "id": run_id,
                    "kind": "train",
                    "status": "completed",
                    "iteration": 100,
                    "max_iter": 100,
                }
            ),
            encoding="utf-8",
        )

    fake_run("train-20260915T120000-aaaa", "yolov8n-seg")
    fake_run("train-20260915T130000-bbbb", "yolov8s-seg")

    items = store.list_checkpoints(record.id)
    assert [item["ref"] for item in items] == [
        "train-20260915T130000-bbbb/model_0100.pt",
        "train-20260915T120000-aaaa/model_0100.pt",
    ]
    assert items[0]["label"].startswith("train-20260915T130000-bbbb / model_0100.pt")
    assert "yolov8s-seg" in items[0]["label"]
    assert "yolov8n-seg" in items[1]["label"]
    assert all(item["run_id"] for item in items)


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


def test_checkpoint_model_card_wins_over_project_default(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("Canvas Free", ["Loop-A"], engine="detectron2")
    assert record.model == "mask_rcnn_r50_fpn"
    run_id = "train-20260916T100000-vitdet"
    output = record.runs_dir / run_id / "output"
    output.mkdir(parents=True)
    checkpoint = output / "model_0100.pth"
    checkpoint.write_bytes(b"weights")
    (output / "backend.json").write_text(
        json.dumps({"model": "mask_rcnn_vitdet_b", "family": "vitdet", "input_size": 256}),
        encoding="utf-8",
    )
    (record.runs_dir / run_id / "status.json").write_text(
        json.dumps({"id": run_id, "kind": "train", "status": "completed", "model": "mask_rcnn_vitdet_b"}),
        encoding="utf-8",
    )
    assert _model_for_checkpoint(record, checkpoint, f"{run_id}/model_0100.pth") == "mask_rcnn_vitdet_b"
    assert _backend_options_for_checkpoint(checkpoint) == {"min_size": 256}


def test_checkpoint_vitdet_family_without_model_id(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("Legacy Sidecar", ["Loop-A"], engine="detectron2")
    checkpoint = record.weights_dir / "model_0008.pth"
    checkpoint.write_bytes(b"weights")
    checkpoint.with_name("model_0008.backend.json").write_text(
        json.dumps({"family": "vitdet", "input_size": 256}),
        encoding="utf-8",
    )
    assert _model_for_checkpoint(record, checkpoint, "model_0008.pth") == "mask_rcnn_vitdet_b"


def test_checkpoint_model_from_run_metrics_when_sidecar_omits_card(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    record = store.create("YOLO Card From Run", ["Loop-A"], engine="ultralytics")
    run_id = "train-20260916T110000-yolo"
    output = record.runs_dir / run_id / "output"
    output.mkdir(parents=True)
    checkpoint = output / "model_0050.pt"
    checkpoint.write_bytes(b"weights")
    (output / "metrics.json").write_text(json.dumps({"model": "yolo26x-seg"}), encoding="utf-8")
    (record.runs_dir / run_id / "status.json").write_text(
        json.dumps({"id": run_id, "kind": "train", "status": "completed"}),
        encoding="utf-8",
    )
    assert _model_for_checkpoint(record, checkpoint, f"{run_id}/model_0050.pt") == "yolo26x-seg"


def test_resolve_checkpoint_loads_weights_from_training_run(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    runner = JobRunner(store, tmp_path / "weights")
    record = store.create("YOLO Resolve", ["Loop-A"], engine="ultralytics")
    run_id = "train-20260915T130000-bbbb"
    output = record.runs_dir / run_id / "output"
    output.mkdir(parents=True)
    (output / "model_0100.pt").write_bytes(b"from-run")
    (record.weights_dir / "model_0100.pt").write_bytes(b"published")
    (record.runs_dir / run_id / "status.json").write_text(
        json.dumps({"id": run_id, "kind": "train", "status": "completed"}),
        encoding="utf-8",
    )
    path = runner._resolve_checkpoint(record, f"{run_id}/model_0100.pt", "ultralytics")
    assert path.resolve() == (output / "model_0100.pt").resolve()
    assert path.read_bytes() == b"from-run"


def test_backend_options_keep_native_zero_and_anchors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_1000.pth"
    checkpoint.write_bytes(b"weights")
    checkpoint.with_name("model_1000.backend.json").write_text(
        json.dumps(
            {
                "model": "mask_rcnn_x101_fpn",
                "family": "mask_rcnn",
                "min_size": 0,
                "min_size_test": 0,
                "anchor_sizes": [8, 16, 32, 64, 128, 256],
                "rpn_post_nms_topk_test": 2000,
            }
        ),
        encoding="utf-8",
    )
    options = _backend_options_for_checkpoint(checkpoint)
    assert options["min_size"] == 0
    assert options["min_size_test"] == 0
    assert options["anchor_sizes"] == [8, 16, 32, 64, 128, 256]
    assert options["rpn_post_nms_topk_test"] == 2000
