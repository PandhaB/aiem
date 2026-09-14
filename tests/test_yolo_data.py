import os
from pathlib import Path

import pytest
from PIL import Image

from engine.ultralytics import ultralytics_data_dir, _configure_ultralytics_env
from engine.yolo_data import write_yolo_seg_dataset, yolo_imgsz


def test_ultralytics_cache_stays_on_weights_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITEM_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    dest = _configure_ultralytics_env()
    assert dest == tmp_path / "weights" / "ultralytics"
    assert dest.is_dir()
    assert Path(os.environ["YOLO_CONFIG_DIR"]) == dest / "config"
    assert Path(os.environ["YOLO_CONFIG_DIR"]).is_dir()
    assert ultralytics_data_dir() == dest


def test_yolo_imgsz_follows_small_tiles() -> None:
    assert yolo_imgsz(image_sizes=[(256, 256)]) == 256
    assert yolo_imgsz(image_sizes=[(200, 180)]) == 224


def test_yolo_imgsz_caps_at_640() -> None:
    assert yolo_imgsz() == 640
    assert yolo_imgsz(image_sizes=[(2048, 1024)]) == 640
    assert yolo_imgsz(min_size=1024, image_sizes=[(256, 256)]) == 640


def test_yolo_imgsz_rejects_smaller_than_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        yolo_imgsz(min_size=16)


def test_write_yolo_seg_dataset_normalises_polygons(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (32, 24), (90, 90, 90)).save(images_dir / "tile.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        """
{
  "images": [{"id": 1, "file_name": "tile.png", "width": 32, "height": 24}],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "segmentation": [[2, 2, 20, 2, 20, 18, 2, 18]],
    "bbox": [2, 2, 18, 16], "area": 288, "iscrowd": 0
  }],
  "categories": [{"id": 1, "name": "Loop-A", "supercategory": "Loop-A"}]
}
""",
        encoding="utf-8",
    )
    dest = tmp_path / "yolo"
    yaml_path = write_yolo_seg_dataset(images_dir, annotations, dest, ["Loop-A"])
    assert yaml_path.is_file()
    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "images/train" in yaml_text
    assert "Loop-A" in yaml_text
    label = (dest / "labels" / "train" / "tile.txt").read_text(encoding="utf-8").strip()
    assert label.startswith("0 ")
    values = [float(item) for item in label.split()[1:]]
    assert values == pytest.approx(
        [2 / 32, 2 / 24, 20 / 32, 2 / 24, 20 / 32, 18 / 24, 2 / 32, 18 / 24],
        abs=1e-6,
    )
    linked = dest / "images" / "train" / "tile.png"
    assert linked.exists()
