from pathlib import Path

from PIL import Image

from engine.stub import StubEngine
from engine.registry import get_engine
from engine.types import InferRequest, TrainRequest


def _write_png(path: Path, color: tuple[int, int, int] = (80, 80, 80)) -> None:
    Image.new("RGB", (64, 48), color).save(path)


def _write_coco(path: Path, image_name: str) -> None:
    path.write_text(
        """
{
  "images": [{"id": 1, "file_name": "%s", "width": 64, "height": 48}],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
    "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0
  }],
  "categories": [{"id": 1, "name": "Loop-A", "supercategory": "Loop-A"}]
}
"""
        % image_name,
        encoding="utf-8",
    )


def test_registry_returns_stub() -> None:
    engine = get_engine("stub")
    assert isinstance(engine, StubEngine)
    assert engine.name() == "stub"


def test_stub_train_and_infer_write_artifacts(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "tile.png"
    _write_png(image_path)
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations, "tile.png")

    engine = StubEngine()
    train = engine.train(
        TrainRequest(
            images_dir=images_dir,
            annotations_path=annotations,
            class_names=["Loop-A"],
            init="random",
            output_dir=tmp_path / "train",
        )
    )
    assert train.checkpoint_path.is_file()
    assert train.metrics_path.is_file()
    engine.load_checkpoint(train.checkpoint_path)

    infer = engine.infer(
        InferRequest(
            images_dir=images_dir,
            checkpoint_path=train.checkpoint_path,
            class_names=["Loop-A"],
            overlay_colors={"Loop-A": "#2a9d8f"},
            output_dir=tmp_path / "infer",
        )
    )
    assert infer.coco_path.is_file()
    assert infer.overlay_dir.is_dir()
    assert infer.masks_dir.is_dir()
    assert (infer.overlay_dir / "tile.png").is_file()
    masks = list(infer.masks_dir.glob("*.png"))
    assert masks
    coco_text = infer.coco_path.read_text(encoding="utf-8")
    assert "Loop-A" in coco_text
    assert "segmentation" in coco_text
