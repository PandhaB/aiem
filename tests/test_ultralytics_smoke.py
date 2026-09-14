from pathlib import Path
import os

import pytest
from PIL import Image

from engine.catalog import ensure_pretrained, ultralytics_installed
from engine.registry import get_engine
from engine.types import InferRequest, TrainRequest


def _ds1_all() -> Path | None:
    candidates = [
        Path(os.environ.get("AITEM_DATASETS_DIR", "Datasets")) / "DS-1" / "all",
        Path("/data/datasets/DS-1/all"),
        Path("Datasets/DS-1/all"),
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _write_png(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    Image.new("RGB", size, (90, 90, 90)).save(path)


def _write_coco(path: Path, image_name: str, width: int, height: int) -> None:
    x2 = min(width - 8, 80)
    y2 = min(height - 8, 80)
    path.write_text(
        f"""
{{
  "images": [{{"id": 1, "file_name": "{image_name}", "width": {width}, "height": {height}}}],
  "annotations": [{{
    "id": 1, "image_id": 1, "category_id": 1,
    "segmentation": [[8, 8, {x2}, 8, {x2}, {y2}, 8, {y2}]],
    "bbox": [8, 8, {x2 - 8}, {y2 - 8}], "area": {(x2 - 8) * (y2 - 8)}, "iscrowd": 0
  }}],
  "categories": [{{"id": 1, "name": "Loop-A", "supercategory": "Loop-A"}}]
}}
""",
        encoding="utf-8",
    )


def _prepare_images(tmp_path: Path) -> tuple[Path, str, int, int]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    source = None
    ds1 = _ds1_all()
    if ds1 is not None:
        pngs = sorted(path for path in ds1.iterdir() if path.suffix.lower() == ".png")
        if pngs:
            source = pngs[0]
    if source is None:
        dest = images_dir / "tile.png"
        _write_png(dest)
    else:
        dest = images_dir / source.name
        Image.open(source).convert("RGB").save(dest)
    with Image.open(dest) as image:
        width, height = image.size
    return images_dir, dest.name, width, height


@pytest.mark.skipif(not ultralytics_installed(), reason="Ultralytics is not installed")
@pytest.mark.skipif(not _cuda(), reason="CUDA is not visible in this environment")
def test_short_ultralytics_train_and_infer(tmp_path: Path) -> None:
    images_dir, image_name, width, height = _prepare_images(tmp_path)
    annotations = tmp_path / "annotations.json"
    _write_coco(annotations, image_name, width, height)
    pretrained = ensure_pretrained(
        Path(os.environ.get("AITEM_WEIGHTS_DIR", "/data/weights")),
        "yolov8n-seg",
    )
    engine = get_engine("ultralytics")
    train = engine.train(
        TrainRequest(
            images_dir=images_dir,
            annotations_path=annotations,
            class_names=["Loop-A"],
            init="pretrained",
            pretrained_weights_path=pretrained,
            output_dir=tmp_path / "train",
            max_iter=1,
            model="yolov8n-seg",
        )
    )
    assert train.checkpoint_path.is_file()
    assert train.checkpoint_path.suffix == ".pt"
    engine.load_checkpoint(train.checkpoint_path)
    infer = engine.infer(
        InferRequest(
            images_dir=images_dir,
            checkpoint_path=train.checkpoint_path,
            class_names=["Loop-A"],
            overlay_colors={"Loop-A": "#e63946"},
            output_dir=tmp_path / "infer",
            model="yolov8n-seg",
        )
    )
    assert infer.coco_path.is_file()
    assert infer.overlay_dir.is_dir()
    assert infer.masks_dir.is_dir()
    overlays = list(infer.overlay_dir.iterdir())
    assert overlays, "Expected at least one overlay image"
