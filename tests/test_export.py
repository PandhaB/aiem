import json
from pathlib import Path

from PIL import Image

from engine.export import compose_instance_overlay, export_instance_visuals, simplify_polygon
from engine.types import ExportRequest


def test_overlapping_masks_accumulate_opacity(tmp_path: Path) -> None:
    image_path = tmp_path / "tile.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(image_path)
    coco = {
        "images": [{"id": 1, "file_name": "tile.png", "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[0, 0, 20, 0, 20, 20, 0, 20]],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            },
        ],
        "categories": [{"id": 1, "name": "Loop-A"}],
    }
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps(coco), encoding="utf-8")
    result = export_instance_visuals(
        ExportRequest(
            predictions_coco_path=predictions,
            images_dir=tmp_path,
            overlay_colors={"Loop-A": "#ff0000"},
            output_dir=tmp_path / "out",
            class_names=["Loop-A"],
        )
    )
    pixels = Image.open(result.overlay_dir / "tile.png").convert("RGB")
    single = pixels.getpixel((5, 5))
    overlap = pixels.getpixel((15, 15))
    empty = pixels.getpixel((30, 2))
    assert empty == (0, 0, 0)
    assert single[0] > empty[0]
    assert overlap[0] > single[0]


def test_filter_coco_instances_keeps_best_per_image() -> None:
    from engine.export import filter_coco_instances

    coco = {
        "images": [
            {"id": 1, "file_name": "a.png"},
            {"id": 2, "file_name": "b.png"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "score": 0.9},
            {"id": 2, "image_id": 1, "score": 0.4},
            {"id": 3, "image_id": 1, "score": 0.7},
            {"id": 4, "image_id": 2, "score": 0.2},
            {"id": 5, "image_id": 2, "score": 0.8},
        ],
    }
    filter_coco_instances(coco, score_threshold=0.5, max_detections=1)
    kept = coco["annotations"]
    assert [(item["image_id"], item["score"]) for item in kept] == [(1, 0.9), (2, 0.8)]
    assert [item["id"] for item in kept] == [1, 2]


def test_compose_instance_overlay_empty() -> None:
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    out = compose_instance_overlay(image, [])
    assert out.getpixel((0, 0)) == (10, 20, 30)


def test_simplify_polygon_drops_colinear_midpoints() -> None:
    # Square with a midpoint on every edge. A small tolerance should keep the four corners.
    noisy = [0, 0, 5, 0, 10, 0, 10, 5, 10, 10, 5, 10, 0, 10, 0, 5]
    simplified = simplify_polygon(noisy, 0.5)
    assert simplified == [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
