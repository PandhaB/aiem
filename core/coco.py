from __future__ import annotations

import json
from pathlib import Path

from engine.types import ClassSpec

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def empty_coco(classes: list[ClassSpec], description: str = "") -> dict:
    return {
        "info": {
            "description": description or "Project annotations",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": item.id, "name": item.name, "supercategory": item.name}
            for item in classes
        ],
    }


def load_coco(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"COCO file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_coco(data)
    return data


def save_coco(path: Path, data: dict) -> None:
    validate_coco(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_coco(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("COCO document must be a JSON object.")
    for key in ("images", "annotations", "categories"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"COCO document is missing a '{key}' list.")
    for category in data["categories"]:
        if "id" not in category or "name" not in category:
            raise ValueError("Each category must have 'id' and 'name'.")
    for image in data["images"]:
        if "id" not in image or "file_name" not in image:
            raise ValueError("Each image must have 'id' and 'file_name'.")
    for annotation in data["annotations"]:
        for field in ("id", "image_id", "category_id", "segmentation"):
            if field not in annotation:
                raise ValueError(f"Each annotation must have '{field}'.")
        segmentation = annotation["segmentation"]
        if not isinstance(segmentation, list) or not segmentation:
            raise ValueError("Annotation segmentation must be a non-empty polygon list.")
        if not isinstance(segmentation[0], list) or len(segmentation[0]) < 6:
            raise ValueError("Each polygon must contain at least three points (6 numbers).")


def polygon_bbox_area(flat: list[float]) -> tuple[list[float], float]:
    xs = flat[0::2]
    ys = flat[1::2]
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    area = 0.0
    for index in range(0, len(flat), 2):
        x1, y1 = flat[index], flat[index + 1]
        x2, y2 = flat[(index + 2) % len(flat)], flat[(index + 3) % len(flat)]
        area += x1 * y2 - x2 * y1
    return [x_min, y_min, width, height], abs(area) / 2.0


def next_id(items: list[dict]) -> int:
    if not items:
        return 1
    return max(int(item["id"]) for item in items) + 1


def add_image(coco: dict, file_name: str, width: int, height: int) -> dict:
    existing = next((item for item in coco["images"] if item["file_name"] == file_name), None)
    if existing is not None:
        existing["width"] = width
        existing["height"] = height
        return existing
    image = {
        "id": next_id(coco["images"]),
        "file_name": file_name,
        "width": width,
        "height": height,
    }
    coco["images"].append(image)
    return image


def replace_annotations(coco: dict, annotations: list[dict]) -> dict:
    rebuilt = []
    for index, annotation in enumerate(annotations, start=1):
        segmentation = annotation["segmentation"]
        polygon = segmentation[0]
        bbox, area = polygon_bbox_area(polygon)
        rebuilt.append(
            {
                "id": index,
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],
                "segmentation": [polygon],
                "bbox": annotation.get("bbox") or bbox,
                "area": annotation.get("area") if annotation.get("area") is not None else area,
                "iscrowd": annotation.get("iscrowd", 0),
            }
        )
    coco["annotations"] = rebuilt
    validate_coco(coco)
    return coco
