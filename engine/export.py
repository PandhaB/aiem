from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from engine.types import ExportRequest, InferResult, ProgressCallback

PREDICTIONS_NAME = "predictions.json"


def export_instance_visuals(
    request: ExportRequest,
    on_progress: ProgressCallback | None = None,
) -> InferResult:
    """Write colour overlays and per-instance masks from a COCO predictions file."""
    if not request.predictions_coco_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {request.predictions_coco_path}")
    coco = json.loads(request.predictions_coco_path.read_text(encoding="utf-8"))
    overlay_dir = request.output_dir / "overlays"
    masks_dir = request.output_dir / "masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

    stopped = False
    images = coco.get("images", [])
    for index, image_info in enumerate(images):
        if request.should_stop is not None and request.should_stop():
            stopped = True
            break
        image_path = request.images_dir / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGBA")
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        annotations = anns_by_image.get(image_info["id"], [])
        for instance_index, annotation in enumerate(annotations, start=1):
            class_name = categories.get(annotation["category_id"], "unknown")
            color = _parse_hex(request.overlay_colors.get(class_name, "#e63946"))
            polygon = _flat_to_pairs(annotation["segmentation"][0])
            if len(polygon) < 3:
                continue
            draw.polygon(polygon, fill=(*color, 96), outline=(*color, 255))
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
            stem = Path(image_info["file_name"]).stem
            mask.save(masks_dir / f"{stem}_{instance_index:03d}.png")
        Image.alpha_composite(image, overlay).convert("RGB").save(overlay_dir / image_info["file_name"])
        if on_progress is not None and images:
            on_progress(0.6 + 0.4 * ((index + 1) / len(images)), f"Exported {image_info['file_name']}")

    coco_path = request.output_dir / PREDICTIONS_NAME
    if request.predictions_coco_path.resolve() != coco_path.resolve():
        coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    if on_progress is not None:
        on_progress(1.0, "Inference stopped" if stopped else "Export complete")
    return InferResult(
        coco_path=coco_path,
        overlay_dir=overlay_dir,
        masks_dir=masks_dir,
        stopped=stopped,
    )


def _flat_to_pairs(flat: list[float]) -> list[tuple[float, float]]:
    return [(flat[index], flat[index + 1]) for index in range(0, len(flat) - 1, 2)]


def _parse_hex(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return (230, 57, 70)
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
