"""Turn COCO instance predictions into colour overlays and per-instance mask files.

All engines should write ``predictions.json`` then call :func:`export_instance_visuals`
instead of drawing in the UI.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from engine.types import ExportRequest, InferResult, ProgressCallback

PREDICTIONS_NAME = "predictions.json"
# Per-instance fill. Overlaps composite on top of each other so stacked masks
# get more opaque instead of the last polygon hiding the previous one.
MASK_FILL_ALPHA = 96


def filter_coco_instances(
    coco: dict,
    score_threshold: float | None = None,
    max_detections: int | None = None,
) -> dict:
    """Keep detections per image: drop low scores, then keep the highest-scoring ones."""
    if score_threshold is None and max_detections is None:
        return coco
    grouped: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations") or []:
        grouped.setdefault(annotation["image_id"], []).append(annotation)
    kept: list[dict] = []
    next_id = 1
    for image in coco.get("images") or []:
        items = list(grouped.get(image["id"], []))
        if score_threshold is not None:
            items = [
                item for item in items if float(item.get("score", 1.0)) >= score_threshold
            ]
        items.sort(key=lambda item: float(item.get("score", 1.0)), reverse=True)
        if max_detections is not None:
            items = items[:max_detections]
        for item in items:
            copy = dict(item)
            copy["id"] = next_id
            next_id += 1
            kept.append(copy)
    coco["annotations"] = kept
    return coco


def export_instance_visuals(
    request: ExportRequest,
    on_progress: ProgressCallback | None = None,
) -> InferResult:
    """Write colour overlays and per-instance masks from a COCO predictions file."""
    if not request.predictions_coco_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {request.predictions_coco_path}")
    coco = json.loads(request.predictions_coco_path.read_text(encoding="utf-8"))
    if request.smooth_tolerance is not None and request.smooth_tolerance > 0:
        simplify_coco_polygons(coco, request.smooth_tolerance)
        request.predictions_coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
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
        annotations = anns_by_image.get(image_info["id"], [])
        instances: list[tuple[list[tuple[float, float]], tuple[int, int, int]]] = []
        for instance_index, annotation in enumerate(annotations, start=1):
            class_name = categories.get(annotation["category_id"], "unknown")
            color = _parse_hex(request.overlay_colors.get(class_name, "#e63946"))
            segmentation = annotation.get("segmentation") or []
            if not segmentation:
                continue
            polygon = _flat_to_pairs(segmentation[0])
            if len(polygon) < 3:
                continue
            instances.append((polygon, color))
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
            stem = Path(image_info["file_name"]).stem
            mask.save(masks_dir / f"{stem}_{instance_index:03d}.png")
        compose_instance_overlay(image, instances).save(overlay_dir / image_info["file_name"])
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


def compose_instance_overlay(
    image: Image.Image,
    instances: list[tuple[list[tuple[float, float]], tuple[int, int, int]]],
) -> Image.Image:
    """Tint instance polygons onto the image. Overlaps accumulate opacity."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for polygon, color in instances:
        if len(polygon) < 3:
            continue
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(polygon, fill=(*color, MASK_FILL_ALPHA))
        overlay = Image.alpha_composite(overlay, layer)
    draw = ImageDraw.Draw(overlay)
    for polygon, color in instances:
        if len(polygon) < 3:
            continue
        draw.polygon(polygon, outline=(*color, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def simplify_polygon(flat: list[float], tolerance: float) -> list[float]:
    """Drop vertices within ``tolerance`` pixels of the simplified outline (Douglas-Peucker)."""
    if tolerance <= 0:
        return [float(value) for value in flat]
    points = _flat_to_pairs(flat)
    if len(points) < 2:
        return [float(value) for value in flat]
    if points[0] == points[-1] and len(points) > 1:
        points = points[:-1]
    if len(points) <= 3:
        return _pairs_to_flat(points)
    closed = points + [points[0]]
    simplified = _rdp(closed, float(tolerance))
    if len(simplified) >= 2 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    if len(simplified) < 3:
        return _pairs_to_flat(points)
    return _pairs_to_flat(simplified)


def simplify_coco_polygons(coco: dict, tolerance: float) -> dict:
    """Rewrite every instance polygon in place and refresh bbox/area."""
    if tolerance <= 0:
        return coco
    rebuilt = []
    for annotation in coco.get("annotations") or []:
        copy = dict(annotation)
        segmentation = list(copy.get("segmentation") or [])
        if segmentation and isinstance(segmentation[0], list):
            polygon = simplify_polygon(segmentation[0], tolerance)
            copy["segmentation"] = [polygon]
            xs = polygon[0::2]
            ys = polygon[1::2]
            x_min, y_min = min(xs), min(ys)
            width, height = max(xs) - x_min, max(ys) - y_min
            copy["bbox"] = [x_min, y_min, width, height]
            copy["area"] = abs(width * height)
        rebuilt.append(copy)
    coco["annotations"] = rebuilt
    return coco


def _flat_to_pairs(flat: list[float]) -> list[tuple[float, float]]:
    return [(float(flat[index]), float(flat[index + 1])) for index in range(0, len(flat) - 1, 2)]


def _pairs_to_flat(points: list[tuple[float, float]]) -> list[float]:
    out: list[float] = []
    for x, y in points:
        out.extend([x, y])
    return out


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        max_dist = -1.0
        farthest = start
        for index in range(start + 1, end):
            dist = _point_line_distance(points[index], points[start], points[end])
            if dist > max_dist:
                max_dist = dist
                farthest = index
        if max_dist > epsilon and farthest != start:
            keep[farthest] = True
            stack.append((start, farthest))
            stack.append((farthest, end))
    return [point for point, flag in zip(points, keep) if flag]


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length


def _parse_hex(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return (230, 57, 70)
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
