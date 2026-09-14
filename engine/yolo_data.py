from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

YOLO_STRIDE = 32
YOLO_DEFAULT_IMGSZ = 640
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def yolo_imgsz(
    min_size: int | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
) -> int:
    """Ultralytics imgsz must be a multiple of 32. Small TEM tiles need not be upscaled to 640."""
    if min_size:
        raw = int(min_size)
        if raw < YOLO_STRIDE:
            raise ValueError(f"YOLO image size must be at least {YOLO_STRIDE} (network stride).")
        return min(YOLO_DEFAULT_IMGSZ, _round_up(raw, YOLO_STRIDE))
    if image_sizes:
        longest = max(max(width, height) for width, height in image_sizes)
        return min(YOLO_DEFAULT_IMGSZ, _round_up(longest, YOLO_STRIDE))
    return YOLO_DEFAULT_IMGSZ


def write_yolo_seg_dataset(
    images_dir: Path,
    annotations_path: Path,
    dest_dir: Path,
    class_names: list[str],
) -> Path:
    """Convert project COCO polygons into an Ultralytics segmentation layout.

    ``dest_dir`` gets ``images/train`` (symlinks), ``labels/train`` (txt), and ``data.yaml``.
    Validation reuses the train split: v1 TEM sets are small and stay in one folder.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = coco.get("images") or []
    if not images:
        raise ValueError("Annotations file has no images.")
    categories = {item["id"]: item["name"] for item in coco.get("categories") or []}
    name_to_index = {name: index for index, name in enumerate(class_names)}
    anns_by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations") or []:
        anns_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    image_root = dest_dir / "images" / "train"
    label_root = dest_dir / "labels" / "train"
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    for item in images:
        file_name = Path(item["file_name"]).name
        source = images_dir / file_name
        if not source.is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width < 1 or height < 1:
            raise ValueError(f"Image {file_name} is missing width/height in the COCO file.")
        target = image_root / file_name
        _link_or_copy(source, target)
        lines = []
        for annotation in anns_by_image.get(int(item["id"]), []):
            if annotation.get("iscrowd"):
                continue
            class_name = categories.get(annotation.get("category_id"))
            if class_name not in name_to_index:
                continue
            polygon = _first_polygon(annotation.get("segmentation"))
            if polygon is None:
                continue
            normalised = _normalise_polygon(polygon, width, height)
            if normalised is None:
                continue
            class_index = name_to_index[class_name]
            coords = " ".join(f"{value:.6f}" for value in normalised)
            lines.append(f"{class_index} {coords}")
        (label_root / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    yaml_path = dest_dir / "data.yaml"
    yaml_path.write_text(_data_yaml(dest_dir, class_names), encoding="utf-8")
    return yaml_path


def image_hw(images_dir: Path, annotations_path: Path | None = None) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    if annotations_path is not None and annotations_path.is_file():
        try:
            coco = json.loads(annotations_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            coco = {}
        for item in coco.get("images") or []:
            width, height = item.get("width"), item.get("height")
            if width and height:
                sizes.append((int(width), int(height)))
        if sizes:
            return sizes
    if not images_dir.is_dir():
        return sizes
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(path) as image:
                sizes.append(image.size)
    return sizes


def _round_up(value: int, step: int) -> int:
    return max(step, math.ceil(value / step) * step)


def _link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(source.resolve())
    except OSError:
        dest.write_bytes(source.read_bytes())


def _first_polygon(segmentation) -> list[float] | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    polygon = segmentation[0]
    if not isinstance(polygon, list) or len(polygon) < 6:
        return None
    return [float(value) for value in polygon]


def _normalise_polygon(polygon: list[float], width: int, height: int) -> list[float] | None:
    if width < 1 or height < 1:
        return None
    values = []
    for index in range(0, len(polygon) - 1, 2):
        x = min(1.0, max(0.0, polygon[index] / width))
        y = min(1.0, max(0.0, polygon[index + 1] / height))
        values.extend((x, y))
    if len(values) < 6:
        return None
    return values


def _data_yaml(root: Path, class_names: list[str]) -> str:
    lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/train",
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {name}")
    return "\n".join(lines) + "\n"
