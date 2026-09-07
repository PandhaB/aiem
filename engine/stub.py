from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from engine.types import (
    ExportRequest,
    InferRequest,
    InferResult,
    ProgressCallback,
    TrainRequest,
    TrainResult,
)

CHECKPOINT_NAME = "model_final.stub.json"
PREDICTIONS_NAME = "predictions.json"


class StubEngine:
    """Fake backend used to exercise the product path without PyTorch or a GPU.

    It writes a JSON checkpoint and synthetic instance outputs so tests and the
    UI can run end-to-end. It is not a real segmenter.
    """

    def name(self) -> str:
        return "stub"

    def train(
        self,
        request: TrainRequest,
        on_progress: ProgressCallback | None = None,
    ) -> TrainResult:
        self._report(on_progress, 0.1, "Validating training inputs")
        if not request.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")
        if not request.annotations_path.is_file():
            raise FileNotFoundError(f"Annotations file not found: {request.annotations_path}")
        with request.annotations_path.open(encoding="utf-8") as handle:
            coco = json.load(handle)
        if "categories" not in coco or "images" not in coco:
            raise ValueError("Annotations file is not valid COCO JSON.")

        request.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = request.output_dir / CHECKPOINT_NAME
        metrics_path = request.output_dir / "metrics.json"

        self._report(on_progress, 0.5, "Writing stub checkpoint")
        payload = {
            "engine": self.name(),
            "init": request.init,
            "class_names": list(request.class_names),
            "pretrained_weights_path": (
                str(request.pretrained_weights_path)
                if request.pretrained_weights_path
                else None
            ),
            "max_iter": request.max_iter,
            "image_count": len(coco.get("images", [])),
            "annotation_count": len(coco.get("annotations", [])),
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        metrics_path.write_text(
            json.dumps({"loss": 0.0, "iterations": request.max_iter or 1}, indent=2),
            encoding="utf-8",
        )
        self._report(on_progress, 1.0, "Training complete")
        return TrainResult(checkpoint_path=checkpoint_path, metrics_path=metrics_path)

    def load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Checkpoint is not valid JSON: {path}") from exc
        if payload.get("engine") != self.name():
            raise ValueError(
                f"Checkpoint was produced by {payload.get('engine')!r}, not {self.name()!r}."
            )

    def infer(
        self,
        request: InferRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        self._report(on_progress, 0.1, "Loading checkpoint")
        self.load_checkpoint(request.checkpoint_path)
        image_paths = _list_images(request.images_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {request.images_dir}")

        request.output_dir.mkdir(parents=True, exist_ok=True)
        coco = _synthetic_coco(image_paths, request.class_names)
        coco_path = request.output_dir / PREDICTIONS_NAME
        coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

        self._report(on_progress, 0.6, "Exporting masks and overlays")
        return self.export_predictions(
            ExportRequest(
                predictions_coco_path=coco_path,
                images_dir=request.images_dir,
                overlay_colors=request.overlay_colors,
                output_dir=request.output_dir,
                class_names=request.class_names,
            ),
            on_progress=on_progress,
        )

    def export_predictions(
        self,
        request: ExportRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        if not request.predictions_coco_path.is_file():
            raise FileNotFoundError(
                f"Predictions file not found: {request.predictions_coco_path}"
            )
        coco = json.loads(request.predictions_coco_path.read_text(encoding="utf-8"))
        overlay_dir = request.output_dir / "overlays"
        masks_dir = request.output_dir / "masks"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
        anns_by_image: dict[int, list[dict]] = {}
        for annotation in coco.get("annotations", []):
            anns_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for index, image_info in enumerate(coco.get("images", [])):
            image_path = request.images_dir / image_info["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert("RGBA")
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay, "RGBA")
            annotations = anns_by_image.get(image_info["id"], [])
            for instance_index, annotation in enumerate(annotations, start=1):
                class_name = categories.get(annotation["category_id"], "unknown")
                color = _parse_hex(
                    request.overlay_colors.get(class_name, "#e63946")
                )
                fill = (*color, 96)
                outline = (*color, 255)
                polygon = _flat_to_pairs(annotation["segmentation"][0])
                draw.polygon(polygon, fill=fill, outline=outline)
                mask = Image.new("L", image.size, 0)
                ImageDraw.Draw(mask).polygon(polygon, fill=255)
                stem = Path(image_info["file_name"]).stem
                mask.save(masks_dir / f"{stem}_{instance_index:03d}.png")
            composed = Image.alpha_composite(image, overlay).convert("RGB")
            composed.save(overlay_dir / image_info["file_name"])
            if coco.get("images"):
                self._report(
                    on_progress,
                    0.6 + 0.4 * ((index + 1) / len(coco["images"])),
                    f"Exported {image_info['file_name']}",
                )

        coco_path = request.output_dir / PREDICTIONS_NAME
        if request.predictions_coco_path.resolve() != coco_path.resolve():
            coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        self._report(on_progress, 1.0, "Export complete")
        return InferResult(coco_path=coco_path, overlay_dir=overlay_dir, masks_dir=masks_dir)

    @staticmethod
    def _report(on_progress: ProgressCallback | None, value: float, message: str) -> None:
        if on_progress is not None:
            on_progress(value, message)


def _list_images(images_dir: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _synthetic_coco(image_paths: list[Path], class_names: list[str]) -> dict:
    categories = [
        {"id": index, "name": name, "supercategory": name}
        for index, name in enumerate(class_names, start=1)
    ]
    images = []
    annotations = []
    next_ann_id = 1
    for image_id, path in enumerate(image_paths, start=1):
        with Image.open(path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": path.name,
                "width": width,
                "height": height,
            }
        )
        if not categories:
            continue
        polygon = _center_diamond(width, height)
        xs = polygon[0::2]
        ys = polygon[1::2]
        x_min, y_min = min(xs), min(ys)
        box_w, box_h = max(xs) - x_min, max(ys) - y_min
        annotations.append(
            {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": categories[0]["id"],
                "segmentation": [polygon],
                "bbox": [x_min, y_min, box_w, box_h],
                "area": box_w * box_h / 2,
                "iscrowd": 0,
                "score": 1.0,
            }
        )
        next_ann_id += 1
    return {
        "info": {"description": "Stub instance-segmentation predictions", "version": "0.1"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _center_diamond(width: int, height: int) -> list[float]:
    cx, cy = width / 2, height / 2
    rx, ry = width * 0.25, height * 0.25
    return [cx, cy - ry, cx + rx, cy, cx, cy + ry, cx - rx, cy]


def _flat_to_pairs(flat: list[float]) -> list[tuple[float, float]]:
    return [(flat[index], flat[index + 1]) for index in range(0, len(flat), 2)]


def _parse_hex(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return (230, 57, 70)
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
