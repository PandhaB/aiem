from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from engine.catalog import (
    DEFAULT_MODEL,
    describe_device,
    get_model,
    reject_incompatible_checkpoint,
)
from engine.types import (
    ExportRequest,
    InferRequest,
    InferResult,
    ProgressCallback,
    TrainRequest,
    TrainResult,
)

CHECKPOINT_NAME = "model_final.pth"
PREDICTIONS_NAME = "predictions.json"


class Detectron2Engine:
    """First real backend: Mask R-CNN via Detectron2. UI must not import this module."""

    def name(self) -> str:
        return "detectron2"

    def train(
        self,
        request: TrainRequest,
        on_progress: ProgressCallback | None = None,
    ) -> TrainResult:
        detectron2 = _detectron2()
        torch = _torch()
        device_info = describe_device()
        self._report(
            on_progress,
            0.02,
            f"Device: {device_info['device']}"
            + ("" if device_info["cuda"] else f" — {device_info['message']}"),
        )
        if not request.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {request.images_dir}")
        if not request.annotations_path.is_file():
            raise FileNotFoundError(f"Annotations file not found: {request.annotations_path}")

        spec = get_model(request.model)
        if spec.engine != self.name():
            raise ValueError(f"Model {spec.id} is not a Detectron2 model.")

        request.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = _unique_dataset_name("train")
        detectron2.data.datasets.register_coco_instances(
            dataset_name,
            {},
            str(request.annotations_path),
            str(request.images_dir),
        )
        try:
            cfg = _build_train_cfg(
                detectron2,
                torch,
                spec.detectron2_config,
                dataset_name,
                n_classes=len(request.class_names),
                init=request.init,
                weights_path=request.pretrained_weights_path,
                max_iter=request.max_iter or 300,
                output_dir=request.output_dir,
            )
            trainer = _ProgressTrainer(cfg, on_progress)
            trainer.resume_or_load(resume=False)
            self._report(on_progress, 0.12, "Starting Detectron2 training")
            trainer.train()
        finally:
            _unregister(detectron2, dataset_name)

        checkpoint_path = request.output_dir / CHECKPOINT_NAME
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Detectron2 did not write {CHECKPOINT_NAME}")
        metrics_path = request.output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "engine": self.name(),
                    "model": spec.id,
                    "init": request.init,
                    "max_iter": request.max_iter or 300,
                    "device": device_info["device"],
                    "cuda": device_info["cuda"],
                    "num_classes": len(request.class_names),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._report(on_progress, 1.0, "Training complete")
        return TrainResult(checkpoint_path=checkpoint_path, metrics_path=metrics_path)

    def load_checkpoint(self, path: Path) -> None:
        reject_incompatible_checkpoint(path, self.name())
        suffix = path.suffix.lower()
        if suffix not in {".pth", ".pkl"}:
            raise ValueError(f"Unsupported Detectron2 checkpoint type: {path.name}")

    def infer(
        self,
        request: InferRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        detectron2 = _detectron2()
        torch = _torch()
        self.load_checkpoint(request.checkpoint_path)
        device_info = describe_device()
        self._report(on_progress, 0.05, f"Device: {device_info['device']}")
        spec = get_model(request.model)
        image_paths = _list_images(request.images_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {request.images_dir}")

        cfg = _build_infer_cfg(
            detectron2,
            torch,
            spec.detectron2_config,
            n_classes=len(request.class_names),
            checkpoint_path=request.checkpoint_path,
        )
        predictor = detectron2.engine.DefaultPredictor(cfg)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {"description": "Detectron2 instance-segmentation predictions", "version": "0.1"},
            "images": [],
            "annotations": [],
            "categories": [
                {"id": index, "name": name, "supercategory": name}
                for index, name in enumerate(request.class_names, start=1)
            ],
        }
        next_ann = 1
        for image_id, image_path in enumerate(image_paths, start=1):
            bgr = detectron2.data.detection_utils.read_image(str(image_path), format="BGR")
            height, width = bgr.shape[:2]
            coco["images"].append(
                {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
            )
            outputs = predictor(bgr)["instances"].to("cpu")
            boxes = outputs.pred_boxes.tensor.numpy() if outputs.has("pred_boxes") else []
            classes = outputs.pred_classes.numpy() if outputs.has("pred_classes") else []
            scores = outputs.scores.numpy() if outputs.has("scores") else []
            masks = outputs.pred_masks.numpy() if outputs.has("pred_masks") else []
            for index in range(len(classes)):
                class_index = int(classes[index])
                if class_index < 0 or class_index >= len(request.class_names):
                    continue
                polygon = _mask_to_polygon(masks[index]) if len(masks) > index else None
                if not polygon:
                    continue
                x1, y1, x2, y2 = [float(v) for v in boxes[index]]
                coco["annotations"].append(
                    {
                        "id": next_ann,
                        "image_id": image_id,
                        "category_id": class_index + 1,
                        "segmentation": [polygon],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0,
                        "score": float(scores[index]) if len(scores) > index else 1.0,
                    }
                )
                next_ann += 1
            self._report(
                on_progress,
                0.1 + 0.5 * (image_id / len(image_paths)),
                f"Predicted {image_path.name}",
            )

        coco_path = request.output_dir / PREDICTIONS_NAME
        coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
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

        images = coco.get("images", [])
        for index, image_info in enumerate(images):
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
                draw.polygon(polygon, fill=(*color, 96), outline=(*color, 255))
                mask = Image.new("L", image.size, 0)
                ImageDraw.Draw(mask).polygon(polygon, fill=255)
                stem = Path(image_info["file_name"]).stem
                mask.save(masks_dir / f"{stem}_{instance_index:03d}.png")
            Image.alpha_composite(image, overlay).convert("RGB").save(
                overlay_dir / image_info["file_name"]
            )
            if images:
                self._report(
                    on_progress,
                    0.6 + 0.4 * ((index + 1) / len(images)),
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


def _detectron2():
    try:
        import detectron2
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
        from detectron2.data import detection_utils
        from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed in this environment. Use the CUDA Docker image."
        ) from exc

    class _NS:
        pass

    ns = _NS()
    ns.model_zoo = model_zoo
    ns.get_cfg = get_cfg
    ns.DatasetCatalog = DatasetCatalog
    ns.MetadataCatalog = MetadataCatalog
    ns.data = _NS()
    ns.data.datasets = _NS()
    ns.data.datasets.register_coco_instances = register_coco_instances
    ns.data.detection_utils = detection_utils
    ns.engine = _NS()
    ns.engine.DefaultPredictor = DefaultPredictor
    ns.engine.DefaultTrainer = DefaultTrainer
    ns.engine.HookBase = HookBase
    return ns


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Detectron2 backend.") from exc
    return torch


def _build_train_cfg(
    detectron2,
    torch,
    config_name: str,
    dataset_name: str,
    n_classes: int,
    init: str,
    weights_path: Path | None,
    max_iter: int,
    output_dir: Path,
):
    cfg = detectron2.get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = max(1, max_iter)
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max(1, max_iter)
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = min(100, max(0, cfg.SOLVER.MAX_ITER // 10))
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    cfg.OUTPUT_DIR = str(output_dir)
    if init == "pretrained":
        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError(
                "Pretrained weights were not found. They should be downloaded into the weights/ volume."
            )
        cfg.MODEL.WEIGHTS = str(weights_path)
    else:
        # FrozenBN + random weights diverges (NaN). Use trainable BN from scratch.
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.SOLVER.BASE_LR = 0.0001
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MASK_FORMAT = "polygon"
    return cfg


def _build_infer_cfg(detectron2, torch, config_name: str, n_classes: int, checkpoint_path: Path):
    cfg = detectron2.get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(config_name))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    cfg.MODEL.WEIGHTS = str(checkpoint_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DATALOADER.NUM_WORKERS = 0
    return cfg


class _ProgressTrainer:
    """Wrapper so Detectron2 is imported only when training actually runs."""

    def __init__(self, cfg, on_progress: ProgressCallback | None) -> None:
        detectron2 = _detectron2()

        class Trainer(detectron2.engine.DefaultTrainer):
            def build_hooks(self):
                built = super().build_hooks()
                if on_progress is not None:
                    built.append(_ProgressHook(detectron2.engine.HookBase, on_progress))
                return built

        self._trainer = Trainer(cfg)

    def resume_or_load(self, resume: bool) -> None:
        self._trainer.resume_or_load(resume=resume)

    def train(self) -> None:
        self._trainer.train()


def _ProgressHook(hook_base, on_progress: ProgressCallback):
    class ProgressHook(hook_base):
        def after_step(self):
            current = self.trainer.iter + 1
            total = max(self.trainer.max_iter, 1)
            on_progress(0.12 + 0.85 * (current / total), f"Training iteration {current}/{total}")

    return ProgressHook()


def _unregister(detectron2, name: str) -> None:
    try:
        detectron2.DatasetCatalog.remove(name)
    except KeyError:
        pass
    try:
        detectron2.MetadataCatalog.remove(name)
    except KeyError:
        pass


def _unique_dataset_name(kind: str) -> str:
    return f"aitem_{kind}_{int(time.time() * 1000)}"


def _list_images(images_dir: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _mask_to_polygon(mask) -> list[float] | None:
    try:
        import cv2
    except ImportError:
        return None
    binary = (mask.astype("uint8") * 255) if mask.dtype != "uint8" else mask
    if binary.max() == 1:
        binary = binary * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return [float(v) for v in contour.flatten().tolist()]


def _flat_to_pairs(flat: list[float]) -> list[tuple[float, float]]:
    return [(flat[index], flat[index + 1]) for index in range(0, len(flat), 2)]


def _parse_hex(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return (230, 57, 70)
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
