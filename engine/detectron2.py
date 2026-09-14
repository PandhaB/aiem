from __future__ import annotations

import json
import logging
import shutil
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
from engine.training import (
    TrainingStopped,
    as_float_metrics,
    checkpoint_filename,
    default_learning_rate,
    format_eta,
)

CHECKPOINT_SUFFIX = ".pth"
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
        max_iter = request.max_iter or 300
        stopped = False
        last_iter = max_iter
        named_checkpoint: Path | None = None
        log_handler = None
        try:
            cfg = _build_train_cfg(
                detectron2,
                torch,
                spec.detectron2_config,
                dataset_name,
                n_classes=len(request.class_names),
                init=request.init,
                weights_path=request.pretrained_weights_path,
                max_iter=max_iter,
                output_dir=request.output_dir,
                learning_rate=request.learning_rate,
                ims_per_batch=request.ims_per_batch,
            )
            trainer = _ProgressTrainer(
                cfg,
                on_progress,
                should_stop=request.should_stop,
                checkpoint_period=request.checkpoint_period,
            )
            log_handler = _install_log_handler(on_progress)
            trainer.resume_or_load(resume=False)
            self._report(on_progress, 0.12, "Starting Detectron2 training", log_line="Starting Detectron2 training")
            try:
                trainer.train()
            except TrainingStopped as exc:
                stopped = True
                last_iter = exc.iteration
                named_checkpoint = exc.checkpoint_path
        finally:
            if log_handler is not None:
                _remove_log_handler(log_handler)
            _unregister(detectron2, dataset_name)

        if named_checkpoint is None or not named_checkpoint.is_file():
            named_checkpoint = _ensure_named_checkpoint(request.output_dir, last_iter)
        if not named_checkpoint.is_file():
            raise FileNotFoundError(
                f"Detectron2 did not write {checkpoint_filename(last_iter, CHECKPOINT_SUFFIX)}"
            )
        metrics_path = request.output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "engine": self.name(),
                    "model": spec.id,
                    "init": request.init,
                    "max_iter": max_iter,
                    "iteration": last_iter,
                    "stopped": stopped,
                    "learning_rate": request.learning_rate or default_learning_rate(request.init),
                    "ims_per_batch": request.ims_per_batch or 1,
                    "checkpoint_period": request.checkpoint_period,
                    "device": device_info["device"],
                    "cuda": device_info["cuda"],
                    "num_classes": len(request.class_names),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if stopped:
            self._report(
                on_progress,
                1.0,
                f"Stopped at iteration {last_iter}",
                log_line=f"Stopped at iteration {last_iter}. Saved {named_checkpoint.name}",
                iteration=last_iter,
                max_iter=max_iter,
                checkpoint_path=str(named_checkpoint),
            )
        else:
            self._report(on_progress, 1.0, "Training complete", log_line="Training complete")
        checkpoints = sorted(
            path
            for path in request.output_dir.glob("model_*.pth")
            if path.is_file() and path.stem.split("_", 1)[-1].isdigit()
        )
        return TrainResult(
            checkpoint_path=named_checkpoint,
            metrics_path=metrics_path,
            stopped=stopped,
            iteration=last_iter,
            checkpoints=checkpoints,
        )

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
    def _report(on_progress: ProgressCallback | None, value: float, message: str, **extra) -> None:
        if on_progress is not None:
            on_progress(value, message, **extra)


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
    learning_rate: float | None = None,
    ims_per_batch: int | None = None,
):
    cfg = detectron2.get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = max(1, ims_per_batch or 1)
    # PeriodicCheckpointer is disabled here; the progress hook writes model_NNNN.pth.
    cfg.SOLVER.CHECKPOINT_PERIOD = max(max_iter + 1, 10**9)
    cfg.SOLVER.BASE_LR = (
        float(learning_rate) if learning_rate is not None else default_learning_rate(init)
    )
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
        if learning_rate is None:
            cfg.SOLVER.BASE_LR = default_learning_rate("random")
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

    def __init__(
        self,
        cfg,
        on_progress: ProgressCallback | None,
        should_stop=None,
        checkpoint_period: int | None = None,
    ) -> None:
        detectron2 = _detectron2()
        started_at = time.monotonic()

        class Trainer(detectron2.engine.DefaultTrainer):
            def build_hooks(self):
                built = super().build_hooks()
                built.append(
                    _ProgressHook(
                        detectron2.engine.HookBase,
                        on_progress,
                        should_stop,
                        checkpoint_period,
                        started_at,
                    )
                )
                return built

        self._trainer = Trainer(cfg)

    def resume_or_load(self, resume: bool) -> None:
        self._trainer.resume_or_load(resume=resume)

    def train(self) -> None:
        self._trainer.train()


def _ProgressHook(hook_base, on_progress, should_stop, checkpoint_period, started_at):
    class ProgressHook(hook_base):
        def after_step(self):
            current = self.trainer.iter + 1
            total = max(self.trainer.max_iter, 1)
            elapsed = time.monotonic() - started_at
            eta = (elapsed / current) * (total - current) if current >= 2 else None
            eta_text = format_eta(eta)
            message = f"Training iteration {current}/{total}"
            if eta_text:
                message += f" — ETA {eta_text}"
            metrics = _metrics_from_trainer(self.trainer)
            log_line = f"iter: {current}/{total}"
            if "total_loss" in metrics:
                log_line += f"  total_loss: {metrics['total_loss']:.4f}"
            if eta_text:
                log_line += f"  eta: {eta_text}"
            saved = None
            if checkpoint_period and current % checkpoint_period == 0:
                saved = _save_named_checkpoint(self.trainer, current)
                log_line += f"  saved {saved.name}"
            stopped = should_stop is not None and should_stop()
            if stopped and saved is None:
                saved = _save_named_checkpoint(self.trainer, current)
                log_line += f"  saved {saved.name}"
            if on_progress is not None:
                on_progress(
                    0.12 + 0.85 * (current / total),
                    message,
                    log_line=log_line,
                    iteration=current,
                    max_iter=total,
                    eta_seconds=eta,
                    metrics=metrics or None,
                    checkpoint_path=str(saved) if saved else None,
                )
            if stopped:
                raise TrainingStopped(current, saved)

        def after_train(self):
            if should_stop is not None and should_stop():
                return
            current = max(int(getattr(self.trainer, "iter", 0)), self.trainer.max_iter)
            saved = _save_named_checkpoint(self.trainer, current)
            if on_progress is not None:
                on_progress(
                    0.97,
                    f"Training iteration {current}/{max(self.trainer.max_iter, 1)}",
                    log_line=f"Saved {saved.name}",
                    iteration=current,
                    max_iter=max(self.trainer.max_iter, 1),
                    checkpoint_path=str(saved),
                )

    return ProgressHook()


def _metrics_from_trainer(trainer) -> dict[str, float]:
    storage = getattr(trainer, "storage", None)
    if storage is None:
        return {}
    try:
        latest = storage.latest()
    except Exception:
        return {}
    return as_float_metrics(latest)


def _save_named_checkpoint(trainer, iteration: int) -> Path:
    stem = checkpoint_filename(iteration, CHECKPOINT_SUFFIX).removesuffix(CHECKPOINT_SUFFIX)
    trainer.checkpointer.save(stem)
    return Path(trainer.cfg.OUTPUT_DIR) / f"{stem}{CHECKPOINT_SUFFIX}"


def _ensure_named_checkpoint(output_dir: Path, iteration: int) -> Path:
    named = output_dir / checkpoint_filename(iteration, CHECKPOINT_SUFFIX)
    if named.is_file():
        return named
    fallback = output_dir / "model_final.pth"
    if fallback.is_file():
        shutil.copy2(fallback, named)
    return named


def _install_log_handler(on_progress: ProgressCallback | None) -> logging.Handler | None:
    if on_progress is None:
        return None
    handler = _ProgressLogHandler(on_progress)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    for name in ("detectron2", "fvcore"):
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(logging.INFO)
    return handler


def _remove_log_handler(handler: logging.Handler) -> None:
    for name in ("detectron2", "fvcore"):
        logging.getLogger(name).removeHandler(handler)


class _ProgressLogHandler(logging.Handler):
    def __init__(self, on_progress: ProgressCallback) -> None:
        super().__init__()
        self._on_progress = on_progress

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info:
            return
        try:
            line = self.format(record)
        except Exception:
            return
        if not line:
            return
        try:
            self._on_progress(None, None, log_line=line)
        except Exception:
            pass


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
