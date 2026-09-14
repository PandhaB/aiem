from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

from engine.catalog import describe_device, get_model, reject_incompatible_checkpoint
from engine.export import PREDICTIONS_NAME, export_instance_visuals
from engine.training import (
    TrainingStopped,
    checkpoint_filename,
    default_learning_rate,
    format_eta,
)
from engine.types import (
    ExportRequest,
    InferRequest,
    InferResult,
    ProgressCallback,
    TrainRequest,
    TrainResult,
)
from engine.yolo_data import image_hw, write_yolo_seg_dataset, yolo_imgsz

CHECKPOINT_SUFFIX = ".pt"


class UltralyticsEngine:
    """YOLO instance-segmentation backend. UI must not import this module."""

    def name(self) -> str:
        return "ultralytics"

    def train(
        self,
        request: TrainRequest,
        on_progress: ProgressCallback | None = None,
    ) -> TrainResult:
        yolo_mod = _ultralytics()
        torch = _torch()
        device_info = describe_device()
        self._report(
            on_progress,
            0.02,
            f"Device: {device_info['device']}"
            + ("" if device_info["cuda"] else f" — {device_info['message']}"),
        )
        spec = get_model(request.model)
        if spec.engine != self.name():
            raise ValueError(f"Model {spec.id} is not an Ultralytics YOLO model.")
        if not spec.ultralytics_name:
            raise ValueError(f"Model {spec.id} is missing its Ultralytics architecture name.")

        epochs = max(1, request.max_iter or 50)
        imgsz = yolo_imgsz(
            (request.backend_options or {}).get("min_size")
            or (request.backend_options or {}).get("imgsz"),
            image_hw(request.images_dir, request.annotations_path),
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = request.output_dir / "yolo_dataset"
        yaml_path = write_yolo_seg_dataset(
            request.images_dir,
            request.annotations_path,
            dataset_dir,
            request.class_names,
        )
        _write_backend_meta(
            request.output_dir,
            {
                "family": spec.family,
                "imgsz": imgsz,
                "input_size": imgsz,
            },
        )
        self._report(
            on_progress,
            0.08,
            f"YOLO image size: {imgsz} px",
            log_line=f"YOLO image size: {imgsz} px (epochs={epochs})",
        )

        model = self._load_train_model(yolo_mod, spec, request)
        device = 0 if torch.cuda.is_available() else "cpu"
        batch = max(1, request.ims_per_batch or 1)
        lr0 = (
            float(request.learning_rate)
            if request.learning_rate is not None
            else (spec.default_lr or default_learning_rate(request.init))
        )
        patience = int((request.backend_options or {}).get("patience") or 0)
        save_period = request.checkpoint_period or -1
        started = time.monotonic()
        saved: list[Path] = []
        stopped = False
        last_iter = request.start_iter
        log_handler = _install_log_handler(on_progress)

        def on_epoch_end(trainer) -> None:
            nonlocal last_iter, stopped
            epoch_num = int(getattr(trainer, "epoch", 0)) + 1
            iteration = request.start_iter + epoch_num
            last_iter = iteration
            metrics = _epoch_metrics(trainer)
            stopping = request.should_stop is not None and request.should_stop()
            save_now = stopping or (
                request.checkpoint_period is not None and epoch_num % request.checkpoint_period == 0
            )
            named = _copy_last_weights(trainer, request.output_dir, iteration) if save_now else None
            if named is not None:
                saved.append(named)
            elapsed = time.monotonic() - started
            remaining = max(0, epochs - epoch_num)
            eta = (elapsed / epoch_num) * remaining if epoch_num else None
            message = f"Training epoch {iteration}/{request.start_iter + epochs}"
            eta_text = format_eta(eta)
            if eta_text:
                message += f" — ETA {eta_text}"
            self._report(
                on_progress,
                0.1 + 0.85 * (epoch_num / epochs),
                message,
                log_line=(
                    f"epoch: {epoch_num}/{epochs}  total_loss: {metrics.get('total_loss', 0):.4f}"
                    + (f"  saved {named.name}" if named is not None else "")
                ),
                iteration=iteration,
                max_iter=request.start_iter + epochs,
                eta_seconds=eta,
                metrics=metrics,
                checkpoint_path=str(named) if named is not None else None,
            )
            if stopping:
                stopped = True
                trainer.stop = True

        try:
            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                device=device,
                workers=0,
                project=str(request.output_dir),
                name="ultralytics",
                exist_ok=True,
                pretrained=request.init == "pretrained",
                patience=patience,
                save_period=save_period,
                plots=False,
                verbose=True,
                amp=bool(torch.cuda.is_available()),
            )
        except TrainingStopped:
            stopped = True
        finally:
            if log_handler is not None:
                _remove_log_handler(log_handler)

        named_checkpoint = _final_named_checkpoint(request.output_dir, last_iter, saved)
        if named_checkpoint is None or not named_checkpoint.is_file():
            raise FileNotFoundError(
                f"Ultralytics did not write {checkpoint_filename(max(last_iter, 1), CHECKPOINT_SUFFIX)}"
            )
        backend_meta = _read_backend_meta(request.output_dir)
        metrics_path = request.output_dir / "metrics.json"
        payload = {
            "engine": self.name(),
            "model": spec.id,
            "init": request.init,
            "max_iter": epochs,
            "iteration": last_iter,
            "stopped": stopped,
            "learning_rate": lr0,
            "ims_per_batch": batch,
            "checkpoint_period": request.checkpoint_period,
            "device": device_info["device"],
            "cuda": device_info["cuda"],
            "num_classes": len(request.class_names),
            "imgsz": imgsz,
        }
        payload.update(backend_meta)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if stopped:
            self._report(
                on_progress,
                1.0,
                f"Stopped at epoch {last_iter}",
                log_line=f"Stopped at epoch {last_iter}. Saved {named_checkpoint.name}",
                iteration=last_iter,
                max_iter=request.start_iter + epochs,
                checkpoint_path=str(named_checkpoint),
            )
        else:
            self._report(on_progress, 1.0, "Training complete", log_line="Training complete")
        checkpoints = sorted(
            path
            for path in request.output_dir.glob("model_*.pt")
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
        if path.suffix.lower() != ".pt":
            raise ValueError(f"Unsupported Ultralytics checkpoint type: {path.name}")

    def infer(
        self,
        request: InferRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        yolo_mod = _ultralytics()
        self.load_checkpoint(request.checkpoint_path)
        device_info = describe_device()
        self._report(on_progress, 0.05, f"Device: {device_info['device']}")
        image_paths = _list_images(request.images_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {request.images_dir}")
        imgsz = yolo_imgsz(
            (request.backend_options or {}).get("min_size")
            or (request.backend_options or {}).get("imgsz"),
            image_hw(request.images_dir),
        )
        self._report(on_progress, 0.06, f"YOLO image size: {imgsz} px")
        model = yolo_mod.YOLO(str(request.checkpoint_path))
        request.output_dir.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {"description": "Ultralytics YOLO instance-segmentation predictions", "version": "0.1"},
            "images": [],
            "annotations": [],
            "categories": [
                {"id": index, "name": name, "supercategory": name}
                for index, name in enumerate(request.class_names, start=1)
            ],
        }
        next_ann = 1
        stopped = False
        for image_id, image_path in enumerate(image_paths, start=1):
            if request.should_stop is not None and request.should_stop():
                stopped = True
                break
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
            coco["images"].append(
                {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
            )
            results = model.predict(
                source=str(image_path),
                imgsz=imgsz,
                conf=0.25,
                retina_masks=True,
                verbose=False,
            )
            result = results[0] if results else None
            if result is not None and getattr(result, "masks", None) is not None:
                boxes = result.boxes
                for index, polygon_xy in enumerate(result.masks.xy):
                    class_index = int(boxes.cls[index].item()) if boxes is not None else 0
                    if class_index < 0 or class_index >= len(request.class_names):
                        continue
                    polygon = [float(value) for point in polygon_xy for value in point]
                    if len(polygon) < 6:
                        continue
                    if boxes is not None and boxes.xyxy is not None:
                        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[index].tolist()]
                    else:
                        xs, ys = polygon[0::2], polygon[1::2]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    score = float(boxes.conf[index].item()) if boxes is not None and boxes.conf is not None else 1.0
                    coco["annotations"].append(
                        {
                            "id": next_ann,
                            "image_id": image_id,
                            "category_id": class_index + 1,
                            "segmentation": [polygon],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": float((x2 - x1) * (y2 - y1)),
                            "iscrowd": 0,
                            "score": score,
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
        result = self.export_predictions(
            ExportRequest(
                predictions_coco_path=coco_path,
                images_dir=request.images_dir,
                overlay_colors=request.overlay_colors,
                output_dir=request.output_dir,
                class_names=request.class_names,
                should_stop=request.should_stop,
            ),
            on_progress=on_progress,
        )
        result.stopped = stopped or result.stopped
        return result

    def export_predictions(
        self,
        request: ExportRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        return export_instance_visuals(request, on_progress)

    def _load_train_model(self, yolo_mod, spec, request: TrainRequest):
        if request.init in {"pretrained", "checkpoint"}:
            if request.pretrained_weights_path is None or not Path(request.pretrained_weights_path).is_file():
                raise FileNotFoundError(
                    "Weights were not found. Use public pretrained weights, or choose a previous checkpoint."
                )
            return yolo_mod.YOLO(str(request.pretrained_weights_path))
        return yolo_mod.YOLO(f"{spec.ultralytics_name}.yaml")

    @staticmethod
    def _report(on_progress: ProgressCallback | None, value: float, message: str, **extra) -> None:
        if on_progress is not None:
            on_progress(value, message, **extra)


def ultralytics_data_dir() -> Path:
    """Writable Ultralytics cache (AMP probe weights, settings). Not used as training output."""
    root = Path(os.environ.get("AITEM_WEIGHTS_DIR") or "/tmp/aitem-weights")
    dest = root / "ultralytics"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _configure_ultralytics_env() -> Path:
    dest = ultralytics_data_dir()
    config = dest / "config"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    return dest


def _ultralytics():
    dest = _configure_ultralytics_env()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The Ultralytics package is required for the YOLO backend. Rebuild the Docker image."
        ) from exc
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS["sync"] = False
        SETTINGS["weights_dir"] = str(dest)
    except Exception:
        pass

    class _NS:
        pass

    ns = _NS()
    ns.YOLO = YOLO
    return ns


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Ultralytics backend.") from exc
    return torch


def _epoch_metrics(trainer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    tloss = getattr(trainer, "tloss", None)
    values: list[float] = []
    if tloss is not None:
        if hasattr(tloss, "detach"):
            raw = tloss.detach().cpu().flatten().tolist()
        elif isinstance(tloss, (list, tuple)):
            raw = list(tloss)
        else:
            try:
                raw = [float(tloss)]
            except (TypeError, ValueError):
                raw = []
        values = [float(item) for item in raw]
    if values:
        metrics["total_loss"] = float(sum(values))
        if len(values) > 1:
            metrics["loss_mask"] = values[1]
        if len(values) > 2:
            metrics["loss_cls"] = values[2]
    return metrics


def _copy_last_weights(trainer, output_dir: Path, iteration: int) -> Path | None:
    last = getattr(trainer, "last", None)
    path = Path(last) if last else None
    if path is None or not path.is_file():
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            candidate = Path(save_dir) / "weights" / "last.pt"
            path = candidate if candidate.is_file() else None
    if path is None or not path.is_file():
        return None
    dest = output_dir / checkpoint_filename(iteration, CHECKPOINT_SUFFIX)
    if dest.resolve() != path.resolve():
        shutil.copy2(path, dest)
    return dest


def _final_named_checkpoint(output_dir: Path, iteration: int, saved: list[Path]) -> Path | None:
    named = output_dir / checkpoint_filename(max(iteration, 1), CHECKPOINT_SUFFIX)
    if named.is_file():
        return named
    last = output_dir / "ultralytics" / "weights" / "last.pt"
    best = output_dir / "ultralytics" / "weights" / "best.pt"
    source = last if last.is_file() else best
    if source is not None and source.is_file():
        shutil.copy2(source, named)
        return named
    return saved[-1] if saved else None


def _write_backend_meta(output_dir: Path, payload: dict) -> Path:
    path = output_dir / "backend.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _read_backend_meta(folder: Path) -> dict:
    path = Path(folder) / "backend.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _install_log_handler(on_progress: ProgressCallback | None) -> logging.Handler | None:
    if on_progress is None:
        return None
    handler = _ProgressLogHandler(on_progress)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("ultralytics").addHandler(handler)
    logging.getLogger("ultralytics").setLevel(logging.INFO)
    return handler


def _remove_log_handler(handler: logging.Handler) -> None:
    logging.getLogger("ultralytics").removeHandler(handler)


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


def _list_images(images_dir: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
