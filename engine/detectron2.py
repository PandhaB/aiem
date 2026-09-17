from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path

from PIL import Image

from engine.catalog import (
    DEFAULT_MODEL,
    describe_device,
    get_model,
    reject_incompatible_checkpoint,
)
from engine.export import PREDICTIONS_NAME, export_instance_visuals, filter_coco_instances
from engine.infer_size import native_capped_size
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
VITDET_PATCH = 16
VITDET_DEFAULT_CANVAS = 1024
BACKEND_META_NAME = "backend.json"
# Detectron2 ResizeShortestEdge treats 0 as a no-op (native pixels). A huge max
# size is only consulted when min size is not 0.
NATIVE_MAX_SIZE = 99999
_SIDECAR_META_KEYS = {"model", "family", "input_size", "img_size", "square_pad"}


def vitdet_canvas_size(
    min_size: int | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
) -> int:
    """Square input canvas for ViTDet. COCO weights use 1024; small TEM tiles need not be upscaled.

    The ViT still loads 1024-px relative-position tables. ``square_pad`` and the resize
    follow this canvas so a 256 px tile is not padded to 1024 (which blows GPU memory).
    """
    if min_size:
        raw = int(min_size)
        if raw < VITDET_PATCH:
            raise ValueError(f"ViTDet min_size must be at least {VITDET_PATCH} (patch size).")
        return _round_up_to_patch(raw)
    if image_sizes:
        longest = max(max(width, height) for width, height in image_sizes)
        return min(VITDET_DEFAULT_CANVAS, _round_up_to_patch(longest))
    return VITDET_DEFAULT_CANVAS


def _round_up_to_patch(value: int) -> int:
    return max(VITDET_PATCH, math.ceil(value / VITDET_PATCH) * VITDET_PATCH)


class Detectron2Engine:
    """First real backend: Mask R-CNN (YAML zoo) and ViTDet (LazyConfig).

    The UI must not import this module. YAML cards share one code path; ViTDet
    uses ``config_kind="lazy"`` and a native-size canvas capped at 1024 px.
    """

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
        last_iter = request.start_iter + max_iter
        named_checkpoint: Path | None = None
        log_handler = None
        try:
            if spec.config_kind == "lazy":
                trainer = _build_lazy_trainer(
                    torch,
                    spec,
                    dataset_name,
                    n_classes=len(request.class_names),
                    request=request,
                    on_progress=on_progress,
                )
            else:
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
                    backend_options=request.backend_options,
                )
                trainer = _ProgressTrainer(
                    cfg,
                    on_progress,
                    should_stop=request.should_stop,
                    checkpoint_period=request.checkpoint_period,
                    start_iter=request.start_iter,
                )
                _write_backend_meta(
                    request.output_dir,
                    _backend_meta_payload(spec, request.backend_options),
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
        metrics = {
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
        }
        backend_meta = _read_backend_meta(request.output_dir)
        if backend_meta:
            metrics.update(backend_meta)
        metrics_path = request.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
        score_threshold = (
            0.5 if request.score_threshold is None else float(request.score_threshold)
        )
        max_detections = (
            100 if request.max_detections is None else int(request.max_detections)
        )

        if spec.config_kind == "lazy":
            image_sizes = _image_hw(request.images_dir)
            if request.max_image_dimension:
                canvas = native_capped_size(
                    image_sizes,
                    request.max_image_dimension,
                    step=VITDET_PATCH,
                    min_value=VITDET_PATCH,
                )
            else:
                canvas = vitdet_canvas_size(
                    (request.backend_options or {}).get("min_size"),
                    image_sizes,
                )
            self._report(
                on_progress,
                0.06,
                f"ViTDet input canvas: {canvas} px",
                log_line=f"ViTDet input canvas: {canvas} px",
            )
            predictor = _LazyPredictor(
                spec,
                n_classes=len(request.class_names),
                checkpoint_path=request.checkpoint_path,
                torch=_torch(),
                canvas=canvas,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
        else:
            cfg = _build_infer_cfg(
                detectron2,
                torch,
                spec.detectron2_config,
                n_classes=len(request.class_names),
                checkpoint_path=request.checkpoint_path,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
            infer_options = dict(request.backend_options or {})
            if request.max_image_dimension:
                size = native_capped_size(
                    _image_hw(request.images_dir),
                    request.max_image_dimension,
                    step=1,
                    min_value=1,
                )
                infer_options["min_size_test"] = size
                infer_options["max_size_test"] = max(size * 2, size)
            _apply_mask_rcnn_options(cfg, infer_options, stage="infer")
            _ensure_rpn_topk_for_detections(cfg, max_detections)
            resize_note = _mask_rcnn_resize_note(cfg)
            if resize_note:
                self._report(
                    on_progress,
                    0.06,
                    f"Mask R-CNN input size: {resize_note}",
                    log_line=f"Mask R-CNN input size: {resize_note}",
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
        stopped = False
        for image_id, image_path in enumerate(image_paths, start=1):
            if request.should_stop is not None and request.should_stop():
                stopped = True
                break
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

        filter_coco_instances(coco, request.score_threshold, request.max_detections)
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
                smooth_tolerance=request.smooth_tolerance,
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


def _build_lazy_trainer(torch, spec, dataset_name: str, n_classes: int, request: TrainRequest, on_progress):
    """ViTDet and other LazyConfig cards. Same job contract as the YAML Mask R-CNN path."""
    from functools import partial

    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate
    from detectron2.engine import AMPTrainer, HookBase, SimpleTrainer
    from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

    max_iter = request.max_iter or 300
    canvas = vitdet_canvas_size(
        (request.backend_options or {}).get("min_size"),
        _image_hw(request.images_dir, request.annotations_path),
    )
    model_cfg = model_zoo.get_config(spec.detectron2_config).model
    model_cfg.roi_heads.num_classes = n_classes
    model_cfg.backbone.square_pad = canvas
    model = instantiate(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader_cfg = model_zoo.get_config("common/data/coco.py").dataloader
    dataloader_cfg.train.dataset.names = dataset_name
    dataloader_cfg.train.total_batch_size = max(1, request.ims_per_batch or 1)
    dataloader_cfg.train.num_workers = 0
    dataloader_cfg.train.mapper.image_format = "RGB"
    aug0 = dataloader_cfg.train.mapper.augmentations[0]
    aug0.short_edge_length = (canvas,)
    aug0.sample_style = "choice"
    aug0.max_size = canvas
    train_loader = instantiate(dataloader_cfg.train)
    _write_backend_meta(
        request.output_dir,
        {
            "model": spec.id,
            "family": spec.family,
            "input_size": canvas,
            "img_size": VITDET_DEFAULT_CANVAS,
            "square_pad": canvas,
        },
    )
    if on_progress is not None:
        on_progress(
            0.1,
            f"ViTDet input canvas: {canvas} px",
            log_line=(
                f"ViTDet input canvas: {canvas} px "
                f"(native tiles are not upscaled to {VITDET_DEFAULT_CANVAS} unless you set min size)"
            ),
        )

    optimizer_cfg = model_zoo.get_config("common/optim.py").AdamW
    optimizer_cfg.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
    optimizer_cfg.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
    optimizer_cfg.lr = (
        float(request.learning_rate)
        if request.learning_rate is not None
        else (spec.default_lr or default_learning_rate(request.init))
    )
    optimizer_cfg.params.model = model
    optim = instantiate(optimizer_cfg)

    inner = AMPTrainer(model, train_loader, optim) if device == "cuda" else SimpleTrainer(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(model, str(request.output_dir), trainer=inner)
    inner.checkpointer = checkpointer
    inner.cfg = type("Cfg", (), {"OUTPUT_DIR": str(request.output_dir)})()
    if request.init in {"pretrained", "checkpoint"}:
        if request.pretrained_weights_path is None or not Path(request.pretrained_weights_path).is_file():
            raise FileNotFoundError(
                "Weights were not found. Use public pretrained weights, or choose a previous checkpoint."
            )
        checkpointer.load(str(request.pretrained_weights_path))
    inner.register_hooks(
        [
            _ProgressHook(
                HookBase,
                on_progress,
                request.should_stop,
                request.checkpoint_period,
                time.monotonic(),
                request.start_iter,
            )
        ]
    )
    return _LazyProgressTrainer(inner, max_iter)


class _LazyProgressTrainer:
    def __init__(self, trainer, max_iter: int) -> None:
        self._trainer = trainer
        self._max_iter = max_iter

    def resume_or_load(self, resume: bool) -> None:
        return None

    def train(self) -> None:
        self._trainer.train(0, self._max_iter)


class _LazyPredictor:
    """Single-image inference for LazyConfig models (ViTDet). DefaultPredictor is YAML-only."""

    def __init__(
        self,
        spec,
        n_classes: int,
        checkpoint_path: Path,
        torch,
        canvas: int = VITDET_DEFAULT_CANVAS,
        score_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> None:
        from detectron2 import model_zoo
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        import detectron2.data.transforms as T

        model_cfg = model_zoo.get_config(spec.detectron2_config).model
        model_cfg.roi_heads.num_classes = n_classes
        model_cfg.backbone.square_pad = canvas
        model = instantiate(model_cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(str(checkpoint_path))
        box_predictor = getattr(model.roi_heads, "box_predictor", None)
        if box_predictor is not None and hasattr(box_predictor, "test_score_thresh"):
            box_predictor.test_score_thresh = score_threshold
        if box_predictor is not None and hasattr(box_predictor, "test_topk_per_image"):
            box_predictor.test_topk_per_image = max_detections
        self.model = model
        self.torch = torch
        self.aug = T.ResizeShortestEdge(short_edge_length=canvas, max_size=canvas)

    def __call__(self, original_bgr):
        height, width = original_bgr.shape[:2]
        image = original_bgr[:, :, ::-1]
        image = self.aug.get_transform(image).apply_image(image)
        tensor = self.torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        with self.torch.no_grad():
            return self.model([{"image": tensor, "height": height, "width": width}])[0]


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
    backend_options: dict | None = None,
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
    if init in {"pretrained", "checkpoint"}:
        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError(
                "Weights were not found. Use public pretrained weights, or choose a previous checkpoint."
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
    _apply_mask_rcnn_options(cfg, backend_options or {})
    return cfg


def _apply_mask_rcnn_options(cfg, options: dict | None, *, stage: str = "train") -> None:
    """Apply Mask R-CNN YAML knobs. ``min_size`` 0 means native pixels (no resize)."""
    options = options or {}
    sizes = options.get("anchor_sizes")
    if sizes:
        values = [int(item) for item in sizes]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[value] for value in values]
    ratios = options.get("anchor_aspect_ratios")
    if ratios:
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[float(item) for item in ratios]]
    post_nms = options.get("rpn_post_nms_topk_test")
    if post_nms is not None:
        value = int(post_nms)
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = value
        if int(cfg.MODEL.RPN.PRE_NMS_TOPK_TEST) < value:
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = value

    min_size_train = options.get("min_size")
    max_size_train = options.get("max_size")
    min_size_test = options.get("min_size_test")
    max_size_test = options.get("max_size_test")
    if min_size_test is None:
        min_size_test = min_size_train
    if max_size_test is None:
        max_size_test = max_size_train

    if stage == "train":
        _set_resize(cfg, min_size_train, max_size_train, train=True)
        _set_resize(cfg, min_size_test, max_size_test, train=False)
    else:
        _set_resize(cfg, min_size_test, max_size_test, train=False)


def _set_resize(cfg, min_size, max_size, *, train: bool) -> None:
    if min_size is None and max_size is None:
        return
    shortest = None if min_size is None else int(min_size)
    longest = None if max_size is None else int(max_size)
    if longest is None and shortest is not None:
        if shortest == 0:
            longest = NATIVE_MAX_SIZE
        else:
            current = int(cfg.INPUT.MAX_SIZE_TRAIN if train else cfg.INPUT.MAX_SIZE_TEST)
            longest = max(current, shortest * 2)
    if train:
        if shortest is not None:
            cfg.INPUT.MIN_SIZE_TRAIN = (shortest,)
        if longest is not None:
            cfg.INPUT.MAX_SIZE_TRAIN = longest
    else:
        if shortest is not None:
            cfg.INPUT.MIN_SIZE_TEST = shortest
        if longest is not None:
            cfg.INPUT.MAX_SIZE_TEST = longest


def _ensure_rpn_topk_for_detections(cfg, max_detections: int) -> None:
    if int(cfg.MODEL.RPN.PRE_NMS_TOPK_TEST) < max_detections:
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = max_detections
    if int(cfg.MODEL.RPN.POST_NMS_TOPK_TEST) < max_detections:
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = max_detections


def _mask_rcnn_resize_note(cfg) -> str | None:
    shortest = int(cfg.INPUT.MIN_SIZE_TEST)
    if shortest == 0:
        return "native pixels (no resize)"
    return f"{shortest} px"


def _backend_meta_payload(spec, backend_options: dict | None) -> dict:
    payload = {"model": spec.id, "family": spec.family}
    for key, value in (backend_options or {}).items():
        if key in _SIDECAR_META_KEYS or value is None:
            continue
        payload[key] = value
    return payload


def _build_infer_cfg(
    detectron2,
    torch,
    config_name: str,
    n_classes: int,
    checkpoint_path: Path,
    score_threshold: float = 0.5,
    max_detections: int = 100,
):
    cfg = detectron2.get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(config_name))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    cfg.MODEL.WEIGHTS = str(checkpoint_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = max_detections
    _ensure_rpn_topk_for_detections(cfg, max_detections)
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
        start_iter: int = 0,
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
                        start_iter,
                    )
                )
                return built

        self._trainer = Trainer(cfg)

    def resume_or_load(self, resume: bool) -> None:
        self._trainer.resume_or_load(resume=resume)

    def train(self) -> None:
        self._trainer.train()


def _ProgressHook(hook_base, on_progress, should_stop, checkpoint_period, started_at, start_iter=0):
    class ProgressHook(hook_base):
        def after_step(self):
            local = self.trainer.iter + 1
            job_total = max(self.trainer.max_iter, 1)
            current = start_iter + local
            display_total = start_iter + job_total
            elapsed = time.monotonic() - started_at
            eta = (elapsed / local) * (job_total - local) if local >= 2 else None
            eta_text = format_eta(eta)
            message = f"Training iteration {current}/{display_total}"
            if eta_text:
                message += f" — ETA {eta_text}"
            metrics = _metrics_from_trainer(self.trainer)
            log_line = f"iter: {current}/{display_total}"
            if "total_loss" in metrics:
                log_line += f"  total_loss: {metrics['total_loss']:.4f}"
            if eta_text:
                log_line += f"  eta: {eta_text}"
            saved = None
            if checkpoint_period and local % checkpoint_period == 0:
                saved = _save_named_checkpoint(self.trainer, current)
                log_line += f"  saved {saved.name}"
            stopped = should_stop is not None and should_stop()
            if stopped and saved is None:
                saved = _save_named_checkpoint(self.trainer, current)
                log_line += f"  saved {saved.name}"
            if on_progress is not None:
                on_progress(
                    0.12 + 0.85 * (local / job_total),
                    message,
                    log_line=log_line,
                    iteration=current,
                    max_iter=display_total,
                    eta_seconds=eta,
                    metrics=metrics or None,
                    checkpoint_path=str(saved) if saved else None,
                )
            if stopped:
                raise TrainingStopped(current, saved)

        def after_train(self):
            if should_stop is not None and should_stop():
                return
            local = max(int(getattr(self.trainer, "iter", 0)), self.trainer.max_iter)
            current = start_iter + local
            saved = _save_named_checkpoint(self.trainer, current)
            if on_progress is not None:
                on_progress(
                    0.97,
                    f"Training iteration {current}/{start_iter + max(self.trainer.max_iter, 1)}",
                    log_line=f"Saved {saved.name}",
                    iteration=current,
                    max_iter=start_iter + max(self.trainer.max_iter, 1),
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
    output_dir = getattr(getattr(trainer, "cfg", None), "OUTPUT_DIR", None) or trainer.checkpointer.save_dir
    return Path(output_dir) / f"{stem}{CHECKPOINT_SUFFIX}"


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


def _image_hw(images_dir: Path, annotations_path: Path | None = None) -> list[tuple[int, int]]:
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
    for path in _list_images(images_dir):
        with Image.open(path) as image:
            sizes.append(image.size)
    return sizes


def _write_backend_meta(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / BACKEND_META_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _read_backend_meta(folder: Path) -> dict:
    path = Path(folder) / BACKEND_META_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


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
