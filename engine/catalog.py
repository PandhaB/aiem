from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_MODEL = "mask_rcnn_r50_fpn"
DEFAULT_ENGINE = "detectron2"
DEFAULT_YOLO_MODEL = "yolov8s-seg"
MODEL_ZOO_URL = "https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md"
ULTRALYTICS_DOCS_URL = "https://docs.ultralytics.com/tasks/segment/"
KNOWN_ENGINES = ("stub", "detectron2", "ultralytics")
ENGINE_ZOO_URLS = {
    "detectron2": MODEL_ZOO_URL,
    "ultralytics": ULTRALYTICS_DOCS_URL,
}
DEFAULT_MODELS = {
    "detectron2": DEFAULT_MODEL,
    "ultralytics": DEFAULT_YOLO_MODEL,
    "stub": DEFAULT_MODEL,
}
CHECKPOINT_SUFFIXES = {
    "detectron2": {".pth", ".pkl"},
    "ultralytics": {".pt"},
    "stub": {".json"},
}


@dataclass(frozen=True)
class ModelSpec:
    """Backend-agnostic model id. Detectron2 config paths stay inside this catalogue."""

    id: str
    label: str
    engine: str
    family: str
    task: str
    detectron2_config: str = ""
    checkpoint_url: str = ""
    checkpoint_filename: str = ""
    config_kind: str = "yaml"
    default_lr: float | None = None
    ultralytics_name: str = ""


MODELS: dict[str, ModelSpec] = {
    "mask_rcnn_r50_fpn": ModelSpec(
        id="mask_rcnn_r50_fpn",
        label="Mask R-CNN R50-FPN",
        engine="detectron2",
        family="mask_rcnn",
        task="instance",
        detectron2_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        checkpoint_url=(
            "https://dl.fbaipublicfiles.com/detectron2/"
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/"
            "137849600/model_final_f10217.pkl"
        ),
        checkpoint_filename="mask_rcnn_r50_fpn.pkl",
    ),
    "mask_rcnn_r101_fpn": ModelSpec(
        id="mask_rcnn_r101_fpn",
        label="Mask R-CNN R101-FPN",
        engine="detectron2",
        family="mask_rcnn",
        task="instance",
        detectron2_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        checkpoint_url=(
            "https://dl.fbaipublicfiles.com/detectron2/"
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/"
            "138363263/model_final_a3ec54.pkl"
        ),
        checkpoint_filename="mask_rcnn_r101_fpn.pkl",
    ),
    "mask_rcnn_x101_fpn": ModelSpec(
        id="mask_rcnn_x101_fpn",
        label="Mask R-CNN X101-FPN",
        engine="detectron2",
        family="mask_rcnn",
        task="instance",
        detectron2_config="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        checkpoint_url=(
            "https://dl.fbaipublicfiles.com/detectron2/"
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/"
            "139653917/model_final_2d9806.pkl"
        ),
        checkpoint_filename="mask_rcnn_x101_fpn.pkl",
    ),
    "mask_rcnn_vitdet_b": ModelSpec(
        id="mask_rcnn_vitdet_b",
        label="ViTDet Mask R-CNN ViT-B",
        engine="detectron2",
        family="vitdet",
        task="instance",
        detectron2_config="common/models/mask_rcnn_vitdet.py",
        checkpoint_url=(
            "https://dl.fbaipublicfiles.com/detectron2/"
            "ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl"
        ),
        checkpoint_filename="mask_rcnn_vitdet_b.pkl",
        config_kind="lazy",
        default_lr=0.0001,
    ),
    "yolov8n-seg": ModelSpec(
        id="yolov8n-seg",
        label="YOLOv8n-seg",
        engine="ultralytics",
        family="yolo",
        task="instance",
        checkpoint_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
        checkpoint_filename="yolov8n-seg.pt",
        default_lr=0.01,
        ultralytics_name="yolov8n-seg",
    ),
    "yolov8s-seg": ModelSpec(
        id="yolov8s-seg",
        label="YOLOv8s-seg",
        engine="ultralytics",
        family="yolo",
        task="instance",
        checkpoint_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt",
        checkpoint_filename="yolov8s-seg.pt",
        default_lr=0.01,
        ultralytics_name="yolov8s-seg",
    ),
    "yolo11n-seg": ModelSpec(
        id="yolo11n-seg",
        label="YOLO11n-seg",
        engine="ultralytics",
        family="yolo",
        task="instance",
        checkpoint_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        checkpoint_filename="yolo11n-seg.pt",
        default_lr=0.01,
        ultralytics_name="yolo11n-seg",
    ),
    "yolo11s-seg": ModelSpec(
        id="yolo11s-seg",
        label="YOLO11s-seg",
        engine="ultralytics",
        family="yolo",
        task="instance",
        checkpoint_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
        checkpoint_filename="yolo11s-seg.pt",
        default_lr=0.01,
        ultralytics_name="yolo11s-seg",
    ),
    "yolo26x-seg": ModelSpec(
        id="yolo26x-seg",
        label="YOLO26x-seg",
        engine="ultralytics",
        family="yolo",
        task="instance",
        checkpoint_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt",
        checkpoint_filename="yolo26x-seg.pt",
        default_lr=0.01,
        ultralytics_name="yolo26x-seg",
    ),
}


def list_models(engine: str | None = None, task: str | None = None) -> list[ModelSpec]:
    items = list(MODELS.values())
    if engine:
        items = [item for item in items if item.engine == engine]
    if task:
        items = [item for item in items if item.task == task]
    return items


def default_model_for_engine(engine: str | None) -> str:
    return DEFAULT_MODELS.get((engine or DEFAULT_ENGINE).strip(), DEFAULT_MODEL)


def engine_zoo_url(engine: str | None) -> str:
    if not engine:
        return MODEL_ZOO_URL
    return ENGINE_ZOO_URLS.get(engine.strip(), MODEL_ZOO_URL)


def checkpoint_suffixes(engine_name: str) -> set[str]:
    return CHECKPOINT_SUFFIXES.get(engine_name, set())


def get_model(model_id: str | None) -> ModelSpec:
    key = (model_id or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        return MODELS[key]
    except KeyError as exc:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model {key!r}. Known models: {known}.") from exc


def pretrained_path(weights_dir: Path, model_id: str | None = None) -> Path:
    spec = get_model(model_id)
    return Path(weights_dir) / spec.engine / spec.checkpoint_filename


def ensure_pretrained(
    weights_dir: Path,
    model_id: str | None = None,
    on_progress=None,
) -> Path:
    """Download public pretrained weights into the host weights volume if missing."""
    spec = get_model(model_id)
    dest = pretrained_path(weights_dir, spec.id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        if on_progress:
            on_progress(0.08, f"Using cached weights {dest.name}")
        return dest
    if on_progress:
        on_progress(0.04, f"Downloading {spec.label} weights (not stored in the image)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(spec.checkpoint_url, tmp)
    tmp.replace(dest)
    if on_progress:
        on_progress(0.1, f"Saved weights to {dest.name}")
    return dest


def detectron2_installed() -> bool:
    try:
        import detectron2  # noqa: F401
    except ImportError:
        return False
    return True


def describe_device() -> dict[str, object]:
    cuda = False
    label = "cpu"
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            label = f"cuda ({torch.cuda.get_device_name(0)})"
    except ImportError:
        pass
    return {
        "device": label,
        "cuda": cuda,
        "gpu_recommended": True,
        "message": (
            None
            if cuda
            else "No GPU visible to this process. Host nvidia-smi is not enough: "
            "the Compose container must see the GPU (NVIDIA Container Toolkit + runtime: nvidia)."
        ),
    }


def ultralytics_installed() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def reject_incompatible_checkpoint(path: Path, engine_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    suffix = path.suffix.lower()
    allowed = checkpoint_suffixes(engine_name)
    if allowed and suffix not in allowed:
        if engine_name == "ultralytics":
            raise ValueError(
                "This file is not an Ultralytics YOLO checkpoint (.pt). Train with the Ultralytics engine."
            )
        if engine_name == "detectron2":
            raise ValueError(
                "This file is not a Detectron2 checkpoint. Train with the Detectron2 engine."
            )
        if engine_name == "stub":
            raise ValueError(
                "This checkpoint does not belong to the stub engine. Switch the project engine."
            )
        raise ValueError(f"Unsupported checkpoint type for {engine_name}: {path.name}")
