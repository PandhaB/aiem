from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_MODEL = "mask_rcnn_r50_fpn"
DEFAULT_ENGINE = "detectron2"


@dataclass(frozen=True)
class ModelSpec:
    """Backend-agnostic model id. Detectron2 config paths stay inside this catalogue."""

    id: str
    label: str
    engine: str
    detectron2_config: str
    checkpoint_url: str
    checkpoint_filename: str


MODELS: dict[str, ModelSpec] = {
    "mask_rcnn_r50_fpn": ModelSpec(
        id="mask_rcnn_r50_fpn",
        label="Mask R-CNN R50-FPN",
        engine="detectron2",
        detectron2_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        checkpoint_url=(
            "https://dl.fbaipublicfiles.com/detectron2/"
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/"
            "137849600/model_final_f10217.pkl"
        ),
        checkpoint_filename="mask_rcnn_r50_fpn.pkl",
    ),
}


def list_models() -> list[ModelSpec]:
    return [MODELS[key] for key in sorted(MODELS)]


def get_model(model_id: str | None) -> ModelSpec:
    key = (model_id or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        return MODELS[key]
    except KeyError as exc:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model {key!r}. Known models: {known}.") from exc


def pretrained_path(weights_dir: Path, model_id: str | None = None) -> Path:
    spec = get_model(model_id)
    return Path(weights_dir) / "detectron2" / spec.checkpoint_filename


def ensure_pretrained(
    weights_dir: Path,
    model_id: str | None = None,
    on_progress=None,
) -> Path:
    """Download public COCO weights into the host weights volume if missing."""
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
            else "No GPU visible. Training will use CPU and will be slow. "
            "Check nvidia-container-toolkit if you expected CUDA."
        ),
    }


def reject_incompatible_checkpoint(path: Path, engine_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    suffix = path.suffix.lower()
    if engine_name == "detectron2" and suffix in {".json"}:
        raise ValueError(
            "This file is not a Detectron2 checkpoint (looks like the stub JSON). Train with the Detectron2 engine."
        )
    if engine_name == "stub" and suffix in {".pth", ".pkl"}:
        raise ValueError("This checkpoint belongs to Detectron2. Switch the project engine or train with the stub.")
