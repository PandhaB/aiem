from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


InitMode = Literal["pretrained", "random"]
ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ClassSpec:
    id: int
    name: str
    color: str


@dataclass
class TrainRequest:
    """Backend-agnostic training inputs. Paths are on disk, not library-specific."""

    images_dir: Path
    annotations_path: Path
    class_names: list[str]
    init: InitMode
    output_dir: Path
    pretrained_weights_path: Path | None = None
    max_iter: int | None = None
    model: str = "mask_rcnn_r50_fpn"


@dataclass
class TrainResult:
    checkpoint_path: Path
    metrics_path: Path


@dataclass
class InferRequest:
    images_dir: Path
    checkpoint_path: Path
    class_names: list[str]
    overlay_colors: dict[str, str]
    output_dir: Path
    model: str = "mask_rcnn_r50_fpn"


@dataclass
class InferResult:
    coco_path: Path
    overlay_dir: Path
    masks_dir: Path


@dataclass
class ExportRequest:
    predictions_coco_path: Path
    images_dir: Path
    overlay_colors: dict[str, str]
    output_dir: Path
    class_names: list[str] = field(default_factory=list)
