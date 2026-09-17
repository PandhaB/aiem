"""Backend-agnostic request/result types. Paths on disk, never library objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


InitMode = Literal["pretrained", "random", "checkpoint"]
ProgressCallback = Callable[..., None]
TaskKind = Literal["instance", "semantic"]


@dataclass(frozen=True)
class ClassSpec:
    """User-defined class on a project (COCO category id, name, overlay colour)."""

    id: int
    name: str
    color: str


@dataclass
class TrainRequest:
    """Backend-agnostic training inputs. Paths are on disk, not library-specific.

    ``model`` is a catalogue id. ``max_iter`` means Detectron2 steps or YOLO epochs
    depending on the engine. ``backend_options`` holds extras such as ViTDet canvas.
    """

    images_dir: Path
    annotations_path: Path
    class_names: list[str]
    init: InitMode
    output_dir: Path
    pretrained_weights_path: Path | None = None
    max_iter: int | None = None
    checkpoint_period: int | None = None
    learning_rate: float | None = None
    ims_per_batch: int | None = None
    model: str = "mask_rcnn_r50_fpn"
    should_stop: Callable[[], bool] | None = None
    start_iter: int = 0
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Final (and optional intermediate) weight files written under the run folder."""
    checkpoint_path: Path
    metrics_path: Path
    stopped: bool = False
    iteration: int | None = None
    checkpoints: list[Path] = field(default_factory=list)


@dataclass
class InferRequest:
    """Run instance segmentation on ``images_dir`` with one checkpoint.

    ``model`` and ``backend_options`` should match the checkpoint (see the run's
    ``backend.json``), not necessarily ``project.model``.
    """
    images_dir: Path
    checkpoint_path: Path
    class_names: list[str]
    overlay_colors: dict[str, str]
    output_dir: Path
    model: str = "mask_rcnn_r50_fpn"
    should_stop: Callable[[], bool] | None = None
    backend_options: dict[str, Any] = field(default_factory=dict)
    score_threshold: float | None = None
    max_detections: int | None = None
    max_image_dimension: int | None = None
    smooth_tolerance: float | None = None


@dataclass
class InferResult:
    """COCO predictions plus overlay/mask directories written under ``output_dir``."""
    coco_path: Path
    overlay_dir: Path
    masks_dir: Path
    stopped: bool = False


@dataclass
class ExportRequest:
    """Turn an existing ``predictions.json`` into overlays and mask files."""
    predictions_coco_path: Path
    images_dir: Path
    overlay_colors: dict[str, str]
    output_dir: Path
    class_names: list[str] = field(default_factory=list)
    should_stop: Callable[[], bool] | None = None
    smooth_tolerance: float | None = None
