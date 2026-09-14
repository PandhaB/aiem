from __future__ import annotations

from pathlib import Path
from typing import Protocol

from engine.types import (
    ExportRequest,
    InferRequest,
    InferResult,
    ProgressCallback,
    TrainRequest,
    TrainResult,
)


class SegmentationEngine(Protocol):
    """Stable contract for training, inference, checkpoints, and prediction export.

    UI, Compose, and project folders must depend on this protocol only — never on
    Detectron2 or another library's APIs.
    """

    def name(self) -> str:
        """Registry key, e.g. ``stub`` or ``detectron2``."""

    def train(
        self,
        request: TrainRequest,
        on_progress: ProgressCallback | None = None,
    ) -> TrainResult:
        """Fine-tune or train from random weights. Writes a checkpoint under output_dir.

        Backends that honour ``request.should_stop`` should save a named checkpoint
        and return ``TrainResult(stopped=True)`` instead of raising to the UI.
        """

    def infer(
        self,
        request: InferRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        """Run instance segmentation and write COCO + masks + colourised overlays."""

    def load_checkpoint(self, path: Path) -> None:
        """Validate and load a checkpoint produced by this backend (or compatible)."""

    def export_predictions(
        self,
        request: ExportRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        """Turn a COCO predictions file into mask files and colourised overlay images."""
