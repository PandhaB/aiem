from __future__ import annotations

from pathlib import Path

from engine.types import (
    ExportRequest,
    InferRequest,
    InferResult,
    ProgressCallback,
    TrainRequest,
    TrainResult,
)


class Detectron2Engine:
    """First real backend candidate. Not implemented in this scaffold.

    Keep this module free of Detectron2 imports so the Docker image does not
    depend on that library until a later iteration.
    """

    def name(self) -> str:
        return "detectron2"

    def train(
        self,
        request: TrainRequest,
        on_progress: ProgressCallback | None = None,
    ) -> TrainResult:
        raise NotImplementedError(
            "The Detectron2 backend is not implemented yet. Use engine 'stub' for now."
        )

    def infer(
        self,
        request: InferRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        raise NotImplementedError(
            "The Detectron2 backend is not implemented yet. Use engine 'stub' for now."
        )

    def load_checkpoint(self, path: Path) -> None:
        raise NotImplementedError(
            "The Detectron2 backend is not implemented yet. Use engine 'stub' for now."
        )

    def export_predictions(
        self,
        request: ExportRequest,
        on_progress: ProgressCallback | None = None,
    ) -> InferResult:
        raise NotImplementedError(
            "The Detectron2 backend is not implemented yet. Use engine 'stub' for now."
        )
