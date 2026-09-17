"""Pluggable instance-segmentation backends.

The rest of the product talks to :class:`SegmentationEngine` only. Detectron2 and
Ultralytics stay behind :func:`get_engine` so the UI never imports them.
"""

from engine.protocol import SegmentationEngine
from engine.registry import available_engines, get_engine
from engine.types import (
    ClassSpec,
    ExportRequest,
    InferRequest,
    InferResult,
    TrainRequest,
    TrainResult,
)

__all__ = [
    "ClassSpec",
    "ExportRequest",
    "InferRequest",
    "InferResult",
    "SegmentationEngine",
    "TrainRequest",
    "TrainResult",
    "available_engines",
    "get_engine",
]
