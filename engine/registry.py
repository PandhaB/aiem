"""Resolve an engine id to a :class:`SegmentationEngine` instance.

Real backends are imported only when requested so host pytest can run without
Detectron2 or Ultralytics installed.
"""

from __future__ import annotations

from engine.catalog import (
    DEFAULT_ENGINE,
    detectron2_installed,
    engine_zoo_url,
    list_models,
    ultralytics_installed,
)
from engine.protocol import SegmentationEngine
from engine.stub import StubEngine


def available_engines() -> list[dict[str, object]]:
    d2 = detectron2_installed()
    yolo = ultralytics_installed()
    return [
        {
            "name": "stub",
            "available": True,
            "label": "Stub (fake engine, no GPU)",
        },
        {
            "name": "detectron2",
            "available": d2,
            "label": "Detectron2" + ("" if d2 else " — not installed here"),
        },
        {
            "name": "ultralytics",
            "available": yolo,
            "label": "Ultralytics YOLO" + ("" if yolo else " — not installed here"),
        },
    ]


def available_models(engine: str | None = None) -> list[dict[str, object]]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "engine": spec.engine,
            "family": spec.family,
            "task": spec.task,
            "config_kind": spec.config_kind,
            "default_lr": spec.default_lr,
        }
        for spec in list_models(engine=engine, task="instance")
    ]


def model_zoo_url(engine: str | None = None) -> str:
    return engine_zoo_url(engine)


def get_engine(name: str) -> SegmentationEngine:
    """Return a backend for ``name``. Unknown ids raise KeyError.

    Detectron2 and Ultralytics modules are imported here, not at module load.
    """
    key = (name or DEFAULT_ENGINE).strip()
    if key == "stub":
        return StubEngine()
    if key == "detectron2":
        from engine.detectron2 import Detectron2Engine

        return Detectron2Engine()
    if key == "ultralytics":
        from engine.ultralytics import UltralyticsEngine

        return UltralyticsEngine()
    raise KeyError(f"Unknown engine {key!r}. Known engines: detectron2, ultralytics, stub.")
