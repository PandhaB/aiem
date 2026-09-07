from __future__ import annotations

from engine.catalog import DEFAULT_ENGINE, detectron2_installed, list_models
from engine.protocol import SegmentationEngine
from engine.stub import StubEngine


def available_engines() -> list[dict[str, object]]:
    d2 = detectron2_installed()
    return [
        {
            "name": "stub",
            "available": True,
            "label": "Stub (fake engine, no GPU)",
        },
        {
            "name": "detectron2",
            "available": d2,
            "label": "Detectron2 (Mask R-CNN)" + ("" if d2 else " — not installed here"),
        },
    ]


def available_models() -> list[dict[str, str]]:
    return [
        {"id": spec.id, "label": spec.label, "engine": spec.engine}
        for spec in list_models()
    ]


def get_engine(name: str) -> SegmentationEngine:
    key = (name or DEFAULT_ENGINE).strip()
    if key == "stub":
        return StubEngine()
    if key == "detectron2":
        from engine.detectron2 import Detectron2Engine

        return Detectron2Engine()
    raise KeyError(f"Unknown engine {key!r}. Known engines: detectron2, stub.")
