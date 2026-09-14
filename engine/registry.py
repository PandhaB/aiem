from __future__ import annotations

from engine.catalog import DEFAULT_ENGINE, MODEL_ZOO_URL, detectron2_installed, list_models
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
            "label": "Detectron2" + ("" if d2 else " — not installed here"),
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


def model_zoo_url() -> str:
    return MODEL_ZOO_URL


def get_engine(name: str) -> SegmentationEngine:
    key = (name or DEFAULT_ENGINE).strip()
    if key == "stub":
        return StubEngine()
    if key == "detectron2":
        from engine.detectron2 import Detectron2Engine

        return Detectron2Engine()
    raise KeyError(f"Unknown engine {key!r}. Known engines: detectron2, stub.")
