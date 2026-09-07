from __future__ import annotations

from engine.detectron2 import Detectron2Engine
from engine.protocol import SegmentationEngine
from engine.stub import StubEngine

_BACKENDS: dict[str, type] = {
    "stub": StubEngine,
    "detectron2": Detectron2Engine,
}


def available_engines() -> list[dict[str, object]]:
    return [
        {"name": "stub", "available": True, "label": "Stub (fake engine for scaffolding)"},
        {
            "name": "detectron2",
            "available": False,
            "label": "Detectron2 (not implemented yet)",
        },
    ]


def get_engine(name: str) -> SegmentationEngine:
    try:
        backend_cls = _BACKENDS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_BACKENDS))
        raise KeyError(f"Unknown engine {name!r}. Known engines: {known}.") from exc
    return backend_cls()
