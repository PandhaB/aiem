from pathlib import Path

import pytest

from engine.catalog import (
    DEFAULT_MODEL,
    get_model,
    pretrained_path,
    reject_incompatible_checkpoint,
)
from engine.detectron2 import Detectron2Engine
from engine.registry import available_engines, get_engine


def test_default_model_is_r50_fpn() -> None:
    spec = get_model(None)
    assert spec.id == DEFAULT_MODEL
    assert spec.id == "mask_rcnn_r50_fpn"
    assert "mask_rcnn_R_50_FPN_3x.yaml" in spec.detectron2_config
    assert spec.engine == "detectron2"


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        get_model("not-a-model")


def test_pretrained_path_stays_on_weights_volume(tmp_path: Path) -> None:
    path = pretrained_path(tmp_path, "mask_rcnn_r50_fpn")
    assert path.parent.name == "detectron2"
    assert path.name.endswith(".pkl")
    assert tmp_path in path.parents


def test_detectron2_engine_can_be_constructed_without_the_library() -> None:
    engine = get_engine("detectron2")
    assert engine.name() == "detectron2"


def test_detectron2_rejects_stub_checkpoint(tmp_path: Path) -> None:
    stub = tmp_path / "model_final.stub.json"
    stub.write_text('{"engine": "stub"}', encoding="utf-8")
    with pytest.raises(ValueError, match="stub"):
        reject_incompatible_checkpoint(stub, "detectron2")
    engine = Detectron2Engine()
    with pytest.raises(ValueError):
        engine.load_checkpoint(stub)
