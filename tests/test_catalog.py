from pathlib import Path

import pytest

from engine.catalog import (
    DEFAULT_MODEL,
    get_model,
    list_models,
    pretrained_path,
    reject_incompatible_checkpoint,
)
from engine.detectron2 import Detectron2Engine
from engine.registry import available_models, get_engine


def test_default_model_is_r50_fpn() -> None:
    spec = get_model(None)
    assert spec.id == DEFAULT_MODEL
    assert spec.id == "mask_rcnn_r50_fpn"
    assert "mask_rcnn_R_50_FPN_3x.yaml" in spec.detectron2_config
    assert spec.engine == "detectron2"


def test_r101_fpn_is_curated() -> None:
    spec = get_model("mask_rcnn_r101_fpn")
    assert spec.engine == "detectron2"
    assert spec.family == "mask_rcnn"
    assert spec.task == "instance"
    assert "mask_rcnn_R_101_FPN_3x.yaml" in spec.detectron2_config


def test_x101_fpn_is_curated() -> None:
    spec = get_model("mask_rcnn_x101_fpn")
    assert spec.engine == "detectron2"
    assert spec.family == "mask_rcnn"
    assert spec.config_kind == "yaml"
    assert "mask_rcnn_X_101_32x8d_FPN_3x.yaml" in spec.detectron2_config


def test_vitdet_b_is_lazy_cfg() -> None:
    spec = get_model("mask_rcnn_vitdet_b")
    assert spec.family == "vitdet"
    assert spec.config_kind == "lazy"
    assert spec.default_lr == 0.0001
    assert "mask_rcnn_vitdet" in spec.detectron2_config
    assert "ViTDet" in spec.checkpoint_url


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        get_model("not-a-model")


def test_pretrained_path_stays_on_weights_volume(tmp_path: Path) -> None:
    path = pretrained_path(tmp_path, "mask_rcnn_r50_fpn")
    assert path.parent.name == "detectron2"
    assert path.name.endswith(".pkl")
    assert tmp_path in path.parents


def test_list_models_keeps_catalogue_order() -> None:
    ids = [spec.id for spec in list_models(engine="detectron2", task="instance")]
    assert ids[0] == "mask_rcnn_r50_fpn"
    assert "mask_rcnn_r101_fpn" in ids
    assert "mask_rcnn_x101_fpn" in ids
    assert "mask_rcnn_vitdet_b" in ids


def test_available_models_are_instance_cards() -> None:
    models = available_models()
    assert {item["id"] for item in models} == {
        "mask_rcnn_r50_fpn",
        "mask_rcnn_r101_fpn",
        "mask_rcnn_x101_fpn",
        "mask_rcnn_vitdet_b",
    }
    vitdet = next(item for item in models if item["id"] == "mask_rcnn_vitdet_b")
    assert vitdet["family"] == "vitdet"
    assert vitdet["config_kind"] == "lazy"
    assert available_models(engine="stub") == []


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
