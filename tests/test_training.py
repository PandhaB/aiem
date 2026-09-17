from types import SimpleNamespace

import pytest

from engine.detectron2 import (
    NATIVE_MAX_SIZE,
    _apply_mask_rcnn_options,
    _backend_meta_payload,
    vitdet_canvas_size,
)
from engine.training import checkpoint_filename, default_learning_rate, format_eta, iteration_from_checkpoint_name


def test_checkpoint_filename_is_zero_padded() -> None:
    assert checkpoint_filename(7, ".pth") == "model_0007.pth"
    assert checkpoint_filename(150, "stub.json") == "model_0150.stub.json"
    assert checkpoint_filename(3, ".pt") == "model_0003.pt"


def test_format_eta() -> None:
    assert format_eta(None) is None
    assert format_eta(0.2) is None
    assert format_eta(9) == "9s"
    assert format_eta(75) == "1m 15s"
    assert format_eta(3661) == "1h 01m"


def test_iteration_from_checkpoint_name() -> None:
    assert iteration_from_checkpoint_name("model_0007.pth") == 7
    assert iteration_from_checkpoint_name("model_0012.stub.json") == 12
    assert iteration_from_checkpoint_name("model_0003.pt") == 3
    assert iteration_from_checkpoint_name("model_final.pth") is None


def test_default_learning_rate() -> None:
    assert default_learning_rate("pretrained") == 0.00025
    assert default_learning_rate("random") == 0.0001


def test_vitdet_canvas_follows_small_tiles() -> None:
    assert vitdet_canvas_size(image_sizes=[(256, 256)]) == 256
    assert vitdet_canvas_size(image_sizes=[(256, 128), (240, 256)]) == 256


def test_vitdet_canvas_rounds_up_to_patch() -> None:
    assert vitdet_canvas_size(min_size=250) == 256
    assert vitdet_canvas_size(image_sizes=[(200, 180)]) == 208


def test_vitdet_canvas_caps_auto_at_1024() -> None:
    assert vitdet_canvas_size() == 1024
    assert vitdet_canvas_size(image_sizes=[(2048, 1024)]) == 1024


def test_vitdet_canvas_explicit_overrides_image_sizes() -> None:
    assert vitdet_canvas_size(min_size=1024, image_sizes=[(256, 256)]) == 1024


def test_vitdet_canvas_rejects_smaller_than_patch() -> None:
    with pytest.raises(ValueError, match="patch size"):
        vitdet_canvas_size(min_size=8)


def test_vitdet_canvas_explicit_overrides_image_sizes() -> None:
    assert vitdet_canvas_size(min_size=1024, image_sizes=[(256, 256)]) == 1024


def _mask_rcnn_cfg():
    return SimpleNamespace(
        MODEL=SimpleNamespace(
            ANCHOR_GENERATOR=SimpleNamespace(
                SIZES=[[32], [64], [128], [256], [512]],
                ASPECT_RATIOS=[[0.5, 1.0, 2.0]],
            ),
            RPN=SimpleNamespace(PRE_NMS_TOPK_TEST=1000, POST_NMS_TOPK_TEST=1000),
        ),
        INPUT=SimpleNamespace(
            MIN_SIZE_TRAIN=(800,),
            MAX_SIZE_TRAIN=1333,
            MIN_SIZE_TEST=800,
            MAX_SIZE_TEST=1333,
        ),
    )


def test_mask_rcnn_empty_options_keep_coco_resize() -> None:
    cfg = _mask_rcnn_cfg()
    _apply_mask_rcnn_options(cfg, {})
    assert cfg.INPUT.MIN_SIZE_TRAIN == (800,)
    assert cfg.INPUT.MIN_SIZE_TEST == 800
    assert cfg.MODEL.RPN.POST_NMS_TOPK_TEST == 1000


def test_mask_rcnn_zero_min_size_is_native_train_and_infer() -> None:
    cfg = _mask_rcnn_cfg()
    _apply_mask_rcnn_options(cfg, {"min_size": 0}, stage="train")
    assert cfg.INPUT.MIN_SIZE_TRAIN == (0,)
    assert cfg.INPUT.MAX_SIZE_TRAIN == NATIVE_MAX_SIZE
    assert cfg.INPUT.MIN_SIZE_TEST == 0
    assert cfg.INPUT.MAX_SIZE_TEST == NATIVE_MAX_SIZE

    infer = _mask_rcnn_cfg()
    _apply_mask_rcnn_options(infer, {"min_size": 0}, stage="infer")
    assert infer.INPUT.MIN_SIZE_TEST == 0
    assert infer.INPUT.MAX_SIZE_TEST == NATIVE_MAX_SIZE
    assert infer.INPUT.MIN_SIZE_TRAIN == (800,)


def test_mask_rcnn_infer_size_can_differ_from_train() -> None:
    cfg = _mask_rcnn_cfg()
    _apply_mask_rcnn_options(
        cfg,
        {"min_size": 0, "min_size_test": 1024, "max_size_test": 2048},
        stage="infer",
    )
    assert cfg.INPUT.MIN_SIZE_TEST == 1024
    assert cfg.INPUT.MAX_SIZE_TEST == 2048


def test_mask_rcnn_anchors_and_rpn_post_nms() -> None:
    cfg = _mask_rcnn_cfg()
    _apply_mask_rcnn_options(
        cfg,
        {
            "anchor_sizes": [8, 16, 32, 64, 128, 256],
            "anchor_aspect_ratios": [0.5, 1, 2],
            "rpn_post_nms_topk_test": 2000,
        },
    )
    assert cfg.MODEL.ANCHOR_GENERATOR.SIZES == [[8], [16], [32], [64], [128], [256]]
    assert cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS == [[0.5, 1.0, 2.0]]
    assert cfg.MODEL.RPN.POST_NMS_TOPK_TEST == 2000
    assert cfg.MODEL.RPN.PRE_NMS_TOPK_TEST == 2000


def test_backend_meta_keeps_native_zero() -> None:
    spec = SimpleNamespace(id="mask_rcnn_x101_fpn", family="mask_rcnn")
    payload = _backend_meta_payload(spec, {"min_size": 0, "rpn_post_nms_topk_test": 2000})
    assert payload["min_size"] == 0
    assert payload["rpn_post_nms_topk_test"] == 2000
    assert payload["model"] == "mask_rcnn_x101_fpn"
