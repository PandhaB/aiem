import pytest

from engine.detectron2 import vitdet_canvas_size
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
