import pytest

from engine.infer_size import longest_side, native_capped_size


def test_longest_side() -> None:
    assert longest_side(None) is None
    assert longest_side([]) is None
    assert longest_side([(256, 256), (512, 128)]) == 512


def test_native_capped_size_follows_image_until_cap() -> None:
    assert native_capped_size([(256, 256)], 1024, step=16, min_value=16) == 256
    assert native_capped_size([(1024, 1024)], 1024, step=16, min_value=16) == 1024
    assert native_capped_size([(2048, 1024)], 1024, step=16, min_value=16) == 1024


def test_native_capped_size_yolo_stride() -> None:
    assert native_capped_size([(1024, 1024)], 1024, step=32, min_value=32) == 1024
    assert native_capped_size([(200, 180)], 1024, step=32, min_value=32) == 224


def test_native_capped_size_does_not_round_above_cap() -> None:
    assert native_capped_size([(2048, 2048)], 1000, step=32, min_value=32) == 992


def test_native_capped_size_without_images_uses_cap() -> None:
    assert native_capped_size(None, 512, step=16, min_value=16) == 512


def test_native_capped_size_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        native_capped_size([(256, 256)], 16, step=32, min_value=32)
