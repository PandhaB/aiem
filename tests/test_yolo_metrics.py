from pathlib import Path
from types import SimpleNamespace

from engine.ultralytics import (
    _epoch_log_line,
    _epoch_metrics,
    _yolo_train_init,
    yolo_train_metrics,
)


def test_yolo_train_metrics_reads_dict_tloss() -> None:
    metrics = yolo_train_metrics(
        {"box_loss": 0.8, "seg_loss": 1.2, "cls_loss": 0.4, "dfl_loss": 0.9}
    )
    assert metrics["box_loss"] == 0.8
    assert metrics["seg_loss"] == 1.2
    assert metrics["loss_mask"] == 1.2
    assert metrics["loss_cls"] == 0.4
    assert metrics["total_loss"] == 3.3


def test_yolo_train_metrics_reads_named_sequence() -> None:
    metrics = yolo_train_metrics(
        [0.5, 1.5, 0.25],
        loss_names=("box_loss", "seg_loss", "cls_loss"),
    )
    assert metrics["box_loss"] == 0.5
    assert metrics["loss_mask"] == 1.5
    assert metrics["total_loss"] == 2.25


def test_yolo_train_metrics_empty() -> None:
    assert yolo_train_metrics(None) == {}
    assert yolo_train_metrics({}) == {}


def test_epoch_metrics_prefers_tloss_dict() -> None:
    trainer = SimpleNamespace(
        tloss={"box_loss": 1.0, "seg_loss": 2.0},
        loss_names=("box_loss", "seg_loss"),
        metrics={"train/box_loss": 9.0},
    )
    metrics = _epoch_metrics(trainer)
    assert metrics["box_loss"] == 1.0
    assert metrics["total_loss"] == 3.0


def test_epoch_log_line_includes_component_losses() -> None:
    line = _epoch_log_line(
        139,
        200,
        {"box_loss": 0.81, "seg_loss": 1.23, "total_loss": 2.04},
        SimpleNamespace(),
        None,
    )
    assert "epoch: 139/200" in line
    assert "box_loss: 0.8100" in line
    assert "seg_loss: 1.2300" in line
    assert "total_loss: 2.0400" in line


def test_yolo_checkpoint_init_keeps_weights_and_does_not_resume(tmp_path: Path) -> None:
    weights = tmp_path / "model_0050.pt"
    weights.write_bytes(b"ckpt")
    flags = _yolo_train_init("checkpoint", weights)
    assert flags["pretrained"] == str(weights)
    assert flags["resume"] is False


def test_yolo_pretrained_and_random_init_flags() -> None:
    assert _yolo_train_init("pretrained", None) == {"pretrained": True, "resume": False}
    assert _yolo_train_init("random", None) == {"pretrained": False, "resume": False}
