from types import SimpleNamespace

from engine.ultralytics import (
    _epoch_log_line,
    _epoch_metrics,
    val_table_lines,
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


def test_val_table_lines_uses_box_and_mask_metrics() -> None:
    trainer = SimpleNamespace(
        metrics={
            "metrics/precision(B)": 0.893,
            "metrics/recall(B)": 0.765,
            "metrics/mAP50(B)": 0.84,
            "metrics/mAP50-95(B)": 0.574,
            "metrics/precision(M)": 0.868,
            "metrics/recall(M)": 0.734,
            "metrics/mAP50(M)": 0.8,
            "metrics/mAP50-95(M)": 0.43,
        },
        validator=SimpleNamespace(seen=44, metrics=SimpleNamespace(nt_per_class=[1066])),
    )
    lines = val_table_lines(trainer)
    assert len(lines) == 2
    assert "Class" in lines[0]
    assert "Box" in lines[0]
    assert "Mask" in lines[0]
    assert lines[1].startswith("all")
    assert "44" in lines[1]
    assert "1066" in lines[1]
    assert "0.893" in lines[1]


def test_val_table_lines_empty_without_metrics() -> None:
    assert val_table_lines(SimpleNamespace(metrics={})) == []
