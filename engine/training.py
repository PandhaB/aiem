from __future__ import annotations

from pathlib import Path


METRIC_KEYS = (
    "total_loss",
    "loss_mask",
    "loss_cls",
    "loss_box_reg",
    "loss_rpn_cls",
    "loss_rpn_loc",
)


class TrainingStopped(Exception):
    """Raised by a backend when the user asks to stop a run in progress."""

    def __init__(self, iteration: int, checkpoint_path: Path | None = None) -> None:
        self.iteration = iteration
        self.checkpoint_path = checkpoint_path
        super().__init__(f"Training stopped at iteration {iteration}.")


def checkpoint_filename(iteration: int, suffix: str) -> str:
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"model_{max(0, int(iteration)):04d}{suffix}"


def default_learning_rate(init: str) -> float:
    return 0.0001 if init == "random" else 0.00025


def iteration_from_checkpoint_name(name: str) -> int | None:
    stem = Path(name).name
    if not stem.startswith("model_"):
        return None
    digits = ""
    for char in stem[6:]:
        if char.isdigit():
            digits += char
        else:
            break
    if not digits:
        return None
    return int(digits)


def format_eta(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    remaining = int(max(0, round(seconds)))
    if remaining < 1:
        return None
    if remaining < 60:
        return f"{remaining}s"
    minutes, secs = divmod(remaining, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def as_float_metrics(raw: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key not in raw:
            continue
        item = raw[key]
        if isinstance(item, tuple):
            item = item[0]
        try:
            values[key] = float(item)
        except (TypeError, ValueError):
            continue
    return values
