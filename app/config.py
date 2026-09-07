from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    projects_dir: Path
    datasets_dir: Path
    weights_dir: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_settings() -> Settings:
    root = repo_root()
    return Settings(
        projects_dir=Path(os.environ.get("AITEM_PROJECTS_DIR", str(root / "projects"))),
        datasets_dir=Path(os.environ.get("AITEM_DATASETS_DIR", str(root / "Datasets"))),
        weights_dir=Path(os.environ.get("AITEM_WEIGHTS_DIR", str(root / "weights"))),
    )
