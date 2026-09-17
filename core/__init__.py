"""Project folders, COCO files, and in-process train/infer jobs.

This layer is the source of truth on disk. It must not import Detectron2 or
Ultralytics; it only asks :func:`engine.registry.get_engine` for a backend.
"""

from core.coco import empty_coco, load_coco, save_coco, validate_coco
from core.jobs import JobError, JobRunner
from core.projects import ProjectStore

__all__ = [
    "JobError",
    "JobRunner",
    "ProjectStore",
    "empty_coco",
    "load_coco",
    "save_coco",
    "validate_coco",
]
