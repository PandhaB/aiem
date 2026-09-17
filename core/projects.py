"""Self-contained project folders: images, COCO polygons, runs, published weights.

``project.json`` stores the *engine*. The architecture card on a checkpoint lives
in that run's ``backend.json``; ``project.model`` is only the default for a new
pretrained/random training job.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from core.coco import (
    IMAGE_SUFFIXES,
    add_image,
    empty_coco,
    load_coco,
    replace_annotations,
    save_coco,
)
from engine.catalog import (
    DEFAULT_ENGINE,
    DEFAULT_MODEL,
    KNOWN_ENGINES,
    checkpoint_suffixes,
    default_model_for_engine,
    get_model,
)
from engine.training import iteration_from_checkpoint_name
from engine.types import ClassSpec

DEFAULT_COLORS = [
    "#e63946",
    "#2a9d8f",
    "#457b9d",
    "#e9c46a",
    "#9b5de5",
    "#f4a261",
    "#00b4d8",
    "#d62828",
]


@dataclass
class ProjectRecord:
    """One self-contained project folder. ``model`` is only the default card for a new train."""
    id: str
    name: str
    engine: str
    model: str
    classes: list[ClassSpec]
    created_at: str
    root: Path

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def annotations_path(self) -> Path:
        return self.root / "annotations.json"

    @property
    def weights_dir(self) -> Path:
        return self.root / "weights"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "model": self.model,
            "classes": [
                {"id": item.id, "name": item.name, "color": item.color}
                for item in self.classes
            ],
            "created_at": self.created_at,
        }


class ProjectStore:
    """Create and update project folders under ``root`` (Compose: ``/data/projects``)."""
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[ProjectRecord]:
        records = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and (path / "project.json").is_file():
                records.append(self._read(path))
        return records

    def get(self, project_id: str) -> ProjectRecord:
        path = self._project_dir(project_id)
        if not (path / "project.json").is_file():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return self._read(path)

    def create(
        self,
        name: str,
        class_names: list[str],
        engine: str = DEFAULT_ENGINE,
        model: str = DEFAULT_MODEL,
        colors: list[str] | None = None,
    ) -> ProjectRecord:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Project name cannot be empty.")
        names = [item.strip() for item in class_names if item.strip()]
        if not names:
            raise ValueError("A project needs at least one class name.")
        if len(set(name.lower() for name in names)) != len(names):
            raise ValueError("Class names must be unique.")
        engine_name = (engine or DEFAULT_ENGINE).strip()
        if engine_name not in KNOWN_ENGINES:
            raise ValueError(f"Unknown engine: {engine_name}")
        requested = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        model_spec = get_model(requested)
        if engine_name != "stub" and model_spec.engine != engine_name:
            if requested == DEFAULT_MODEL:
                model_spec = get_model(default_model_for_engine(engine_name))
            else:
                raise ValueError(f"Model {model_spec.id} does not belong to {engine_name}.")

        project_id = self._unique_slug(cleaned_name)
        classes = [
            ClassSpec(
                id=index,
                name=label,
                color=(colors[index - 1] if colors and index - 1 < len(colors) else DEFAULT_COLORS[(index - 1) % len(DEFAULT_COLORS)]),
            )
            for index, label in enumerate(names, start=1)
        ]
        record = ProjectRecord(
            id=project_id,
            name=cleaned_name,
            engine=engine_name,
            model=model_spec.id,
            classes=classes,
            created_at=datetime.now(timezone.utc).isoformat(),
            root=self._project_dir(project_id),
        )
        record.images_dir.mkdir(parents=True, exist_ok=True)
        record.weights_dir.mkdir(parents=True, exist_ok=True)
        record.runs_dir.mkdir(parents=True, exist_ok=True)
        (record.root / "project.json").write_text(
            json.dumps(record.to_dict(), indent=2),
            encoding="utf-8",
        )
        save_coco(record.annotations_path, empty_coco(classes, description=cleaned_name))
        return record

    def delete(self, project_id: str) -> None:
        record = self.get(project_id)
        root = record.root.resolve()
        store_root = self.root.resolve()
        if root == store_root or not root.is_relative_to(store_root):
            raise ValueError("Refusing to delete that path.")
        shutil.rmtree(root)

    def update_classes_colors(self, project_id: str, colors: dict[str, str]) -> ProjectRecord:
        record = self.get(project_id)
        updated = []
        for item in record.classes:
            color = colors.get(item.name, item.color)
            updated.append(ClassSpec(id=item.id, name=item.name, color=color))
        record.classes = updated
        payload = record.to_dict()
        (record.root / "project.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return record

    def update_model(self, project_id: str, model: str) -> ProjectRecord:
        record = self.get(project_id)
        model_spec = get_model(model)
        if record.engine != "stub" and model_spec.engine != record.engine:
            raise ValueError(f"Model {model_spec.id} does not belong to {record.engine}.")
        record.model = model_spec.id
        (record.root / "project.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        return record

    def add_image(self, project_id: str, filename: str, data: bytes) -> dict:
        record = self.get(project_id)
        safe_name = Path(filename).name
        if Path(safe_name).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {safe_name}")
        target = record.images_dir / safe_name
        target.write_bytes(data)
        with Image.open(target) as image:
            width, height = image.size
        coco = load_coco(record.annotations_path)
        image_info = add_image(coco, safe_name, width, height)
        save_coco(record.annotations_path, coco)
        return image_info

    def copy_images_from(self, project_id: str, source_dir: Path, limit: int | None = None) -> list[dict]:
        record = self.get(project_id)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {source_dir}")
        imported = []
        paths = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            imported.append(self.add_image(project_id, path.name, path.read_bytes()))
        return imported

    def save_annotations(self, project_id: str, coco: dict) -> dict:
        record = self.get(project_id)
        existing = load_coco(record.annotations_path)
        coco["images"] = existing["images"]
        coco["categories"] = [
            {"id": item.id, "name": item.name, "supercategory": item.name}
            for item in record.classes
        ]
        replace_annotations(coco, coco.get("annotations") or [])
        save_coco(record.annotations_path, coco)
        return load_coco(record.annotations_path)

    def list_runs(self, project_id: str) -> list[dict]:
        record = self.get(project_id)
        runs = []
        if not record.runs_dir.is_dir():
            return runs
        for path in sorted(record.runs_dir.iterdir(), reverse=True):
            status_path = path / "status.json"
            if status_path.is_file():
                runs.append(json.loads(status_path.read_text(encoding="utf-8")))
        return runs

    def list_prediction_runs(self, project_id: str) -> list[dict]:
        """Completed or stopped infer runs that still have a ``predictions.json``."""
        record = self.get(project_id)
        items = []
        for run in self.list_runs(project_id):
            run_id = run.get("id")
            if run.get("kind") != "infer" or not run_id:
                continue
            path = record.runs_dir / Path(str(run_id)).name / "output" / "predictions.json"
            if path.is_file():
                items.append(run)
        return items

    def list_prediction_files(self, project_id: str, run_id: str) -> list[str]:
        record = self.get(project_id)
        run_dir = record.runs_dir / Path(str(run_id)).name
        output = run_dir / "output"
        if not str(run_dir.resolve()).startswith(str(record.runs_dir.resolve())) or not output.is_dir():
            return []
        names = []
        for path in sorted(output.glob("*.json")):
            try:
                load_coco(path)
            except (ValueError, OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            names.append(path.name)
        if "predictions.json" in names:
            names.remove("predictions.json")
            names.insert(0, "predictions.json")
        return names

    def list_images(self, project_id: str) -> list[dict]:
        record = self.get(project_id)
        coco = load_coco(record.annotations_path)
        return coco["images"]

    def image_path(self, project_id: str, filename: str) -> Path:
        record = self.get(project_id)
        path = (record.images_dir / Path(filename).name).resolve()
        if not str(path).startswith(str(record.images_dir.resolve())):
            raise ValueError("Invalid image path.")
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def list_checkpoints(self, project_id: str) -> list[dict]:
        record = self.get(project_id)
        items: list[dict] = []
        seen_run_files: set[str] = set()
        for run in self.list_runs(project_id):
            if run.get("kind") != "train" or run.get("status") not in {"completed", "stopped"}:
                continue
            run_id = run.get("id")
            if not run_id:
                continue
            files = _checkpoint_files(record.runs_dir / run_id / "output", record.engine)
            files.sort(key=lambda path: iteration_from_checkpoint_name(path.name) or 0, reverse=True)
            model = _model_for_run(run, record)
            for path in files:
                seen_run_files.add(path.name)
                step = iteration_from_checkpoint_name(path.name)
                items.append(
                    {
                        "name": path.name,
                        "ref": f"{run_id}/{path.name}",
                        "run_id": run_id,
                        "label": _checkpoint_label(run_id, path.name, step, run.get("status"), model),
                        "iteration": step,
                        "status": run.get("status"),
                        "model": model,
                    }
                )
        for path in _checkpoint_files(record.weights_dir, record.engine):
            if path.name in seen_run_files:
                continue
            step = iteration_from_checkpoint_name(path.name)
            items.append(
                {
                    "name": path.name,
                    "ref": path.name,
                    "run_id": None,
                    "label": f"project weights / {path.name}",
                    "iteration": step,
                    "status": None,
                    "model": record.model,
                }
            )
        return items

    def _read(self, path: Path) -> ProjectRecord:
        payload = json.loads((path / "project.json").read_text(encoding="utf-8"))
        classes = [
            ClassSpec(id=item["id"], name=item["name"], color=item["color"])
            for item in payload["classes"]
        ]
        return ProjectRecord(
            id=payload["id"],
            name=payload["name"],
            engine=payload.get("engine", DEFAULT_ENGINE),
            model=payload.get("model", DEFAULT_MODEL),
            classes=classes,
            created_at=payload["created_at"],
            root=path,
        )

    def _project_dir(self, project_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project_id):
            raise ValueError(f"Invalid project id: {project_id}")
        return self.root / project_id

    def _unique_slug(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"
        candidate = slug
        suffix = 2
        while (self.root / candidate / "project.json").is_file():
            candidate = f"{slug}-{suffix}"
            suffix += 1
        return candidate


def _checkpoint_files(folder: Path, engine: str) -> list[Path]:
    if not folder.is_dir():
        return []
    suffixes = checkpoint_suffixes(engine)
    items = []
    for path in folder.iterdir():
        if not path.is_file() or path.name.endswith(".backend.json"):
            continue
        if iteration_from_checkpoint_name(path.name) is None:
            continue
        if engine == "stub":
            if not path.name.endswith(".stub.json"):
                continue
        elif suffixes and path.suffix.lower() not in suffixes:
            continue
        items.append(path)
    return items


def _checkpoint_label(
    run_id: str,
    filename: str,
    step: int | None,
    status: str | None,
    model: str | None,
) -> str:
    extras = []
    if step is not None:
        extras.append(f"step {step}")
    if model:
        extras.append(model)
    if status:
        extras.append(status)
    extra = ", ".join(extras)
    if extra:
        return f"{run_id} / {filename} ({extra})"
    return f"{run_id} / {filename}"


def _model_for_run(run: dict, record: ProjectRecord) -> str:
    if run.get("model"):
        return run["model"]
    run_id = run.get("id")
    if run_id:
        metrics_path = record.runs_dir / run_id / "output" / "metrics.json"
        if metrics_path.is_file():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if payload.get("model"):
                return payload["model"]
    return record.model
