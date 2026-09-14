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
from engine.catalog import DEFAULT_ENGINE, DEFAULT_MODEL, get_model
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
        if engine_name not in {"stub", "detectron2"}:
            raise ValueError(f"Unknown engine: {engine_name}")
        model_spec = get_model(model)
        if engine_name == "detectron2" and model_spec.engine != "detectron2":
            raise ValueError(f"Model {model_spec.id} does not belong to Detectron2.")

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
        if record.engine == "detectron2" and model_spec.engine != "detectron2":
            raise ValueError(f"Model {model_spec.id} does not belong to Detectron2.")
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
        items = []
        for path in sorted(record.weights_dir.glob("*")):
            if path.is_file():
                items.append({"name": path.name, "path": str(path)})
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
