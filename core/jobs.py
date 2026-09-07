from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.projects import ProjectStore
from engine.registry import get_engine
from engine.types import InferRequest, TrainRequest


class JobError(Exception):
    pass


class JobRunner:
    """In-process jobs. Status is written to disk under the project run folder."""

    def __init__(self, store: ProjectStore, shared_weights_dir: Path) -> None:
        self.store = store
        self.shared_weights_dir = shared_weights_dir
        self.shared_weights_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._busy = False
        self._jobs: dict[str, Path] = {}

    def get(self, job_id: str) -> dict:
        with self._lock:
            path = self._jobs.get(job_id)
        if path is None:
            path = self._find_status(job_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Job not found: {job_id}")
        return _read_json(path)

    def start_train(self, project_id: str, init: str, max_iter: int | None = None) -> dict:
        if init not in {"pretrained", "random"}:
            raise ValueError("init must be 'pretrained' or 'random'.")
        self._ensure_idle()
        record = self.store.get(project_id)
        run_dir = self._new_run_dir(record.runs_dir, "train")
        status = self._write_status(
            run_dir,
            {
                "id": run_dir.name,
                "project_id": project_id,
                "kind": "train",
                "status": "queued",
                "progress": 0.0,
                "message": "Queued",
                "error": None,
                "created_at": _now(),
                "finished_at": None,
                "result": None,
            },
        )
        self._submit(self._run_train, record.id, init, max_iter, run_dir)
        return status

    def start_infer(
        self,
        project_id: str,
        overlay_colors: dict[str, str] | None = None,
        checkpoint_name: str | None = None,
    ) -> dict:
        self._ensure_idle()
        record = self.store.get(project_id)
        colors = {item.name: item.color for item in record.classes}
        if overlay_colors:
            colors.update(overlay_colors)
            self.store.update_classes_colors(project_id, colors)
        run_dir = self._new_run_dir(record.runs_dir, "infer")
        status = self._write_status(
            run_dir,
            {
                "id": run_dir.name,
                "project_id": project_id,
                "kind": "infer",
                "status": "queued",
                "progress": 0.0,
                "message": "Queued",
                "error": None,
                "created_at": _now(),
                "finished_at": None,
                "result": None,
            },
        )
        self._submit(self._run_infer, record.id, colors, checkpoint_name, run_dir)
        return status

    def _ensure_idle(self) -> None:
        with self._lock:
            if self._busy:
                raise JobError("A job is already running. Wait for it to finish.")

    def _submit(self, target, *args) -> None:
        with self._lock:
            if self._busy:
                raise JobError("A job is already running. Wait for it to finish.")
            self._busy = True
            job_id = args[-1].name
            self._jobs[job_id] = args[-1] / "status.json"
        thread = threading.Thread(target=self._guarded, args=(target, *args), daemon=True)
        thread.start()

    def _guarded(self, target, *args) -> None:
        try:
            target(*args)
        finally:
            with self._lock:
                self._busy = False

    def _run_train(
        self,
        project_id: str,
        init: str,
        max_iter: int | None,
        run_dir: Path,
    ) -> None:
        record = self.store.get(project_id)
        status_path = run_dir / "status.json"

        def on_progress(value: float, message: str) -> None:
            current = _read_json(status_path)
            current["status"] = "running"
            current["progress"] = value
            current["message"] = message
            self._write_status(run_dir, current)

        try:
            on_progress(0.01, "Starting training")
            engine = get_engine(record.engine)
            pretrained = None
            if init == "pretrained":
                pretrained = self._optional_shared_weights()
            result = engine.train(
                TrainRequest(
                    images_dir=record.images_dir,
                    annotations_path=record.annotations_path,
                    class_names=[item.name for item in record.classes],
                    init=init,  # type: ignore[arg-type]
                    output_dir=run_dir / "output",
                    pretrained_weights_path=pretrained,
                    max_iter=max_iter,
                ),
                on_progress=on_progress,
            )
            dest = record.weights_dir / result.checkpoint_path.name
            shutil.copy2(result.checkpoint_path, dest)
            current = _read_json(status_path)
            current["status"] = "completed"
            current["progress"] = 1.0
            current["message"] = "Training complete"
            current["finished_at"] = _now()
            current["result"] = {
                "checkpoint": str(dest),
                "run_checkpoint": str(result.checkpoint_path),
                "metrics": str(result.metrics_path),
            }
            self._write_status(run_dir, current)
        except Exception as exc:
            current = _read_json(status_path)
            current["status"] = "failed"
            current["message"] = "Training failed"
            current["error"] = str(exc)
            current["finished_at"] = _now()
            self._write_status(run_dir, current)

    def _run_infer(
        self,
        project_id: str,
        overlay_colors: dict[str, str],
        checkpoint_name: str | None,
        run_dir: Path,
    ) -> None:
        record = self.store.get(project_id)
        status_path = run_dir / "status.json"

        def on_progress(value: float, message: str) -> None:
            current = _read_json(status_path)
            current["status"] = "running"
            current["progress"] = value
            current["message"] = message
            self._write_status(run_dir, current)

        try:
            on_progress(0.01, "Starting inference")
            engine = get_engine(record.engine)
            checkpoint = self._resolve_checkpoint(record.weights_dir, checkpoint_name)
            result = engine.infer(
                InferRequest(
                    images_dir=record.images_dir,
                    checkpoint_path=checkpoint,
                    class_names=[item.name for item in record.classes],
                    overlay_colors=overlay_colors,
                    output_dir=run_dir / "output",
                ),
                on_progress=on_progress,
            )
            current = _read_json(status_path)
            current["status"] = "completed"
            current["progress"] = 1.0
            current["message"] = "Inference complete"
            current["finished_at"] = _now()
            current["result"] = {
                "coco": str(result.coco_path),
                "overlays": str(result.overlay_dir),
                "masks": str(result.masks_dir),
            }
            self._write_status(run_dir, current)
        except Exception as exc:
            current = _read_json(status_path)
            current["status"] = "failed"
            current["message"] = "Inference failed"
            current["error"] = str(exc)
            current["finished_at"] = _now()
            self._write_status(run_dir, current)

    def _resolve_checkpoint(self, weights_dir: Path, checkpoint_name: str | None) -> Path:
        if checkpoint_name:
            path = (weights_dir / Path(checkpoint_name).name).resolve()
            if not str(path).startswith(str(weights_dir.resolve())):
                raise ValueError("Invalid checkpoint path.")
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
            return path
        files = sorted(path for path in weights_dir.iterdir() if path.is_file())
        if not files:
            raise FileNotFoundError("No checkpoint in the project weights folder. Train first.")
        return files[-1]

    def _optional_shared_weights(self) -> Path | None:
        files = sorted(path for path in self.shared_weights_dir.iterdir() if path.is_file())
        return files[-1] if files else None

    def _new_run_dir(self, runs_dir: Path, kind: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = runs_dir / f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_status(self, run_dir: Path, payload: dict) -> dict:
        path = run_dir / "status.json"
        _write_json_atomic(path, payload)
        with self._lock:
            self._jobs[payload["id"]] = path
        return payload

    def _find_status(self, job_id: str) -> Path | None:
        for project_dir in self.store.root.iterdir():
            status = project_dir / "runs" / job_id / "status.json"
            if status.is_file():
                return status
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    last_error: Exception | None = None
    for _ in range(20):
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            last_error = exc
        time.sleep(0.01)
    if last_error:
        raise last_error
    raise json.JSONDecodeError("Expecting value", "", 0)
