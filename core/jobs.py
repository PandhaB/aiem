"""One in-process train/infer job at a time, with status on disk under ``runs/``.

The runner picks a backend with :func:`engine.registry.get_engine`, downloads
public weights into the shared ``weights/`` volume, and copies checkpoints into
the project ``weights/`` folder. It must not import Detectron2 itself.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.coco import load_coco, replace_annotations, save_coco
from core.paths import resolve_under
from core.projects import ProjectStore
from engine.catalog import (
    DEFAULT_MODEL,
    checkpoint_suffixes,
    ensure_pretrained,
    get_model,
    reject_incompatible_checkpoint,
)
from engine.export import PREDICTIONS_NAME, export_instance_visuals
from engine.registry import get_engine
from engine.training import format_eta, iteration_from_checkpoint_name
from engine.types import ExportRequest, InferRequest, TrainRequest


class JobError(Exception):
    pass


class JobRunner:
    """In-process jobs. Status is written to disk under the project run folder."""

    def __init__(self, store: ProjectStore, shared_weights_dir: Path, datasets_dir: Path | None = None) -> None:
        self.store = store
        self.shared_weights_dir = shared_weights_dir
        self.shared_weights_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = datasets_dir
        self._lock = threading.Lock()
        self._busy = False
        self._jobs: dict[str, Path] = {}
        self._stop_events: dict[str, threading.Event] = {}

    def active(self) -> dict | None:
        """The job currently occupying the runner, if any. One job at a time."""
        with self._lock:
            if not self._busy:
                return None
            paths = [
                self._jobs[job_id]
                for job_id in self._stop_events
                if job_id in self._jobs
            ]
        for path in paths:
            if path is None or not path.is_file():
                continue
            payload = _read_json(path)
            if payload.get("status") not in {"queued", "running"}:
                continue
            return {
                "id": payload.get("id"),
                "project_id": payload.get("project_id"),
                "kind": payload.get("kind"),
                "status": payload.get("status"),
                "progress": payload.get("progress"),
                "message": payload.get("message"),
                "iteration": payload.get("iteration"),
                "max_iter": payload.get("max_iter"),
                "eta": payload.get("eta"),
            }
        return None

    def get(self, job_id: str) -> dict:
        with self._lock:
            path = self._jobs.get(job_id)
        if path is None:
            path = self._find_status(job_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Job not found: {job_id}")
        payload = _read_json(path)
        run_dir = path.parent
        payload["history"] = _read_jsonl(run_dir / "history.jsonl")
        payload["log"] = _tail_lines(run_dir / "train.log", 400)
        if payload.get("eta_seconds") is not None and not payload.get("eta"):
            payload["eta"] = format_eta(payload["eta_seconds"])
        return payload

    def start_train(
        self,
        project_id: str,
        init: str,
        max_iter: int | None = None,
        checkpoint_period: int | None = None,
        learning_rate: float | None = None,
        ims_per_batch: int | None = None,
        model: str | None = None,
        resume_run_id: str | None = None,
        resume_checkpoint: str | None = None,
        backend_options: dict | None = None,
    ) -> dict:
        if init not in {"pretrained", "random", "checkpoint"}:
            raise ValueError("init must be 'pretrained', 'random', or 'checkpoint'.")
        if max_iter is not None and max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if checkpoint_period is not None and checkpoint_period < 1:
            raise ValueError("checkpoint_period must be at least 1.")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if ims_per_batch is not None and ims_per_batch < 1:
            raise ValueError("ims_per_batch must be at least 1.")
        self._ensure_idle()
        record = self.store.get(project_id)
        resume = None
        resolved_options = dict(backend_options or {})
        if init == "checkpoint":
            resume = self._resolve_resume(record, resume_run_id, resume_checkpoint)
            model_id = _model_for_checkpoint(record, resume["checkpoint"], resume_run_id)
            resolved_options = {**_backend_options_for_checkpoint(resume["checkpoint"]), **resolved_options}
        else:
            if model:
                record = self.store.update_model(project_id, model)
            model_id = record.model
        run_dir = self._new_run_dir(record.runs_dir, "train")
        start_iter = resume["start_iter"] if resume else 0
        if resume and resume.get("history_path"):
            shutil.copy2(resume["history_path"], run_dir / "history.jsonl")
        resolved_max = max_iter or 300
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
                "iteration": start_iter,
                "max_iter": start_iter + resolved_max,
                "eta_seconds": None,
                "eta": None,
                "checkpoints": [],
                "model": model_id,
            },
        )
        params = {
            "init": init,
            "max_iter": resolved_max,
            "checkpoint_period": checkpoint_period,
            "learning_rate": learning_rate,
            "ims_per_batch": ims_per_batch,
            "start_iter": start_iter,
            "resume_weights": str(resume["checkpoint"]) if resume else None,
            "backend_options": resolved_options,
            "model": model_id,
        }
        self._submit(self._run_train, record.id, params, run_dir)
        return status

    def request_stop(self, job_id: str) -> dict:
        try:
            status = self.get(job_id)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Job not found: {job_id}") from exc
        if status.get("kind") not in {"train", "infer"}:
            raise JobError("Only a running train or infer job can be stopped.")
        if status.get("status") in {"completed", "failed", "stopped"}:
            return status
        with self._lock:
            event = self._stop_events.get(job_id)
        if event is None:
            raise JobError("Only a running train or infer job can be stopped.")
        event.set()
        path = self._jobs.get(job_id) or self._find_status(job_id)
        if path is not None and path.is_file():
            with self._lock:
                current = _read_json(path)
                if current.get("status") in {"queued", "running"}:
                    if status.get("kind") == "infer":
                        current["message"] = "Stopping after this image…"
                    else:
                        current["message"] = "Stopping after this iteration…"
                        _append_line(path.parent / "train.log", "Stop requested")
                    self._write_status_locked(path.parent, current)
        return self.get(job_id)

    def start_infer(
        self,
        project_id: str,
        overlay_colors: dict[str, str] | None = None,
        checkpoint_name: str | None = None,
        source: str = "project",
        relative_path: str | None = None,
        uploaded_files: list[tuple[str, bytes]] | None = None,
        score_threshold: float | None = None,
        max_detections: int | None = None,
        max_image_dimension: int | None = None,
        smooth_tolerance: float | None = None,
    ) -> dict:
        if source not in {"project", "dataset", "upload"}:
            raise ValueError("source must be 'project', 'dataset', or 'upload'.")
        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1.")
        if max_detections is not None and max_detections < 1:
            raise ValueError("max_detections must be at least 1.")
        if max_image_dimension is not None and max_image_dimension < 1:
            raise ValueError("max_image_dimension must be at least 1.")
        if smooth_tolerance is not None and smooth_tolerance <= 0:
            raise ValueError("smooth_tolerance must be greater than 0.")
        self._ensure_idle()
        record = self.store.get(project_id)
        colors = {item.name: item.color for item in record.classes}
        if overlay_colors:
            colors.update(overlay_colors)
            self.store.update_classes_colors(project_id, colors)
        run_dir = self._new_run_dir(record.runs_dir, "infer")
        images_dir = self._resolve_infer_images(record, run_dir, source, relative_path, uploaded_files)
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
                "source": source,
                "relative_path": relative_path,
                "score_threshold": score_threshold,
                "max_detections": max_detections,
                "max_image_dimension": max_image_dimension,
                "smooth_tolerance": smooth_tolerance,
                "overlay_colors": colors,
            },
        )
        self._submit(
            self._run_infer,
            record.id,
            colors,
            checkpoint_name,
            str(images_dir),
            score_threshold,
            max_detections,
            max_image_dimension,
            smooth_tolerance,
            run_dir,
        )
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
            run_dir = args[-1]
            job_id = run_dir.name
            self._jobs[job_id] = run_dir / "status.json"
            self._stop_events[job_id] = threading.Event()
        thread = threading.Thread(target=self._guarded, args=(target, *args), daemon=True)
        thread.start()

    def _guarded(self, target, *args) -> None:
        run_dir = args[-1]
        try:
            target(*args)
        finally:
            with self._lock:
                self._busy = False
                self._stop_events.pop(run_dir.name, None)

    def _run_train(
        self,
        project_id: str,
        params: dict,
        run_dir: Path,
    ) -> None:
        record = self.store.get(project_id)
        status_path = run_dir / "status.json"
        stop_event = self._stop_events.get(run_dir.name) or threading.Event()

        def on_progress(value, message, **extra) -> None:
            log_line = extra.get("log_line")
            if log_line:
                _append_line(run_dir / "train.log", str(log_line))
            metrics = extra.get("metrics")
            if metrics:
                point = {"iter": extra.get("iteration")}
                point.update(metrics)
                _append_jsonl(run_dir / "history.jsonl", point)
            with self._lock:
                current = _read_json(status_path)
                if current.get("status") in {"completed", "failed", "stopped"}:
                    return
                current["status"] = "running"
                if value is not None:
                    current["progress"] = value
                if message:
                    current["message"] = message
                if extra.get("iteration") is not None:
                    current["iteration"] = extra["iteration"]
                if extra.get("max_iter") is not None:
                    current["max_iter"] = extra["max_iter"]
                if extra.get("eta_seconds") is not None:
                    current["eta_seconds"] = extra["eta_seconds"]
                    current["eta"] = format_eta(extra["eta_seconds"])
                checkpoint_path = extra.get("checkpoint_path")
                if checkpoint_path:
                    published = self._publish_checkpoint(record.weights_dir, Path(checkpoint_path))
                    if published is not None:
                        names = list(current.get("checkpoints") or [])
                        if published.name not in names:
                            names.append(published.name)
                        current["checkpoints"] = names
                self._write_status_locked(run_dir, current)

        try:
            on_progress(0.01, "Starting training", log_line="Starting training")
            engine = get_engine(record.engine)
            pretrained = None
            init = params["init"]
            model_id = params.get("model") or record.model or DEFAULT_MODEL
            if init == "checkpoint":
                pretrained = Path(params["resume_weights"]) if params.get("resume_weights") else None
            elif init == "pretrained" and record.engine in {"detectron2", "ultralytics"}:
                # A new engine must be listed here or pretrained weights are not downloaded.
                pretrained = ensure_pretrained(
                    self.shared_weights_dir,
                    model_id,
                    on_progress=on_progress,
                )
            elif init == "pretrained":
                pretrained = self._optional_shared_weights()
            result = engine.train(
                TrainRequest(
                    images_dir=record.images_dir,
                    annotations_path=record.annotations_path,
                    class_names=[item.name for item in record.classes],
                    init=init,
                    output_dir=run_dir / "output",
                    pretrained_weights_path=pretrained,
                    max_iter=params["max_iter"],
                    checkpoint_period=params["checkpoint_period"],
                    learning_rate=params["learning_rate"],
                    ims_per_batch=params["ims_per_batch"],
                    model=model_id,
                    should_stop=stop_event.is_set,
                    start_iter=params.get("start_iter") or 0,
                    backend_options=params.get("backend_options") or {},
                ),
                on_progress=on_progress,
            )
            published = []
            for path in list(result.checkpoints) + [result.checkpoint_path]:
                copied = self._publish_checkpoint(record.weights_dir, path)
                if copied is not None:
                    published.append(copied.name)
            dest = self._publish_checkpoint(record.weights_dir, result.checkpoint_path)
            current = _read_json(status_path)
            current["status"] = "stopped" if result.stopped else "completed"
            current["progress"] = 1.0
            if result.stopped:
                name = dest.name if dest is not None else "no checkpoint"
                current["message"] = f"Stopped at iteration {result.iteration}. Saved {name}."
            else:
                current["message"] = "Training complete"
            current["finished_at"] = _now()
            current["iteration"] = result.iteration
            current["checkpoints"] = list(dict.fromkeys(published))
            current["result"] = {
                "checkpoint": str(dest) if dest is not None else None,
                "run_checkpoint": str(result.checkpoint_path),
                "metrics": str(result.metrics_path),
                "stopped": result.stopped,
                "iteration": result.iteration,
            }
            self._write_status(run_dir, current)
            _append_line(run_dir / "train.log", current["message"])
        except Exception as exc:
            current = _read_json(status_path)
            current["status"] = "failed"
            current["message"] = "Training failed"
            current["error"] = str(exc)
            current["finished_at"] = _now()
            self._write_status(run_dir, current)
            _append_line(run_dir / "train.log", f"Training failed: {exc}")

    def _run_infer(
        self,
        project_id: str,
        overlay_colors: dict[str, str],
        checkpoint_name: str | None,
        images_dir: str,
        score_threshold: float | None,
        max_detections: int | None,
        max_image_dimension: int | None,
        smooth_tolerance: float | None,
        run_dir: Path,
    ) -> None:
        record = self.store.get(project_id)
        status_path = run_dir / "status.json"

        stop_event = self._stop_events.get(run_dir.name) or threading.Event()

        def on_progress(value, message, **extra) -> None:
            current = _read_json(status_path)
            if current.get("status") in {"completed", "failed", "stopped"}:
                return
            current["status"] = "running"
            if value is not None:
                current["progress"] = value
            if message:
                current["message"] = message
            self._write_status(run_dir, current)

        try:
            on_progress(0.01, "Starting inference")
            engine = get_engine(record.engine)
            checkpoint = self._resolve_checkpoint(record, checkpoint_name, record.engine)
            reject_incompatible_checkpoint(checkpoint, record.engine)
            model_id = _model_for_checkpoint(record, checkpoint, checkpoint_name)
            result = engine.infer(
                InferRequest(
                    images_dir=Path(images_dir),
                    checkpoint_path=checkpoint,
                    class_names=[item.name for item in record.classes],
                    overlay_colors=overlay_colors,
                    output_dir=run_dir / "output",
                    model=model_id,
                    should_stop=stop_event.is_set,
                    backend_options=_backend_options_for_checkpoint(checkpoint),
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    max_image_dimension=max_image_dimension,
                    smooth_tolerance=smooth_tolerance,
                ),
                on_progress=on_progress,
            )
            overlays = sorted(
                path.name for path in result.overlay_dir.iterdir() if path.is_file()
            ) if result.overlay_dir.is_dir() else []
            current = _read_json(status_path)
            current["status"] = "stopped" if result.stopped else "completed"
            current["progress"] = 1.0
            current["message"] = "Inference stopped" if result.stopped else "Inference complete"
            current["finished_at"] = _now()
            current["result"] = {
                "coco": str(result.coco_path),
                "overlays": str(result.overlay_dir),
                "masks": str(result.masks_dir),
                "preview": overlays[0] if overlays else None,
                "images": overlays,
                "stopped": result.stopped,
            }
            self._write_status(run_dir, current)
        except Exception as exc:
            current = _read_json(status_path)
            current["status"] = "failed"
            current["message"] = "Inference failed"
            current["error"] = str(exc)
            current["finished_at"] = _now()
            self._write_status(run_dir, current)

    def _publish_checkpoint(self, weights_dir: Path, source: Path | None) -> Path | None:
        if source is None:
            return None
        path = Path(source)
        if not path.is_file():
            return None
        weights_dir.mkdir(parents=True, exist_ok=True)
        dest = weights_dir / path.name
        if dest.resolve() != path.resolve():
            shutil.copy2(path, dest)
        sidecar = path.with_name("backend.json")
        if sidecar.is_file():
            target = dest.with_name(f"{dest.stem}.backend.json")
            if sidecar.resolve() != target.resolve():
                shutil.copy2(sidecar, target)
        return dest

    def _resolve_checkpoint(
        self,
        record,
        checkpoint_name: str | None,
        engine_name: str,
    ) -> Path:
        if not checkpoint_name:
            listed = self.store.list_checkpoints(record.id)
            if not listed:
                raise FileNotFoundError("No checkpoint in the project weights folder. Train first.")
            checkpoint_name = listed[0].get("ref") or listed[0]["name"]
        text = checkpoint_name.strip().strip("/")
        if "/" in text:
            run_id, file_name = text.split("/", 1)
            run_id = Path(run_id).name
            file_name = Path(file_name).name
            run_dir = (record.runs_dir / run_id).resolve()
            if not str(run_dir).startswith(str(record.runs_dir.resolve())) or not run_dir.is_dir():
                raise FileNotFoundError(f"Training run not found: {run_id}")
            output = (run_dir / "output").resolve()
            path = (output / file_name).resolve()
            if not str(path).startswith(str(output)) or not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {file_name} in {run_id}")
            return path
        weights_dir = record.weights_dir
        path = (weights_dir / Path(text).name).resolve()
        if not str(path).startswith(str(weights_dir.resolve())):
            raise ValueError("Invalid checkpoint path.")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
        return path

    def _optional_shared_weights(self) -> Path | None:
        files = sorted(path for path in self.shared_weights_dir.iterdir() if path.is_file())
        return files[-1] if files else None

    def _resolve_infer_images(
        self,
        record,
        run_dir: Path,
        source: str,
        relative_path: str | None,
        uploaded_files: list[tuple[str, bytes]] | None,
    ) -> Path:
        if source == "upload" or uploaded_files:
            target = run_dir / "input_images"
            target.mkdir(parents=True, exist_ok=True)
            if not uploaded_files:
                raise ValueError("Upload at least one image.")
            for filename, data in uploaded_files:
                name = Path(filename).name
                (target / name).write_bytes(data)
            return target
        if source == "dataset":
            if not relative_path:
                raise ValueError("Choose a folder under Datasets.")
            if self.datasets_dir is None:
                raise FileNotFoundError("Datasets folder is not configured.")
            folder = resolve_under(self.datasets_dir, relative_path)
            if not folder.is_dir():
                raise FileNotFoundError(f"Dataset folder not found: {relative_path}")
            return folder
        return record.images_dir

    def infer_images_dir(self, record, run_dir: Path) -> Path:
        status_path = run_dir / "status.json"
        status = _read_json(status_path) if status_path.is_file() else {}
        source = status.get("source") or "project"
        if source == "upload":
            folder = run_dir / "input_images"
        elif source == "dataset":
            relative = status.get("relative_path") or ""
            if self.datasets_dir is None:
                raise FileNotFoundError("Datasets folder is not configured.")
            folder = resolve_under(self.datasets_dir, relative)
        else:
            folder = record.images_dir
        if not folder.is_dir():
            raise FileNotFoundError("Inference images folder not found.")
        return folder

    def infer_original_path(self, record, run_dir: Path, filename: str) -> Path:
        folder = self.infer_images_dir(record, run_dir)
        name = Path(filename).name
        path = (folder / name).resolve()
        if not str(path).startswith(str(folder.resolve())) or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def save_run_predictions(
        self,
        project_id: str,
        run_id: str,
        coco: dict,
        filename: str | None = None,
        rebuild_visuals: bool | None = None,
    ) -> dict:
        """Write refined COCO predictions. Optionally rebuild overlay/mask files.

        Saving to a name other than ``predictions.json`` keeps the model output
        intact. Overlays and Infer downloads still follow ``predictions.json``.
        """
        record = self.store.get(project_id)
        run_dir = _run_dir(record, run_id)
        output_dir = run_dir / "output"
        original_path = output_dir / PREDICTIONS_NAME
        if not original_path.is_file():
            raise FileNotFoundError("COCO predictions not found.")
        name = safe_prediction_filename(filename)
        target_path = output_dir / name
        source_path = target_path if target_path.is_file() else original_path
        existing = load_coco(source_path)
        image_ids = {item["id"] for item in existing["images"]}
        category_ids = {item["id"] for item in existing["categories"]}
        incoming = []
        for annotation in coco.get("annotations") or []:
            if annotation.get("image_id") not in image_ids:
                raise ValueError("Annotation image_id does not match this run.")
            if annotation.get("category_id") not in category_ids:
                raise ValueError("Annotation category_id is not in this run.")
            copy = dict(annotation)
            if copy.get("score") is None:
                copy["score"] = 1.0
            incoming.append(copy)
        payload = {
            "info": existing.get("info") or {},
            "licenses": existing.get("licenses") or [],
            "images": existing["images"],
            "categories": existing["categories"],
            "annotations": incoming,
        }
        replace_annotations(payload, incoming)
        save_coco(target_path, payload)

        should_rebuild = rebuild_visuals if rebuild_visuals is not None else name == PREDICTIONS_NAME
        if should_rebuild:
            status_path = run_dir / "status.json"
            status = _read_json(status_path) if status_path.is_file() else {}
            colors = status.get("overlay_colors") or {item.name: item.color for item in record.classes}
            if not isinstance(colors, dict):
                colors = {item.name: item.color for item in record.classes}
            masks_dir = output_dir / "masks"
            if masks_dir.is_dir():
                for path in masks_dir.iterdir():
                    if path.is_file():
                        path.unlink()
            result = export_instance_visuals(
                ExportRequest(
                    predictions_coco_path=target_path,
                    images_dir=self.infer_images_dir(record, run_dir),
                    overlay_colors=colors,
                    output_dir=output_dir,
                    class_names=[item.name for item in record.classes],
                )
            )
            overlays = (
                sorted(path.name for path in result.overlay_dir.iterdir() if path.is_file())
                if result.overlay_dir.is_dir()
                else []
            )
            if status:
                current = dict(status)
                result_payload = dict(current.get("result") or {})
                result_payload["preview"] = overlays[0] if overlays else None
                result_payload["images"] = overlays
                current["result"] = result_payload
                current["message"] = "Predictions refined"
                self._write_status(run_dir, current)
        return load_coco(target_path)

    def _resolve_resume(self, record, resume_run_id: str | None, resume_checkpoint: str | None) -> dict:
        checkpoint: Path | None = None
        history_path: Path | None = None
        if resume_run_id:
            run_dir = (record.runs_dir / Path(resume_run_id).name).resolve()
            if not str(run_dir).startswith(str(record.runs_dir.resolve())) or not run_dir.is_dir():
                raise FileNotFoundError(f"Training run not found: {resume_run_id}")
            history_candidate = run_dir / "history.jsonl"
            if history_candidate.is_file():
                history_path = history_candidate
            output_dir = run_dir / "output"
            if resume_checkpoint:
                checkpoint = _named_file(output_dir, resume_checkpoint) or _named_file(
                    record.weights_dir, resume_checkpoint
                )
            else:
                checkpoint = _latest_named_checkpoint(output_dir, record.engine)
            if checkpoint is None and resume_checkpoint:
                checkpoint = _named_file(record.weights_dir, resume_checkpoint)
        elif resume_checkpoint:
            checkpoint = _named_file(record.weights_dir, resume_checkpoint)
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError("Choose a previous training run or a checkpoint file.")
        start_iter = 0
        if history_path is not None:
            rows = _read_jsonl(history_path)
            iters = [int(row["iter"]) for row in rows if row.get("iter") is not None]
            if iters:
                start_iter = max(iters)
        named = iteration_from_checkpoint_name(checkpoint.name)
        if named is not None:
            start_iter = max(start_iter, named)
        return {"checkpoint": checkpoint, "history_path": history_path, "start_iter": start_iter}

    def _new_run_dir(self, runs_dir: Path, kind: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = runs_dir / f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_status(self, run_dir: Path, payload: dict) -> dict:
        with self._lock:
            return self._write_status_locked(run_dir, payload)

    def _write_status_locked(self, run_dir: Path, payload: dict) -> dict:
        path = run_dir / "status.json"
        _write_json_atomic(path, payload)
        self._jobs[payload["id"]] = path
        return payload

    def _find_status(self, job_id: str) -> Path | None:
        for project_dir in self.store.root.iterdir():
            status = project_dir / "runs" / job_id / "status.json"
            if status.is_file():
                return status
        return None


def _run_dir(record, run_id: str) -> Path:
    path = (record.runs_dir / Path(run_id).name).resolve()
    if not str(path).startswith(str(record.runs_dir.resolve())) or not path.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return path


_PREDICTION_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.json$")
_RESERVED_PREDICTION_NAMES = {"metrics.json", "backend.json", "status.json"}


def safe_prediction_filename(name: str | None) -> str:
    raw = (name or PREDICTIONS_NAME).strip()
    text = Path(raw).name
    if raw != text or not _PREDICTION_FILE_RE.fullmatch(text) or text in _RESERVED_PREDICTION_NAMES:
        raise ValueError(
            "Prediction file must be a simple .json name (letters, numbers, dash, underscore)."
        )
    return text


def _named_file(folder: Path, name: str) -> Path | None:
    if not folder.is_dir():
        return None
    path = (folder / Path(name).name).resolve()
    if not str(path).startswith(str(folder.resolve())) or not path.is_file():
        return None
    return path


def _latest_named_checkpoint(folder: Path, engine_name: str) -> Path | None:
    if not folder.is_dir():
        return None
    files = sorted(path for path in folder.iterdir() if path.is_file())
    files = [path for path in files if not path.name.endswith(".backend.json")]
    suffixes = checkpoint_suffixes(engine_name)
    if suffixes:
        files = [path for path in files if path.suffix.lower() in suffixes]
    numbered = [path for path in files if iteration_from_checkpoint_name(path.name) is not None]
    if numbered:
        files = sorted(numbered, key=lambda path: iteration_from_checkpoint_name(path.name) or 0)
    return files[-1] if files else None


def _read_checkpoint_sidecar(checkpoint: Path) -> dict:
    sidecar = checkpoint.with_name(f"{checkpoint.stem}.backend.json")
    if not sidecar.is_file():
        sibling = checkpoint.with_name("backend.json")
        sidecar = sibling if sibling.is_file() else sidecar
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


_SIDECAR_META_KEYS = {"model", "family", "input_size", "img_size", "square_pad"}


def _backend_options_for_checkpoint(checkpoint: Path) -> dict:
    """Replay train-time backend knobs at infer. ``min_size`` 0 (native) must not be dropped."""
    data = _read_checkpoint_sidecar(checkpoint)
    options = {
        key: value
        for key, value in data.items()
        if key not in _SIDECAR_META_KEYS and value is not None
    }
    if "min_size" not in options:
        size = data.get("input_size")
        if size:
            options["min_size"] = int(size)
    return options


def _model_for_checkpoint(record, checkpoint: Path, checkpoint_ref: str | None = None) -> str:
    """Architecture card lives on the run/checkpoint, not on the project."""
    data = _read_checkpoint_sidecar(checkpoint)
    named = _valid_engine_model(record.engine, data.get("model"))
    if named:
        return named
    family = data.get("family")
    if family == "vitdet":
        return "mask_rcnn_vitdet_b"
    run_dir = _run_dir_for_checkpoint(record, checkpoint, checkpoint_ref)
    if run_dir is not None:
        from_run = _model_from_run_dir(run_dir)
        named = _valid_engine_model(record.engine, from_run)
        if named:
            return named
    return record.model or DEFAULT_MODEL


def _valid_engine_model(engine_name: str, model_id: str | None) -> str | None:
    if not model_id:
        return None
    try:
        spec = get_model(str(model_id))
    except KeyError:
        return None
    if engine_name != "stub" and spec.engine != engine_name:
        raise ValueError(f"Checkpoint model {spec.id} does not belong to {engine_name}.")
    return spec.id


def _run_dir_for_checkpoint(record, checkpoint: Path, checkpoint_ref: str | None) -> Path | None:
    if checkpoint_ref:
        run_id = Path(checkpoint_ref.strip().strip("/").split("/", 1)[0]).name
        path = record.runs_dir / run_id
        if path.is_dir():
            return path
    try:
        resolved = checkpoint.resolve()
        resolved.relative_to(record.runs_dir.resolve())
    except ValueError:
        return None
    if resolved.parent.name == "output":
        return resolved.parent.parent
    return None


def _model_from_run_dir(run_dir: Path) -> str | None:
    for path in (run_dir / "status.json", run_dir / "output" / "metrics.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("model"):
            return str(payload["model"])
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> dict:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return payload


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


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit < 1:
        return lines
    return lines[-limit:]
