"""JSON API used by the browser (create project, save COCO, start train/infer).

These handlers stay engine-agnostic: they pass ``engine`` / ``model`` ids through
to :class:`core.jobs.JobRunner` and :class:`core.projects.ProjectStore`.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.coco import load_coco
from core.jobs import JobError, safe_prediction_filename
from core.paths import list_image_folders
from engine.catalog import (
    DEFAULT_ENGINE,
    DEFAULT_MODEL,
    KNOWN_ENGINES,
    default_model_for_engine,
    describe_device,
    engine_zoo_url,
    get_model,
)
from engine.registry import available_engines, available_models

router = APIRouter(prefix="/api")


class CreateProjectBody(BaseModel):
    name: str
    classes: list[str] = Field(min_length=1)
    engine: str = DEFAULT_ENGINE
    model: str = DEFAULT_MODEL


class SaveAnnotationsBody(BaseModel):
    images: list[dict] = Field(default_factory=list)
    annotations: list[dict] = Field(default_factory=list)
    categories: list[dict] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)
    licenses: list = Field(default_factory=list)


class TrainBody(BaseModel):
    init: str = "pretrained"
    max_iter: int | None = Field(default=None, ge=1)
    checkpoint_period: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    ims_per_batch: int | None = Field(default=None, ge=1)
    model: str | None = None
    resume_run_id: str | None = None
    resume_checkpoint: str | None = None
    backend_options: dict = Field(default_factory=dict)


class InferBody(BaseModel):
    overlay_colors: dict[str, str] = Field(default_factory=dict)
    checkpoint_name: str | None = None
    source: str = "project"
    relative_path: str | None = None
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    max_detections: int | None = Field(default=None, ge=1)
    max_image_dimension: int | None = Field(default=None, ge=1)
    smooth_tolerance: float | None = Field(default=None, gt=0)


class ImportDatasetBody(BaseModel):
    relative_path: str = "DS-1/all"
    limit: int | None = 8


def _store(request: Request):
    return request.app.state.store


def _jobs(request: Request):
    return request.app.state.jobs


def _settings(request: Request):
    return request.app.state.settings


def _project_or_404(request: Request, project_id: str):
    try:
        return _store(request).get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/engines")
def list_engines() -> dict:
    return {"engines": available_engines()}


@router.get("/models")
def list_models(engine: str | None = None) -> dict:
    return {
        "models": available_models(engine=engine),
        "default": default_model_for_engine(engine or DEFAULT_ENGINE),
        "zoo_url": engine_zoo_url(engine),
    }


@router.get("/datasets")
def list_datasets(request: Request) -> dict:
    return {"datasets": list_image_folders(_settings(request).datasets_dir)}


@router.get("/status")
def runtime_status() -> dict:
    return describe_device()


@router.get("/projects")
def list_projects(request: Request) -> dict:
    projects = [_project_payload(request, item) for item in _store(request).list_projects()]
    return {"projects": projects}


@router.post("/projects")
def create_project(request: Request, body: CreateProjectBody) -> dict:
    if body.engine not in KNOWN_ENGINES:
        raise HTTPException(status_code=400, detail="Unknown engine.")
    try:
        get_model(body.model)
        record = _store(request).create(
            body.name,
            body.classes,
            engine=body.engine,
            model=body.model,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _project_payload(request, record)


@router.delete("/projects/{project_id}")
def delete_project(request: Request, project_id: str) -> dict:
    _project_or_404(request, project_id)
    try:
        _store(request).delete(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "id": project_id}


@router.get("/projects/{project_id}")
def get_project(request: Request, project_id: str) -> dict:
    record = _project_or_404(request, project_id)
    return _project_payload(request, record)


@router.post("/projects/{project_id}/images")
async def upload_images(
    request: Request,
    project_id: str,
    files: list[UploadFile] = File(...),
) -> dict:
    _project_or_404(request, project_id)
    imported = []
    try:
        for upload in files:
            data = await upload.read()
            imported.append(_store(request).add_image(project_id, upload.filename or "image.png", data))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"images": imported}


@router.post("/projects/{project_id}/import-dataset")
def import_dataset(request: Request, project_id: str, body: ImportDatasetBody) -> dict:
    _project_or_404(request, project_id)
    datasets_dir: Path = _settings(request).datasets_dir
    relative = Path(body.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(status_code=400, detail="Invalid dataset path.")
    source = (datasets_dir / relative).resolve()
    if not str(source).startswith(str(datasets_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid dataset path.")
    try:
        imported = _store(request).copy_images_from(project_id, source, limit=body.limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"images": imported, "source": str(relative)}


@router.get("/projects/{project_id}/images/{filename}")
def get_image(request: Request, project_id: str, filename: str) -> FileResponse:
    _project_or_404(request, project_id)
    try:
        path = _store(request).image_path(project_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path)


@router.get("/projects/{project_id}/annotations")
def get_annotations(request: Request, project_id: str) -> dict:
    record = _project_or_404(request, project_id)
    return load_coco(record.annotations_path)


@router.put("/projects/{project_id}/annotations")
def put_annotations(request: Request, project_id: str, body: SaveAnnotationsBody) -> dict:
    _project_or_404(request, project_id)
    try:
        return _store(request).save_annotations(project_id, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/projects/{project_id}/jobs/train")
def start_train(request: Request, project_id: str, body: TrainBody) -> dict:
    _project_or_404(request, project_id)
    try:
        return _jobs(request).start_train(
            project_id,
            init=body.init,
            max_iter=body.max_iter,
            checkpoint_period=body.checkpoint_period,
            learning_rate=body.learning_rate,
            ims_per_batch=body.ims_per_batch,
            model=body.model,
            resume_run_id=body.resume_run_id,
            resume_checkpoint=body.resume_checkpoint,
            backend_options=body.backend_options,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/projects/{project_id}/jobs/infer")
def start_infer(request: Request, project_id: str, body: InferBody) -> dict:
    _project_or_404(request, project_id)
    try:
        return _jobs(request).start_infer(
            project_id,
            overlay_colors=body.overlay_colors,
            checkpoint_name=body.checkpoint_name,
            source=body.source,
            relative_path=body.relative_path,
            score_threshold=body.score_threshold,
            max_detections=body.max_detections,
            max_image_dimension=body.max_image_dimension,
            smooth_tolerance=body.smooth_tolerance,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/projects/{project_id}/jobs/infer-upload")
async def start_infer_upload(
    request: Request,
    project_id: str,
    files: list[UploadFile] = File(...),
    checkpoint_name: str | None = Form(None),
    overlay_colors: str = Form("{}"),
    score_threshold: str | None = Form(None),
    max_detections: str | None = Form(None),
    max_image_dimension: str | None = Form(None),
    smooth_tolerance: str | None = Form(None),
) -> dict:
    _project_or_404(request, project_id)
    try:
        colors = json.loads(overlay_colors) if overlay_colors else {}
        if not isinstance(colors, dict):
            raise ValueError("overlay_colors must be a JSON object.")
        uploaded = [(upload.filename or "image.png", await upload.read()) for upload in files]
        return _jobs(request).start_infer(
            project_id,
            overlay_colors=colors,
            checkpoint_name=checkpoint_name,
            source="upload",
            uploaded_files=uploaded,
            score_threshold=_optional_form_float(score_threshold),
            max_detections=_optional_form_int(max_detections),
            max_image_dimension=_optional_form_int(max_image_dimension),
            smooth_tolerance=_optional_form_float(smooth_tolerance),
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/active")
def active_job(request: Request) -> dict:
    return {"job": _jobs(request).active()}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    try:
        return _jobs(request).get(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/stop")
def stop_job(request: Request, job_id: str) -> dict:
    try:
        return _jobs(request).request_stop(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/projects/{project_id}/runs/{run_id}/predictions")
def get_predictions(request: Request, project_id: str, run_id: str, file: str | None = None) -> dict:
    record = _project_or_404(request, project_id)
    run_dir = _safe_run_dir(record.root / "runs", run_id)
    try:
        name = safe_prediction_filename(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = run_dir / "output" / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="COCO predictions not found.")
    try:
        return load_coco(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/projects/{project_id}/runs/{run_id}/predictions")
def put_predictions(
    request: Request,
    project_id: str,
    run_id: str,
    body: SaveAnnotationsBody,
    file: str | None = None,
    rebuild_visuals: bool | None = None,
) -> dict:
    _project_or_404(request, project_id)
    try:
        return _jobs(request).save_run_predictions(
            project_id,
            run_id,
            body.model_dump(),
            filename=file,
            rebuild_visuals=rebuild_visuals,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/projects/{project_id}/runs/{run_id}/download/{kind}")
def download_run(request: Request, project_id: str, run_id: str, kind: str):
    record = _project_or_404(request, project_id)
    run_dir = _safe_run_dir(record.root / "runs", run_id)
    output = run_dir / "output"
    if kind == "coco":
        path = output / "predictions.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="COCO predictions not found.")
        return FileResponse(path, filename="predictions.json")
    if kind in {"overlays", "masks"}:
        folder = output / kind
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail=f"{kind} folder not found.")
        buffer = _zip_directory(folder)
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{kind}.zip"'},
        )
    raise HTTPException(status_code=400, detail="kind must be coco, overlays, or masks.")


@router.get("/projects/{project_id}/runs/{run_id}/overlays/{filename}")
def get_overlay(request: Request, project_id: str, run_id: str, filename: str) -> FileResponse:
    record = _project_or_404(request, project_id)
    run_dir = _safe_run_dir(record.root / "runs", run_id)
    path = (run_dir / "output" / "overlays" / Path(filename).name).resolve()
    overlay_root = (run_dir / "output" / "overlays").resolve()
    if not str(path).startswith(str(overlay_root)) or not path.is_file():
        raise HTTPException(status_code=404, detail="Overlay not found.")
    return FileResponse(path)


@router.get("/projects/{project_id}/runs/{run_id}/inputs/{filename}")
def get_run_input(request: Request, project_id: str, run_id: str, filename: str) -> FileResponse:
    record = _project_or_404(request, project_id)
    run_dir = _safe_run_dir(record.root / "runs", run_id)
    try:
        path = _jobs(request).infer_original_path(record, run_dir, filename)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Original image not found.") from exc
    return FileResponse(path)


def _project_payload(request: Request, record) -> dict:
    coco = load_coco(record.annotations_path)
    payload = record.to_dict()
    payload["image_count"] = len(coco["images"])
    payload["annotation_count"] = len(coco["annotations"])
    payload["checkpoints"] = _store(request).list_checkpoints(record.id)
    payload["runs"] = _store(request).list_runs(record.id)
    return payload


def _safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    path = (runs_dir / Path(run_id).name).resolve()
    if not str(path).startswith(str(runs_dir.resolve())) or not path.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    return path


def _zip_directory(directory: Path) -> BytesIO:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    buffer.seek(0)
    return buffer


def _optional_form_float(raw: str | None) -> float | None:
    if raw is None or not str(raw).strip():
        return None
    return float(raw)


def _optional_form_int(raw: str | None) -> int | None:
    if raw is None or not str(raw).strip():
        return None
    return int(raw)
