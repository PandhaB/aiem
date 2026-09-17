"""HTML routes. ``/`` is the homepage; the instance-segmentation list is ``/projects``."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.coco import load_coco
from engine.catalog import describe_device

router = APIRouter()
templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "templates")
)


@router.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Product homepage (intent). Projects live at ``/projects``."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"device": describe_device()},
    )


@router.get("/projects", response_class=HTMLResponse)
def projects_page(request: Request) -> HTMLResponse:
    """Instance-segmentation project list and create form."""
    projects = request.app.state.store.list_projects()
    return templates.TemplateResponse(
        request,
        "projects.html",
        {"projects": projects, "device": describe_device()},
    )


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(request: Request, project_id: str):
    try:
        record = request.app.state.store.get(project_id)
    except FileNotFoundError:
        return RedirectResponse("/projects", status_code=302)
    coco = load_coco(record.annotations_path)
    prediction_runs = request.app.state.store.list_prediction_runs(project_id)
    return templates.TemplateResponse(
        request,
        "project.html",
        {
            "project": record,
            "image_count": len(coco["images"]),
            "annotation_count": len(coco["annotations"]),
            "checkpoints": request.app.state.store.list_checkpoints(project_id),
            "runs": request.app.state.store.list_runs(project_id),
            "prediction_run_ids": {item["id"] for item in prediction_runs},
            "device": describe_device(),
        },
    )


@router.get("/projects/{project_id}/annotate", response_class=HTMLResponse)
def annotate_page(request: Request, project_id: str):
    try:
        record = request.app.state.store.get(project_id)
    except FileNotFoundError:
        return RedirectResponse("/projects", status_code=302)
    return templates.TemplateResponse(
        request,
        "annotate.html",
        {"project": record},
    )


@router.get("/projects/{project_id}/train", response_class=HTMLResponse)
def train_page(request: Request, project_id: str):
    try:
        record = request.app.state.store.get(project_id)
    except FileNotFoundError:
        return RedirectResponse("/projects", status_code=302)
    runs = request.app.state.store.list_runs(project_id)
    train_runs = [
        item
        for item in runs
        if item.get("kind") == "train" and item.get("status") in {"completed", "stopped"}
    ]
    return templates.TemplateResponse(
        request,
        "train.html",
        {
            "project": record,
            "device": describe_device(),
            "train_runs": train_runs,
            "checkpoints": request.app.state.store.list_checkpoints(project_id),
        },
    )


@router.get("/projects/{project_id}/infer", response_class=HTMLResponse)
def infer_page(request: Request, project_id: str):
    try:
        record = request.app.state.store.get(project_id)
    except FileNotFoundError:
        return RedirectResponse("/projects", status_code=302)
    return templates.TemplateResponse(
        request,
        "infer.html",
        {
            "project": record,
            "checkpoints": request.app.state.store.list_checkpoints(project_id),
            "device": describe_device(),
        },
    )


@router.get("/projects/{project_id}/refine", response_class=HTMLResponse)
def refine_page(request: Request, project_id: str, run: str | None = None, file: str | None = None):
    try:
        record = request.app.state.store.get(project_id)
    except FileNotFoundError:
        return RedirectResponse("/projects", status_code=302)
    prediction_runs = request.app.state.store.list_prediction_runs(project_id)
    selected = Path(run).name if run else None
    if selected and not any(item.get("id") == selected for item in prediction_runs):
        selected = None
    if selected is None and prediction_runs:
        selected = prediction_runs[0]["id"]
    prediction_files = (
        request.app.state.store.list_prediction_files(project_id, selected) if selected else []
    )
    selected_file = Path(file).name if file else None
    if selected_file and selected_file not in prediction_files:
        selected_file = None
    if selected_file is None:
        if "predictions_refined.json" in prediction_files:
            selected_file = "predictions_refined.json"
        elif prediction_files:
            selected_file = prediction_files[0]
        else:
            selected_file = "predictions.json"
    return templates.TemplateResponse(
        request,
        "refine.html",
        {
            "project": record,
            "prediction_runs": prediction_runs,
            "selected_run_id": selected,
            "prediction_files": prediction_files,
            "selected_file": selected_file,
        },
    )
