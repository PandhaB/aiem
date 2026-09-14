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
    projects = request.app.state.store.list_projects()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"projects": projects, "device": describe_device()},
    )


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_page(request: Request, project_id: str):
    try:
        record = request.app.state.store.get(project_id)
    except FileNotFoundError:
        return RedirectResponse("/", status_code=302)
    coco = load_coco(record.annotations_path)
    return templates.TemplateResponse(
        request,
        "project.html",
        {
            "project": record,
            "image_count": len(coco["images"]),
            "annotation_count": len(coco["annotations"]),
            "checkpoints": request.app.state.store.list_checkpoints(project_id),
            "runs": request.app.state.store.list_runs(project_id),
            "device": describe_device(),
        },
    )


@router.get("/projects/{project_id}/annotate", response_class=HTMLResponse)
def annotate_page(request: Request, project_id: str):
    try:
        record = request.app.state.store.get(project_id)
    except FileNotFoundError:
        return RedirectResponse("/", status_code=302)
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
        return RedirectResponse("/", status_code=302)
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
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        request,
        "infer.html",
        {
            "project": record,
            "checkpoints": request.app.state.store.list_checkpoints(project_id),
            "device": describe_device(),
        },
    )
