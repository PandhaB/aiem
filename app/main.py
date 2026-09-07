from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import Settings, load_settings
from app.routers.api import router as api_router
from app.routers.pages import router as pages_router
from core.jobs import JobRunner
from core.projects import ProjectStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.weights_dir.mkdir(parents=True, exist_ok=True)

    application = FastAPI(
        title="TEM instance segmentation",
        description="Local tool to annotate, train, and run instance segmentation.",
    )
    application.state.settings = settings
    application.state.store = ProjectStore(settings.projects_dir)
    application.state.jobs = JobRunner(application.state.store, settings.weights_dir)

    static_dir = Path(__file__).resolve().parent / "static"
    application.mount("/static", StaticFiles(directory=static_dir), name="static")
    application.include_router(api_router)
    application.include_router(pages_router)
    return application


app = create_app()
