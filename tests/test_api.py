import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        projects_dir=tmp_path / "projects",
        datasets_dir=tmp_path / "datasets",
        weights_dir=tmp_path / "weights",
    )
    settings.datasets_dir.mkdir()
    ds_all = settings.datasets_dir / "DS-1" / "all"
    ds_all.mkdir(parents=True)
    Image.new("RGB", (32, 32), (90, 90, 90)).save(ds_all / "0_0.png")
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (40, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _wait_job(client: TestClient, job_id: str) -> dict:
    for _ in range(80):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not finish in time.")


def test_create_upload_annotate_train_infer(client: TestClient) -> None:
    created = client.post(
        "/api/projects",
        json={"name": "API Demo", "classes": ["Loop-A", "Loop-B"], "engine": "stub"},
    )
    assert created.status_code == 200
    project_id = created.json()["id"]
    assert project_id == "api-demo"

    upload = client.post(
        f"/api/projects/{project_id}/images",
        files=[("files", ("tile.png", _png_bytes(), "image/png"))],
    )
    assert upload.status_code == 200
    image_id = upload.json()["images"][0]["id"]

    saved = client.put(
        f"/api/projects/{project_id}/annotations",
        json={
            "images": [],
            "categories": [],
            "annotations": [
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [[2, 2, 20, 2, 20, 18, 2, 18]],
                }
            ],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["annotations"][0]["segmentation"][0][0] == 2

    train = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random"},
    )
    assert train.status_code == 200
    train_job = _wait_job(client, train.json()["id"])
    assert train_job["status"] == "completed", train_job.get("error")

    infer = client.post(
        f"/api/projects/{project_id}/jobs/infer",
        json={"overlay_colors": {"Loop-A": "#112233"}},
    )
    assert infer.status_code == 200
    infer_job = _wait_job(client, infer.json()["id"])
    assert infer_job["status"] == "completed", infer_job.get("error")

    run_id = infer_job["id"]
    coco = client.get(f"/api/projects/{project_id}/runs/{run_id}/download/coco")
    assert coco.status_code == 200
    overlays = client.get(f"/api/projects/{project_id}/runs/{run_id}/download/overlays")
    assert overlays.status_code == 200
    assert overlays.headers["content-type"].startswith("application/zip")
    masks = client.get(f"/api/projects/{project_id}/runs/{run_id}/download/masks")
    assert masks.status_code == 200

    preview_name = client.get(f"/api/projects/{project_id}/annotations").json()["images"][0]["file_name"]
    preview = client.get(f"/api/projects/{project_id}/runs/{run_id}/overlays/{preview_name}")
    assert preview.status_code == 200


def test_import_dataset_limit(client: TestClient) -> None:
    created = client.post(
        "/api/projects",
        json={"name": "From DS", "classes": ["Loop-A"]},
    )
    project_id = created.json()["id"]
    imported = client.post(
        f"/api/projects/{project_id}/import-dataset",
        json={"relative_path": "DS-1/all", "limit": 1},
    )
    assert imported.status_code == 200
    assert len(imported.json()["images"]) == 1


def test_delete_project_via_api(client: TestClient) -> None:
    created = client.post(
        "/api/projects",
        json={"name": "Disposable", "classes": ["Loop-A"]},
    )
    project_id = created.json()["id"]
    deleted = client.delete(f"/api/projects/{project_id}")
    assert deleted.status_code == 200
    missing = client.get(f"/api/projects/{project_id}")
    assert missing.status_code == 404


def test_engines_models_and_status(client: TestClient) -> None:
    engines = client.get("/api/engines")
    assert engines.status_code == 200
    names = {item["name"] for item in engines.json()["engines"]}
    assert names == {"stub", "detectron2"}
    models = client.get("/api/models")
    assert models.status_code == 200
    payload = models.json()
    assert payload["default"] == "mask_rcnn_r50_fpn"
    assert payload["models"][0]["id"] == "mask_rcnn_r50_fpn"
    status = client.get("/api/status")
    assert status.status_code == 200
    body = status.json()
    assert "cuda" in body
    assert "device" in body
    assert body["gpu_recommended"] is True
