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
    for _ in range(200):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"completed", "failed", "stopped"}:
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
        json={"init": "random", "max_iter": 4, "checkpoint_period": 2},
    )
    assert train.status_code == 200
    train_job = _wait_job(client, train.json()["id"])
    assert train_job["status"] == "completed", train_job.get("error")
    assert train_job["log"]
    assert train_job["history"]
    assert train_job["history"][0]["total_loss"] > 0
    weights = client.get(f"/api/projects/{project_id}").json()["checkpoints"]
    names = {item["name"] for item in weights}
    assert "model_0002.stub.json" in names
    assert "model_0004.stub.json" in names
    assert all(item["ref"].startswith(train_job["id"] + "/") for item in weights)
    assert all(train_job["id"] in item["label"] for item in weights)

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
    original = client.get(f"/api/projects/{project_id}/runs/{run_id}/inputs/{preview_name}")
    assert original.status_code == 200
    assert preview_name in infer_job["result"]["images"]


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
    assert names == {"stub", "detectron2", "ultralytics"}
    models = client.get("/api/models")
    assert models.status_code == 200
    payload = models.json()
    assert payload["default"] == "mask_rcnn_r50_fpn"
    assert payload["zoo_url"].endswith("MODEL_ZOO.md")
    ids = [item["id"] for item in payload["models"]]
    assert ids[0] == "mask_rcnn_r50_fpn"
    assert "mask_rcnn_r101_fpn" in ids
    assert "mask_rcnn_x101_fpn" in ids
    assert "mask_rcnn_vitdet_b" in ids
    assert payload["models"][0]["family"] == "mask_rcnn"
    yolo = client.get("/api/models", params={"engine": "ultralytics"})
    assert yolo.status_code == 200
    yolo_payload = yolo.json()
    assert yolo_payload["default"] == "yolov8s-seg"
    assert "segment" in yolo_payload["zoo_url"]
    yolo_ids = [item["id"] for item in yolo_payload["models"]]
    assert yolo_ids == ["yolov8n-seg", "yolov8s-seg", "yolo11n-seg", "yolo11s-seg", "yolo26x-seg"]
    created_yolo = client.post(
        "/api/projects",
        json={"name": "YOLO Demo", "classes": ["Loop-A"], "engine": "ultralytics"},
    )
    assert created_yolo.status_code == 200
    assert created_yolo.json()["engine"] == "ultralytics"
    assert created_yolo.json()["model"] == "yolov8s-seg"
    infer_page = client.get(f"/projects/{created_yolo.json()['id']}/infer")
    assert infer_page.status_code == 200
    assert 'id="preview-original"' in infer_page.text
    assert 'id="preview-overlay"' in infer_page.text
    stub_models = client.get("/api/models", params={"engine": "stub"})
    assert stub_models.json()["models"] == []
    datasets = client.get("/api/datasets")
    assert datasets.status_code == 200
    paths = {item["relative_path"] for item in datasets.json()["datasets"]}
    assert "DS-1/all" in paths
    status = client.get("/api/status")
    assert status.status_code == 200
    body = status.json()
    assert "cuda" in body
    assert "device" in body
    assert body["gpu_recommended"] is True
    home = client.get("/")
    assert home.status_code == 200
    assert 'id="active-job-banner"' in home.text
    assert 'value="ultralytics"' in home.text
    idle = client.get("/api/jobs/active")
    assert idle.status_code == 200
    assert idle.json() == {"job": None}


def test_active_job_is_listed_until_it_finishes(client: TestClient) -> None:
    project_id = _stub_project_with_image(client, "Active Banner")
    train = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 80},
    )
    assert train.status_code == 200
    job_id = train.json()["id"]
    active = client.get("/api/jobs/active")
    assert active.status_code == 200
    job = active.json()["job"]
    assert job is not None
    assert job["id"] == job_id
    assert job["project_id"] == project_id
    assert job["kind"] == "train"
    assert job["status"] in {"queued", "running"}
    stopped = client.post(f"/api/jobs/{job_id}/stop")
    assert stopped.status_code == 200
    finished = _wait_job(client, job_id)
    assert finished["status"] == "stopped", finished.get("error")
    idle = client.get("/api/jobs/active")
    assert idle.status_code == 200
    assert idle.json()["job"] is None


def test_stop_training_saves_named_checkpoint(client: TestClient) -> None:
    created = client.post(
        "/api/projects",
        json={"name": "Stop Me", "classes": ["Loop-A"], "engine": "stub"},
    )
    project_id = created.json()["id"]
    client.post(
        f"/api/projects/{project_id}/images",
        files=[("files", ("tile.png", _png_bytes(), "image/png"))],
    )
    train = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 80},
    )
    assert train.status_code == 200
    job_id = train.json()["id"]
    stopped = client.post(f"/api/jobs/{job_id}/stop")
    assert stopped.status_code == 200
    job = _wait_job(client, job_id)
    assert job["status"] == "stopped", job.get("error")
    assert job["iteration"] >= 1
    assert job["iteration"] < 80
    weights = client.get(f"/api/projects/{project_id}").json()["checkpoints"]
    assert any(item["name"].startswith("model_") for item in weights)


def test_stop_inference(client: TestClient) -> None:
    project_id = _stub_project_with_image(client, "Stop Infer")
    for index in range(7):
        client.post(
            f"/api/projects/{project_id}/images",
            files=[("files", (f"tile_{index}.png", _png_bytes(), "image/png"))],
        )
    train = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 2},
    )
    train_job = _wait_job(client, train.json()["id"])
    assert train_job["status"] == "completed", train_job.get("error")
    infer = client.post(f"/api/projects/{project_id}/jobs/infer", json={})
    assert infer.status_code == 200
    job_id = infer.json()["id"]
    stopped = client.post(f"/api/jobs/{job_id}/stop")
    assert stopped.status_code == 200
    job = _wait_job(client, job_id)
    assert job["status"] == "stopped", job.get("error")


def test_stop_unknown_job(client: TestClient) -> None:
    response = client.post("/api/jobs/missing-job/stop")
    assert response.status_code == 404


def _stub_project_with_image(client: TestClient, name: str) -> str:
    created = client.post(
        "/api/projects",
        json={"name": name, "classes": ["Loop-A"], "engine": "stub"},
    )
    project_id = created.json()["id"]
    client.post(
        f"/api/projects/{project_id}/images",
        files=[("files", ("tile.png", _png_bytes(), "image/png"))],
    )
    return project_id


def test_resume_training_copies_history_into_new_run(client: TestClient) -> None:
    project_id = _stub_project_with_image(client, "Resume History")
    first = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 4},
    )
    assert first.status_code == 200
    first_job = _wait_job(client, first.json()["id"])
    assert first_job["status"] == "completed", first_job.get("error")
    first_history = list(first_job["history"])
    assert len(first_history) == 4
    first_iters = [row["iter"] for row in first_history]

    second = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={
            "init": "checkpoint",
            "max_iter": 2,
            "resume_run_id": first_job["id"],
        },
    )
    assert second.status_code == 200
    second_job = _wait_job(client, second.json()["id"])
    assert second_job["status"] == "completed", second_job.get("error")
    assert second_job["id"] != first_job["id"]
    assert second_job["iteration"] == 6
    second_iters = [row["iter"] for row in second_job["history"]]
    assert second_iters[:4] == first_iters
    assert second_iters[-2:] == [5, 6]

    original = client.get(f"/api/jobs/{first_job['id']}")
    assert original.status_code == 200
    assert [row["iter"] for row in original.json()["history"]] == first_iters

    weights = {item["name"] for item in client.get(f"/api/projects/{project_id}").json()["checkpoints"]}
    assert "model_0004.stub.json" in weights
    assert "model_0006.stub.json" in weights


def test_resume_without_run_or_checkpoint_fails(client: TestClient) -> None:
    project_id = _stub_project_with_image(client, "Resume Missing")
    response = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "checkpoint", "max_iter": 1},
    )
    assert response.status_code == 400


def test_infer_can_select_checkpoint_from_a_specific_run(client: TestClient) -> None:
    project_id = _stub_project_with_image(client, "Infer Named Run")
    first = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 2},
    )
    first_job = _wait_job(client, first.json()["id"])
    assert first_job["status"] == "completed", first_job.get("error")
    second = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 4},
    )
    second_job = _wait_job(client, second.json()["id"])
    assert second_job["status"] == "completed", second_job.get("error")

    listed = client.get(f"/api/projects/{project_id}").json()["checkpoints"]
    first_ref = f"{first_job['id']}/model_0002.stub.json"
    second_ref = f"{second_job['id']}/model_0004.stub.json"
    refs = {item["ref"] for item in listed}
    assert first_ref in refs
    assert second_ref in refs

    infer = client.post(
        f"/api/projects/{project_id}/jobs/infer",
        json={"checkpoint_name": first_ref},
    )
    assert infer.status_code == 200
    infer_job = _wait_job(client, infer.json()["id"])
    assert infer_job["status"] == "completed", infer_job.get("error")


def test_infer_from_dataset_folder(client: TestClient) -> None:
    project_id = _stub_project_with_image(client, "Infer Dataset")
    train = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 2},
    )
    train_job = _wait_job(client, train.json()["id"])
    assert train_job["status"] == "completed", train_job.get("error")

    infer = client.post(
        f"/api/projects/{project_id}/jobs/infer",
        json={"source": "dataset", "relative_path": "DS-1/all"},
    )
    assert infer.status_code == 200
    infer_job = _wait_job(client, infer.json()["id"])
    assert infer_job["status"] == "completed", infer_job.get("error")
    preview = infer_job["result"]["preview"]
    assert preview == "0_0.png"
    assert "0_0.png" in infer_job["result"]["images"]
    overlay = client.get(f"/api/projects/{project_id}/runs/{infer_job['id']}/overlays/{preview}")
    assert overlay.status_code == 200
    original = client.get(f"/api/projects/{project_id}/runs/{infer_job['id']}/inputs/{preview}")
    assert original.status_code == 200


def test_infer_upload_uses_run_preview(client: TestClient) -> None:
    project_id = _stub_project_with_image(client, "Infer Upload")
    train = client.post(
        f"/api/projects/{project_id}/jobs/train",
        json={"init": "random", "max_iter": 2},
    )
    train_job = _wait_job(client, train.json()["id"])
    assert train_job["status"] == "completed", train_job.get("error")

    infer = client.post(
        f"/api/projects/{project_id}/jobs/infer-upload",
        files=[("files", ("extra.png", _png_bytes(), "image/png"))],
        data={"overlay_colors": '{"Loop-A": "#112233"}'},
    )
    assert infer.status_code == 200
    infer_job = _wait_job(client, infer.json()["id"])
    assert infer_job["status"] == "completed", infer_job.get("error")
    assert infer_job["result"]["preview"] == "extra.png"
    overlay = client.get(
        f"/api/projects/{project_id}/runs/{infer_job['id']}/overlays/extra.png"
    )
    assert overlay.status_code == 200
    original = client.get(
        f"/api/projects/{project_id}/runs/{infer_job['id']}/inputs/extra.png"
    )
    assert original.status_code == 200


def test_create_project_with_r101_model(client: TestClient) -> None:
    created = client.post(
        "/api/projects",
        json={
            "name": "R101 Project",
            "classes": ["Loop-A"],
            "engine": "detectron2",
            "model": "mask_rcnn_r101_fpn",
        },
    )
    assert created.status_code == 200
    assert created.json()["model"] == "mask_rcnn_r101_fpn"


def test_train_page_explains_vitdet_native_size(client: TestClient) -> None:
    created = client.post(
        "/api/projects",
        json={"name": "ViTDet Page", "classes": ["Loop-A"], "engine": "detectron2"},
    )
    project_id = created.json()["id"]
    page = client.get(f"/projects/{project_id}/train")
    assert page.status_code == 200
    html = page.text
    assert "vitdet-note" in html
    assert "not upscaled to the COCO 1024 recipe" in html
    assert "Native image size (max 1024)" in html or "Detectron2 default" in html
