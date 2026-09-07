# TEM instance segmentation tool

Local web app: annotate TEM images with polygons, train, and run instance segmentation. English UI. After Compose is up, work happens in the browser.

## Start

From this repository (Linux). The first two values make files in `projects/` belong to **you**, not to root:

```bash
docker compose -f Docker/docker-compose.yml up --build
# Legacy ?
#HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose -f Docker/docker-compose.yml up --build
# Open http://localhost:8000
```

The container also prints that URL when it starts.

If your user id is 1000 (common on a personal Ubuntu machine), you can omit `HOST_UID` / `HOST_GID`.

## What `docker compose` is doing

Docker **Compose** reads `Docker/docker-compose.yml` and starts the services described there. Here there is only one service, `app`: it **builds** an image (the software: Python, FastAPI, the stub engine) and **runs** a container (a running copy of that image). Your folders `projects/`, `Datasets/` and `weights/` are **mounted** into the container: the program inside sees them, but the files stay on your disk. Model weight files are never copied into the image.

Usual loop:

1. `up --build` — build if needed, then start. Logs stay in this terminal.
2. Open http://localhost:8000 and use the app.
3. Stop the app: in that same terminal, <kbd>Ctrl</kbd>+<kbd>C</kbd>. That shuts the container down.

### The <kbd>w</kbd> and <kbd>d</kbd> prompts

When Compose asks something like “enable watch” / “detach”, it is offering **developer shortcuts**, not extra features of this TEM tool.

- **<kbd>d</kbd> Detach** — start in the background and give you the terminal back. The container **keeps running**. The app is still at http://localhost:8000 until you stop it on purpose (see below). This is normal.
- **<kbd>w</kbd> Enable Watch** — Compose Watch: rebuild or copy files when you edit the code. Useful for development, unused for a colleague who only wants to annotate. You can ignore it.

To detach on purpose from the command line: add `-d` (the flag, same idea as the <kbd>d</kbd> key):

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose -f Docker/docker-compose.yml up --build -d
```

### Stopping a container that was detached

The process is still alive in the background. From the repository:

```bash
docker compose -f Docker/docker-compose.yml stop    # pause; data in projects/ is kept
docker compose -f Docker/docker-compose.yml start   # start again without rebuilding
docker compose -f Docker/docker-compose.yml down    # stop and remove the container; projects/ on disk stay
```

<kbd>Ctrl</kbd>+<kbd>C</kbd> only works if that terminal is still attached to `up` (you did not press <kbd>d</kbd>).

## Layout

| Path             | Role                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------ |
| `app/`           | Web server (FastAPI) and HTML/JS pages                                               |
| `engine/`        | Training/inference contract and backends (`stub` now, Detectron2 later)              |
| `core/`          | Projects on disk, COCO files, background jobs                                        |
| `Datasets/DS-1/` | Dummy dataset (PNG tiles in `all/`, COCO in `train.json` / `val.json` / `test.json`) |
| `projects/`      | Your projects (host folder, mounted into the container)                              |
| `weights/`       | Downloaded model weight files (host folder, **not** baked into the Docker image)     |

The current training backend is a **stub** (a fake engine). It writes dummy checkpoints, COCO, masks, and overlays so the full click-path can be tested without a GPU.

## Checks (tests)

These are **automatic checks**, not a judgement of your micrographs. They never look at scientific quality. They only ask: “if I create a project, save a polygon, and run the fake train/infer path, does the tool still write the files it promised?”

You run them after a code change, or when something feels broken, to see a red/green answer in a few seconds instead of clicking through every screen. Colleagues who only use the browser can ignore them.

From the repository, with Python available:

```bash
pip install -r requirements.txt
pytest
```

Inside a running Compose service:

```bash
docker compose -f Docker/docker-compose.yml exec app pytest -q
```

Four files, in plain language:

- **Engine** (`tests/test_engine_stub.py`) — the fake backend still produces a checkpoint, a COCO file, mask images and a colour overlay; Detectron2 is still explicitly “not implemented”.
- **COCO** (`tests/test_coco.py`) — a polygon JSON can be saved and opened again without losing classes or vertices.
- **Project folder** (`tests/test_projects.py`) — creating (and deleting) a project still builds the expected folders on disk.
- **Web API** (`tests/test_api.py`) — the same path a browser uses: create, upload, annotate, train, infer, download, delete.

## Data note

`Datasets/` is mounted from the host.

## Acknowledgements

Created with aid from Cursor AI ([https://www.cursor.com](https://www.cursor.com)) IDE and various LLM models.Created with aid from Cursor AI ([https://www.cursor.com](https://www.cursor.com)) IDE and its Composer/Grok models.
