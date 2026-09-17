# Web UI (HTML, CSS, JavaScript)

English UI. Templates are Jinja2 (server fills `{{ ... }}` before the browser sees the page).
Comments in the files explain a concept **the first time it appears**; this page is the map.

Product layout of the whole repo: [`structure.md`](structure.md).

## Pages

| URL | Template | What it is |
| --- | --- | --- |
| `/` | `index.html` | Homepage (intent). |
| `/projects` | `projects.html` | Create / list instance-segmentation projects. |
| `/projects/<id>` | `project.html` | One project: upload, counts, delete. |
| `.../annotate` | `annotate.html` + `annotate.js` | Polygon canvas. |
| `.../train` | `train.html` | Start/stop training, log, loss curves. |
| `.../infer` | `infer.html` | Run inference, preview, downloads. |

`base.html` is the chrome (title, banner, CSS, `app.js`). Child pages `{% extends "base.html" %}` and fill `{% block content %}` and `{% block scripts %}`.

## Who talks to the API

- **Jinja** prints values the Python route already knows (`project.name`, `device.device`).
- **fetch()** in page scripts talks to `/api/...` for create, upload, train, infer, save COCO.
- **`app.js`** (every page): job banner, `pollJob`, loss-curve drawing.
- **`annotate.js`**: canvas only. It cannot see Jinja, so `annotate.html` sets `window.PROJECT_ID` first.

Files cannot go in JSON: uploads use `FormData` (multipart).

## CSS (`style.css`)

One file. Tokens live in `:root` (`--ink`, `--accent`, …). Layout is flex (header, actions) or grid (card rows). `[hidden]` is forced to `display: none` so JS `el.hidden = true` wins over flex/grid.

## First-occurrence cheat sheet

| Idea | Where it is introduced |
| --- | --- |
| Template inheritance, `{% block %}`, `hidden` | `base.html` |
| `{{ value }}`, `class="button"` on a link | `index.html` |
| `{% if %}` / `{% for %}`, `<form>`, `fetch` POST | `projects.html` |
| `<input type="file" multiple>`, `FormData`, `confirm()` | `project.html` |
| `<canvas>`, `data-color`, `window.PROJECT_ID` | `annotate.html` |
| `{% elif %}`, radio `name=`, `<details>`, `<progress>` | `train.html` |
| `<input type="color">`, `encodeURIComponent` | `infer.html` |
| CSS variables, `rem`, flex vs grid, `@media` | `style.css` (header comment) |
| `async`/`await`, polling, canvas 2D chart | `app.js` |
| IIFE, image vs canvas coordinates, hit-test | `annotate.js` |
