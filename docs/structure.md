# Structure du code

Ce fichier décrit **comment le dépôt est rangé** et **qui parle à qui**.
Le brief produit (objectifs, non-objectifs, décisions) reste dans [`PROJECT.md`](../PROJECT.md).
Pour ajouter un moteur ou une carte de modèle, voir [`nouvelles-implementations.md`](nouvelles-implementations.md).

## En une phrase

Une application web locale envoie des jobs d’entraînement et d’inférence à un **moteur interchangeable**. L’interface ne connaît pas Detectron2 ni Ultralytics : elle connaît des dossiers projet, du COCO JSON, et le nom d’un moteur.

## Couches

```
Navigateur (HTML / JS)
        │  HTTP
        ▼
app/          FastAPI : pages + API JSON
        │
        ▼
core/         Projets sur disque + jobs en arrière-plan
        │
        ▼
engine/       Contrat SegmentationEngine
              ├── stub          (faux moteur, tests / démo)
              ├── detectron2    (Mask R-CNN, y compris ViTDet)
              └── ultralytics   (YOLO-seg)
        │
        ▼
Volumes hôtes (montés dans Docker)
  projects/   données de l’utilisateur
  Datasets/   jeux de test (ex. DS-1)
  weights/    poids publics téléchargés, jamais dans l’image
```

Règle d’or : **UI, format d’annotation et layout Docker ne doivent pas importer Detectron2**. Si demain on remplace la bibliothèque, on change `engine/`, pas le produit.

Deux notions à ne pas confondre :

| Mot | Ce que c’est | Où ça vit |
| --- | --- | --- |
| **Moteur** (`engine`) | Une bibliothèque / un backend (`detectron2`, `ultralytics`, `stub`) | `project.json` → champ `engine` |
| **Carte de modèle** (`model`) | Une architecture précise (`mask_rcnn_r50_fpn`, `yolov8s-seg`, …) | Catalogue `engine/catalog.py`. Sur un **run** / checkpoint (`backend.json`), pas comme identité du projet. `project.model` n’est que le défaut pour un nouvel entraînement pretrained/random |

Un nouveau YOLO dans la famille Ultralytics = **nouvelle carte**. Une nouvelle bibliothèque (SMP, un autre framework) = **nouveau moteur**.

## Dossiers

| Chemin | Rôle |
| --- | --- |
| `app/` | Serveur web. Templates Jinja, CSS/JS vanilla, routes HTML et `/api`. |
| `core/` | Vérité disque : dossiers projet, COCO, file de jobs. |
| `engine/` | Contrat d’entraînement / inférence / export, plus les backends. |
| `tests/` | Contrats automatiques (stub sans GPU ; smoke CUDA skippés hors Compose). |
| `Docker/` | Image unique `app` (PyTorch + CUDA + Detectron2 + Ultralytics). |
| `projects/` | Projets utilisateur (monté, **pas** versionné comme données scientifiques). |
| `Datasets/` | Jeux montés depuis l’hôte. `DS-1` est le jeu de test par défaut. |
| `weights/` | Checkpoints publics téléchargés (`weights/detectron2/`, `weights/ultralytics/`). |
| `docs/` | Documentation développeur (ce fichier et le guide d’extension). |
| `.cursor/` | Conventions pour les agents (architecture, tests, UI en anglais). |

Le dépôt contient aussi du **legs** Jupyter / Detectron2 (`Detectron2/`, etc.). Le produit v1 ne s’appuie pas dessus : on réutilise surtout le format COCO et `Datasets/DS-1`.

## Flux utilisateur (instance segmentation)

1. Accueil `/` — texte d’intention.
2. `/projects` — créer un projet (nom, classes, moteur).
3. Page projet — images, polygones COCO, checkpoints, runs.
4. Annotate — canvas polygones, sauvegarde `annotations.json`.
5. Train — job asynchrone, courbes de loss, checkpoints.
6. Infer — overlay + masques + COCO, téléchargements zip.

Les pages HTML sont en **anglais** (règle produit). L’API JSON est le même chemin que le navigateur.

## `app/` — interface et API

| Fichier | Rôle |
| --- | --- |
| `app/main.py` | Construit l’app FastAPI : `ProjectStore`, `JobRunner`, static, routers. |
| `app/config.py` | Chemins via `AITEM_PROJECTS_DIR`, `AITEM_DATASETS_DIR`, `AITEM_WEIGHTS_DIR`. |
| `app/routers/pages.py` | HTML : `/`, `/projects`, pages projet / annotate / train / infer. |
| `app/routers/api.py` | JSON : projets, annotations, jobs, téléchargements, catalogue moteurs/modèles. |
| `app/templates/` | Jinja. `base.html` (bandeau), `index.html` (accueil), `projects.html`, etc. |
| `app/static/style.css` | Apparence. |
| `app/static/app.js` | Bannière de job actif, polling, courbes de loss. |
| `app/static/annotate.js` | Dessin des polygones (coordonnées en pixels natifs). |

L’UI **choisit** un `engine` et une carte `model`, puis appelle `/api/projects/.../jobs/train` ou `.../jobs/infer`. Elle n’ouvre jamais un fichier `.yaml` Detectron2.

## `core/` — projets et jobs

### Un projet sur disque

Sous `projects/<id>/` :

```
project.json          moteur, classes, modèle par défaut
images/               fichiers image
annotations.json      COCO (polygones)
weights/              copies des checkpoints publiés après train
runs/
  train-<id>/         status.json, train.log, history.jsonl, output/
  infer-<id>/         status.json, output/ (predictions.json, overlays/, masks/)
```

`core/projects.py` (`ProjectStore`) crée, lit, importe des images, sauve le COCO, liste runs et checkpoints.

`core/coco.py` valide le sous-ensemble COCO utilisé ici : images, catégories, polygones (au moins 3 points).

`core/paths.py` empêche un chemin utilisateur de sortir de `Datasets/` (`..` interdit).

### Jobs

`core/jobs.py` (`JobRunner`) :

- **Un seul job à la fois** (entraînement GPU et inférence se marcheraient dessus).
- Chaque job = un dossier `runs/<kind>-<uuid>/` avec `status.json`.
- Le thread appelle `get_engine(project.engine)` puis `engine.train` / `engine.infer`.
- Les poids publics sont téléchargés dans `weights/<moteur>/` via `engine.catalog.ensure_pretrained`.
- Un checkpoint d’un run est **copié** vers `projects/<id>/weights/` (et le sidecar `backend.json` avec).
- Continuer un run : **nouveau** dossier `train-…`, copie de `history.jsonl`, chargement des poids (`init=checkpoint`). On ne réécrit pas le run source. Dans Detectron2, `resume=False` (pas d’état d’optimiseur).

`max_iter` côté Detectron2 = pas supplémentaires (le compteur global continue). Côté YOLO, la même case UI veut dire **epochs**.

## `engine/` — contrat et backends

| Fichier | Rôle |
| --- | --- |
| `protocol.py` | `SegmentationEngine` : `name`, `train`, `infer`, `load_checkpoint`, `export_predictions`. |
| `types.py` | `TrainRequest`, `InferRequest`, etc. Chemins sur disque, pas d’objets Detectron2. |
| `catalog.py` | Cartes de modèles, URL de poids, suffixes de checkpoint, détection CUDA. |
| `registry.py` | `get_engine("detectron2")` → instance. Import **paresseux** des vrais backends. |
| `training.py` | Noms de checkpoints `model_0004.pth`, ETA, exception `TrainingStopped`. |
| `export.py` | COCO prédictions → overlays couleur + masques. **Partagé** par tous les moteurs. |
| `stub.py` | Faux réseau : JSON + losanges synthétiques. Sert aux tests et à la démo CPU. |
| `detectron2.py` | Mask R-CNN YAML + ViTDet LazyConfig. |
| `ultralytics.py` | YOLO-seg. Convertit le COCO projet via `yolo_data.py`. |
| `yolo_data.py` | `images/train`, `labels/train`, `data.yaml` (split val = train en v1). |

Les moteurs Detectron2 et Ultralytics s’importent seulement dans `get_engine`, pour que `pytest` sur l’hôte (sans ces libs) reste possible.

Sorties d’inférence attendues (tous moteurs) :

- `predictions.json` — COCO avec `score`
- `overlays/` — PNG colorés
- `masks/` — un fichier par instance

L’UI et l’API ne savent que ça.

## Docker

`Docker/docker-compose.yml` : un service `app`, GPU `runtime: nvidia`, volumes `projects/`, `Datasets/`, `weights/`.

`Docker/Dockerfile` copie `app/`, `engine/`, `core/`, `tests/`. Les poids **ne sont pas** dans l’image.

`Docker/entrypoint.sh` aligne UID/GID hôte, puis lance uvicorn sur le port 8000.

Modifier le code Python **dans l’image** demande un rebuild (`up --build`). Les données projet, elles, restent sur l’hôte.

## Tests (ce qu’ils prouvent)

Pas la qualité scientifique des masques. Seulement : « si j’annote, j’entraîne, j’infère, est-ce que les fichiers promis existent encore ? »

- Hôte, sans GPU : stub, catalogue, COCO, projets, API.
- Compose + CUDA : smoke Detectron2 et Ultralytics (skippés sinon).

Voir la section Checks de [`README.md`](../README.md).
