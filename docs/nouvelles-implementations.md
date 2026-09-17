# Ajouter un moteur ou un modèle

Guide pratique. La carte du dépôt est dans [`structure.md`](structure.md).
Les décisions produit (instance seulement en v1, pas de post-traitement matériaux) sont dans [`PROJECT.md`](../PROJECT.md).

## D’abord : moteur ou carte ?

Pose-toi **une** question : est-ce la même bibliothèque, ou une autre ?

```
Je veux un nouveau réseau
        │
        ├─ Même lib déjà branchée
        │     Detectron2  → nouvelle carte dans le moteur detectron2
        │     Ultralytics → nouvelle carte YOLO-seg dans le moteur ultralytics
        │
        └─ Autre lib (SMP, MMDetection, un wrapper maison, …)
              → nouveau moteur (nouveau fichier + registre)
```

Exemples concrets :

| Demande | Quoi faire |
| --- | --- |
| Mask R-CNN R101 (déjà là) / un autre YAML Mask R-CNN du zoo Detectron2 | Carte `ModelSpec`, souvent **sans** toucher `detectron2.py` |
| ViTDet, ou un LazyConfig différent | Carte **et** une branche dans `engine/detectron2.py` (`config_kind="lazy"`) |
| YOLOv8m-seg, YOLO11l-seg, … | Carte Ultralytics (`ultralytics_name` + URL `.pt`) |
| Un YOLO qui n’est pas un checkpoint Ultralytics-seg | Probablement **nouveau moteur**, pas une carte |
| Mask2Former via Detectron2 | Encore Detectron2 : nouvelle **famille** (`family`) + code cfg, pas un 4ᵉ moteur |
| Segmentation sémantique | Hors v1. Nouveau `task`, pas seulement une carte instance. Voir plus bas. |

`timm` charge un **backbone**, pas un moteur de segmentation. L’ajouter n’est pas « un nouveau moteur » : ça se cacherait derrière Detectron2 (ou un autre) si un jour on custom-backbone.

Les checkpoints **ne sont pas interchangeables** : `.pth` / `.pkl` Detectron2 ≠ `.pt` YOLO ≠ `.stub.json`.

---

## A. Nouvelle carte dans un moteur existant

C’est le cas le plus fréquent. Le fichier central est [`engine/catalog.py`](../engine/catalog.py).

### 1. Décrire la carte (`ModelSpec`)

Dans le dictionnaire `MODELS`, ajoute une entrée **avant** de coder le train. Champs utiles :

| Champ | Rôle |
| --- | --- |
| `id` | Identifiant stable (celui de l’UI et de `project.model`). |
| `label` | Texte anglais affiché. |
| `engine` | `detectron2` ou `ultralytics` (doit matcher le projet). |
| `family` | Groupe : `mask_rcnn`, `vitdet`, `yolo`, … L’UI s’en sert pour des notes (ViTDet, YOLO). |
| `task` | Aujourd’hui `instance`. Filtre `list_models(..., task="instance")`. |
| `detectron2_config` | Chemin zoo YAML **ou** LazyConfig, selon `config_kind`. |
| `config_kind` | `"yaml"` (défaut) ou `"lazy"` (ViTDet). |
| `checkpoint_url` | Poids publics. Téléchargés dans `weights/<engine>/`. |
| `checkpoint_filename` | Nom du fichier sur le volume `weights/`. |
| `default_lr` | Si le zoo ne va pas avec 0.00025 (ViTDet / YOLO ont les leurs). |
| `ultralytics_name` | Nom passé à `YOLO(...)` (souvent égal à `id`). |

Contraintes :

- `id` unique, lisible, stable (ne pas le renommer une fois des projets créés).
- L’URL doit pointer vers des poids **instance-seg** compatibles (pas un classifieur).
- Pour YOLO : fichier `.pt` Ultralytics-seg, pas un `.pth` Detectron2.

### 2. Detectron2 YAML (le chemin simple)

Si la recette est un YAML du Model Zoo du même style que R50/R101/X101 :

1. Carte avec `config_kind="yaml"` (ou omis).
2. `detectron2_config="COCO-InstanceSegmentation/....yaml"`.
3. Tests catalogue (voir §4).
4. En général **aucun** changement dans `engine/detectron2.py` : le moteur charge déjà `spec.detectron2_config`.

Entraîne une fois dans Compose (quelques itérations) pour vérifier que le YAML et l’URL existent encore.

### 3. Detectron2 LazyConfig (comme ViTDet)

Si ce n’est pas un YAML `get_cfg()` :

1. `config_kind="lazy"`.
2. Regarde comment `Detectron2Engine` branche déjà sur `spec.config_kind == "lazy"` (canvas, `square_pad`, lr).
3. Si le nouveau cfg a **les mêmes** besoins que ViTDet, réutiliser cette branche.
4. Sinon, étendre `detectron2.py` **sans** casser les cartes YAML. Préfère un `family` / `config_kind` plutôt qu’un `if model_id == "..."`.

Écris aussi `backend.json` dans le dossier du run (taille d’entrée, famille). L’inférence et le « continue from checkpoint » relisent ça, pas `project.model`.

### 4. Ultralytics (nouvelle taille / année YOLO-seg)

1. Carte `engine="ultralytics"`, `family="yolo"`, `ultralytics_name="yolo11m-seg"` (exemple).
2. URL GitHub `ultralytics/assets` vers le `.pt` **seg**.
3. `checkpoint_filename` identique au fichier officiel.
4. Vérifier la version `ultralytics` dans `requirements.txt` (YOLO26 exige déjà 8.4+).
5. Souvent **aucun** changement dans `engine/ultralytics.py` : il fait `YOLO(spec.ultralytics_name)`.

Nano / small / medium changent la VRAM, pas l’API. Documente-le dans `train.html` seulement si l’utilisateur doit changer un réglage (batch 1, imgsz).

### 5. Tests minimum pour une carte

Dans [`tests/test_catalog.py`](../tests/test_catalog.py) :

- `get_model("ton-id")` → bon `engine`, `family`, fichier de config ou `ultralytics_name`.
- `available_models(engine="…")` contient le nouvel `id` (ordre du catalogue respecté).
- `pretrained_path` tombe sous `weights/<engine>/`.
- Un moteur refuse un checkpoint de **l’autre** (suffixes).

Puis, si CUDA + lib sont là : smoke existant, ou quelques itérations à la main sur `Datasets/DS-1`.

L’UI (`train.html`) charge `/api/models?engine=…` : une carte catalogue apparaît **sans** dupliquer la liste en HTML. Ajoute une note (`#vitdet-note`, `#yolo-note`) seulement si le comportement sort du lot.

### 6. Ce qu’il ne faut pas faire

- Mettre le chemin YAML Detectron2 dans un template HTML.
- Créer un moteur `detectron2_r101` alors que R101 est une carte.
- Stocker l’architecture uniquement dans `project.json` en ignorant `backend.json` du checkpoint (l’inférence suivrait alors le mauvais réseau).

---

## B. Nouveau moteur (nouvelle bibliothèque)

Un moteur, c’est une classe qui **respecte** `SegmentationEngine` dans [`engine/protocol.py`](../engine/protocol.py).

### Contrat à implémenter

```text
name() -> str                         # clé registre, ex. "smp"
train(TrainRequest) -> TrainResult
infer(InferRequest) -> InferResult
load_checkpoint(path) -> None         # valide + refuse l’incompatible
export_predictions(ExportRequest)     # ou déléguer à engine.export
```

Entrées (`engine/types.py`) : dossiers, JSON COCO, noms de classes, `init` ∈ {`pretrained`, `random`, `checkpoint`}. Pas d’objet `CfgNode` qui fuit vers `core/` ou `app/`.

Sorties train :

- Au moins un fichier poids nommé comme les autres : `model_XXXX.<suffixe>` (`engine.training.checkpoint_filename`).
- `metrics.json`.
- Idéalement `backend.json` (id de carte, taille d’entrée, famille) pour l’inférence et le resume.
- Honorer `request.should_stop` : sauver, renvoyer `TrainResult(stopped=True)` (voir le stub).

Sorties infer :

- Écrire `predictions.json` (COCO + `score`).
- Appeler `engine.export.export_instance_visuals` pour overlays + masques — **ne réinvente pas** le dessin dans l’UI.
- Filtrer avec `filter_coco_instances` si tu respectes `score_threshold` / `max_detections`.

Imports **paresseux** : `import ta_lib` seulement dans les méthodes (comme Detectron2), pour que `get_engine("tonmoteur")` et pytest hôte ne cassent pas.

Inspire-toi de [`engine/stub.py`](../engine/stub.py) pour le déroulé (progress, stop, export), et d’un vrai backend pour le chargement des poids.

### Fichiers à toucher (checklist)

1. **`engine/tonmoteur.py`** — la classe.
2. **`engine/registry.py`**
   - `available_engines()` : entrée `name` / `available` / `label`.
   - `get_engine()` : `if key == "tonmoteur": return TonMoteur()`.
3. **`engine/catalog.py`**
   - `KNOWN_ENGINES`
   - `CHECKPOINT_SUFFIXES` (ex. `{".pt"}`)
   - `DEFAULT_MODELS`
   - `ENGINE_ZOO_URLS` (lien docs)
   - au moins une `ModelSpec` avec `engine="tonmoteur"`
   - messages dans `reject_incompatible_checkpoint`
4. **`core/jobs.py`** — **piège fréquent** : le téléchargement pretrained n’est fait que pour `detectron2` et `ultralytics` :

   ```python
   elif init == "pretrained" and record.engine in {"detectron2", "ultralytics"}:
       pretrained = ensure_pretrained(...)
   ```

   Ajoute ton nom de moteur à cet ensemble, sinon « pretrained » ne téléchargera rien.
5. **`app/templates/projects.html`** — `<option>` dans le sélecteur de moteur.
6. **`app/templates/train.html`** — texte d’aide si le vocabulaire change (epochs vs iter).
7. **`requirements.txt`** + **`Docker/Dockerfile`** si le paquet n’est pas déjà là. Rebuild Compose.
8. **Tests**
   - Catalogue : cartes, suffixes, refus de checkpoints étrangers.
   - Stub-like ou smoke CUDA (skippé sans GPU), sur `Datasets/DS-1`.
   - L’API crée un projet avec `engine="tonmoteur"` (`tests/test_api.py` a le précédent YOLO).

Le stub reste le filet de sécurité : un nouveau moteur **ne remplace pas** le stub.

### Dépendances Docker

- Paquet Python → `requirements.txt` puis `docker compose -f Docker/docker-compose.yml up --build`.
- Poids : URL dans la carte, dossier `weights/tonmoteur/`. Jamais `COPY` des `.pth` dans l’image.
- Si l’outil écrit chez lui au runtime (Ultralytics AMP), suis le modèle de `Docker/entrypoint.sh` (`YOLO_CONFIG_DIR` sur le volume).

### UI

Anglais uniquement. Le bandeau et `/projects` parlent d’**instance segmentation** : un moteur sémantique n’est pas « une option de plus dans le même `<select>` » sans travail de tâche (voir C).

---

## C. Semantic segmentation (plus tard)

`TaskKind` existe déjà (`instance` | `semantic`) et `ModelSpec.task` aussi. **La v1 ne doit pas** tordre l’UI, le COCO polygones, ni l’export instance autour de ça.

Quand ce sera le moment, ce n’est en général **pas** « une carte Mask R-CNN de plus ». C’est plutôt :

- des cartes `task="semantic"` (éventuellement un moteur dédié, ou une famille Detectron2) ;
- un autre contrat d’annotation (souvent un masque plein écran, pas N polygones d’instances) ;
- un autre export (une carte de classes, pas une instance par fichier) ;
- une entrée d’accueil du type « Semantic segmentation projects », parallèle à l’instance.

Jusque-là : n’ajoute pas de routes `/semantic` « pour préparer le terrain ».

---

## Vérifications après un ajout

1. `pytest tests/test_catalog.py` (et les tests du nouveau module).
2. Créer un projet stub : le reste de l’app n’a pas régressé.
3. Dans Compose, un train court + infer sur `DS-1` avec **la nouvelle carte**.
4. Reprendre depuis le checkpoint : la bonne architecture se relit (`backend.json` / sidecar).
5. Un checkpoint de l’autre moteur est **refusé** avec un message clair.

Si un test hôte se met à `import detectron2` au chargement du module, l’import n’est plus paresseux : à corriger.
