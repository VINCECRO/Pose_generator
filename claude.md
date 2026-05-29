# CLAUDE.md — Dataset Builder pour GestureAI

## Contexte du projet

Pipeline de construction d'un dataset de poses pour entraîner des modèles de détection spécialisés (construction Loomis, orientation volumes). Ce projet est **indépendant de l'app principale** — c'est un outil de préparation de données.

Le pipeline prend des photos de poses humaines en entrée et produit :
1. Des **images squelette OpenPose** (pour alimenter ControlNet + Stable Diffusion)
2. Des **métriques JSON** par image (angles, orientations, centre de gravité, 33 landmarks complets — pour tester les prompts LLM)
3. Des **recadrages de visages** (tête + cou) pour le sous-pipeline facial
4. Des **overlays de landmarks faciaux** (478 points MediaPipe) sur les crops visage

### Source des images

Les images source proviennent de SketchDaily References (sketchdaily.net), un site de référence pour artistes qui agrège des photos de poses libres de droits. Elles sont collectées via le scraper Playwright (voir section dédiée) et placées dans `data/raw/`. Le pipeline de traitement ne télécharge rien — il traite les images déjà présentes dans ce dossier.

Les images sont des photos de référence pour artistes (poses académiques, dynamiques, gesture drawing). Le pipeline extrait uniquement des données géométriques (keypoints) — les images source ne sont jamais incluses dans le dataset ML final.

---

## Structure du projet

```
gesture-ai-dataset/
├── CLAUDE.md                    ← ce fichier (dans .gitignore)
├── pyproject.toml               ← config uv
├── uv.lock
├── README.md
├── .gitignore
│
├── models/                      ← modèles MediaPipe (téléchargés auto, dans .gitignore)
│   ├── pose_landmarker_heavy.task
│   ├── blaze_face_full_range.tflite
│   └── face_landmarker.task
│
├── data/
│   ├── raw/                     ← images source (non committées)
│   ├── filtered/                ← images retenues après filtrage confiance
│   ├── skeletons/               ← squelettes OpenPose + overlays poses
│   ├── metrics/                 ← JSON métriques + 33 landmarks par image
│   ├── faces/                   ← recadrages visage+cou (non committés)
│   └── face_landmarks/          ← overlays + JSON 478 landmarks faciaux
│
├── src/
│   └── dataset_builder/
│       ├── __init__.py
│       ├── pipeline.py                ← orchestration principale poses
│       ├── pose_detector.py           ← wrapper MediaPipe PoseLandmarker
│       ├── skeleton_renderer.py       ← rendu squelette OpenPose sur canvas
│       ├── metrics_extractor.py       ← calcul des métriques géométriques
│       ├── filter.py                  ← filtrage par score de confiance
│       ├── face_cropper.py            ← détection + recadrage visage/cou
│       ├── face_landmark_detector.py  ← wrapper MediaPipe FaceLandmarker (478 pts)
│       ├── face_landmark_renderer.py  ← rendu tessellation + contours faciaux
│       └── scraper.py                 ← scraper Playwright (dans .gitignore)
│
├── scripts/
│   ├── run_pipeline.py          ← point d'entrée pipeline poses
│   ├── run_face_crop.py         ← point d'entrée recadrage visages
│   ├── run_face_landmarks.py    ← point d'entrée landmarks faciaux
│   └── scrape_sketchdaily.py    ← point d'entrée scraper (dans .gitignore)
│
└── tests/
    ├── fixtures/
    │   └── test_pose.jpg        ← image de test incluse dans le repo
    ├── test_pose_detector.py
    ├── test_metrics_extractor.py
    └── test_skeleton_renderer.py
```

---

## .gitignore

```gitignore
# Ce fichier — instructions privées du projet
CLAUDE.md

# Scraper — usage privé uniquement, ne pas publier
src/dataset_builder/scraper.py
scripts/scrape_sketchdaily.py

# Images source et dérivées — non committées
data/raw/
data/filtered/
data/faces/

# Outputs — régénérables, trop lourds pour git
data/skeletons/
data/metrics/
data/face_landmarks/

# Modèles MediaPipe — téléchargés automatiquement au premier lancement
models/

# Python
__pycache__/
.venv/
*.pyc
*.pyo
dist/
*.egg-info/
```

---

## Stack technique

- **Python 3.11+**
- **uv** pour la gestion des dépendances
- **MediaPipe** (`mediapipe>=0.10`) — API Tasks uniquement (`mp.solutions` absent en 0.10.35)
  - `PoseLandmarker` — 33 landmarks corps entier
  - `FaceDetector` — bounding box visage (modèle `blaze_face_full_range`)
  - `FaceLandmarker` — 478 landmarks faciaux + blendshapes
- **OpenCV** (`opencv-python`) pour le rendu et la manipulation d'images
- **Pillow** pour le chargement d'images
- **numpy** pour les calculs géométriques
- **tqdm** pour les barres de progression
- **playwright** (`playwright>=1.40`) pour le scraper (dans .gitignore)
- **requests** pour le téléchargement des images interceptées

---

## pyproject.toml

```toml
[project]
name = "gesture-ai-dataset"
version = "0.1.0"
description = "Pipeline d'extraction de keypoints et génération de squelettes OpenPose depuis une banque de poses de référence"
requires-python = ">=3.11"

[project.dependencies]
mediapipe = ">=0.10"
opencv-python = ">=4.8"
Pillow = ">=10.0"
numpy = ">=1.24"
tqdm = ">=4.65"
playwright = ">=1.40"
requests = ">=2.31"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Scraper Playwright — FICHIER PRIVÉ (dans .gitignore)

### Principe

SketchDaily (sketchdaily.net) est un SPA React qui affiche des photos de poses de référence pour artistes en mode timer. Chaque pose s'affiche pendant une durée configurable (30s, 1min, etc.). Playwright lance un vrai navigateur Chromium, ce qui contourne la détection bot.

Les images sont servies depuis `files.sketchdaily.net`. Playwright intercepte toutes les requêtes réseau sortantes vers ce domaine et télécharge chaque image. La déduplication se fait par hash MD5 du contenu — plus fiable que la taille du fichier.

### URL cible

```
https://www.sketchdaily.net/en/view/FullBodies?Time=30&recentImagesOnly=false
```

Paramètres utiles :
- `Time=30` — durée d'affichage de chaque pose en secondes (30, 60, 120)
- `recentImagesOnly=false` — inclure toutes les poses, pas seulement les récentes
- Catégories disponibles : `FullBodies`, `Faces`, `Hands`, `Animals`

### `scraper.py` — comportement attendu

```python
class SketchDailyScraper:
    def __init__(
        self,
        output_dir: Path,
        url: str = "https://www.sketchdaily.net/en/view/FullBodies?Time=30&recentImagesOnly=false",
        duration_minutes: int = 30,
        headless: bool = True
    )
    async def run(self) -> ScraperReport

class ScraperReport:
    total_intercepted: int
    total_saved: int
    total_duplicates: int
    output_dir: Path
```

Comportement :
1. Charger les hashes MD5 des images déjà présentes dans `output_dir` (reprise possible)
2. Lancer Playwright Chromium via `channel="chrome"` (vrai Chrome système)
3. Intercepter toutes les requêtes vers `files.sketchdaily.net`
4. Pour chaque image : télécharger → MD5 → doublon ? ignorer : sauvegarder
5. Tourner pendant `duration_minutes`, afficher compteur temps réel
6. Retourner un `ScraperReport`

### `scripts/scrape_sketchdaily.py` — FICHIER PRIVÉ (dans .gitignore)

```bash
uv run python scripts/scrape_sketchdaily.py                        # 30 min, ~60 poses
uv run python scripts/scrape_sketchdaily.py --duration 120         # 2 heures
uv run python scripts/scrape_sketchdaily.py --no-headless          # navigateur visible
uv run python scripts/scrape_sketchdaily.py --category Faces       # visages
uv run python scripts/scrape_sketchdaily.py --time 60              # 1 min/pose
```

Note post-install : `uv run playwright install chromium` nécessaire après `uv sync`.

---

## Modules pipeline

### `pose_detector.py`

Wrapper autour de MediaPipe PoseLandmarker (API Tasks, mediapipe 0.10+).

Les modèles sont téléchargés automatiquement dans `models/` au premier lancement.

```python
class PoseDetector:
    def __init__(self, model_complexity: int = 2, min_confidence: float = 0.3)
    def detect(self, image_path: Path) -> DetectionResult | None
    def detect_batch(self, image_paths: list[Path]) -> list[DetectionResult | None]

class DetectionResult:
    landmarks: list[Landmark]           # 33 keypoints, coordonnées normalisées (0→1)
    landmarks_3d: list[Landmark]        # coordonnées monde 3D
    image_width: int
    image_height: int
    confidence_scores: dict[int, float] # score de visibilité par keypoint
```

**`model_complexity` :**
| Valeur | Modèle | Usage |
|--------|--------|-------|
| `0` | `pose_landmarker_lite` | Prototypage rapide |
| `1` | `pose_landmarker_full` | Équilibré |
| `2` (défaut) | `pose_landmarker_heavy` | Production — meilleure précision sur membres partiellement visibles |

**Règle** : toujours utiliser `model_complexity=2` en production.

**Seuils** : `PoseDetector(min_confidence=0.3)` pour la détection large, `PoseFilter(confidence_threshold=0.8)` pour la sélection qualité. Ne pas confondre les deux — mettre 0.8 dans le détecteur rejette trop d'images valides.

Indices MediaPipe landmarks corps :
```
0  = nez
11 = épaule gauche    12 = épaule droite
15 = poignet gauche   16 = poignet droit
23 = hanche gauche    24 = hanche droite
25 = genou gauche     26 = genou droit
27 = cheville gauche  28 = cheville droite
```

### `face_cropper.py`

Détecte les visages et produit des recadrages tête + cou depuis les images `data/raw/`.
Utilise MediaPipe `FaceDetector` (modèle `blaze_face_full_range.tflite`, téléchargé auto).

**Important** : utiliser `blaze_face_full_range` (pas `short_range`). Le modèle short_range est conçu pour les selfies et échoue sur les poses corps entier où les visages sont petits et à distance variable.

```python
class FaceCropper:
    def __init__(self, min_confidence: float = 0.3, metrics_dir: Path | None = None)
    def crop(self, image_path: Path, debug_dir: Path | None = None) -> list[CropResult]
    def crop_batch(self, image_paths: list[Path], debug_dir: Path | None = None) -> list[CropResult]

class CropResult:
    image: np.ndarray    # image recadrée BGR
    source_path: Path
    face_index: int      # 0 si un seul visage, 0/1/... si plusieurs
    method: str          # "detector" | "landmarks"
```

Marges appliquées autour de la bounding box :
- **Haut** : +65% (crâne entier)
- **Côtés** : +50%
- **Bas** : +140% (cou complet + amorce épaules)

**Fallback landmarks** : si `metrics_dir` est fourni et que le détecteur ne trouve rien, le crop est calculé depuis les landmarks faciaux du JSON de métriques (nez, yeux, oreilles, bouche). Garantit un crop pour toute image ayant passé le pipeline poses.

### `scripts/run_face_crop.py`

```bash
uv run python scripts/run_face_crop.py                            # data/raw/ → data/faces/
uv run python scripts/run_face_crop.py --confidence 0.3           # seuil (défaut)
uv run python scripts/run_face_crop.py --debug                    # sauvegarde bounding boxes dans data/faces_debug/
uv run python scripts/run_face_crop.py --input /autre/dossier
```

Idempotent — skip les fichiers déjà présents dans `data/faces/`.

### `face_landmark_detector.py`

Wrapper autour de MediaPipe FaceLandmarker (478 landmarks faciaux + blendshapes).
Modèle : `face_landmarker.task` (téléchargé auto dans `models/`).

```python
class FaceLandmarkDetector:
    def __init__(self, min_confidence: float = 0.5)
    def detect(self, image_path: Path) -> FaceLandmarkResult | None
    def close(self) -> None

class FaceLandmarkResult:
    landmarks: list[FaceLandmark]   # 478 points, coordonnées normalisées
    image_width: int
    image_height: int
    blendshapes: dict[str, float]   # 52 blendshapes (eyeBlinkLeft, jawOpen, etc.)

    def px(self, idx: int) -> tuple[int, int]  # coordonnées pixel
```

Landmarks 468-477 = iris (gauche 468-472, droit 473-477).

### `face_landmark_renderer.py`

Utilise `FaceLandmarksConnections` du Tasks API (pas de code hardcodé).

```python
class FaceLandmarkRenderer:
    def render(self, result: FaceLandmarkResult, output_size: tuple = (512, 512)) -> np.ndarray
    def render_on_photo(self, result: FaceLandmarkResult, image: np.ndarray) -> np.ndarray
```

Rendu en couches :
- Tessellation complète (2556 connexions, gris sombre) — toute la surface du visage
- Contours par-dessus : ovale (gris clair), yeux (jaune), sourcils (jaune clair), nez (rose), lèvres (orange), iris (blanc)

### `scripts/run_face_landmarks.py`

```bash
uv run python scripts/run_face_landmarks.py                       # data/faces/ → data/face_landmarks/
uv run python scripts/run_face_landmarks.py --confidence 0.5      # seuil (défaut)
uv run python scripts/run_face_landmarks.py --no-json             # overlays uniquement
```

Outputs par image :
- `nom_face_overlay.png` — landmarks dessinés sur la photo
- `nom_face_mesh.png` — maillage sur fond noir 512×512
- `nom_face.json` — 478 coords (x, y, z, x_px, y_px) + 52 blendshapes

Idempotent — skip si overlay déjà présent.

### `metrics_extractor.py`

Calcule les métriques géométriques à partir des landmarks (coordonnées converties en pixels absolus).

```python
class MetricsExtractor:
    def extract(self, result: DetectionResult) -> PoseMetrics

class PoseMetrics:
    shoulder_angle: float          # bascule ligne épaules (degrés / horizontale)
    hip_angle: float               # bascule ligne hanches
    shoulder_midpoint: tuple       # milieu ligne épaules (pixels absolus)
    hip_midpoint: tuple            # milieu ligne hanches
    spine_vector: tuple            # vecteur colonne (shoulder_mid → hip_mid)
    center_of_gravity: tuple       # approximation bassin
    weight_side: str               # "left" | "right" | "balanced"
    spine_curve: str               # "straight" | "slight_S" | "pronounced_S" | "C_curve"
    support_type: str              # "bipodal" | "left_dominant" | "right_dominant"

    def to_json(self) -> dict
    def to_llm_prompt(self) -> dict
```

### `skeleton_renderer.py`

```python
class SkeletonRenderer:
    def render(self, result: DetectionResult, output_size: tuple = (512, 512)) -> np.ndarray
    def render_on_photo(self, result: DetectionResult, original_image: np.ndarray) -> np.ndarray
```

33 connexions MediaPipe complètes : tronc, bras, jambes, tête→épaules, visage (yeux/bouche), mains (pouce/index/auriculaire), pieds (talon/orteil).

Rendu visibilité : liens solides si score ≥ 0.5, pointillés sinon. Même code couleur.

Couleurs : torse blanc, bras gauche bleu, bras droit vert, jambe gauche jaune, jambe droite orange, tête/visage rose.

Outputs : `data/skeletons/nom.png` (fond noir 512×512) + `data/skeletons/nom_overlay.png` (debug).

### `filter.py`

```python
KEY_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28]  # épaules, hanches, genoux, chevilles

class PoseFilter:
    def __init__(self, confidence_threshold: float = 0.8)
    def is_valid(self, result: DetectionResult) -> bool
    def filter_batch(self, results: list[tuple[Path, DetectionResult]]) -> list[Path]
    def generate_report(self, total: int, kept: int) -> str
```

### `pipeline.py`

```python
def run_pipeline(
    input_dir: Path = Path("data/raw"),
    skeleton_dir: Path = Path("data/skeletons"),
    metrics_dir: Path = Path("data/metrics"),
    filtered_dir: Path = Path("data/filtered"),
    confidence_threshold: float = 0.8,
    verbose: bool = True
) -> PipelineReport
```

Comportement :
1. Lister toutes les images dans `input_dir` (jpg, jpeg, png, webp)
2. Ignorer les images déjà traitées (JSON correspondant présent dans `metrics/`) — idempotent
3. Pour chaque image : détecter (seuil 0.3) → filtrer qualité (seuil 0.8) → si valide → squelette + métriques (33 landmarks complets) + copie `filtered/`
4. Barre de progression `tqdm`
5. Rapport final : total, retenu, rejeté, taux de réussite

---

## Ordre d'exécution recommandé

```bash
# 1. Collecter les poses (scraper privé)
uv run python scripts/scrape_sketchdaily.py --duration 60

# 2. Traiter les poses complètes → skeletons + metrics + filtered
uv run python scripts/run_pipeline.py

# 3. Recadrer les visages
uv run python scripts/run_face_crop.py

# 4. Détecter les landmarks faciaux sur les crops
uv run python scripts/run_face_landmarks.py
```

---

## Format de sortie

### `data/metrics/nom_image.json`
```json
{
  "source_image": "pose_001.jpg",
  "image_dimensions": {"width": 1527, "height": 2048},
  "detection_confidence": {
    "shoulder_left": 0.9999, "shoulder_right": 0.9997,
    "hip_left": 0.9996, "hip_right": 0.9996
  },
  "landmarks": {
    "nose":          {"x": 0.583542, "y": 0.194978, "z": -0.446994, "visibility": 0.9999, "x_px": 891, "y_px": 399},
    "shoulder_left": {"x": 0.743296, "y": 0.308126, "z": -0.216491, "visibility": 0.9999, "x_px": 1135, "y_px": 631},
    "...":           "33 landmarks au total (nose → foot_index_right)"
  },
  "metrics": {
    "shoulder_angle": -8.3,
    "hip_angle": 14.7,
    "shoulder_midpoint": [412, 280],
    "hip_midpoint": [398, 680],
    "spine_vector": [-14, 400],
    "center_of_gravity": [398, 680],
    "weight_side": "right",
    "spine_curve": "slight_S",
    "support_type": "right_dominant"
  },
  "llm_ready": {
    "bascule_epaules": "droite_8deg",
    "bascule_bassin": "gauche_15deg",
    "centre_gravite": "jambe_droite",
    "courbe_colonne": "legere_S",
    "appui": "monopodal_droit"
  }
}
```

### `data/face_landmarks/nom_face.json`
```json
{
  "source_image": "nom_face.jpg",
  "image_dimensions": {"width": 512, "height": 680},
  "landmarks": [
    {"x": 0.512, "y": 0.423, "z": -0.031, "x_px": 262, "y_px": 287},
    "... 478 entrées (indices 0-477)"
  ],
  "blendshapes": {
    "eyeBlinkLeft": 0.012,
    "eyeBlinkRight": 0.008,
    "jawOpen": 0.041,
    "... 52 blendshapes au total": "..."
  }
}
```

---

## Tests

```bash
uv run pytest tests/ -v
```

- `test_pose_detector.py` : détection sur `tests/fixtures/test_pose.jpg`
- `test_metrics_extractor.py` : calculs géométriques avec landmarks fictifs connus
- `test_skeleton_renderer.py` : image générée 512×512, non entièrement noire

---

## README.md

**Ne jamais mentionner SketchDaily, le scraper, ni aucune source d'images spécifique.**

Sections : titre + badges → description → schéma ASCII pipeline → prérequis → installation → usage (poses + face crop + face landmarks) → structure des outputs → tests → images source (générique) → licence MIT.

---

## Contraintes importantes

- **Python pur** — aucune dépendance Node.js
- **Pas de GPU requise** — MediaPipe sur CPU ; RTX 2070 intervient pour Stable Diffusion (hors scope)
- **API Tasks uniquement** — `mediapipe.python.solutions` absent en 0.10.35, ne pas tenter de l'importer. Utiliser `mediapipe.tasks.python.vision` et `FaceLandmarksConnections` pour les connexions du mesh facial.
- **Modèle heavy par défaut** — `model_complexity=2` pour une meilleure précision sur les membres partiellement visibles ou hors cadre
- **Seuil de détection ≠ seuil de filtre** — `PoseDetector(min_confidence=0.3)` large pour ne rien rater, `PoseFilter(confidence_threshold=0.8)` strict pour la qualité. Confondre les deux rejette trop d'images.
- **FaceDetector : full_range** — `blaze_face_full_range.tflite` obligatoire pour les poses corps entier. Le modèle `short_range` est conçu pour les selfies et échoue à distance.
- **Idempotent** — ne retraite pas les images déjà traitées
- **Pas de base de données** — tout est fichier
- **Attribution préservée** — nom du fichier source conservé dans le JSON
- **CLAUDE.md dans .gitignore** — ne jamais committer
- **scraper.py et scrape_sketchdaily.py dans .gitignore** — usage privé uniquement
- **README générique** — aucune mention de SketchDaily, du scraper, ou de source spécifique
- **Post-install Playwright** : `uv run playwright install chromium` nécessaire pour le scraper (pas pour le pipeline)
