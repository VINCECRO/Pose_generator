# gesture-ai-dataset

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

Pipeline Python d'extraction de keypoints et de métriques géométriques depuis une banque de photos de poses de référence pour artistes. Produit, pour chaque image retenue, un squelette OpenPose prêt pour ControlNet et un JSON structuré utilisable dans des prompts LLM. Composant du projet **GestureAI** — outil pédagogique de gesture drawing.

---

## Pipeline

```
Banque de photos (data/raw/)
         │
         ▼
MediaPipe PoseLandmarker
(33 landmarks corps entier)
         │
         ▼
Filtrage qualité (confiance > 0.8)
         │
    ┌────┴─────────────┐
    ▼                  ▼
Squelette          Métriques JSON
OpenPose           (angles, centre de gravité,
(data/skeletons/)   33 landmarks + llm_ready)
    │              (data/metrics/)
    ▼
ControlNet + Stable Diffusion
    ▼
Dataset ML libre de droits
```

**Sous-pipeline visage (depuis les images filtrées) :**

```
data/raw/
    │
    ▼
FaceDetector (BlazeFace full-range)
    │
    ▼
Recadrage tête + cou (data/faces/)
    │
    ▼
FaceLandmarker (478 landmarks + blendshapes)
    │
    ▼
Overlays + JSON (data/face_landmarks/)
```

---

## État d'avancement

### Terminé

- **Pipeline poses complet** — détection, filtrage qualité, rendu squelette OpenPose (fond noir 512×512 + overlay debug), extraction métriques géométriques (angles épaules/hanches, centre de gravité, type d'appui, courbe colonne), export JSON avec champ `llm_ready`
- **Sous-pipeline visage complet** — détection et recadrage tête+cou (marges calibrées, fallback landmarks si détecteur échoue), détection 478 landmarks faciaux + 52 blendshapes, rendu tessellation + contours (overlay photo + maillage fond noir)
- **Idempotence** — les trois scripts skippent les images déjà traitées, reprises possibles sans retraitement
- **Suite de tests** — détection sur image réelle, calculs métriques sur landmarks connus, vérification rendu squelette

### En cours

- Constitution de la banque de photos source (`data/raw/`)
- Validation qualitative des outputs squelettes en vue de l'usage ControlNet

### Prochaines étapes

1. **Génération du dataset synthétique** — conditionner Stable Diffusion via ControlNet sur les squelettes pour produire des images libres de droits à partir des poses extraites
2. **Enrichissement des métriques** — ajouter orientations volumiques (tête, torse, bassin en 3D depuis les landmarks z) pour préparer les prompts d'entraînement
3. **Modèle de détection spécialisé** — entraîner un modèle de reconnaissance des constructions Loomis (proportions, volumes de tête/cage thoracique/bassin) sur le dataset généré
4. **Intégration app principale** — exposer l'analyse de pose comme service appelé par GestureAI pour annoter les poses affichées en temps réel

---

## Prérequis

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- Pas de GPU requise — MediaPipe tourne entièrement sur CPU

---

## Installation

```bash
git clone https://github.com/ton-user/gesture-ai-dataset
cd gesture-ai-dataset
uv sync
```

Les modèles MediaPipe sont téléchargés automatiquement dans `models/` au premier lancement.

---

## Usage

Placez vos images de référence dans `data/raw/`, puis lancez les trois étapes dans l'ordre :

### 1. Pipeline poses

```bash
# Traiter toutes les images dans data/raw/
uv run python scripts/run_pipeline.py

# Options
uv run python scripts/run_pipeline.py --confidence 0.85   # seuil de confiance (défaut 0.8)
uv run python scripts/run_pipeline.py --quiet             # sans barre de progression
uv run python scripts/run_pipeline.py --input /chemin/vers/images
```

| Option | Défaut | Description |
|---|---|---|
| `--input` | `data/raw` | Dossier source des images |
| `--skeletons` | `data/skeletons` | Dossier de sortie squelettes |
| `--metrics` | `data/metrics` | Dossier de sortie métriques JSON |
| `--filtered` | `data/filtered` | Images retenues copiées ici |
| `--confidence` | `0.8` | Seuil de confiance MediaPipe (0.0–1.0) |
| `--quiet` | — | Désactive la barre de progression |

### 2. Recadrage visages

```bash
uv run python scripts/run_face_crop.py                    # data/raw/ → data/faces/
uv run python scripts/run_face_crop.py --confidence 0.3   # seuil détection (défaut)
uv run python scripts/run_face_crop.py --debug            # sauvegarde les bounding boxes
```

### 3. Landmarks faciaux

```bash
uv run python scripts/run_face_landmarks.py               # data/faces/ → data/face_landmarks/
uv run python scripts/run_face_landmarks.py --no-json     # overlays uniquement, sans JSON
```

Les trois scripts sont **idempotents** — relancer ne retraite pas les fichiers déjà produits.

---

## Structure des outputs

```
data/
├── filtered/                      ← images retenues (confiance > 0.8)
├── skeletons/
│   ├── pose_001.png               ← squelette OpenPose, fond noir 512×512 (ControlNet)
│   └── pose_001_overlay.png       ← squelette superposé à la photo (debug)
├── metrics/
│   └── pose_001.json              ← métriques + 33 landmarks + champ llm_ready
├── faces/
│   └── pose_001_face0.jpg         ← recadrage tête+cou
└── face_landmarks/
    ├── pose_001_face0_face.json      ← 478 landmarks + 52 blendshapes
    ├── pose_001_face0_face_overlay.png  ← landmarks sur photo
    └── pose_001_face0_face_mesh.png     ← maillage sur fond noir 512×512
```

### Format métriques JSON

```json
{
  "source_image": "pose_001.jpg",
  "image_dimensions": { "width": 1527, "height": 2048 },
  "detection_confidence": {
    "shoulder_left": 0.9999, "shoulder_right": 0.9997,
    "hip_left": 0.9996, "hip_right": 0.9996
  },
  "landmarks": {
    "nose":          { "x": 0.583, "y": 0.194, "z": -0.446, "visibility": 0.9999, "x_px": 891, "y_px": 399 },
    "shoulder_left": { "x": 0.743, "y": 0.308, "z": -0.216, "visibility": 0.9999, "x_px": 1135, "y_px": 631 },
    "...": "33 landmarks au total"
  },
  "metrics": {
    "shoulder_angle": -8.3,
    "hip_angle": 14.7,
    "shoulder_midpoint": [412, 280],
    "hip_midpoint": [398, 680],
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

Le champ `llm_ready` est une version condensée directement injectable dans un prompt LLM (Groq, Claude, etc.) pour générer des descriptions de poses ou tester des systèmes d'analyse gestuelle.

---

## Lancer les tests

```bash
uv run pytest tests/ -v
```

---

## Images source

Le dossier `data/raw/` n'est pas inclus dans ce dépôt. Placez-y vos propres photos de poses de référence avant de lancer le pipeline. Vérifiez les licences applicables avant tout usage.

---

## Licence

MIT — voir [LICENSE](LICENSE) pour le code source.

Les images de poses de référence ne sont pas distribuées avec ce projet.
