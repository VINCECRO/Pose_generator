# gesture-ai-dataset

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

Python pipeline for extracting keypoints and geometric metrics from a reference photo bank of human poses. For each valid image, it produces an OpenPose-style skeleton ready for ControlNet and a structured JSON usable in LLM prompts. Part of the **GestureAI** project — a gesture drawing learning tool.

---

## Pipeline

```
Photo bank (data/raw/)
         │
         ▼
MediaPipe PoseLandmarker
(33 full-body landmarks)
         │
         ▼
Quality filter (confidence > 0.8)
         │
    ┌────┴──────────────────┐
    ▼                       ▼
OpenPose skeleton       Metrics JSON
(data/skeletons/)       (angles, center of gravity,
    │                    33 landmarks + llm_ready)
    ▼                   (data/metrics/)
ControlNet + Stable Diffusion
    ▼
Royalty-free ML dataset
```

**Face sub-pipeline (from filtered images):**

```
data/raw/
    │
    ▼
FaceDetector (BlazeFace full-range)
    │
    ▼
Head + neck crop (data/faces/)
    │
    ▼
FaceLandmarker (478 landmarks + blendshapes)
    │
    ▼
Overlays + JSON (data/face_landmarks/)
```

---

## Status

### Done

- **Full pose pipeline** — detection, quality filtering, OpenPose skeleton rendering (black background 512×512 + debug overlay), geometric metrics extraction (shoulder/hip angles, center of gravity, support type, spine curve), JSON export with `llm_ready` field
- **Full face sub-pipeline** — head+neck detection and cropping (calibrated margins, landmark fallback if detector fails), 478 facial landmarks + 52 blendshapes detection, tessellation + contour rendering (photo overlay + black background mesh)
- **Idempotency** — all three scripts skip already-processed images; runs can be resumed without reprocessing
- **Test suite** — detection on a real image, metric calculations on known landmarks, skeleton render verification

### In progress

- Building the source photo bank (`data/raw/`)
- Qualitative validation of skeleton outputs for ControlNet use

### Next steps

1. **Synthetic dataset generation** — condition Stable Diffusion via ControlNet on the extracted skeletons to produce royalty-free images
2. **Metrics enrichment** — add volumetric orientations (head, torso, pelvis in 3D from z landmarks) to improve training prompts
3. **Specialized detection model** — train a Loomis construction recognition model (proportions, head/ribcage/pelvis volumes) on the generated dataset
4. **Main app integration** — expose pose analysis as a service called by GestureAI to annotate displayed poses in real time

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- No GPU required — MediaPipe runs entirely on CPU

---

## Installation

```bash
git clone https://github.com/ton-user/gesture-ai-dataset
cd gesture-ai-dataset
uv sync
```

MediaPipe models are downloaded automatically into `models/` on first run.

---

## Usage

Place your reference images in `data/raw/`, then run the three steps in order:

### 1. Pose pipeline

```bash
# Process all images in data/raw/
uv run python scripts/run_pipeline.py

# Options
uv run python scripts/run_pipeline.py --confidence 0.85   # confidence threshold (default 0.8)
uv run python scripts/run_pipeline.py --quiet             # no progress bar
uv run python scripts/run_pipeline.py --input /path/to/images
```

| Option | Default | Description |
|---|---|---|
| `--input` | `data/raw` | Source image folder |
| `--skeletons` | `data/skeletons` | Skeleton output folder |
| `--metrics` | `data/metrics` | Metrics JSON output folder |
| `--filtered` | `data/filtered` | Retained images copied here |
| `--confidence` | `0.8` | MediaPipe confidence threshold (0.0–1.0) |
| `--quiet` | — | Disable progress bar |

### 2. Face cropping

```bash
uv run python scripts/run_face_crop.py                    # data/raw/ → data/faces/
uv run python scripts/run_face_crop.py --confidence 0.3   # detection threshold (default)
uv run python scripts/run_face_crop.py --debug            # save bounding boxes
```

### 3. Face landmarks

```bash
uv run python scripts/run_face_landmarks.py               # data/faces/ → data/face_landmarks/
uv run python scripts/run_face_landmarks.py --no-json     # overlays only, no JSON
```

All three scripts are **idempotent** — re-running does not reprocess already produced files.

---

## Output structure

```
data/
├── filtered/                         ← retained images (confidence > 0.8)
├── skeletons/
│   ├── pose_001.png                  ← OpenPose skeleton, black background 512×512 (ControlNet)
│   └── pose_001_overlay.png          ← skeleton overlaid on photo (debug)
├── metrics/
│   └── pose_001.json                 ← metrics + 33 landmarks + llm_ready field
├── faces/
│   └── pose_001_face0.jpg            ← head + neck crop
└── face_landmarks/
    ├── pose_001_face0_face.json         ← 478 landmarks + 52 blendshapes
    ├── pose_001_face0_face_overlay.png  ← landmarks drawn on photo
    └── pose_001_face0_face_mesh.png     ← mesh on black background 512×512
```

### Metrics JSON format

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
    "...": "33 landmarks total"
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

The `llm_ready` field is a condensed version directly injectable into an LLM prompt (Groq, Claude, etc.) to generate pose descriptions or test gesture analysis systems.

---

## Running tests

```bash
uv run pytest tests/ -v
```

---

## Source images

The `data/raw/` folder is not included in this repository. Place your own reference pose photos there before running the pipeline. Check applicable licenses before using images from third-party sources.

---

## License

MIT — see [LICENSE](LICENSE) for the source code.

Reference pose images are not distributed with this project.
