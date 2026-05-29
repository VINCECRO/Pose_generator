from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from PIL import Image

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_MODEL_NAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_model() -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / _MODEL_NAME
    if not path.exists():
        print(f"Téléchargement du modèle FaceLandmarker ({_MODEL_NAME})…")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print(f"Modèle sauvegardé : {path}")
    return path


@dataclass
class FaceLandmark:
    x: float
    y: float
    z: float


@dataclass
class FaceLandmarkResult:
    landmarks: list[FaceLandmark]   # 478 points, coordonnées normalisées
    image_width: int
    image_height: int
    blendshapes: dict[str, float] = field(default_factory=dict)

    def px(self, idx: int) -> tuple[int, int]:
        lm = self.landmarks[idx]
        return (int(lm.x * self.image_width), int(lm.y * self.image_height))


class FaceLandmarkDetector:
    def __init__(self, min_confidence: float = 0.5) -> None:
        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, image_path: Path) -> FaceLandmarkResult | None:
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size
        frame = np.array(pil_img)

        mp_image = MpImage(image_format=ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        raw = result.face_landmarks[0]
        landmarks = [FaceLandmark(lm.x, lm.y, lm.z) for lm in raw]

        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                blendshapes[cat.category_name] = round(cat.score, 4)

        return FaceLandmarkResult(
            landmarks=landmarks,
            image_width=w,
            image_height=h,
            blendshapes=blendshapes,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
