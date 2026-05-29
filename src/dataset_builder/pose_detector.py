from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from PIL import Image

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"

_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


def _ensure_model(complexity: int) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / _MODEL_NAMES[complexity]
    if not path.exists():
        url = _MODEL_URLS[complexity]
        print(f"Téléchargement du modèle MediaPipe ({path.name})…")
        urllib.request.urlretrieve(url, path)
        print(f"Modèle sauvegardé : {path}")
    return path


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 0.0


@dataclass
class DetectionResult:
    landmarks: list[Landmark]
    landmarks_3d: list[Landmark]
    image_width: int
    image_height: int
    confidence_scores: dict[int, float] = field(default_factory=dict)


class PoseDetector:
    def __init__(self, model_complexity: int = 2, min_confidence: float = 0.3) -> None:
        model_path = _ensure_model(model_complexity)
        options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, image_path: Path) -> DetectionResult | None:
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size
        frame = np.array(pil_img)

        mp_image = MpImage(image_format=ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        raw = result.pose_landmarks[0]
        raw_3d = result.pose_world_landmarks[0] if result.pose_world_landmarks else raw

        landmarks = [Landmark(lm.x, lm.y, lm.z, lm.visibility) for lm in raw]
        landmarks_3d = [Landmark(lm.x, lm.y, lm.z, lm.visibility) for lm in raw_3d]
        confidence_scores = {i: lm.visibility for i, lm in enumerate(raw)}

        return DetectionResult(
            landmarks=landmarks,
            landmarks_3d=landmarks_3d,
            image_width=w,
            image_height=h,
            confidence_scores=confidence_scores,
        )

    def detect_batch(self, image_paths: list[Path]) -> list[DetectionResult | None]:
        return [self.detect(p) for p in image_paths]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
