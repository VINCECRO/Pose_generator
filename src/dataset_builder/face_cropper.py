from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
from PIL import Image

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_MODEL_NAME = "blaze_face_full_range.tflite"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)

_PAD_TOP    = 0.65   # crâne entier au-dessus du front
_PAD_SIDES  = 0.50   # air de chaque côté
_PAD_BOTTOM = 1.40   # cou complet + amorce des épaules

_FACE_LANDMARK_KEYS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
]


def _ensure_model() -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / _MODEL_NAME
    if not path.exists():
        print(f"Téléchargement du modèle FaceDetector ({_MODEL_NAME})…")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print(f"Modèle sauvegardé : {path}")
    return path


@dataclass
class CropResult:
    image: np.ndarray
    source_path: Path
    face_index: int = 0
    method: str = "detector"   # "detector" | "landmarks"


class FaceCropper:
    def __init__(
        self,
        min_confidence: float = 0.3,
        metrics_dir: Path | None = None,
    ) -> None:
        model_path = _ensure_model()
        options = FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = FaceDetector.create_from_options(options)
        self._metrics_dir = metrics_dir

    def crop(self, image_path: Path, debug_dir: Path | None = None) -> list[CropResult]:
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size
        frame = np.array(pil_img)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mp_image = MpImage(image_format=ImageFormat.SRGB, data=frame)
        result = self._detector.detect(mp_image)

        crops: list[CropResult] = []

        if result.detections:
            if debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)

            for idx, detection in enumerate(result.detections):
                bb = detection.bounding_box
                fx, fy = bb.origin_x, bb.origin_y
                fw, fh = bb.width, bb.height

                pad_top    = int(fh * _PAD_TOP)
                pad_sides  = int(fw * _PAD_SIDES)
                pad_bottom = int(fh * _PAD_BOTTOM)

                x1 = max(0, fx - pad_sides)
                y1 = max(0, fy - pad_top)
                x2 = min(w, fx + fw + pad_sides)
                y2 = min(h, fy + fh + pad_bottom)

                if debug_dir:
                    dbg = bgr.copy()
                    cv2.rectangle(dbg, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    conf = detection.categories[0].score if detection.categories else 0.0
                    cv2.putText(dbg, f"det {conf:.2f}", (fx, max(0, fy - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imwrite(str(debug_dir / f"{image_path.stem}_dbg{idx}.jpg"), dbg)

                crop_bgr = bgr[y1:y2, x1:x2]
                if crop_bgr.size > 0:
                    crops.append(CropResult(
                        image=crop_bgr, source_path=image_path,
                        face_index=idx, method="detector",
                    ))

        # Fallback : recadrage depuis les landmarks de pose déjà calculés
        if not crops and self._metrics_dir:
            crops = self._crop_from_landmarks(image_path, bgr, w, h, debug_dir)

        return crops

    def _crop_from_landmarks(
        self,
        image_path: Path,
        bgr: np.ndarray,
        w: int,
        h: int,
        debug_dir: Path | None,
    ) -> list[CropResult]:
        json_path = self._metrics_dir / (image_path.stem + ".json")
        if not json_path.exists():
            return []

        data = json.loads(json_path.read_text())
        landmarks = data.get("landmarks", {})

        xs, ys = [], []
        for key in _FACE_LANDMARK_KEYS:
            lm = landmarks.get(key)
            if lm and lm.get("visibility", 0) >= 0.3:
                xs.append(lm["x_px"])
                ys.append(lm["y_px"])

        if len(xs) < 3:
            return []

        fx, fy = min(xs), min(ys)
        fw = max(xs) - fx
        fh = max(ys) - fy

        if fw < 5 or fh < 5:
            return []

        pad_top    = int(fh * _PAD_TOP)
        pad_sides  = int(fw * _PAD_SIDES)
        pad_bottom = int(fh * _PAD_BOTTOM)

        x1 = max(0, fx - pad_sides)
        y1 = max(0, fy - pad_top)
        x2 = min(w, fx + fw + pad_sides)
        y2 = min(h, fy + fh + pad_bottom)

        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            dbg = bgr.copy()
            cv2.rectangle(dbg, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(dbg, "landmarks", (fx, max(0, fy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imwrite(str(debug_dir / f"{image_path.stem}_dbg_lm.jpg"), dbg)

        crop_bgr = bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return []

        return [CropResult(
            image=crop_bgr, source_path=image_path,
            face_index=0, method="landmarks",
        )]

    def crop_batch(
        self,
        image_paths: list[Path],
        debug_dir: Path | None = None,
    ) -> list[CropResult]:
        results = []
        for p in image_paths:
            results.extend(self.crop(p, debug_dir=debug_dir))
        return results

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "FaceCropper":
        return self

    def __exit__(self, *_) -> None:
        self.close()
