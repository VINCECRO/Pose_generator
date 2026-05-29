from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from .face_landmark_detector import FaceLandmarkResult

_C = FaceLandmarksConnections

# ── Couleurs BGR ──────────────────────────────────────────────────────────────

_TESS    = ( 60,  60,  60)   # tessellation fond : gris sombre
_OVAL    = (200, 200, 200)   # ovale / mâchoire
_EYE     = (255, 220,  80)   # yeux
_BROW    = (100, 220, 255)   # sourcils
_NOSE    = (180, 130, 255)   # nez
_LIPS    = ( 80, 130, 255)   # lèvres
_IRIS    = (255, 255, 255)   # iris

# (connexions MediaPipe, couleur BGR, épaisseur)
_LAYERS: list[tuple[list, tuple[int, int, int], int]] = [
    (_C.FACE_LANDMARKS_TESSELATION,    _TESS, 1),
    (_C.FACE_LANDMARKS_FACE_OVAL,      _OVAL, 2),
    (_C.FACE_LANDMARKS_LEFT_EYE,       _EYE,  1),
    (_C.FACE_LANDMARKS_RIGHT_EYE,      _EYE,  1),
    (_C.FACE_LANDMARKS_LEFT_EYEBROW,   _BROW, 1),
    (_C.FACE_LANDMARKS_RIGHT_EYEBROW,  _BROW, 1),
    (_C.FACE_LANDMARKS_NOSE,           _NOSE, 1),
    (_C.FACE_LANDMARKS_LIPS,           _LIPS, 1),
    (_C.FACE_LANDMARKS_LEFT_IRIS,      _IRIS, 1),
    (_C.FACE_LANDMARKS_RIGHT_IRIS,     _IRIS, 1),
]


class FaceLandmarkRenderer:
    def render(
        self,
        result: FaceLandmarkResult,
        output_size: tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        ow, oh = output_size

        def px(idx: int) -> tuple[int, int]:
            lm = result.landmarks[idx]
            return (int(lm.x * ow), int(lm.y * oh))

        self._draw(canvas, px, len(result.landmarks))
        return canvas

    def render_on_photo(
        self,
        result: FaceLandmarkResult,
        image: np.ndarray,
    ) -> np.ndarray:
        overlay = image.copy()
        oh, ow = image.shape[:2]

        def px(idx: int) -> tuple[int, int]:
            lm = result.landmarks[idx]
            return (int(lm.x * ow), int(lm.y * oh))

        self._draw(overlay, px, len(result.landmarks))
        return overlay

    def _draw(self, canvas: np.ndarray, px_fn, n: int) -> None:
        for connections, color, thickness in _LAYERS:
            for conn in connections:
                a, b = conn.start, conn.end
                if a >= n or b >= n:
                    continue
                cv2.line(canvas, px_fn(a), px_fn(b), color, thickness, lineType=cv2.LINE_AA)

        # Points clés : centres yeux, iris, bout du nez, commissures lèvres
        for i in [1, 4, 33, 133, 263, 362, 61, 291, 0, 17, 468, 473]:
            if i < n:
                cv2.circle(canvas, px_fn(i), radius=2, color=(255, 255, 255),
                           thickness=-1, lineType=cv2.LINE_AA)
