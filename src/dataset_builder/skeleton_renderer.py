from __future__ import annotations

import cv2
import numpy as np

from .pose_detector import DetectionResult

# ── Connexions ────────────────────────────────────────────────────────────────

POSE_CONNECTIONS = [
    # Tronc
    (11, 12), (23, 24), (11, 23), (12, 24),
    # Bras gauche
    (11, 13), (13, 15),
    # Bras droit
    (12, 14), (14, 16),
    # Jambe gauche
    (23, 25), (25, 27),
    # Jambe droite
    (24, 26), (26, 28),
    # Tête → épaules
    (0, 11), (0, 12),
    # Visage
    (0, 1), (1, 2), (2, 3), (3, 7),    # œil gauche
    (0, 4), (4, 5), (5, 6), (6, 8),    # œil droit
    (9, 10),                            # bouche
    # Main gauche
    (15, 17), (15, 19), (15, 21), (17, 19),
    # Main droite
    (16, 18), (16, 20), (16, 22), (18, 20),
    # Pied gauche
    (27, 29), (29, 31), (27, 31),
    # Pied droit
    (28, 30), (30, 32), (28, 32),
]

# ── Couleurs BGR ──────────────────────────────────────────────────────────────

_TORSO     = (255, 255, 255)
_ARM_LEFT  = (255, 100,  50)   # bleu
_ARM_RIGHT = ( 50, 200,  50)   # vert
_LEG_LEFT  = (  0, 220, 220)   # jaune
_LEG_RIGHT = (  0, 130, 255)   # orange
_HEAD      = (180, 105, 255)   # rose

_CONNECTION_COLORS: dict[tuple[int, int], tuple[int, int, int]] = {
    # Tronc
    (11, 12): _TORSO, (23, 24): _TORSO, (11, 23): _TORSO, (12, 24): _TORSO,
    # Bras gauche
    (11, 13): _ARM_LEFT,  (13, 15): _ARM_LEFT,
    # Bras droit
    (12, 14): _ARM_RIGHT, (14, 16): _ARM_RIGHT,
    # Jambe gauche
    (23, 25): _LEG_LEFT,  (25, 27): _LEG_LEFT,
    # Jambe droite
    (24, 26): _LEG_RIGHT, (26, 28): _LEG_RIGHT,
    # Tête
    (0, 11): _HEAD, (0, 12): _HEAD,
    # Visage
    (0, 1): _HEAD, (1, 2): _HEAD, (2, 3): _HEAD, (3, 7): _HEAD,
    (0, 4): _HEAD, (4, 5): _HEAD, (5, 6): _HEAD, (6, 8): _HEAD,
    (9, 10): _HEAD,
    # Main gauche
    (15, 17): _ARM_LEFT,  (15, 19): _ARM_LEFT,  (15, 21): _ARM_LEFT,  (17, 19): _ARM_LEFT,
    # Main droite
    (16, 18): _ARM_RIGHT, (16, 20): _ARM_RIGHT, (16, 22): _ARM_RIGHT, (18, 20): _ARM_RIGHT,
    # Pied gauche
    (27, 29): _LEG_LEFT,  (29, 31): _LEG_LEFT,  (27, 31): _LEG_LEFT,
    # Pied droit
    (28, 30): _LEG_RIGHT, (30, 32): _LEG_RIGHT, (28, 32): _LEG_RIGHT,
}

_VISIBILITY_THRESHOLD = 0.5   # en dessous → lien en pointillé


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_dashed_line(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 8,
    gap_len: int = 6,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = max(int(np.hypot(x2 - x1, y2 - y1)), 1)
    step = dash_len + gap_len
    for start in range(0, length, step):
        t0 = start / length
        t1 = min((start + dash_len) / length, 1.0)
        pa = (int(x1 + t0 * (x2 - x1)), int(y1 + t0 * (y2 - y1)))
        pb = (int(x1 + t1 * (x2 - x1)), int(y1 + t1 * (y2 - y1)))
        cv2.line(canvas, pa, pb, color, thickness, lineType=cv2.LINE_AA)


class SkeletonRenderer:
    def render(
        self,
        result: DetectionResult,
        output_size: tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        ow, oh = output_size

        def px(idx: int) -> tuple[int, int]:
            lm = result.landmarks[idx]
            return (int(lm.x * ow), int(lm.y * oh))

        self._draw_skeleton(canvas, px, result.confidence_scores)
        return canvas

    def render_on_photo(
        self,
        result: DetectionResult,
        original_image: np.ndarray,
    ) -> np.ndarray:
        overlay = original_image.copy()
        oh, ow = original_image.shape[:2]

        def px(idx: int) -> tuple[int, int]:
            lm = result.landmarks[idx]
            return (int(lm.x * ow), int(lm.y * oh))

        self._draw_skeleton(overlay, px, result.confidence_scores)
        return overlay

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        px_fn,
        confidence_scores: dict[int, float],
    ) -> None:
        n = len(confidence_scores) if confidence_scores else 33

        for start, end in POSE_CONNECTIONS:
            if start >= n or end >= n:
                continue
            p1 = px_fn(start)
            p2 = px_fn(end)
            color = _CONNECTION_COLORS.get((start, end), _TORSO)

            vis_start = confidence_scores.get(start, 0.0)
            vis_end   = confidence_scores.get(end,   0.0)
            visible   = vis_start >= _VISIBILITY_THRESHOLD and vis_end >= _VISIBILITY_THRESHOLD

            if visible:
                cv2.line(canvas, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
            else:
                _draw_dashed_line(canvas, p1, p2, color, thickness=2)

        # Points — taille proportionnelle à la visibilité
        for i in range(n):
            vis = confidence_scores.get(i, 0.0)
            if vis >= _VISIBILITY_THRESHOLD:
                cv2.circle(canvas, px_fn(i), radius=5, color=(255, 255, 255),
                           thickness=-1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(canvas, px_fn(i), radius=3, color=(80, 80, 80),
                           thickness=1, lineType=cv2.LINE_AA)
