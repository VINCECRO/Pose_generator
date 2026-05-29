import numpy as np
import pytest

from dataset_builder.pose_detector import DetectionResult, Landmark
from dataset_builder.skeleton_renderer import SkeletonRenderer


def _make_landmark(x: float, y: float) -> Landmark:
    return Landmark(x=x, y=y, z=0.0, visibility=0.95)


def _make_spread_result() -> DetectionResult:
    """33 landmarks spread across the canvas so skeleton is visible."""
    positions = {
        0:  (0.5,  0.05),
        1:  (0.48, 0.08),
        2:  (0.52, 0.08),
        3:  (0.46, 0.10),
        4:  (0.54, 0.10),
        5:  (0.44, 0.10),
        6:  (0.56, 0.10),
        7:  (0.43, 0.12),
        8:  (0.57, 0.12),
        9:  (0.48, 0.15),
        10: (0.52, 0.15),
        11: (0.35, 0.28),   # épaule gauche
        12: (0.65, 0.28),   # épaule droite
        13: (0.25, 0.44),
        14: (0.75, 0.44),
        15: (0.20, 0.60),
        16: (0.80, 0.60),
        17: (0.19, 0.62),
        18: (0.81, 0.62),
        19: (0.18, 0.64),
        20: (0.82, 0.64),
        21: (0.21, 0.61),
        22: (0.79, 0.61),
        23: (0.38, 0.58),   # hanche gauche
        24: (0.62, 0.58),   # hanche droite
        25: (0.37, 0.75),
        26: (0.63, 0.75),
        27: (0.36, 0.90),
        28: (0.64, 0.90),
        29: (0.35, 0.93),
        30: (0.65, 0.93),
        31: (0.34, 0.95),
        32: (0.66, 0.95),
    }
    lms = []
    for i in range(33):
        x, y = positions.get(i, (0.5, 0.5))
        lms.append(_make_landmark(x, y))
    return DetectionResult(
        landmarks=lms,
        landmarks_3d=lms,
        image_width=512,
        image_height=512,
        confidence_scores={i: 0.95 for i in range(33)},
    )


def test_render_output_shape():
    renderer = SkeletonRenderer()
    result = _make_spread_result()
    img = renderer.render(result, output_size=(512, 512))
    assert img.shape == (512, 512, 3)


def test_render_custom_size():
    renderer = SkeletonRenderer()
    result = _make_spread_result()
    img = renderer.render(result, output_size=(256, 256))
    assert img.shape == (256, 256, 3)


def test_render_not_all_black():
    renderer = SkeletonRenderer()
    result = _make_spread_result()
    img = renderer.render(result, output_size=(512, 512))
    assert img.max() > 0, "Le rendu squelette est entièrement noir"


def test_render_returns_numpy_array():
    renderer = SkeletonRenderer()
    result = _make_spread_result()
    img = renderer.render(result)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8


def test_render_on_photo_preserves_shape():
    renderer = SkeletonRenderer()
    result = _make_spread_result()
    original = np.ones((480, 640, 3), dtype=np.uint8) * 128
    overlay = renderer.render_on_photo(result, original)
    assert overlay.shape == (480, 640, 3)


def test_render_on_photo_modifies_image():
    renderer = SkeletonRenderer()
    result = _make_spread_result()
    original = np.zeros((512, 512, 3), dtype=np.uint8)
    overlay = renderer.render_on_photo(result, original)
    assert overlay.max() > 0, "L'overlay est entièrement noir"


def test_render_background_is_black():
    renderer = SkeletonRenderer()
    result = _make_spread_result()
    img = renderer.render(result, output_size=(512, 512))
    # Corner pixel (0,0) should be black (no skeleton there)
    assert img[0, 0].tolist() == [0, 0, 0]
