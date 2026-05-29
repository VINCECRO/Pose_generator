import math

import pytest

from dataset_builder.pose_detector import DetectionResult, Landmark
from dataset_builder.metrics_extractor import MetricsExtractor, PoseMetrics


def _make_landmark(x: float, y: float, z: float = 0.0, vis: float = 0.95) -> Landmark:
    return Landmark(x=x, y=y, z=z, visibility=vis)


def _make_result(positions: dict[int, tuple[float, float]]) -> DetectionResult:
    """Build a DetectionResult with 33 landmarks. Positions are (x,y) normalized 0→1."""
    lms = []
    for i in range(33):
        x, y = positions.get(i, (0.5, 0.5))
        lms.append(_make_landmark(x, y))
    return DetectionResult(
        landmarks=lms,
        landmarks_3d=lms,
        image_width=800,
        image_height=1200,
        confidence_scores={i: 0.95 for i in range(33)},
    )


def _straight_pose() -> DetectionResult:
    return _make_result({
        0:  (0.5, 0.05),   # nez
        11: (0.4, 0.25),   # épaule gauche
        12: (0.6, 0.25),   # épaule droite
        23: (0.4, 0.55),   # hanche gauche
        24: (0.6, 0.55),   # hanche droite
        25: (0.4, 0.75),   # genou gauche
        26: (0.6, 0.75),   # genou droit
        27: (0.4, 0.92),   # cheville gauche
        28: (0.6, 0.92),   # cheville droite
        13: (0.3, 0.45),   # coude gauche
        14: (0.7, 0.45),   # coude droit
        15: (0.25, 0.60),  # poignet gauche
        16: (0.75, 0.60),  # poignet droit
    })


def test_extract_returns_pose_metrics():
    extractor = MetricsExtractor()
    result = _straight_pose()
    metrics = extractor.extract(result)
    assert isinstance(metrics, PoseMetrics)


def test_shoulder_angle_horizontal():
    extractor = MetricsExtractor()
    result = _straight_pose()
    metrics = extractor.extract(result)
    # Épaules alignées horizontalement → angle ≈ 0°
    assert abs(metrics.shoulder_angle) < 5.0


def test_hip_angle_horizontal():
    extractor = MetricsExtractor()
    result = _straight_pose()
    metrics = extractor.extract(result)
    assert abs(metrics.hip_angle) < 5.0


def test_shoulder_midpoint():
    extractor = MetricsExtractor()
    result = _straight_pose()
    metrics = extractor.extract(result)
    # Milieu épaules attendu : x=0.5*800=400, y=0.25*1200=300
    assert abs(metrics.shoulder_midpoint[0] - 400) <= 2
    assert abs(metrics.shoulder_midpoint[1] - 300) <= 2


def test_hip_midpoint():
    extractor = MetricsExtractor()
    result = _straight_pose()
    metrics = extractor.extract(result)
    assert abs(metrics.hip_midpoint[0] - 400) <= 2
    assert abs(metrics.hip_midpoint[1] - 660) <= 2


def test_weight_balanced_for_straight_pose():
    extractor = MetricsExtractor()
    result = _straight_pose()
    metrics = extractor.extract(result)
    assert metrics.weight_side == "balanced"


def test_spine_straight_for_upright_pose():
    extractor = MetricsExtractor()
    result = _straight_pose()
    metrics = extractor.extract(result)
    assert metrics.spine_curve in ("straight", "slight_S")


def test_to_json_keys():
    extractor = MetricsExtractor()
    metrics = extractor.extract(_straight_pose())
    j = metrics.to_json()
    expected_keys = {
        "shoulder_angle", "hip_angle", "shoulder_midpoint", "hip_midpoint",
        "spine_vector", "center_of_gravity", "weight_side", "spine_curve", "support_type",
    }
    assert expected_keys.issubset(j.keys())


def test_to_llm_prompt_keys():
    extractor = MetricsExtractor()
    metrics = extractor.extract(_straight_pose())
    llm = metrics.to_llm_prompt()
    assert set(llm.keys()) == {"bascule_epaules", "bascule_bassin", "centre_gravite", "courbe_colonne", "appui"}


def test_tilted_shoulder_angle():
    result = _make_result({
        0:  (0.5, 0.05),
        11: (0.35, 0.20),  # épaule gauche plus haute
        12: (0.65, 0.30),  # épaule droite plus basse
        23: (0.4, 0.55),
        24: (0.6, 0.55),
        25: (0.4, 0.75),
        26: (0.6, 0.75),
        27: (0.4, 0.92),
        28: (0.6, 0.92),
    })
    extractor = MetricsExtractor()
    metrics = extractor.extract(result)
    # Angle doit être non nul (pente positive)
    assert abs(metrics.shoulder_angle) > 1.0
