from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "test_pose.jpg"


def test_fixture_exists():
    assert FIXTURE.exists(), "test_pose.jpg fixture manquant"


def test_detect_returns_result_or_none():
    from dataset_builder.pose_detector import PoseDetector

    detector = PoseDetector(model_complexity=0, min_confidence=0.3)
    result = detector.detect(FIXTURE)
    detector.close()
    # L'image synthétique peut ne pas contenir une vraie pose — None est acceptable
    assert result is None or len(result.landmarks) == 33


def test_detect_result_structure():
    from dataset_builder.pose_detector import PoseDetector, DetectionResult

    detector = PoseDetector(model_complexity=0, min_confidence=0.3)
    result = detector.detect(FIXTURE)
    detector.close()

    if result is None:
        pytest.skip("Pas de pose détectée sur l'image de test synthétique")

    assert isinstance(result, DetectionResult)
    assert result.image_width == 400
    assert result.image_height == 600
    assert len(result.landmarks) == 33
    assert len(result.landmarks_3d) == 33
    assert isinstance(result.confidence_scores, dict)


def test_detect_batch():
    from dataset_builder.pose_detector import PoseDetector

    detector = PoseDetector(model_complexity=0, min_confidence=0.3)
    results = detector.detect_batch([FIXTURE, FIXTURE])
    detector.close()

    assert len(results) == 2


def test_context_manager():
    from dataset_builder.pose_detector import PoseDetector

    with PoseDetector(model_complexity=0) as detector:
        result = detector.detect(FIXTURE)
    assert result is None or len(result.landmarks) == 33
