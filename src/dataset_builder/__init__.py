from __future__ import annotations


def __getattr__(name: str):
    if name in ("PoseDetector", "DetectionResult", "Landmark"):
        from .pose_detector import PoseDetector, DetectionResult, Landmark
        globals()["PoseDetector"] = PoseDetector
        globals()["DetectionResult"] = DetectionResult
        globals()["Landmark"] = Landmark
        return globals()[name]
    if name in ("MetricsExtractor", "PoseMetrics"):
        from .metrics_extractor import MetricsExtractor, PoseMetrics
        globals()["MetricsExtractor"] = MetricsExtractor
        globals()["PoseMetrics"] = PoseMetrics
        return globals()[name]
    if name == "SkeletonRenderer":
        from .skeleton_renderer import SkeletonRenderer
        globals()["SkeletonRenderer"] = SkeletonRenderer
        return SkeletonRenderer
    if name == "PoseFilter":
        from .filter import PoseFilter
        globals()["PoseFilter"] = PoseFilter
        return PoseFilter
    if name in ("run_pipeline", "PipelineReport"):
        from .pipeline import run_pipeline, PipelineReport
        globals()["run_pipeline"] = run_pipeline
        globals()["PipelineReport"] = PipelineReport
        return globals()[name]
    raise AttributeError(f"module 'dataset_builder' has no attribute {name!r}")


__all__ = [
    "PoseDetector",
    "DetectionResult",
    "Landmark",
    "MetricsExtractor",
    "PoseMetrics",
    "SkeletonRenderer",
    "PoseFilter",
    "run_pipeline",
    "PipelineReport",
]
