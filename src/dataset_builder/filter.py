from __future__ import annotations

from pathlib import Path

from .pose_detector import DetectionResult

KEY_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28]


class PoseFilter:
    def __init__(self, confidence_threshold: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold

    def is_valid(self, result: DetectionResult) -> bool:
        for idx in KEY_LANDMARKS:
            score = result.confidence_scores.get(idx, 0.0)
            if score < self.confidence_threshold:
                return False
        return True

    def filter_batch(
        self,
        results: list[tuple[Path, DetectionResult | None]],
    ) -> list[Path]:
        kept = []
        for path, result in results:
            if result is not None and self.is_valid(result):
                kept.append(path)
        return kept

    def generate_report(self, total: int, kept: int) -> str:
        rejected = total - kept
        rate = (kept / total * 100) if total > 0 else 0.0
        return (
            f"Pipeline report: {total} images processed — "
            f"{kept} kept ({rate:.1f}%), {rejected} rejected"
        )
