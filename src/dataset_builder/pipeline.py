from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from .filter import PoseFilter
from .metrics_extractor import MetricsExtractor
from .pose_detector import PoseDetector
from .skeleton_renderer import SkeletonRenderer

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_LANDMARK_NAMES = {
    0:  "nose",
    1:  "left_eye_inner",   2:  "left_eye",       3:  "left_eye_outer",
    4:  "right_eye_inner",  5:  "right_eye",       6:  "right_eye_outer",
    7:  "left_ear",         8:  "right_ear",
    9:  "mouth_left",       10: "mouth_right",
    11: "shoulder_left",    12: "shoulder_right",
    13: "elbow_left",       14: "elbow_right",
    15: "wrist_left",       16: "wrist_right",
    17: "pinky_left",       18: "pinky_right",
    19: "index_left",       20: "index_right",
    21: "thumb_left",       22: "thumb_right",
    23: "hip_left",         24: "hip_right",
    25: "knee_left",        26: "knee_right",
    27: "ankle_left",       28: "ankle_right",
    29: "heel_left",        30: "heel_right",
    31: "foot_index_left",  32: "foot_index_right",
}


@dataclass
class PipelineReport:
    total: int
    kept: int
    rejected: int
    skipped: int

    @property
    def success_rate(self) -> float:
        processed = self.total - self.skipped
        return (self.kept / processed * 100) if processed > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Pipeline finished — "
            f"{self.total} found, {self.skipped} skipped (already done), "
            f"{self.kept} kept, {self.rejected} rejected "
            f"({self.success_rate:.1f}% pass rate)"
        )


def run_pipeline(
    input_dir: Path = Path("data/raw"),
    skeleton_dir: Path = Path("data/skeletons"),
    metrics_dir: Path = Path("data/metrics"),
    filtered_dir: Path = Path("data/filtered"),
    confidence_threshold: float = 0.8,
    verbose: bool = True,
) -> PipelineReport:
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    # Seuil bas pour la détection : laisser MediaPipe tenter sur toutes les images.
    # Le filtre de qualité (PoseFilter) fait la sélection stricte ensuite.
    detector = PoseDetector(min_confidence=0.3)
    pose_filter = PoseFilter(confidence_threshold=confidence_threshold)
    extractor = MetricsExtractor()
    renderer = SkeletonRenderer()

    total = len(image_paths)
    skipped = 0
    kept = 0
    rejected = 0

    iterator = tqdm(image_paths, desc="Processing poses", disable=not verbose)

    for image_path in iterator:
        json_path = metrics_dir / (image_path.stem + ".json")
        if json_path.exists():
            skipped += 1
            continue

        result = detector.detect(image_path)

        if result is None or not pose_filter.is_valid(result):
            rejected += 1
            continue

        metrics = extractor.extract(result)

        skeleton = renderer.render(result, output_size=(512, 512))
        skeleton_path = skeleton_dir / (image_path.stem + ".png")
        cv2.imwrite(str(skeleton_path), skeleton)

        original = cv2.imread(str(image_path))
        if original is not None:
            overlay = renderer.render_on_photo(result, original)
            overlay_path = skeleton_dir / (image_path.stem + "_overlay.png")
            cv2.imwrite(str(overlay_path), overlay)

        payload = {
            "source_image": image_path.name,
            "image_dimensions": {
                "width": result.image_width,
                "height": result.image_height,
            },
            "detection_confidence": {
                "shoulder_left": round(result.confidence_scores.get(11, 0.0), 4),
                "shoulder_right": round(result.confidence_scores.get(12, 0.0), 4),
                "hip_left": round(result.confidence_scores.get(23, 0.0), 4),
                "hip_right": round(result.confidence_scores.get(24, 0.0), 4),
            },
            "landmarks": {
                _LANDMARK_NAMES[i]: {
                    "x":          round(result.landmarks[i].x, 6),
                    "y":          round(result.landmarks[i].y, 6),
                    "z":          round(result.landmarks[i].z, 6),
                    "visibility": round(result.confidence_scores.get(i, 0.0), 4),
                    "x_px":       int(result.landmarks[i].x * result.image_width),
                    "y_px":       int(result.landmarks[i].y * result.image_height),
                }
                for i in range(33)
            },
            "metrics": metrics.to_json(),
            "llm_ready": metrics.to_llm_prompt(),
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        shutil.copy2(image_path, filtered_dir / image_path.name)
        kept += 1

    detector.close()

    report = PipelineReport(
        total=total,
        kept=kept,
        rejected=rejected,
        skipped=skipped,
    )
    if verbose:
        print(report)
    return report
