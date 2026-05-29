from __future__ import annotations

import math
from dataclasses import dataclass

from .pose_detector import DetectionResult


@dataclass
class PoseMetrics:
    shoulder_angle: float
    hip_angle: float
    shoulder_midpoint: tuple[int, int]
    hip_midpoint: tuple[int, int]
    spine_vector: tuple[int, int]
    center_of_gravity: tuple[int, int]
    weight_side: str
    spine_curve: str
    support_type: str

    def to_json(self) -> dict:
        return {
            "shoulder_angle": round(self.shoulder_angle, 2),
            "hip_angle": round(self.hip_angle, 2),
            "shoulder_midpoint": list(self.shoulder_midpoint),
            "hip_midpoint": list(self.hip_midpoint),
            "spine_vector": list(self.spine_vector),
            "center_of_gravity": list(self.center_of_gravity),
            "weight_side": self.weight_side,
            "spine_curve": self.spine_curve,
            "support_type": self.support_type,
        }

    def to_llm_prompt(self) -> dict:
        shoulder_dir = "droite" if self.shoulder_angle < 0 else "gauche"
        hip_dir = "droite" if self.hip_angle < 0 else "gauche"

        spine_map = {
            "straight": "droite",
            "slight_S": "legere_S",
            "pronounced_S": "S_prononcee",
            "C_curve": "courbe_C",
        }
        support_map = {
            "bipodal": "bipodal",
            "left_dominant": "monopodal_gauche",
            "right_dominant": "monopodal_droit",
        }
        gravity_map = {
            "left": "jambe_gauche",
            "right": "jambe_droite",
            "balanced": "centre",
        }

        return {
            "bascule_epaules": f"{shoulder_dir}_{abs(round(self.shoulder_angle))}deg",
            "bascule_bassin": f"{hip_dir}_{abs(round(self.hip_angle))}deg",
            "centre_gravite": gravity_map.get(self.weight_side, "centre"),
            "courbe_colonne": spine_map.get(self.spine_curve, self.spine_curve),
            "appui": support_map.get(self.support_type, self.support_type),
        }


def _angle_from_horizontal(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def _midpoint(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


class MetricsExtractor:
    def extract(self, result: DetectionResult) -> PoseMetrics:
        w, h = result.image_width, result.image_height
        lm = result.landmarks

        def px(idx: int) -> tuple[int, int]:
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        shoulder_l, shoulder_r = px(11), px(12)
        hip_l, hip_r = px(23), px(24)
        ankle_l, ankle_r = px(27), px(28)
        knee_l, knee_r = px(25), px(26)

        shoulder_mid = _midpoint(shoulder_l, shoulder_r)
        hip_mid = _midpoint(hip_l, hip_r)

        shoulder_angle = _angle_from_horizontal(shoulder_l, shoulder_r)
        hip_angle = _angle_from_horizontal(hip_l, hip_r)

        spine_vector = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])

        center_of_gravity = hip_mid

        ankle_mid_x = (ankle_l[0] + ankle_r[0]) / 2
        cog_x = center_of_gravity[0]
        offset_ratio = (cog_x - ankle_mid_x) / max(abs(ankle_r[0] - ankle_l[0]), 1)
        if offset_ratio < -0.2:
            weight_side = "left"
        elif offset_ratio > 0.2:
            weight_side = "right"
        else:
            weight_side = "balanced"

        support_type = _classify_support(ankle_l, ankle_r, center_of_gravity)
        spine_curve = _classify_spine_curve(shoulder_mid, hip_mid, knee_l, knee_r)

        return PoseMetrics(
            shoulder_angle=shoulder_angle,
            hip_angle=hip_angle,
            shoulder_midpoint=shoulder_mid,
            hip_midpoint=hip_mid,
            spine_vector=spine_vector,
            center_of_gravity=center_of_gravity,
            weight_side=weight_side,
            spine_curve=spine_curve,
            support_type=support_type,
        )


def _classify_support(
    ankle_l: tuple[int, int],
    ankle_r: tuple[int, int],
    cog: tuple[int, int],
) -> str:
    ankle_span = abs(ankle_r[0] - ankle_l[0])
    if ankle_span < 20:
        return "bipodal"
    cog_x = cog[0]
    mid_x = (ankle_l[0] + ankle_r[0]) / 2
    offset = (cog_x - mid_x) / max(ankle_span, 1)
    if offset < -0.25:
        return "left_dominant"
    elif offset > 0.25:
        return "right_dominant"
    return "bipodal"


def _classify_spine_curve(
    shoulder_mid: tuple[int, int],
    hip_mid: tuple[int, int],
    knee_l: tuple[int, int],
    knee_r: tuple[int, int],
) -> str:
    knee_mid_x = (knee_l[0] + knee_r[0]) / 2
    spine_mid_x = (shoulder_mid[0] + hip_mid[0]) / 2
    lateral_deviation = abs(knee_mid_x - spine_mid_x)

    spine_len = math.hypot(
        hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]
    )
    if spine_len == 0:
        return "straight"

    ratio = lateral_deviation / spine_len
    if ratio < 0.05:
        return "straight"
    elif ratio < 0.15:
        return "slight_S"
    elif ratio < 0.30:
        return "pronounced_S"
    return "C_curve"
