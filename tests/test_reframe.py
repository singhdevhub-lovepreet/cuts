from __future__ import annotations

import pytest

from cuts.edl import CropKeyframe
from cuts.nodes.reframe import CropPathPlanner


def test_crop_bounds_clamp_on_wide_source_keeps_vertical_center_fixed() -> None:
    planner = CropPathPlanner()
    clamped_x, clamped_y = planner.clamp_center(0.95, 0.12, 1920, 1080)
    assert clamped_x == pytest.approx(0.841796875)
    assert clamped_y == pytest.approx(0.5)


def test_crop_smoothing_limits_velocity() -> None:
    planner = CropPathPlanner(ema_alpha=1.0, max_velocity_per_second=0.1)
    points = [
        CropKeyframe(t=0.0, center_x=0.2, center_y=0.5),
        CropKeyframe(t=1.0, center_x=0.9, center_y=0.5),
    ]
    smoothed = planner.smooth(points, 1920, 1080)
    assert smoothed[0].center_x == pytest.approx(0.2)
    assert smoothed[1].center_x == pytest.approx(0.3)
    assert smoothed[1].center_y == pytest.approx(0.5)
