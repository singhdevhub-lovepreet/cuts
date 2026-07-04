from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cuts.edl import CropKeyframe
from cuts.nodes.reframe import (
    CropPathPlanner,
    ReframeNode,
    SubjectDetection,
    _max_frames_to_track,
)


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


def test_max_frames_to_track_is_inclusive() -> None:
    assert _max_frames_to_track(0, 0) == 0
    assert _max_frames_to_track(1, 37) == 35


class _FakePredictor:
    def __init__(self) -> None:
        self.add_frame_idx: int | None = None
        self.propagate_calls: list[tuple[int | None, int | None, bool]] = []

    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        async_loading_frames: bool = False,
    ) -> object:
        return object()

    def add_new_points_or_box(
        self,
        inference_state: object,
        frame_idx: int,
        obj_id: int,
        points: object = None,
        labels: object = None,
        clear_old_points: bool = True,
        normalize_coords: bool = False,
        box: object = None,
    ) -> object:
        self.add_frame_idx = frame_idx
        return object()

    def propagate_in_video(
        self,
        inference_state: object,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], object]]:
        self.propagate_calls.append((start_frame_idx, max_frame_num_to_track, reverse))
        return iter(())


def test_track_subject_clamps_sam2_indices_to_extracted_frames(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(36):
        (frame_dir / f"{index:06d}.jpg").write_bytes(b"stub")

    node = ReframeNode()
    predictor = _FakePredictor()
    seed = SubjectDetection(
        frame_idx=999,
        box=(10.0, 10.0, 20.0, 20.0),
        score=1.0,
    )

    tracked = node._track_subject(
        predictor=predictor,
        frame_dir=frame_dir,
        start_frame=0,
        end_frame=173,
        frame_width=1920,
        frame_height=1080,
        seed=seed,
    )

    assert tracked == {}
    assert predictor.add_frame_idx == 35
    assert predictor.propagate_calls == [
        (35, 35, True),
        (35, 35, False),
    ]
