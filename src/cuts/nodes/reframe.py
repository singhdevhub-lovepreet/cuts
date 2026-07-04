from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from cuts.edl import CropKeyframe
from cuts.graph import Context, Node

if TYPE_CHECKING:
    import numpy as np


@dataclass(slots=True, frozen=True)
class SubjectDetection:
    frame_idx: int
    box: tuple[float, float, float, float]
    score: float


@dataclass(slots=True, frozen=True)
class CropBounds:
    min_center_x: float
    max_center_x: float
    min_center_y: float
    max_center_y: float


@dataclass(slots=True)
class CropPathPlanner:
    crop_aspect: float = 9.0 / 16.0
    sample_step_seconds: float = 0.5
    ema_alpha: float = 0.35
    max_velocity_per_second: float = 0.65

    def bounds(self, frame_width: int, frame_height: int) -> CropBounds:
        if frame_width <= 0 or frame_height <= 0:
            return CropBounds(0.5, 0.5, 0.5, 0.5)
        source_aspect = frame_width / frame_height
        if source_aspect >= self.crop_aspect:
            crop_width_fraction = self.crop_aspect / source_aspect
            crop_height_fraction = 1.0
        else:
            crop_width_fraction = 1.0
            crop_height_fraction = source_aspect / self.crop_aspect
        min_center_x = 0.5 if crop_width_fraction >= 1.0 else crop_width_fraction / 2.0
        max_center_x = 0.5 if crop_width_fraction >= 1.0 else 1.0 - crop_width_fraction / 2.0
        min_center_y = 0.5 if crop_height_fraction >= 1.0 else crop_height_fraction / 2.0
        max_center_y = 0.5 if crop_height_fraction >= 1.0 else 1.0 - crop_height_fraction / 2.0
        return CropBounds(min_center_x, max_center_x, min_center_y, max_center_y)

    def clamp_center(
        self, center_x: float, center_y: float, frame_width: int, frame_height: int
    ) -> tuple[float, float]:
        bounds = self.bounds(frame_width, frame_height)
        return (
            min(max(center_x, bounds.min_center_x), bounds.max_center_x),
            min(max(center_y, bounds.min_center_y), bounds.max_center_y),
        )

    def smooth(
        self, points: list[CropKeyframe], frame_width: int, frame_height: int
    ) -> list[CropKeyframe]:
        if not points:
            return []
        ordered = sorted(points, key=lambda item: (item.t, item.center_x, item.center_y))
        smoothed: list[CropKeyframe] = []
        prev_t = ordered[0].t
        prev_x, prev_y = self.clamp_center(
            ordered[0].center_x, ordered[0].center_y, frame_width, frame_height
        )
        smoothed.append(CropKeyframe(t=prev_t, center_x=prev_x, center_y=prev_y))
        for point in ordered[1:]:
            clamped_x, clamped_y = self.clamp_center(
                point.center_x, point.center_y, frame_width, frame_height
            )
            dt = max(point.t - prev_t, 1e-6)
            ema_x = prev_x + self.ema_alpha * (clamped_x - prev_x)
            ema_y = prev_y + self.ema_alpha * (clamped_y - prev_y)
            max_step = self.max_velocity_per_second * dt
            next_x = self._limit_step(prev_x, ema_x, max_step)
            next_y = self._limit_step(prev_y, ema_y, max_step)
            smoothed.append(CropKeyframe(t=point.t, center_x=next_x, center_y=next_y))
            prev_t = point.t
            prev_x = next_x
            prev_y = next_y
        return smoothed

    def _limit_step(self, previous: float, target: float, max_step: float) -> float:
        delta = target - previous
        if delta > max_step:
            return previous + max_step
        if delta < -max_step:
            return previous - max_step
        return target


class _CaptureLike(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802

    def get(self, prop_id: int) -> float: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray]: ...

    def release(self) -> None: ...


class _OpenCVModule(Protocol):
    CAP_PROP_POS_FRAMES: int
    COLOR_BGR2RGB: int

    def cvtColor(self, src: np.ndarray, code: int) -> np.ndarray: ...  # noqa: N802


class _RelativeBoundingBoxLike(Protocol):
    xmin: float
    ymin: float
    width: float
    height: float


class _FaceLocationDataLike(Protocol):
    relative_bounding_box: _RelativeBoundingBoxLike


class _FaceDetectionLike(Protocol):
    score: list[float]
    location_data: _FaceLocationDataLike


class _FaceResultsLike(Protocol):
    detections: list[_FaceDetectionLike] | None


class _PoseLandmarkLike(Protocol):
    x: float
    y: float
    visibility: float


class _PoseLandmarksLike(Protocol):
    landmark: list[_PoseLandmarkLike]


class _PoseResultsLike(Protocol):
    pose_landmarks: _PoseLandmarksLike | None


class _FaceDetectorLike(Protocol):
    def process(self, image: np.ndarray) -> _FaceResultsLike: ...


class _PoseDetectorLike(Protocol):
    def process(self, image: np.ndarray) -> _PoseResultsLike: ...


class _SAM2Predictor(Protocol):
    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = ...,
        offload_state_to_cpu: bool = ...,
        async_loading_frames: bool = ...,
    ) -> object: ...

    def add_new_points_or_box(
        self,
        inference_state: object,
        frame_idx: int,
        obj_id: int,
        points: object = ...,
        labels: object = ...,
        clear_old_points: bool = ...,
        normalize_coords: bool = ...,
        box: object = ...,
    ) -> object: ...

    def propagate_in_video(
        self,
        inference_state: object,
        start_frame_idx: int | None = ...,
        max_frame_num_to_track: int | None = ...,
        reverse: bool = ...,
    ) -> Iterator[tuple[int, list[int], object]]: ...


class ReframeNode(Node):
    name = "reframe"
    requires = ("timeline",)
    provides = ()

    def __init__(self) -> None:
        self._planner = CropPathPlanner()
        self._predictor: _SAM2Predictor | None = None
        self._predictor_unavailable = False

    def run(self, context: Context) -> Context:
        timeline = context.timeline
        if timeline is None:
            return context
        for clip in timeline.clips:
            crop_path = self._build_crop_path_for_clip(
                clip.source_path, clip.source_in, clip.source_out
            )
            if crop_path:
                clip.crop_aspect = self._planner.crop_aspect
                clip.crop_path = crop_path
        context.timeline = timeline
        return context

    def _build_crop_path_for_clip(
        self, source_path: Path, source_in: float, source_out: float
    ) -> list[CropKeyframe]:
        predictor = self._load_predictor()
        if predictor is None:
            return []
        try:
            import cv2
        except ImportError:
            return []

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            return []
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
                return []

            start_frame = max(0, int(round(source_in * fps)))
            end_frame = min(frame_count - 1, int(round(source_out * fps)))
            if end_frame <= start_frame:
                return []

            seed = self._find_subject_seed(
                capture, start_frame, end_frame, frame_width, frame_height, fps
            )
            if seed is None:
                return []

            tracked_centers = self._track_subject(
                predictor=predictor,
                frame_dir=self._sam2_frame_dir(source_path, source_in, source_out),
                start_frame=start_frame,
                end_frame=end_frame,
                frame_width=frame_width,
                frame_height=frame_height,
                seed=seed,
            )
            if not tracked_centers:
                return []

            duration = source_out - source_in
            sample_count = max(2, ceil(duration / self._planner.sample_step_seconds) + 1)
            sample_times = self._sample_times(duration, sample_count)
            fallback_center = (
                (seed.box[0] + seed.box[2]) / 2.0 / frame_width,
                (seed.box[1] + seed.box[3]) / 2.0 / frame_height,
            )
            sampled_points = [
                CropKeyframe(
                    t=sample_time,
                    center_x=self._center_for_time(
                        source_in + sample_time,
                        fps,
                        tracked_centers,
                        fallback_center,
                        axis=0,
                    ),
                    center_y=self._center_for_time(
                        source_in + sample_time,
                        fps,
                        tracked_centers,
                        fallback_center,
                        axis=1,
                    ),
                )
                for sample_time in sample_times
            ]
            return self._planner.smooth(sampled_points, frame_width, frame_height)
        finally:
            capture.release()

    def _load_predictor(self) -> _SAM2Predictor | None:
        if self._predictor_unavailable:
            return None
        if self._predictor is not None:
            return self._predictor
        checkpoint = self._default_checkpoint_path()
        if not checkpoint.exists():
            self._predictor_unavailable = True
            return None
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            self._predictor_unavailable = True
            return None
        try:
            predictor = build_sam2_video_predictor(
                "sam2_hiera_t.yaml",
                ckpt_path=str(checkpoint),
                device="cpu",
                apply_postprocessing=False,
                vos_optimized=False,
            )
        except Exception:
            self._predictor_unavailable = True
            return None
        self._predictor = predictor
        return predictor

    def _default_checkpoint_path(self) -> Path:
        return Path.home() / ".cache" / "cuts" / "sam2" / "sam2_hiera_tiny.pt"

    def _find_subject_seed(
        self,
        capture: _CaptureLike,
        start_frame: int,
        end_frame: int,
        frame_width: int,
        frame_height: int,
        fps: float,
    ) -> SubjectDetection | None:
        sample_span = min(end_frame - start_frame, max(1, int(round(fps * 1.5))))
        candidate_frames = sorted(
            {start_frame + int(round(index * sample_span / 4)) for index in range(5)}
        )
        try:
            import cv2 as cv2_module
            import mediapipe as mp
        except ImportError:
            return None
        cv2 = cast(_OpenCVModule, cv2_module)
        best: SubjectDetection | None = None
        with (
            mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            ) as face_detector,
            mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            ) as pose_detector,
        ):
            for frame_idx in candidate_frames:
                frame = self._read_frame(capture, frame_idx)
                if frame is None:
                    continue
                detection = self._detect_subject(
                    frame,
                    frame_idx,
                    frame_width,
                    frame_height,
                    cv2,
                    cast(_FaceDetectorLike, face_detector),
                    cast(_PoseDetectorLike, pose_detector),
                )
                if detection is None:
                    continue
                if best is None or detection.score > best.score:
                    best = detection
        return best

    def _detect_subject(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_width: int,
        frame_height: int,
        cv2: _OpenCVModule,
        face_detector: _FaceDetectorLike,
        pose_detector: _PoseDetectorLike,
    ) -> SubjectDetection | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        best_detection: SubjectDetection | None = None

        face_results = face_detector.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                left = max(0.0, bounding_box.xmin * frame_width)
                top = max(0.0, bounding_box.ymin * frame_height)
                right = min(
                    float(frame_width),
                    (bounding_box.xmin + bounding_box.width) * frame_width,
                )
                bottom = min(
                    float(frame_height),
                    (bounding_box.ymin + bounding_box.height) * frame_height,
                )
                width = right - left
                height = bottom - top
                area = width * height
                center_x = left + width / 2.0
                center_y = top + height / 2.0
                center_distance = (
                    abs(center_x - frame_center_x) / frame_width
                    + abs(center_y - frame_center_y) / frame_height
                )
                score = float(detection.score[0]) * area * (1.0 - 0.25 * center_distance)
                candidate = SubjectDetection(
                    frame_idx=frame_idx,
                    box=(left, top, right, bottom),
                    score=score,
                )
                if best_detection is None or candidate.score > best_detection.score:
                    best_detection = candidate

        pose_results = pose_detector.process(rgb_frame)
        if pose_results.pose_landmarks:
            xs: list[float] = []
            ys: list[float] = []
            for landmark in pose_results.pose_landmarks.landmark:
                if landmark.visibility < 0.3:
                    continue
                xs.append(landmark.x * frame_width)
                ys.append(landmark.y * frame_height)
            if xs and ys:
                left = max(0.0, min(xs))
                right = min(float(frame_width), max(xs))
                top = max(0.0, min(ys))
                bottom = min(float(frame_height), max(ys))
                width = right - left
                height = bottom - top
                if width > 0.0 and height > 0.0:
                    center_x = left + width / 2.0
                    center_y = top + height / 2.0
                    center_distance = (
                        abs(center_x - frame_center_x) / frame_width
                        + abs(center_y - frame_center_y) / frame_height
                    )
                    score = width * height * (1.0 - 0.25 * center_distance)
                    candidate = SubjectDetection(
                        frame_idx=frame_idx,
                        box=(left, top, right, bottom),
                        score=score,
                    )
                    if best_detection is None or candidate.score > best_detection.score:
                        best_detection = candidate

        return best_detection

    def _track_subject(
        self,
        predictor: _SAM2Predictor,
        frame_dir: Path,
        start_frame: int,
        end_frame: int,
        frame_width: int,
        frame_height: int,
        seed: SubjectDetection,
    ) -> dict[int, tuple[float, float]]:
        try:
            import numpy as np
        except ImportError:
            return {}
        inference_state = predictor.init_state(
            str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        box = np.asarray(seed.box, dtype=float)
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=seed.frame_idx,
            obj_id=1,
            box=box,
        )
        tracked_centers: dict[int, tuple[float, float]] = {}
        max_frames_to_track = end_frame - start_frame
        for reverse in (True, False):
            for frame_idx, object_ids, masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=seed.frame_idx,
                max_frame_num_to_track=max_frames_to_track,
                reverse=reverse,
            ):
                if not object_ids:
                    continue
                center = self._mask_to_center(masks, frame_width, frame_height)
                if center is not None:
                    tracked_centers[frame_idx] = center
        return tracked_centers

    def _sam2_frame_dir(self, source_path: Path, source_in: float, source_out: float) -> Path:
        cache_dir = Path.home() / ".cache" / "cuts" / "sam2" / "segments"
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_in_text = f"{source_in:.6f}".rstrip("0").rstrip(".")
        duration = source_out - source_in
        duration_text = f"{duration:.6f}".rstrip("0").rstrip(".")
        cache_key = f"{source_path.resolve()}::{source_in_text}::{duration_text}"
        frame_dir = cache_dir / hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
        if frame_dir.exists() and any(frame_dir.iterdir()):
            return frame_dir
        frame_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            source_in_text,
            "-t",
            duration_text,
            "-i",
            str(source_path),
            "-an",
            "-vsync",
            "0",
            "-q:v",
            "2",
            str(frame_dir / "%06d.jpg"),
        ]
        subprocess.run(command, check=True)
        return frame_dir

    def _mask_to_center(
        self, masks: object, frame_width: int, frame_height: int
    ) -> tuple[float, float] | None:
        try:
            import numpy as np
        except ImportError:
            return None
        array = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        if array.ndim == 4:
            array = array[0, 0]
        elif array.ndim == 3:
            array = array[0]
        if array.size == 0:
            return None
        ys, xs = np.nonzero(array > 0.0)
        if xs.size == 0 or ys.size == 0:
            return None
        return float(xs.mean()) / frame_width, float(ys.mean()) / frame_height

    def _sample_times(self, duration: float, count: int) -> list[float]:
        if count <= 1:
            return [0.0]
        step = duration / (count - 1)
        return [round(index * step, 6) for index in range(count)]

    def _center_for_time(
        self,
        time_s: float,
        fps: float,
        tracked_centers: dict[int, tuple[float, float]],
        fallback_center: tuple[float, float],
        axis: int,
    ) -> float:
        frame_idx = self._time_to_frame(time_s, fps)
        if frame_idx in tracked_centers:
            return tracked_centers[frame_idx][axis]
        nearest_frame = min(
            tracked_centers,
            key=lambda candidate: (abs(candidate - frame_idx), candidate),
        )
        return tracked_centers.get(nearest_frame, fallback_center)[axis]

    def _read_frame(self, capture: _CaptureLike, frame_idx: int) -> np.ndarray | None:
        import cv2 as cv2_module

        cv2 = cast(_OpenCVModule, cv2_module)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx)):
            return None
        success, frame = capture.read()
        if not success:
            return None
        return frame

    def _time_to_frame(self, time_s: float, fps: float) -> int:
        return max(0, int(round(time_s * fps)))
