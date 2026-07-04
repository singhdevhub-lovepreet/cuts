from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuts.domain import MotionWasteSegment
from cuts.graph import Context, Node


@dataclass(slots=True)
class MotionAnalysisResult:
    waste_segments: list[MotionWasteSegment]


class MotionNode(Node):
    name = "motion"
    requires = ("clips",)
    provides = ("motion_segments",)

    def run(self, context: Context) -> Context:
        context.motion_segments = []
        for clip in context.clips:
            context.motion_segments.extend(
                self._analyze_clip(clip.path, clip.clip_id, clip.duration, context)
            )
        return context

    def _analyze_clip(
        self, path: Path, clip_id: str, duration: float, context: Context
    ) -> list[MotionWasteSegment]:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError:
            return []

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return []

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 1:
            capture.release()
            return []

        sample_stride = max(1, int(round(fps * context.config.motion_stride_seconds)))
        window_frames = max(2, int(round(fps * context.config.motion_window_seconds)))
        samples: list[tuple[float, float, float]] = []
        previous_gray = None
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_stride != 0:
                frame_index += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            motion = 0.0
            if previous_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    0.5,
                    1,
                    12,
                    2,
                    5,
                    1.2,
                    0,
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                motion = float(np.mean(magnitude))
            samples.append((frame_index / fps, sharpness, motion))
            previous_gray = gray
            frame_index += 1

        capture.release()
        if not samples:
            return []

        waste_segments: list[MotionWasteSegment] = []
        for index in range(0, len(samples), window_frames):
            window = samples[index : index + window_frames]
            if not window:
                continue
            start = window[0][0]
            end = min(duration, window[-1][0] + context.config.motion_window_seconds)
            avg_sharpness = sum(item[1] for item in window) / len(window)
            avg_motion = sum(item[2] for item in window) / len(window)
            if (
                avg_sharpness < context.config.motion_sharpness_threshold
                or avg_motion > context.config.motion_flow_threshold
            ):
                score = (context.config.motion_sharpness_threshold - avg_sharpness) / max(
                    context.config.motion_sharpness_threshold, 1.0
                ) + avg_motion
                reason = (
                    "blur"
                    if avg_sharpness < context.config.motion_sharpness_threshold
                    else "heavy-motion"
                )
                waste_segments.append(
                    MotionWasteSegment(
                        clip_id=clip_id, start=start, end=end, score=score, reason=reason
                    )
                )
        return waste_segments
