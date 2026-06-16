"""Scene motion and horizon estimation for stable tracer rendering."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.pipeline_errors import PipelineError


@dataclass(frozen=True)
class SceneFrameTransform:
    frame_index: int
    cumulative_dx: float
    cumulative_dy: float
    horizon_y: float


@dataclass(frozen=True)
class SceneStabilizationConfig:
    enabled: bool = True
    max_motion_px: float = 35.0
    horizon_ratio: float = 0.42
    horizon_smoothing: float = 0.85


def estimate_scene_transforms(
    video_path: Path,
    config: SceneStabilizationConfig | None = None,
) -> list[SceneFrameTransform]:
    config = config or SceneStabilizationConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError("invalid_video", f"Could not read video: {video_path}")

    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fallback_horizon = height * config.horizon_ratio
    transforms: list[SceneFrameTransform] = []
    previous_gray: np.ndarray | None = None
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    smoothed_horizon = fallback_horizon
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if config.enabled:
                motion = _estimate_camera_motion(previous_gray, gray, config.max_motion_px)
                if motion is not None:
                    cumulative_dx += motion[0]
                    cumulative_dy += motion[1]

            horizon_y = _estimate_horizon_y(gray, fallback_horizon)
            smoothed_horizon = (
                smoothed_horizon * config.horizon_smoothing
                + horizon_y * (1.0 - config.horizon_smoothing)
            )
            transforms.append(
                SceneFrameTransform(
                    frame_index=frame_index,
                    cumulative_dx=cumulative_dx,
                    cumulative_dy=cumulative_dy,
                    horizon_y=smoothed_horizon,
                ),
            )
            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    return transforms


def _estimate_camera_motion(
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
    max_motion_px: float,
) -> tuple[float, float] | None:
    if previous_gray is None:
        return None

    features = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
    )
    if features is None:
        return None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        features,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return None

    previous_points = features[status.ravel() == 1].reshape(-1, 2)
    current_points = next_points[status.ravel() == 1].reshape(-1, 2)
    if len(previous_points) < 12:
        return None

    displacements = current_points - previous_points
    median = np.median(displacements, axis=0)
    dx = float(np.clip(median[0], -max_motion_px, max_motion_px))
    dy = float(np.clip(median[1], -max_motion_px, max_motion_px))
    if np.hypot(dx, dy) < 0.25:
        return None
    return (dx, dy)


def _estimate_horizon_y(gray: np.ndarray, fallback_horizon: float) -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 140)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=max(40, gray.shape[1] // 4),
        maxLineGap=18,
    )
    if lines is None:
        return fallback_horizon

    candidates: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = x2 - x1
        if dx == 0:
            continue
        slope = abs((y2 - y1) / dx)
        length = float(np.hypot(dx, y2 - y1))
        midpoint_y = (y1 + y2) / 2.0
        if slope <= 0.12 and length >= gray.shape[1] * 0.25 and midpoint_y <= gray.shape[0] * 0.8:
            candidates.append(midpoint_y)

    if not candidates:
        return fallback_horizon
    return float(np.median(candidates))
