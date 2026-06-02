"""Motion-based golf club head tracking and swing path estimation."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.impact import estimate_impact_frame
from app.services.pipeline_errors import PipelineError
from app.services.tracker import (
    TrackerConfig,
    TrackPoint,
    _create_kalman_filter,
    _distance,
    _initialize_kalman,
    _measurement_array,
    _smooth_track,
)


@dataclass(frozen=True)
class SwingTrack:
    points: list[TrackPoint]
    impact_frame_index: int | None
    impact_x: float | None
    impact_y: float | None


@dataclass(frozen=True)
class ClubTrackerConfig:
    impact_detection: bool = True
    backswing_frames: int = 24
    follow_through_frames: int = 18
    impact_pre_roll_frames: int = 2
    motion_threshold: float = 16.0
    min_motion_blob_area: int = 60
    roi_top_ratio: float = 0.22
    max_gap_frames: int = 10
    smooth_window: int = 5
    detection_gate_px: float = 140.0
    optical_flow_gate_px: float = 90.0
    kalman_confidence_decay: float = 0.08
    camera_motion_compensation: bool = True
    camera_motion_max_px: float = 35.0
    fallback_impact_search_frames: int = 45


def build_club_swing_track(
    video_path: Path,
    config: ClubTrackerConfig | None = None,
    impact_frame: int | None = None,
) -> SwingTrack:
    config = config or ClubTrackerConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError("invalid_video", f"Could not read video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if impact_frame is None and config.impact_detection:
        impact_estimate = estimate_impact_frame(video_path, fps)
        impact_frame = impact_estimate.frame_index if impact_estimate is not None else None

    motion_scores = _scan_motion_scores(capture, config)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if impact_frame is None:
        impact_frame = _estimate_impact_from_motion(motion_scores, config)

    pre_impact = config.backswing_frames + config.impact_pre_roll_frames
    track_start = max(0, (impact_frame or 0) - pre_impact)
    track_end = (
        min(frame_count - 1, (impact_frame or 0) + config.follow_through_frames)
        if impact_frame is not None
        else len(motion_scores) - 1
    )

    track: list[TrackPoint] = []
    kalman = _create_kalman_filter()
    kalman_initialized = False
    previous_gray: np.ndarray | None = None
    previous_point: np.ndarray | None = None
    frame_index = 0
    gap_frames = 0
    impact_point: tuple[float, float] | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_index < track_start:
                previous_gray = gray
                frame_index += 1
                continue

            if frame_index > track_end:
                break

            camera_motion = _estimate_camera_motion(previous_gray, gray, config)
            if camera_motion is not None and kalman_initialized:
                _shift_kalman_position(kalman, camera_motion)

            prediction: tuple[float, float] | None = None
            if kalman_initialized:
                predicted_state = kalman.predict()
                prediction = (float(predicted_state[0][0]), float(predicted_state[1][0]))

            measurement: tuple[float, float] | None = None
            confidence = 0.0
            source = "none"

            motion_point = _club_head_from_motion(
                previous_gray,
                gray,
                prediction,
                config,
            )
            if motion_point is not None and _passes_gate(
                motion_point,
                prediction,
                config.detection_gate_px,
                gap_frames,
            ):
                measurement = motion_point
                if frame_index < len(motion_scores):
                    confidence = min(0.95, motion_scores[frame_index] / 255.0)
                else:
                    confidence = 0.7
                source = "motion"
            elif previous_gray is not None and previous_point is not None:
                flow_point = _track_optical_flow(
                    previous_gray,
                    gray,
                    previous_point,
                    prediction,
                    config,
                )
                if flow_point is not None:
                    measurement = flow_point
                    confidence = 0.45
                    source = "optical_flow"

            if measurement is not None:
                if not kalman_initialized:
                    _initialize_kalman(kalman, measurement)
                    filtered_x, filtered_y = measurement
                    kalman_initialized = True
                else:
                    corrected = kalman.correct(_measurement_array(measurement))
                    filtered_x = float(corrected[0][0])
                    filtered_y = float(corrected[1][0])
                previous_point = np.array([[[filtered_x, filtered_y]]], dtype=np.float32)
                gap_frames = 0
                point = TrackPoint(
                    frame_index=frame_index,
                    x=filtered_x,
                    y=filtered_y,
                    confidence=confidence,
                    source=source,
                )
                track.append(point)
                if impact_frame is not None and frame_index == impact_frame:
                    impact_point = (filtered_x, filtered_y)
            elif (
                kalman_initialized
                and prediction is not None
                and gap_frames < config.max_gap_frames
            ):
                gap_frames += 1
                confidence = max(0.1, 0.4 - gap_frames * config.kalman_confidence_decay)
                x, y = prediction
                previous_point = np.array([[[x, y]]], dtype=np.float32)
                track.append(
                    TrackPoint(
                        frame_index=frame_index,
                        x=x,
                        y=y,
                        confidence=confidence,
                        source="kalman",
                    )
                )
                if impact_frame is not None and frame_index == impact_frame:
                    impact_point = (x, y)
            else:
                previous_point = None

            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    if not track:
        raise PipelineError(
            "club_not_detected",
            (
                "Could not track club motion in the uploaded video. "
                "Try a clearer down-the-line or face-on swing clip."
            ),
        )

    smoothed = _smooth_track(
        track,
        TrackerConfig(
            smooth_window=config.smooth_window,
            max_gap_frames=config.max_gap_frames,
        ),
    )

    resolved_impact_frame = impact_frame
    resolved_impact_x: float | None = None
    resolved_impact_y: float | None = None

    if resolved_impact_frame is not None:
        impact_candidates = [
            point for point in smoothed if point.frame_index == resolved_impact_frame
        ]
        if impact_candidates:
            resolved_impact_x = impact_candidates[0].x
            resolved_impact_y = impact_candidates[0].y
        elif impact_point is not None:
            resolved_impact_x, resolved_impact_y = impact_point
        else:
            nearest = min(
                smoothed,
                key=lambda point: abs(point.frame_index - resolved_impact_frame),
            )
            resolved_impact_x = nearest.x
            resolved_impact_y = nearest.y
    else:
        peak = max(smoothed, key=lambda point: point.confidence)
        resolved_impact_frame = peak.frame_index
        resolved_impact_x = peak.x
        resolved_impact_y = peak.y

    return SwingTrack(
        points=smoothed,
        impact_frame_index=resolved_impact_frame,
        impact_x=resolved_impact_x,
        impact_y=resolved_impact_y,
    )


def _scan_motion_scores(capture: cv2.VideoCapture, config: ClubTrackerConfig) -> list[float]:
    previous_gray: np.ndarray | None = None
    scores: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            scores.append(_motion_score(previous_gray, gray, config))
        else:
            scores.append(0.0)
        previous_gray = gray
    return scores


def _estimate_impact_from_motion(scores: list[float], config: ClubTrackerConfig) -> int | None:
    if len(scores) < 5:
        return None
    window = min(config.fallback_impact_search_frames, len(scores))
    segment = np.array(scores[:window], dtype=np.float32)
    baseline = float(np.median(segment))
    mad = float(np.median(np.abs(segment - baseline)))
    peak_index = int(np.argmax(segment))
    peak = float(segment[peak_index])
    threshold = baseline + max(4.0 * mad, 12.0)
    if peak < threshold:
        return None
    return peak_index


def _club_head_from_motion(
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
    prediction: tuple[float, float] | None,
    config: ClubTrackerConfig,
) -> tuple[float, float] | None:
    if previous_gray is None:
        return None

    diff = cv2.absdiff(previous_gray, gray)
    roi_top = int(gray.shape[0] * config.roi_top_ratio)
    diff[:roi_top, :] = 0
    _, mask = cv2.threshold(diff, int(config.motion_threshold), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates: list[tuple[float, float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_motion_blob_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        ix, iy = int(cx), int(cy)
        in_bounds = 0 <= iy < diff.shape[0] and 0 <= ix < diff.shape[1]
        motion_strength = float(np.mean(diff[iy, ix])) if in_bounds else float(area)
        candidates.append((cx, cy, area, motion_strength))

    if not candidates:
        return None

    if prediction is not None:
        ranked = sorted(
            candidates,
            key=lambda item: (
                _distance((item[0], item[1]), prediction) - item[3] * 0.05,
            ),
        )
        best = ranked[0]
        return (best[0], best[1])

    best = max(candidates, key=lambda item: item[2] * 0.5 + item[3])
    return (best[0], best[1])


def _motion_score(previous_gray: np.ndarray, gray: np.ndarray, config: ClubTrackerConfig) -> float:
    diff = cv2.absdiff(previous_gray, gray)
    roi_top = int(gray.shape[0] * config.roi_top_ratio)
    lower = diff[roi_top:, :]
    return float(np.percentile(lower, 92))


def _passes_gate(
    point: tuple[float, float],
    prediction: tuple[float, float] | None,
    gate_px: float,
    gap_frames: int,
) -> bool:
    if prediction is None:
        return True
    allowed_distance = gate_px * (1.0 + min(gap_frames, 5) * 0.2)
    return _distance(point, prediction) <= allowed_distance


def _estimate_camera_motion(
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
    config: ClubTrackerConfig,
) -> tuple[float, float] | None:
    if previous_gray is None or not config.camera_motion_compensation:
        return None

    features = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=200,
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
    if len(previous_points) < 10:
        return None

    displacements = current_points - previous_points
    median = np.median(displacements, axis=0)
    dx = float(np.clip(median[0], -config.camera_motion_max_px, config.camera_motion_max_px))
    dy = float(np.clip(median[1], -config.camera_motion_max_px, config.camera_motion_max_px))
    if np.hypot(dx, dy) < 0.25:
        return None
    return (dx, dy)


def _shift_kalman_position(kalman: cv2.KalmanFilter, motion: tuple[float, float]) -> None:
    dx, dy = motion
    for state in (kalman.statePre, kalman.statePost):
        if state is not None and state.shape[0] >= 2:
            state[0][0] = np.float32(state[0][0] + dx)
            state[1][0] = np.float32(state[1][0] + dy)


def _track_optical_flow(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    previous_point: np.ndarray,
    prediction: tuple[float, float] | None,
    config: ClubTrackerConfig,
) -> tuple[float, float] | None:
    next_point, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        previous_point,
        None,
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    if next_point is None or status is None or status[0][0] != 1:
        return None

    flow_xy = tuple(float(value) for value in next_point[0][0])
    if not _passes_gate(flow_xy, prediction, config.optical_flow_gate_px, gap_frames=0):
        return None
    return flow_xy
