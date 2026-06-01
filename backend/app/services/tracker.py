from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.detector import Detection, GolfBallDetector
from app.services.pipeline_errors import PipelineError


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    x: float
    y: float
    confidence: float
    source: str


@dataclass(frozen=True)
class TrackerConfig:
    max_gap_frames: int = 12
    detection_gate_px: float = 120.0
    optical_flow_gate_px: float = 80.0
    optical_flow_max_error: float = 24.0
    optical_flow_forward_backward_px: float = 3.0
    kalman_confidence_decay: float = 0.08
    smooth_window: int = 5
    stationary_address_frames: int = 4
    stationary_address_radius_px: float = 8.0
    swing_motion_min_px: float = 2.0
    swing_motion_roi_px: int = 140
    swing_launch_speed_px: float = 7.0
    stale_track_frames: int = 30
    stale_track_radius_px: float = 80.0
    synthetic_launch_frames: int = 45
    synthetic_launch_upward_bias: float = 0.85
    camera_motion_compensation: bool = True
    camera_motion_max_px: float = 35.0


def build_video_track(
    video_path: Path,
    detector: GolfBallDetector,
    config: TrackerConfig | None = None,
) -> list[TrackPoint]:
    config = config or TrackerConfig()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError("invalid_video", f"Could not read video: {video_path}")

    track: list[TrackPoint] = []
    kalman = _create_kalman_filter()
    kalman_initialized = False
    previous_gray: np.ndarray | None = None
    previous_point: np.ndarray | None = None
    frame_index = 0
    gap_frames = 0
    address_point: tuple[float, float] | None = None
    previous_detection_point: tuple[float, float] | None = None
    stationary_address_hits = 0
    stale_track_hits = 0
    launch_vector: tuple[float, float] | None = None
    synthetic_launch_remaining = 0
    synthetic_launch_started = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            camera_motion = _estimate_camera_motion(previous_gray, gray, config)
            if camera_motion is not None:
                if address_point is not None:
                    address_point = _add_points(address_point, camera_motion)
                if previous_detection_point is not None:
                    previous_detection_point = _add_points(previous_detection_point, camera_motion)
                if kalman_initialized:
                    _shift_kalman_position(kalman, camera_motion)

            detections = detector.detect_frame(frame)
            best_detection = _best_detection(detections)
            prediction: tuple[float, float] | None = None
            if kalman_initialized:
                predicted_state = kalman.predict()
                prediction = (float(predicted_state[0][0]), float(predicted_state[1][0]))

            measurement: tuple[float, float] | None = None
            confidence = 0.0
            source = "none"
            swing_vector = _estimate_swing_vector(previous_gray, gray, address_point, config)

            if best_detection is not None:
                x, y = best_detection.center
                detection_point = (x, y)
                if address_point is None:
                    address_point = detection_point
                stale_track_hits = _next_stale_track_hits(
                    detection_point,
                    previous_detection_point,
                    address_point,
                    stale_track_hits,
                    config,
                )
                previous_detection_point = detection_point
                is_stuck_at_address = _is_stuck_at_address(
                    detection_point,
                    address_point,
                    stationary_address_hits,
                    stale_track_hits,
                    swing_vector,
                    config,
                )

                if is_stuck_at_address:
                    launch_vector = _launch_vector_from_swing(swing_vector, config) or launch_vector
                    if not synthetic_launch_started:
                        synthetic_launch_remaining = config.synthetic_launch_frames
                        synthetic_launch_started = True
                elif _passes_gate(
                    detection_point,
                    prediction,
                    config.detection_gate_px,
                    gap_frames,
                ):
                    measurement = detection_point
                    confidence = best_detection.confidence
                    source = "yolo"
                    if (
                        _distance(detection_point, address_point)
                        <= config.stationary_address_radius_px
                    ):
                        stationary_address_hits += 1
                    else:
                        launch_vector = _unit_vector_from_points(address_point, detection_point)
                        stationary_address_hits = 0
                else:
                    launch_vector = _launch_vector_from_swing(swing_vector, config) or launch_vector
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
                    confidence = 0.5
                    source = "optical_flow"
                    if address_point is not None:
                        launch_vector = _unit_vector_from_points(address_point, flow_point)
                    stationary_address_hits = 0

            if (
                measurement is None
                and kalman_initialized
                and launch_vector is not None
                and synthetic_launch_remaining > 0
            ):
                prediction = _nudge_prediction_from_launch_vector(
                    kalman,
                    prediction,
                    launch_vector,
                    config,
                )
                synthetic_launch_remaining -= 1
                synthetic_prediction = True
            else:
                synthetic_prediction = False

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
                track.append(
                    TrackPoint(
                        frame_index=frame_index,
                        x=filtered_x,
                        y=filtered_y,
                        confidence=confidence,
                        source=source,
                    )
                )
            elif (
                kalman_initialized
                and prediction is not None
                and (gap_frames < config.max_gap_frames or synthetic_prediction)
            ):
                gap_frames += 1
                confidence = (
                    0.35
                    if synthetic_prediction
                    else max(0.1, 0.45 - gap_frames * config.kalman_confidence_decay)
                )
                x, y = prediction
                previous_point = np.array([[[x, y]]], dtype=np.float32)
                track.append(
                    TrackPoint(
                        frame_index=frame_index,
                        x=x,
                        y=y,
                        confidence=confidence,
                        source="synthetic_launch" if synthetic_prediction else "kalman",
                    )
                )
            else:
                previous_point = None

            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    if not track:
        raise PipelineError(
            "ball_not_detected",
            "No golf ball detections were found in the uploaded video.",
        )

    return _smooth_track(track, config)


def _best_detection(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return max(detections, key=lambda detection: detection.confidence)


def _create_kalman_filter() -> cv2.KalmanFilter:
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        np.float32,
    )
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        np.float32,
    )
    kalman.processNoiseCov = np.diag([8.0, 8.0, 120.0, 120.0]).astype(np.float32)
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 6.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


def _initialize_kalman(kalman: cv2.KalmanFilter, point: tuple[float, float]) -> None:
    x, y = point
    state = np.array([[np.float32(x)], [np.float32(y)], [0.0], [0.0]], dtype=np.float32)
    kalman.statePre = state.copy()
    kalman.statePost = state.copy()


def _measurement_array(point: tuple[float, float]) -> np.ndarray:
    x, y = point
    return np.array([[np.float32(x)], [np.float32(y)]])


def _passes_gate(
    point: tuple[float, float],
    prediction: tuple[float, float] | None,
    gate_px: float,
    gap_frames: int,
) -> bool:
    if prediction is None:
        return True
    if gap_frames >= 5:
        return True
    allowed_distance = gate_px * (1.0 + min(gap_frames, 5) * 0.25)
    return _distance(point, prediction) <= allowed_distance


def _is_stuck_at_address(
    point: tuple[float, float],
    address_point: tuple[float, float],
    stationary_address_hits: int,
    stale_track_hits: int,
    swing_vector: tuple[float, float] | None,
    config: TrackerConfig,
) -> bool:
    address_stuck = (
        stationary_address_hits >= config.stationary_address_frames
        and _distance(point, address_point) <= config.stationary_address_radius_px
    )
    stale_near_address = stale_track_hits >= config.stale_track_frames and _distance(
        point,
        address_point,
    ) <= config.stale_track_radius_px
    return (address_stuck or stale_near_address) and (
        swing_vector is not None or stale_track_hits >= config.stale_track_frames
    )


def _next_stale_track_hits(
    point: tuple[float, float],
    previous_point: tuple[float, float] | None,
    address_point: tuple[float, float],
    current_hits: int,
    config: TrackerConfig,
) -> int:
    del previous_point
    is_near_address = _distance(point, address_point) <= config.stale_track_radius_px
    return current_hits + 1 if is_near_address else 0


def _estimate_swing_vector(
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
    address_point: tuple[float, float] | None,
    config: TrackerConfig,
) -> tuple[float, float] | None:
    if previous_gray is None or address_point is None:
        return None

    x, y = address_point
    radius = config.swing_motion_roi_px
    height, width = gray.shape[:2]
    x1 = max(0, int(round(x - radius)))
    y1 = max(0, int(round(y - radius)))
    x2 = min(width, int(round(x + radius)))
    y2 = min(height, int(round(y + radius)))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    previous_roi = previous_gray[y1:y2, x1:x2]
    current_roi = gray[y1:y2, x1:x2]
    features = cv2.goodFeaturesToTrack(
        previous_roi,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=5,
    )
    if features is None:
        return None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_roi,
        current_roi,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return None

    good_previous = features[status.ravel() == 1].reshape(-1, 2)
    good_next = next_points[status.ravel() == 1].reshape(-1, 2)
    if len(good_previous) < 3:
        return None

    displacements = good_next - good_previous
    magnitudes = np.linalg.norm(displacements, axis=1)
    moving = displacements[magnitudes >= config.swing_motion_min_px]
    if len(moving) < 2:
        return None

    # Use the strongest local motion as a rough club-path proxy through impact.
    strongest_count = max(2, len(moving) // 4)
    strongest_indices = np.argsort(np.linalg.norm(moving, axis=1))[-strongest_count:]
    vector = np.median(moving[strongest_indices], axis=0)
    norm = float(np.linalg.norm(vector))
    if norm < config.swing_motion_min_px:
        return None
    return (float(vector[0] / norm), float(vector[1] / norm))


def _estimate_camera_motion(
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
    config: TrackerConfig,
) -> tuple[float, float] | None:
    if previous_gray is None or not config.camera_motion_compensation:
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


def _add_points(
    point: tuple[float, float],
    motion: tuple[float, float],
) -> tuple[float, float]:
    return (point[0] + motion[0], point[1] + motion[1])


def _nudge_prediction_from_launch_vector(
    kalman: cv2.KalmanFilter,
    prediction: tuple[float, float] | None,
    launch_vector: tuple[float, float],
    config: TrackerConfig,
) -> tuple[float, float] | None:
    if prediction is None:
        return None

    vx = np.float32(launch_vector[0] * config.swing_launch_speed_px)
    vy = np.float32(launch_vector[1] * config.swing_launch_speed_px)
    x = np.float32(prediction[0])
    y = np.float32(prediction[1])
    state = np.array([[x], [y], [vx], [vy]], dtype=np.float32)
    kalman.statePost = state
    return (float(x), float(y))


def _launch_vector_from_swing(
    swing_vector: tuple[float, float] | None,
    config: TrackerConfig,
) -> tuple[float, float]:
    # Camera footage usually shows the ball climbing upward after impact; blend local
    # club motion with an upward bias so stale address detections cannot pin the trace.
    if swing_vector is None:
        raw = np.array([0.0, -1.0], dtype=np.float32)
    else:
        raw = np.array(
            [
                swing_vector[0],
                swing_vector[1] - config.synthetic_launch_upward_bias,
            ],
            dtype=np.float32,
        )
    norm = float(np.linalg.norm(raw))
    if norm == 0:
        return (0.0, -1.0)
    return (float(raw[0] / norm), float(raw[1] / norm))


def _unit_vector_from_points(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float] | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        return None
    return (dx / norm, dy / norm)


def _track_optical_flow(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    previous_point: np.ndarray,
    prediction: tuple[float, float] | None,
    config: TrackerConfig,
) -> tuple[float, float] | None:
    next_point, status, error = cv2.calcOpticalFlowPyrLK(
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
    if error is not None and float(error[0][0]) > config.optical_flow_max_error:
        return None

    back_point, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        next_point,
        None,
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    if back_point is None or back_status is None or back_status[0][0] != 1:
        return None
    previous_xy = tuple(float(value) for value in previous_point[0][0])
    back_xy = tuple(float(value) for value in back_point[0][0])
    if _distance(previous_xy, back_xy) > config.optical_flow_forward_backward_px:
        return None

    flow_xy = tuple(float(value) for value in next_point[0][0])
    if not _passes_gate(flow_xy, prediction, config.optical_flow_gate_px, gap_frames=0):
        return None
    return flow_xy


def _smooth_track(track: list[TrackPoint], config: TrackerConfig) -> list[TrackPoint]:
    if config.smooth_window <= 1 or len(track) < 3:
        return track

    radius = max(1, config.smooth_window // 2)
    smoothed: list[TrackPoint] = []
    for index, point in enumerate(track):
        start = max(0, index - radius)
        end = min(len(track), index + radius + 1)
        neighbors = [
            neighbor
            for neighbor in track[start:end]
            if abs(neighbor.frame_index - point.frame_index) <= config.max_gap_frames
        ]
        weights = np.array([max(neighbor.confidence, 0.1) for neighbor in neighbors])
        xs = np.array([neighbor.x for neighbor in neighbors])
        ys = np.array([neighbor.y for neighbor in neighbors])
        smoothed.append(
            TrackPoint(
                frame_index=point.frame_index,
                x=float(np.average(xs, weights=weights)),
                y=float(np.average(ys, weights=weights)),
                confidence=point.confidence,
                source=point.source,
            )
        )
    return smoothed


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(np.hypot(first[0] - second[0], first[1] - second[1]))
