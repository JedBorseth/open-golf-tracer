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

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect_frame(frame)
            best_detection = _best_detection(detections)
            prediction: tuple[float, float] | None = None
            if kalman_initialized:
                predicted_state = kalman.predict()
                prediction = (float(predicted_state[0][0]), float(predicted_state[1][0]))

            measurement: tuple[float, float] | None = None
            confidence = 0.0
            source = "none"

            if best_detection is not None and _passes_gate(
                best_detection.center,
                prediction,
                config.detection_gate_px,
                gap_frames,
            ):
                x, y = best_detection.center
                measurement = (x, y)
                confidence = best_detection.confidence
                source = "yolo"
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
                and gap_frames < config.max_gap_frames
            ):
                gap_frames += 1
                confidence = max(0.1, 0.45 - gap_frames * config.kalman_confidence_decay)
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
