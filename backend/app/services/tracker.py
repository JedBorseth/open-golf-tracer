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


def build_video_track(video_path: Path, detector: GolfBallDetector) -> list[TrackPoint]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError("invalid_video", f"Could not read video: {video_path}")

    track: list[TrackPoint] = []
    kalman = _create_kalman_filter()
    previous_gray: np.ndarray | None = None
    previous_point: np.ndarray | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect_frame(frame)
            best_detection = _best_detection(detections)

            if best_detection is not None:
                x, y = best_detection.center
                measurement = np.array([[np.float32(x)], [np.float32(y)]])
                kalman.correct(measurement)
                previous_point = np.array([[[x, y]]], dtype=np.float32)
                track.append(
                    TrackPoint(
                        frame_index=frame_index,
                        x=x,
                        y=y,
                        confidence=best_detection.confidence,
                        source="yolo",
                    )
                )
            elif previous_gray is not None and previous_point is not None:
                next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    gray,
                    previous_point,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if next_point is not None and status is not None and status[0][0] == 1:
                    x, y = next_point[0][0]
                    kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
                    previous_point = next_point
                    track.append(
                        TrackPoint(
                            frame_index=frame_index,
                            x=float(x),
                            y=float(y),
                            confidence=0.5,
                            source="optical_flow",
                        )
                    )
                else:
                    prediction = kalman.predict()
                    x = float(prediction[0][0])
                    y = float(prediction[1][0])
                    previous_point = np.array([[[x, y]]], dtype=np.float32)
                    track.append(
                        TrackPoint(
                            frame_index=frame_index,
                            x=x,
                            y=y,
                            confidence=0.25,
                            source="kalman",
                        )
                    )

            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    if not track:
        raise PipelineError(
            "ball_not_detected",
            "No golf ball detections were found in the uploaded video.",
        )

    return track


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
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    return kalman
