from pathlib import Path

import cv2
import numpy as np

from app.services.detector import Detection
from app.services.tracker import TrackerConfig, build_video_track


class FakeDetector:
    def __init__(self, detections_by_frame: list[list[Detection]]) -> None:
        self.detections_by_frame = detections_by_frame
        self.frame_index = 0

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        del frame
        if self.frame_index >= len(self.detections_by_frame):
            return []
        detections = self.detections_by_frame[self.frame_index]
        self.frame_index += 1
        return detections


def test_tracker_fills_short_gaps_with_kalman(tmp_path: Path) -> None:
    video_path = _write_blank_video(tmp_path / "shot.mp4", frame_count=5)
    detector = FakeDetector(
        [
            [_detection(20, 40)],
            [_detection(30, 38)],
            [],
            [],
            [_detection(60, 32)],
        ]
    )

    track = build_video_track(
        video_path,
        detector,  # type: ignore[arg-type]
        TrackerConfig(max_gap_frames=4, smooth_window=1),
    )

    assert [point.frame_index for point in track] == [0, 1, 2, 3, 4]
    assert {point.source for point in track[2:4]} == {"kalman"}


def test_tracker_rejects_outlier_detection(tmp_path: Path) -> None:
    video_path = _write_blank_video(tmp_path / "shot.mp4", frame_count=4)
    detector = FakeDetector(
        [
            [_detection(20, 40)],
            [_detection(30, 38)],
            [_detection(250, 150)],
            [_detection(40, 36)],
        ]
    )

    track = build_video_track(
        video_path,
        detector,  # type: ignore[arg-type]
        TrackerConfig(detection_gate_px=35, max_gap_frames=3, smooth_window=1),
    )

    frame_two = next(point for point in track if point.frame_index == 2)
    assert frame_two.source != "yolo"
    assert frame_two.x < 100


def _detection(center_x: float, center_y: float, confidence: float = 0.9) -> Detection:
    return Detection(
        x=center_x - 2,
        y=center_y - 2,
        width=4,
        height=4,
        confidence=confidence,
        class_id=0,
        label="golf_ball",
    )


def _write_blank_video(path: Path, frame_count: int) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()
    return path
