from pathlib import Path

import cv2
import numpy as np

from app.services.detector import Detection
from app.services.tracker import (
    TrackerConfig,
    TrackPoint,
    build_physics_flight,
    build_video_track,
    find_ball_address,
    find_ball_address_by_vision,
)


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


def test_tracker_uses_swing_motion_when_detection_sticks_at_address(tmp_path: Path) -> None:
    video_path = _write_swing_motion_video(tmp_path / "shot.mp4", frame_count=5)
    detector = FakeDetector([[_detection(80, 100)] for _ in range(5)])

    track = build_video_track(
        video_path,
        detector,  # type: ignore[arg-type]
        TrackerConfig(
            stationary_address_frames=1,
            stationary_address_radius_px=5,
            swing_launch_speed_px=12,
            max_gap_frames=4,
            smooth_window=1,
        ),
    )

    assert track[0].source == "yolo"
    assert any(point.source == "synthetic_launch" and point.x > 90 for point in track[1:])


def test_tracker_forces_launch_when_yolo_detection_stays_stale(tmp_path: Path) -> None:
    video_path = _write_blank_video(tmp_path / "shot.mp4", frame_count=8)
    detector = FakeDetector([[_detection(80, 100)] for _ in range(8)])

    track = build_video_track(
        video_path,
        detector,  # type: ignore[arg-type]
        TrackerConfig(
            stale_track_frames=3,
            stale_track_radius_px=20,
            synthetic_launch_frames=8,
            swing_launch_speed_px=10,
            max_gap_frames=8,
            smooth_window=1,
        ),
    )

    assert any(point.source == "synthetic_launch" for point in track[3:])
    assert track[-1].y < track[0].y


def test_tracker_compensates_for_camera_motion_before_stale_check(tmp_path: Path) -> None:
    video_path = _write_panning_background_video(tmp_path / "shot.mp4", frame_count=8)
    detector = FakeDetector([[_detection(80 + index * 5, 100)] for index in range(8)])

    track = build_video_track(
        video_path,
        detector,  # type: ignore[arg-type]
        TrackerConfig(
            stale_track_frames=3,
            stale_track_radius_px=12,
            synthetic_launch_frames=8,
            swing_launch_speed_px=6,
            max_gap_frames=8,
            smooth_window=1,
            camera_motion_compensation=True,
        ),
    )

    assert any(point.source == "synthetic_launch" for point in track[3:])


def test_find_ball_address_uses_stable_pre_impact_cluster(tmp_path: Path) -> None:
    video_path = _write_blank_video(tmp_path / "shot.mp4", frame_count=6)
    detector = FakeDetector(
        [
            [_detection(80, 100, 0.8)],
            [_detection(82, 99, 0.9)],
            [_detection(81, 101, 0.85)],
            [_detection(180, 40, 0.95)],
            [_detection(220, 30, 0.95)],
            [_detection(230, 25, 0.95)],
        ],
    )

    address = find_ball_address(
        video_path,
        detector,  # type: ignore[arg-type]
        TrackerConfig(stationary_address_frames=3),
        before_frame=3,
    )

    assert address is not None
    assert address.source == "ball_address"
    assert abs(address.x - 81) <= 1
    assert abs(address.y - 100) <= 1


def test_find_ball_address_by_vision_uses_stable_bright_ball(tmp_path: Path) -> None:
    video_path = _write_ball_address_video(tmp_path / "address.mp4", frame_count=6)

    address = find_ball_address_by_vision(
        video_path,
        TrackerConfig(
            stationary_address_frames=3,
            vision_ball_min_brightness=150,
            vision_ball_max_area_px=200,
        ),
        before_frame=5,
    )

    assert address is not None
    assert address.source == "ball_address_vision"
    assert abs(address.x - 78) <= 2
    assert abs(address.y - 124) <= 2


def test_physics_flight_moves_up_then_falls_from_club_velocity() -> None:
    address = TrackPoint(5, 100.0, 120.0, 0.9, "ball_address")
    club_points = [
        TrackPoint(3, 80.0, 140.0, 0.8, "motion"),
        TrackPoint(5, 100.0, 120.0, 0.9, "motion"),
        TrackPoint(7, 130.0, 105.0, 0.8, "motion"),
    ]

    flight = build_physics_flight(
        address,
        club_points,
        impact_frame=5,
        config=TrackerConfig(
            synthetic_launch_frames=18,
            swing_launch_speed_px=8,
            flight_speed_multiplier=2.0,
            flight_gravity_px_per_frame=1.8,
        ),
    )

    assert [point.source for point in flight] == ["physics_flight"] * 18
    assert flight[0].x > address.x
    assert min(point.y for point in flight) < address.y
    assert flight[-1].y > min(point.y for point in flight)


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


def _write_swing_motion_video(path: Path, frame_count: int) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        x = 40 + index * 16
        cv2.rectangle(frame, (x, 85), (x + 28, 115), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _write_panning_background_video(path: Path, frame_count: int) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    base_points = [(30 + (index * 37) % 260, 20 + (index * 29) % 130) for index in range(40)]
    for frame_index in range(frame_count):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for x, y in base_points:
            cv2.circle(frame, (x + frame_index * 5, y), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _write_ball_address_video(path: Path, frame_count: int) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (78, 124), 5, (245, 245, 245), -1)
        cv2.rectangle(frame, (25 + index * 14, 90), (40 + index * 14, 120), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path
