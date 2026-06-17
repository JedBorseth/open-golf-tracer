from pathlib import Path

import cv2
import numpy as np

from app.services.club_tracker import ClubTrackerConfig, _motion_score, build_club_swing_track


def test_club_tracker_follows_moving_club_head(tmp_path: Path) -> None:
    video_path = _write_club_motion_video(tmp_path / "swing.mp4", frame_count=12)

    swing = build_club_swing_track(
        video_path,
        ClubTrackerConfig(
            impact_detection=False,
            backswing_frames=4,
            follow_through_frames=4,
            impact_pre_roll_frames=0,
            smooth_window=1,
        ),
        impact_frame=6,
    )

    assert len(swing.points) >= 6
    assert swing.impact_frame_index == 6
    assert swing.impact_x is not None
    xs = [point.x for point in swing.points]
    assert max(xs) > min(xs) + 20


def test_club_tracker_fills_short_gaps(tmp_path: Path) -> None:
    video_path = _write_intermittent_club_video(tmp_path / "swing.mp4", frame_count=10)

    swing = build_club_swing_track(
        video_path,
        ClubTrackerConfig(
            impact_detection=False,
            backswing_frames=2,
            follow_through_frames=2,
            max_gap_frames=4,
            smooth_window=1,
        ),
        impact_frame=5,
    )

    frame_indices = [point.frame_index for point in swing.points]
    assert 4 in frame_indices or 5 in frame_indices
    assert any(point.source == "kalman" for point in swing.points)


def test_club_tracker_waits_for_late_high_motion_shot(tmp_path: Path) -> None:
    video_path = _write_preshot_then_late_swing_video(tmp_path / "late-swing.mp4")

    swing = build_club_swing_track(
        video_path,
        ClubTrackerConfig(
            backswing_frames=6,
            follow_through_frames=8,
            impact_pre_roll_frames=0,
            motion_threshold=8,
            min_motion_blob_area=20,
            smooth_window=1,
        ),
    )

    assert swing.impact_frame_index is not None
    assert 55 <= swing.impact_frame_index <= 64
    assert min(point.frame_index for point in swing.points) >= 49


def test_club_tracker_rejects_preshot_only_motion_as_impact(tmp_path: Path) -> None:
    video_path = _write_preshot_only_video(tmp_path / "preshot-only.mp4")

    swing = build_club_swing_track(
        video_path,
        ClubTrackerConfig(
            backswing_frames=6,
            follow_through_frames=8,
            impact_pre_roll_frames=0,
            motion_threshold=8,
            min_motion_blob_area=20,
            smooth_window=1,
        ),
    )

    assert swing.impact_frame_index is None
    assert swing.impact_x is None
    assert swing.impact_y is None


def test_motion_score_ignores_sparse_camera_pan() -> None:
    previous_gray = _camera_pan_frame(0)
    gray = _camera_pan_frame(1)

    score = _motion_score(
        previous_gray,
        gray,
        ClubTrackerConfig(motion_threshold=8, min_motion_blob_area=20),
    )

    assert score == 0.0


def _write_club_motion_video(path: Path, frame_count: int) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        x = 30 + index * 18
        cv2.rectangle(frame, (x, 90), (x + 24, 118), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _write_preshot_then_late_swing_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(80):
        frame = _shot_discrimination_frame(index, include_late_swing=True)
        writer.write(frame)
    writer.release()
    return path


def _write_preshot_only_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(45):
        frame = _shot_discrimination_frame(index, include_late_swing=False)
        writer.write(frame)
    writer.release()
    return path


def _shot_discrimination_frame(index: int, include_late_swing: bool) -> np.ndarray:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2.line(frame, (0, 70), (319, 70), (75, 75, 75), 2)
    cv2.circle(frame, (78, 124), 4, (255, 255, 255), -1)
    if index < 25:
        x = 50 + (index % 2) * 16
        cv2.rectangle(frame, (x, 100), (x + 14, 126), (180, 180, 180), -1)
    elif include_late_swing and index >= 55:
        x = 35 + (index - 55) * 22
        y = 135 - min(index - 55, 9) * 4
        cv2.rectangle(frame, (x, y), (x + 24, y + 24), (255, 255, 255), -1)
    return frame


def _camera_pan_frame(frame_index: int) -> np.ndarray:
    frame = np.zeros((180, 320), dtype=np.uint8)
    points = [(30 + (index * 37) % 260, 45 + (index * 29) % 110) for index in range(80)]
    for x, y in points:
        cv2.circle(frame, (x + frame_index * 5, y), 2, 255, -1)
    return frame


def _write_intermittent_club_video(path: Path, frame_count: int) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        if index not in {3, 4}:
            x = 40 + index * 14
            cv2.rectangle(frame, (x, 95), (x + 20, 115), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path
