from pathlib import Path

import cv2
import numpy as np

from app.services.club_tracker import ClubTrackerConfig, build_club_swing_track


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
