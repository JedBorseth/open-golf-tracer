import numpy as np

from app.services.club_tracker import SwingTrack
from app.services.render import RenderConfig, _draw_swing_path, _segment_thickness
from app.services.tracker import TrackPoint


def test_swing_path_draws_visible_segments() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    points = [
        TrackPoint(frame_index=0, x=30, y=140, confidence=0.9, source="motion"),
        TrackPoint(frame_index=1, x=60, y=120, confidence=0.8, source="motion"),
        TrackPoint(frame_index=8, x=90, y=100, confidence=0.8, source="motion"),
    ]

    _draw_swing_path(
        frame,
        points,
        frame_index=8,
        frame_height=180,
        impact_frame=4,
        config=RenderConfig(tracer_max_gap_frames=2),
    )

    assert frame.sum() > 0


def test_swing_thickness_shrinks_near_horizon() -> None:
    config = RenderConfig(tracer_thickness=8, tracer_horizon_ratio=0.4)
    near_camera = TrackPoint(frame_index=0, x=50, y=170, confidence=1.0, source="motion")
    near_horizon = TrackPoint(frame_index=1, x=50, y=80, confidence=1.0, source="motion")

    assert _segment_thickness(near_camera, 180, config) > _segment_thickness(
        near_horizon,
        180,
        config,
    )


def test_impact_marker_fields_on_swing_track() -> None:
    swing = SwingTrack(
        points=[TrackPoint(5, 100.0, 120.0, 0.9, "motion")],
        impact_frame_index=5,
        impact_x=100.0,
        impact_y=120.0,
    )
    assert swing.impact_frame_index == 5
    assert swing.impact_x == 100.0
