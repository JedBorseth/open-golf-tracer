import numpy as np

from app.services.render import RenderConfig, _draw_perspective_trace, _segment_thickness
from app.services.tracker import TrackPoint


def test_perspective_trace_draws_visible_segments() -> None:
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    points = [
        TrackPoint(frame_index=0, x=30, y=140, confidence=0.9, source="yolo"),
        TrackPoint(frame_index=1, x=60, y=120, confidence=0.8, source="yolo"),
        TrackPoint(frame_index=8, x=90, y=100, confidence=0.8, source="yolo"),
    ]

    _draw_perspective_trace(
        frame,
        points,
        frame_index=8,
        frame_height=180,
        config=RenderConfig(tracer_max_gap_frames=2),
    )

    assert frame.sum() > 0


def test_perspective_thickness_shrinks_near_horizon() -> None:
    config = RenderConfig(tracer_thickness=8, tracer_horizon_ratio=0.4)
    near_camera = TrackPoint(frame_index=0, x=50, y=170, confidence=1.0, source="yolo")
    near_horizon = TrackPoint(frame_index=1, x=50, y=80, confidence=1.0, source="yolo")

    assert _segment_thickness(near_camera, 180, config) > _segment_thickness(
        near_horizon,
        180,
        config,
    )
