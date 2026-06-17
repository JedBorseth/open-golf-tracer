from pathlib import Path

import cv2
import numpy as np

import app.services.pipeline as pipeline_module
from app.models.job import JobRecord
from app.models.trace import TraceAdjustments
from app.services.club_tracker import ClubTrackerConfig, SwingTrack
from app.services.detector import Detection
from app.services.pipeline import TracerPipeline
from app.services.render import RenderConfig
from app.services.tracker import TrackerConfig


class FakeDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.frame_index = 0

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        del frame
        self.frame_index += 1
        if self.frame_index <= 6:
            return [_detection(75, 125)]
        return []


def test_hybrid_pipeline_produces_video_with_physics_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = _write_hybrid_swing_video(tmp_path / "swing.mp4", frame_count=16)
    model_path = tmp_path / "fake-ball-model.pt"
    model_path.write_text("fake")
    output_dir = tmp_path / "outputs"
    swing = _swing_track()
    monkeypatch.setattr(
        pipeline_module,
        "build_club_swing_track",
        lambda input_path, config: swing,
    )
    pipeline = TracerPipeline(
        output_dir=output_dir,
        detector=FakeDetector(model_path),  # type: ignore[arg-type]
        club_config=ClubTrackerConfig(
            impact_detection=False,
            backswing_frames=5,
            follow_through_frames=5,
            impact_pre_roll_frames=0,
            motion_threshold=8,
            min_motion_blob_area=20,
            min_impact_motion_score=40,
            smooth_window=1,
        ),
        tracker_config=TrackerConfig(
            synthetic_launch_frames=8,
            swing_launch_speed_px=7,
            smooth_window=1,
        ),
        render_config=RenderConfig(
            tracer_tail_frames=24,
            stabilize_tracer=True,
            tracer_max_gap_frames=8,
        ),
    )

    output_path = pipeline.process(
        JobRecord(
            id="job-1",
            media_kind="video",
            original_filename="swing.mp4",
            content_type="video/mp4",
            input_path=video_path,
        ),
    )

    assert output_path.exists()
    assert output_path.suffix == ".mp4"
    assert output_path.stat().st_size > 0
    trace = pipeline.load_trace(
        JobRecord(
            id="job-1",
            media_kind="video",
            original_filename="swing.mp4",
            content_type="video/mp4",
            input_path=video_path,
            trace_path=pipeline.trace_path_for(
                JobRecord(
                    id="job-1",
                    media_kind="video",
                    original_filename="swing.mp4",
                    content_type="video/mp4",
                    input_path=video_path,
                ),
            ),
        ),
    )
    assert trace.video.width == 320
    assert trace.ball_address.source == "ball_address"
    assert len(trace.ball_flight) == 8


def test_hybrid_pipeline_draws_predicted_flight_without_model_on_final_impact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = _write_blank_video(tmp_path / "late-impact.mp4", frame_count=6)
    output_dir = tmp_path / "outputs"
    swing = SwingTrack(
        points=[],
        impact_frame_index=5,
        impact_x=100.0,
        impact_y=130.0,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_club_swing_track",
        lambda input_path, config: swing,
    )
    pipeline = TracerPipeline(
        output_dir=output_dir,
        detector=None,
        tracker_config=TrackerConfig(
            synthetic_launch_frames=10,
            swing_launch_speed_px=7,
            smooth_window=1,
        ),
        render_config=RenderConfig(
            tracer_tail_frames=24,
            stabilize_tracer=False,
        ),
    )

    output_path = pipeline.process(
        JobRecord(
            id="late-impact",
            media_kind="video",
            original_filename="late-impact.mp4",
            content_type="video/mp4",
            input_path=video_path,
        ),
    )

    final_frame = _read_frame(output_path, frame_index=5)

    assert _ball_flight_pixel_count(final_frame) > 25


def test_pipeline_resolves_ball_address_with_vision_when_model_is_missing(
    tmp_path: Path,
) -> None:
    video_path = _write_hybrid_swing_video(tmp_path / "swing.mp4", frame_count=12)
    pipeline = TracerPipeline(output_dir=tmp_path / "outputs", detector=None)
    swing = SwingTrack(
        points=[],
        impact_frame_index=8,
        impact_x=120.0,
        impact_y=120.0,
    )

    address = pipeline._resolve_ball_address(video_path, swing)

    assert address.source == "ball_address_vision"
    assert abs(address.x - 75) <= 2
    assert abs(address.y - 125) <= 2


def test_pipeline_rerender_applies_adjustments_without_retracking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = _write_hybrid_swing_video(tmp_path / "swing.mp4", frame_count=16)
    output_dir = tmp_path / "outputs"
    swing = _swing_track()
    monkeypatch.setattr(
        pipeline_module,
        "build_club_swing_track",
        lambda input_path, config: swing,
    )
    pipeline = TracerPipeline(
        output_dir=output_dir,
        detector=None,
        club_config=ClubTrackerConfig(
            backswing_frames=5,
            follow_through_frames=5,
            impact_pre_roll_frames=0,
            motion_threshold=8,
            min_motion_blob_area=20,
            min_impact_motion_score=40,
            smooth_window=1,
        ),
        tracker_config=TrackerConfig(synthetic_launch_frames=8, smooth_window=1),
        render_config=RenderConfig(stabilize_tracer=False, tracer_max_gap_frames=8),
    )
    job = JobRecord(
        id="adjustable",
        media_kind="video",
        original_filename="swing.mp4",
        content_type="video/mp4",
        input_path=video_path,
    )
    pipeline.process(job)
    trace_path = pipeline.trace_path_for(job)
    trace = pipeline.load_trace(job.model_copy(update={"trace_path": trace_path}))
    _, address, flight = pipeline_module.adjusted_trace_parts(
        trace,
        TraceAdjustments(x_offset_px=25, y_offset_px=-12, arc_scale=1.5),
    )

    def fail_tracking(input_path, config):  # noqa: ANN001
        raise AssertionError("rerender should use saved trace geometry")

    monkeypatch.setattr(pipeline_module, "build_club_swing_track", fail_tracking)
    output_path = pipeline.rerender(
        job.model_copy(update={"trace_path": trace_path}),
        TraceAdjustments(x_offset_px=25, y_offset_px=-12, arc_scale=1.5),
    )

    assert output_path.exists()
    assert address.x == trace.ball_address.x + 25
    assert address.y == trace.ball_address.y - 12
    assert flight[0].y < trace.ball_flight[0].y


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


def _swing_track() -> SwingTrack:
    return SwingTrack(
        points=[
            pipeline_module.TrackPoint(3, 80.0, 140.0, 0.8, "motion"),
            pipeline_module.TrackPoint(5, 100.0, 125.0, 0.9, "motion"),
            pipeline_module.TrackPoint(7, 130.0, 110.0, 0.8, "motion"),
        ],
        impact_frame_index=5,
        impact_x=100.0,
        impact_y=125.0,
    )


def _write_hybrid_swing_video(path: Path, frame_count: int) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.line(frame, (0, 74), (319, 74), (90, 90, 90), 2)
        cv2.circle(frame, (75, 125), 4, (255, 255, 255), -1)
        x = 35 + index * 15
        y = 135 - min(index, 9) * 4
        cv2.rectangle(frame, (x, y), (x + 18, y + 20), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


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


def _read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    assert ok
    return frame


def _ball_flight_pixel_count(frame: np.ndarray) -> int:
    mask = cv2.inRange(
        frame,
        np.array([0, 140, 160], dtype=np.uint8),
        np.array([100, 255, 255], dtype=np.uint8),
    )
    return int(cv2.countNonZero(mask))
