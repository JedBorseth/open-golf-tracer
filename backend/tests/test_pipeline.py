from pathlib import Path

import cv2
import numpy as np

from app.models.job import JobRecord
from app.services.club_tracker import ClubTrackerConfig
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


def test_hybrid_pipeline_produces_video_with_physics_trace(tmp_path: Path) -> None:
    video_path = _write_hybrid_swing_video(tmp_path / "swing.mp4", frame_count=16)
    model_path = tmp_path / "fake-ball-model.pt"
    model_path.write_text("fake")
    output_dir = tmp_path / "outputs"
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
