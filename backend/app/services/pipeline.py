from pathlib import Path

from app.models.job import JobRecord
from app.services.club_tracker import ClubTrackerConfig, build_club_swing_track
from app.services.detector import GolfBallDetector
from app.services.pipeline_errors import PipelineError
from app.services.render import RenderConfig, render_hybrid_trace, render_image_swing_hint
from app.services.tracker import TrackerConfig, TrackPoint, build_physics_flight, find_ball_address


class TracerPipeline:
    def __init__(
        self,
        output_dir: Path,
        detector: GolfBallDetector | None = None,
        club_config: ClubTrackerConfig | None = None,
        tracker_config: TrackerConfig | None = None,
        render_config: RenderConfig | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.detector = detector
        self.club_config = club_config or ClubTrackerConfig()
        self.tracker_config = tracker_config or TrackerConfig()
        self.render_config = render_config or RenderConfig()

    def process(self, job: JobRecord) -> Path:
        if job.media_kind == "image":
            return self._process_image(job)
        if job.media_kind == "video":
            return self._process_video(job)
        raise PipelineError("unsupported_media", f"Unsupported media kind: {job.media_kind}")

    def _process_image(self, job: JobRecord) -> Path:
        output_path = self.output_dir / f"{job.id}.jpg"
        render_image_swing_hint(job.input_path, output_path)
        return output_path

    def _process_video(self, job: JobRecord) -> Path:
        swing = build_club_swing_track(job.input_path, self.club_config)
        ball_address = self._find_pre_impact_ball(job.input_path, swing.impact_frame_index)
        if ball_address is None and swing.impact_x is not None and swing.impact_y is not None:
            ball_address = TrackPoint(
                frame_index=swing.impact_frame_index or 0,
                x=swing.impact_x,
                y=swing.impact_y,
                confidence=0.45,
                source="impact_fallback",
            )
        ball_flight = build_physics_flight(
            ball_address,
            swing.points,
            swing.impact_frame_index,
            self.tracker_config,
        )
        output_path = self.output_dir / f"{job.id}.mp4"
        render_hybrid_trace(
            job.input_path,
            output_path,
            swing,
            ball_address,
            ball_flight,
            self.render_config,
        )
        return output_path

    def _find_pre_impact_ball(
        self,
        input_path: Path,
        impact_frame: int | None,
    ) -> TrackPoint | None:
        if self.detector is None or not self.detector.model_path.exists():
            return None
        return find_ball_address(
            input_path,
            self.detector,
            self.tracker_config,
            before_frame=impact_frame,
        )
