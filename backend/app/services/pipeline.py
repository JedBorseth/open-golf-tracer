import json
from pathlib import Path

from app.models.job import JobRecord
from app.models.trace import (
    TraceAdjustments,
    TraceData,
    adjusted_trace_parts,
    trace_from_parts,
)
from app.services.club_tracker import ClubTrackerConfig, SwingTrack, build_club_swing_track
from app.services.detector import GolfBallDetector
from app.services.pipeline_errors import PipelineError
from app.services.render import RenderConfig, render_hybrid_trace, render_image_swing_hint
from app.services.tracker import (
    TrackerConfig,
    TrackPoint,
    build_physics_flight,
    find_ball_address,
    find_ball_address_by_vision,
)


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
        trace = self.analyze_video(job)
        self.save_trace(job, trace)
        output_path = self.output_dir / f"{job.id}.mp4"
        self.render_trace(job, trace, output_path)
        return output_path

    def analyze_video(self, job: JobRecord) -> TraceData:
        swing = build_club_swing_track(job.input_path, self.club_config)
        ball_address = self._resolve_ball_address(job.input_path, swing)
        ball_flight = build_physics_flight(
            ball_address,
            swing.points,
            swing.impact_frame_index,
            self.tracker_config,
        )
        return trace_from_parts(job.input_path, swing, ball_address, ball_flight)

    def render_trace(
        self,
        job: JobRecord,
        trace: TraceData,
        output_path: Path,
        adjustments: TraceAdjustments | None = None,
    ) -> Path:
        swing, ball_address, ball_flight = adjusted_trace_parts(trace, adjustments)
        render_hybrid_trace(
            job.input_path,
            output_path,
            swing,
            ball_address,
            ball_flight,
            self.render_config,
        )
        return output_path

    def rerender(
        self,
        job: JobRecord,
        adjustments: TraceAdjustments,
    ) -> Path:
        trace = self.load_trace(job)
        output_path = self.output_dir / f"{job.id}.mp4"
        return self.render_trace(job, trace, output_path, adjustments)

    def trace_path_for(self, job: JobRecord) -> Path:
        return self.output_dir / f"{job.id}.trace.json"

    def save_trace(self, job: JobRecord, trace: TraceData) -> Path:
        trace_path = self.trace_path_for(job)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = trace_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(trace.model_dump(mode="json"), file, indent=2, sort_keys=True)
        tmp_path.replace(trace_path)
        return trace_path

    def load_trace(self, job: JobRecord) -> TraceData:
        trace_path = job.trace_path or self.trace_path_for(job)
        if not trace_path.exists():
            raise PipelineError(
                "trace_not_found",
                "Trace geometry is not ready for this job.",
            )
        with trace_path.open("r", encoding="utf-8") as file:
            return TraceData.model_validate(json.load(file))

    def _resolve_ball_address(
        self,
        input_path: Path,
        swing: SwingTrack,
    ) -> TrackPoint:
        impact_frame = swing.impact_frame_index
        if self.detector is not None and self.detector.model_path.exists():
            detected = find_ball_address(
                input_path,
                self.detector,
                self.tracker_config,
                before_frame=impact_frame,
            )
            if detected is not None:
                return detected

        vision_address = find_ball_address_by_vision(
            input_path,
            self.tracker_config,
            before_frame=impact_frame,
        )
        if vision_address is not None:
            return vision_address

        if swing.impact_x is not None and swing.impact_y is not None:
            return TrackPoint(
                frame_index=impact_frame or 0,
                x=swing.impact_x,
                y=swing.impact_y,
                confidence=0.35,
                source="ball_address_from_impact",
            )

        raise PipelineError(
            "ball_not_detected",
            "Could not find the golf ball before impact in the uploaded video.",
        )
