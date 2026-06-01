from pathlib import Path

from app.models.job import JobRecord
from app.services.detector import GolfBallDetector
from app.services.pipeline_errors import PipelineError
from app.services.render import RenderConfig, render_image_trace, render_video_trace
from app.services.tracker import TrackerConfig, build_video_track


class TracerPipeline:
    def __init__(
        self,
        detector: GolfBallDetector,
        output_dir: Path,
        tracker_config: TrackerConfig | None = None,
        render_config: RenderConfig | None = None,
    ) -> None:
        self.detector = detector
        self.output_dir = output_dir
        self.tracker_config = tracker_config or TrackerConfig()
        self.render_config = render_config or RenderConfig()

    def process(self, job: JobRecord) -> Path:
        if job.media_kind == "image":
            return self._process_image(job)
        if job.media_kind == "video":
            return self._process_video(job)
        raise PipelineError("unsupported_media", f"Unsupported media kind: {job.media_kind}")

    def _process_image(self, job: JobRecord) -> Path:
        detections = self.detector.detect_image(job.input_path)
        output_path = self.output_dir / f"{job.id}.jpg"
        render_image_trace(job.input_path, output_path, detections)
        return output_path

    def _process_video(self, job: JobRecord) -> Path:
        track = build_video_track(job.input_path, self.detector, self.tracker_config)
        output_path = self.output_dir / f"{job.id}.mp4"
        render_video_trace(job.input_path, output_path, track, self.render_config)
        return output_path
