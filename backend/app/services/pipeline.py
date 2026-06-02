from pathlib import Path

from app.models.job import JobRecord
from app.services.club_tracker import ClubTrackerConfig, build_club_swing_track
from app.services.pipeline_errors import PipelineError
from app.services.render import RenderConfig, render_image_swing_hint, render_swing_trace


class TracerPipeline:
    def __init__(
        self,
        output_dir: Path,
        club_config: ClubTrackerConfig | None = None,
        render_config: RenderConfig | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.club_config = club_config or ClubTrackerConfig()
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
        output_path = self.output_dir / f"{job.id}.mp4"
        render_swing_trace(job.input_path, output_path, swing, self.render_config)
        return output_path
