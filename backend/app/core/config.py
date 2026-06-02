from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Golf Tracer API"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    upload_dir: Path = Path("/app/uploads")
    output_dir: Path = Path("/app/outputs")
    job_store_dir: Path = Path("/app/job-store")
    model_path: Path = Path("/app/models/yolov11s-golf-ball.pt")

    max_upload_mb: int = 512
    require_cuda: bool = False

    club_backswing_frames: int = 24
    club_follow_through_frames: int = 18
    club_motion_threshold: float = 16.0
    club_roi_top_ratio: float = 0.22

    tracker_max_gap_frames: int = 12
    tracker_detection_gate_px: float = 120.0
    tracker_optical_flow_gate_px: float = 80.0
    tracker_smooth_window: int = 5
    tracker_stationary_address_frames: int = 4
    tracker_stationary_address_radius_px: float = 8.0
    tracker_swing_motion_roi_px: int = 140
    tracker_swing_launch_speed_px: float = 7.0
    tracker_stale_track_frames: int = 30
    tracker_stale_track_radius_px: float = 80.0
    tracker_synthetic_launch_frames: int = 45
    tracker_synthetic_launch_upward_bias: float = 0.85
    tracker_camera_motion_compensation: bool = True
    tracker_camera_motion_max_px: float = 35.0
    tracker_impact_detection: bool = True
    tracker_impact_pre_roll_frames: int = 2
    tracker_post_impact_stale_frames: int = 4

    tracer_thickness: int = 8
    tracer_tail_frames: int = 120
    tracer_horizon_ratio: float = 0.42

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for directory in (
        settings.upload_dir,
        settings.output_dir,
        settings.job_store_dir,
        settings.model_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return settings
