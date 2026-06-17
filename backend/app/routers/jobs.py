from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings
from app.models.job import JobRecord, JobResponse, MediaKind
from app.services.club_tracker import ClubTrackerConfig
from app.services.detector import GolfBallDetector
from app.services.job_store import JobStore
from app.services.pipeline import PipelineError, TracerPipeline
from app.services.render import RenderConfig
from app.services.tracker import TrackerConfig

router = APIRouter()


def get_job_store(settings: Annotated[Settings, Depends(get_settings)]) -> JobStore:
    return JobStore(settings.job_store_dir)


def get_pipeline(settings: Annotated[Settings, Depends(get_settings)]) -> TracerPipeline:
    return _build_pipeline(settings)


@router.post("", response_model=JobResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[JobStore, Depends(get_job_store)],
) -> JobResponse:
    media_kind = _media_kind(file.content_type)
    if media_kind is None:
        raise HTTPException(
            status_code=415,
            detail="Upload must be an image/* or video/* file.",
        )

    job_id = uuid4().hex
    extension = Path(file.filename or "").suffix or _default_extension(media_kind)
    input_path = settings.upload_dir / f"{job_id}{extension}"
    bytes_written = 0

    with input_path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > settings.max_upload_bytes:
                input_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            output.write(chunk)

    record = store.create(
        JobRecord(
            id=job_id,
            media_kind=media_kind,
            original_filename=file.filename or input_path.name,
            content_type=file.content_type or "application/octet-stream",
            input_path=input_path,
        )
    )

    background_tasks.add_task(run_job, job_id, settings)
    return JobResponse.from_record(record)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> JobResponse:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse.from_record(record)


@router.get("/{job_id}/result")
def get_result(
    job_id: str,
    store: Annotated[JobStore, Depends(get_job_store)],
) -> FileResponse:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if record.output_path is None or not record.output_path.exists():
        raise HTTPException(status_code=404, detail="Result is not ready.")
    return FileResponse(
        path=record.output_path,
        filename=record.output_path.name,
        media_type=record.content_type,
    )


def run_job(job_id: str, settings: Settings) -> None:
    store = JobStore(settings.job_store_dir)
    pipeline = _build_pipeline(settings)

    try:
        job = store.mark_running(job_id)
        output_path = pipeline.process(job)
        store.mark_complete(job_id, output_path, f"/api/jobs/{job_id}/result")
    except PipelineError as error:
        store.mark_failed(job_id, error.code, error.message)
    except Exception as error:  # noqa: BLE001 - background jobs must surface failures.
        store.mark_failed(job_id, "processing_failed", str(error))


def _media_kind(content_type: str | None) -> MediaKind | None:
    if content_type is None:
        return None
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    return None


def _default_extension(media_kind: MediaKind) -> str:
    return ".jpg" if media_kind == "image" else ".mp4"


def _build_pipeline(settings: Settings) -> TracerPipeline:
    return TracerPipeline(
        output_dir=settings.output_dir,
        detector=GolfBallDetector(
            settings.model_path,
            settings.yolo_device,
            settings.yolo_confidence,
        ),
        club_config=ClubTrackerConfig(
            impact_detection=settings.tracker_impact_detection,
            backswing_frames=settings.club_backswing_frames,
            follow_through_frames=settings.club_follow_through_frames,
            impact_pre_roll_frames=settings.tracker_impact_pre_roll_frames,
            motion_threshold=settings.club_motion_threshold,
            roi_top_ratio=settings.club_roi_top_ratio,
            min_impact_motion_score=settings.club_min_impact_motion_score,
            max_camera_motion_area_ratio=settings.club_max_camera_motion_area_ratio,
            max_gap_frames=settings.tracker_max_gap_frames,
            smooth_window=settings.tracker_smooth_window,
            detection_gate_px=settings.tracker_detection_gate_px,
            optical_flow_gate_px=settings.tracker_optical_flow_gate_px,
            camera_motion_compensation=settings.tracker_camera_motion_compensation,
            camera_motion_max_px=settings.tracker_camera_motion_max_px,
        ),
        tracker_config=TrackerConfig(
            max_gap_frames=settings.tracker_max_gap_frames,
            detection_gate_px=settings.tracker_detection_gate_px,
            optical_flow_gate_px=settings.tracker_optical_flow_gate_px,
            smooth_window=settings.tracker_smooth_window,
            stationary_address_frames=settings.tracker_stationary_address_frames,
            stationary_address_radius_px=settings.tracker_stationary_address_radius_px,
            vision_ball_min_area_px=settings.tracker_vision_ball_min_area_px,
            vision_ball_max_area_px=settings.tracker_vision_ball_max_area_px,
            vision_ball_min_brightness=settings.tracker_vision_ball_min_brightness,
            vision_ball_roi_top_ratio=settings.tracker_vision_ball_roi_top_ratio,
            swing_motion_roi_px=settings.tracker_swing_motion_roi_px,
            swing_launch_speed_px=settings.tracker_swing_launch_speed_px,
            flight_speed_multiplier=settings.tracker_flight_speed_multiplier,
            flight_gravity_px_per_frame=settings.tracker_flight_gravity_px_per_frame,
            stale_track_frames=settings.tracker_stale_track_frames,
            stale_track_radius_px=settings.tracker_stale_track_radius_px,
            synthetic_launch_frames=settings.tracker_synthetic_launch_frames,
            synthetic_launch_upward_bias=settings.tracker_synthetic_launch_upward_bias,
            camera_motion_compensation=settings.tracker_camera_motion_compensation,
            camera_motion_max_px=settings.tracker_camera_motion_max_px,
            impact_detection=settings.tracker_impact_detection,
            impact_pre_roll_frames=settings.tracker_impact_pre_roll_frames,
            post_impact_stale_frames=settings.tracker_post_impact_stale_frames,
        ),
        render_config=RenderConfig(
            tracer_thickness=settings.tracer_thickness,
            tracer_tail_frames=settings.tracer_tail_frames,
            tracer_horizon_ratio=settings.tracer_horizon_ratio,
            stabilize_tracer=settings.tracer_stabilize,
            scene_motion_max_px=settings.tracker_camera_motion_max_px,
        ),
    )
