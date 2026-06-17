from pathlib import Path

import cv2
from pydantic import BaseModel, Field

from app.services.club_tracker import SwingTrack
from app.services.pipeline_errors import PipelineError
from app.services.tracker import TrackPoint


class TracePointModel(BaseModel):
    frame_index: int
    x: float
    y: float
    confidence: float
    source: str


class SwingTraceModel(BaseModel):
    points: list[TracePointModel]
    impact_frame_index: int | None
    impact_x: float | None
    impact_y: float | None


class VideoTraceMetadata(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int


class TraceData(BaseModel):
    video: VideoTraceMetadata
    swing: SwingTraceModel
    ball_address: TracePointModel
    ball_flight: list[TracePointModel]


class TraceAdjustments(BaseModel):
    x_offset_px: float = 0.0
    y_offset_px: float = 0.0
    arc_scale: float = Field(default=1.0, ge=0.25, le=2.5)


def trace_from_parts(
    video_path: Path,
    swing: SwingTrack,
    ball_address: TrackPoint,
    ball_flight: list[TrackPoint],
) -> TraceData:
    return TraceData(
        video=_video_metadata(video_path),
        swing=SwingTraceModel(
            points=[_point_to_model(point) for point in swing.points],
            impact_frame_index=swing.impact_frame_index,
            impact_x=swing.impact_x,
            impact_y=swing.impact_y,
        ),
        ball_address=_point_to_model(ball_address),
        ball_flight=[_point_to_model(point) for point in ball_flight],
    )


def adjusted_trace_parts(
    trace: TraceData,
    adjustments: TraceAdjustments | None = None,
) -> tuple[SwingTrack, TrackPoint, list[TrackPoint]]:
    adjustments = adjustments or TraceAdjustments()
    swing = SwingTrack(
        points=[_model_to_point(point) for point in trace.swing.points],
        impact_frame_index=trace.swing.impact_frame_index,
        impact_x=trace.swing.impact_x,
        impact_y=trace.swing.impact_y,
    )
    base_address = _model_to_point(trace.ball_address)
    adjusted_address = _adjust_point(base_address, base_address, adjustments)
    adjusted_flight = [
        _adjust_point(_model_to_point(point), base_address, adjustments)
        for point in trace.ball_flight
    ]
    return swing, adjusted_address, adjusted_flight


def _adjust_point(
    point: TrackPoint,
    base_address: TrackPoint,
    adjustments: TraceAdjustments,
) -> TrackPoint:
    return TrackPoint(
        frame_index=point.frame_index,
        x=point.x + adjustments.x_offset_px,
        y=base_address.y
        + adjustments.y_offset_px
        + (point.y - base_address.y) * adjustments.arc_scale,
        confidence=point.confidence,
        source=point.source,
    )


def _video_metadata(video_path: Path) -> VideoTraceMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError("invalid_video", f"Could not read video: {video_path}")
    try:
        return VideoTraceMetadata(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 30),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
    finally:
        capture.release()


def _point_to_model(point: TrackPoint) -> TracePointModel:
    return TracePointModel(
        frame_index=point.frame_index,
        x=point.x,
        y=point.y,
        confidence=point.confidence,
        source=point.source,
    )


def _model_to_point(point: TracePointModel) -> TrackPoint:
    return TrackPoint(
        frame_index=point.frame_index,
        x=point.x,
        y=point.y,
        confidence=point.confidence,
        source=point.source,
    )
