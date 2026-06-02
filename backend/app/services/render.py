import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.club_tracker import SwingTrack
from app.services.pipeline_errors import PipelineError
from app.services.tracker import TrackPoint

SWING_PATH_COLOR = (255, 180, 0)  # BGR gold — downswing / follow-through
BACKSWING_COLOR = (0, 200, 255)  # BGR amber-cyan — backswing
IMPACT_COLOR = (60, 60, 255)  # BGR red
IMPACT_RING_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class RenderConfig:
    tracer_thickness: int = 8
    tracer_tail_frames: int = 120
    tracer_min_alpha: float = 0.2
    tracer_max_alpha: float = 0.95
    tracer_horizon_ratio: float = 0.42
    tracer_max_gap_frames: int = 6
    tracer_min_confidence: float = 0.12
    marker_min_confidence: float = 0.2
    impact_marker_radius: int = 14
    impact_label: str = "IMPACT"


def render_swing_trace(
    input_path: Path,
    output_path: Path,
    swing: SwingTrack,
    config: RenderConfig | None = None,
) -> None:
    config = config or RenderConfig()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise PipelineError("invalid_video", f"Could not read video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = output_path.with_suffix(".raw.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise PipelineError("render_failed", f"Could not open video writer: {temp_path}")

    points_by_frame = {point.frame_index: point for point in swing.points}
    drawn_points: list[TrackPoint] = []
    frame_index = 0
    impact_frame = swing.impact_frame_index

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            point = points_by_frame.get(frame_index)
            if point is not None:
                drawn_points.append(point)

            _draw_swing_path(frame, drawn_points, frame_index, height, impact_frame, config)

            if (
                impact_frame is not None
                and frame_index == impact_frame
                and swing.impact_x is not None
                and swing.impact_y is not None
            ):
                _draw_impact_marker(
                    frame,
                    int(round(swing.impact_x)),
                    int(round(swing.impact_y)),
                    config,
                )

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    _transcode_with_ffmpeg(temp_path, output_path)
    temp_path.unlink(missing_ok=True)


def render_image_swing_hint(
    input_path: Path,
    output_path: Path,
) -> None:
    """Static images cannot show a swing path — return a labeled still."""
    image = cv2.imread(str(input_path))
    if image is None:
        raise PipelineError("invalid_image", f"Could not read image: {input_path}")

    message = "Upload a swing video to trace club path and impact"
    cv2.putText(
        image,
        message,
        (24, max(40, image.shape[0] // 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        SWING_PATH_COLOR,
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise PipelineError("render_failed", f"Could not write image: {output_path}")


def _draw_swing_path(
    frame: np.ndarray,
    points: list[TrackPoint],
    frame_index: int,
    frame_height: int,
    impact_frame: int | None,
    config: RenderConfig,
) -> None:
    visible_points = [
        point
        for point in points
        if frame_index - point.frame_index <= config.tracer_tail_frames
        and point.confidence >= config.tracer_min_confidence
    ]
    if len(visible_points) < 2:
        return

    for previous, current in zip(visible_points, visible_points[1:], strict=False):
        if current.frame_index - previous.frame_index > config.tracer_max_gap_frames:
            continue
        age = frame_index - current.frame_index
        recency = 1.0 - min(age / max(config.tracer_tail_frames, 1), 1.0)
        confidence = min((previous.confidence + current.confidence) / 2.0, 1.0)
        alpha = _lerp(config.tracer_min_alpha, config.tracer_max_alpha, recency) * confidence
        thickness = _segment_thickness(current, frame_height, config)
        segment_color = _segment_color(previous, impact_frame)
        _draw_alpha_line(
            frame,
            _screen_point(previous),
            _screen_point(current),
            segment_color,
            thickness,
            alpha,
        )

    latest = visible_points[-1]
    if latest.confidence >= config.marker_min_confidence:
        radius = max(4, int(round(_segment_thickness(latest, frame_height, config) * 0.9)))
        _draw_alpha_circle(
            frame,
            _screen_point(latest),
            radius,
            SWING_PATH_COLOR,
            0.75,
        )


def _segment_color(point: TrackPoint, impact_frame: int | None) -> tuple[int, int, int]:
    if impact_frame is None:
        return SWING_PATH_COLOR
    if point.frame_index <= impact_frame:
        return BACKSWING_COLOR
    return SWING_PATH_COLOR


def _draw_impact_marker(
    frame: np.ndarray,
    x: int,
    y: int,
    config: RenderConfig,
) -> None:
    radius = config.impact_marker_radius
    _draw_alpha_circle(frame, (x, y), radius + 4, IMPACT_RING_COLOR, 0.9)
    _draw_alpha_circle(frame, (x, y), radius, IMPACT_COLOR, 0.92)
    label_y = max(24, y - radius - 12)
    cv2.putText(
        frame,
        config.impact_label,
        (max(8, x - 36), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        IMPACT_RING_COLOR,
        2,
        cv2.LINE_AA,
    )


def _segment_thickness(point: TrackPoint, frame_height: int, config: RenderConfig) -> int:
    horizon_y = frame_height * config.tracer_horizon_ratio
    depth = np.clip((point.y - horizon_y) / max(frame_height - horizon_y, 1.0), 0.0, 1.0)
    perspective_scale = 0.45 + depth * 0.75
    confidence_scale = 0.65 + min(point.confidence, 1.0) * 0.35
    return max(2, int(round(config.tracer_thickness * perspective_scale * confidence_scale)))


def _draw_alpha_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)
    clamped_alpha = float(np.clip(alpha, 0.0, 1.0))
    cv2.addWeighted(overlay, clamped_alpha, frame, 1.0 - clamped_alpha, 0, frame)


def _draw_alpha_circle(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    clamped_alpha = float(np.clip(alpha, 0.0, 1.0))
    cv2.addWeighted(overlay, clamped_alpha, frame, 1.0 - clamped_alpha, 0, frame)


def _screen_point(point: TrackPoint) -> tuple[int, int]:
    return (int(round(point.x)), int(round(point.y)))


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * np.clip(amount, 0.0, 1.0)


def _transcode_with_ffmpeg(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise PipelineError(
            "ffmpeg_failed",
            result.stderr.strip() or "ffmpeg failed while rendering the swing trace video.",
        )
