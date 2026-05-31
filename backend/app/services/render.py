import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.services.detector import Detection
from app.services.pipeline_errors import PipelineError
from app.services.tracker import TrackPoint


TRACE_COLOR = (0, 242, 255)
BOX_COLOR = (38, 255, 116)


def render_image_trace(
    input_path: Path,
    output_path: Path,
    detections: list[Detection],
) -> None:
    image = cv2.imread(str(input_path))
    if image is None:
        raise PipelineError("invalid_image", f"Could not read image: {input_path}")

    if not detections:
        raise PipelineError(
            "ball_not_detected",
            "No golf ball detections were found in the uploaded image.",
        )

    for detection in detections[:3]:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(
            image,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise PipelineError("render_failed", f"Could not write image: {output_path}")


def render_video_trace(
    input_path: Path,
    output_path: Path,
    track: list[TrackPoint],
) -> None:
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

    points_by_frame = {point.frame_index: point for point in track}
    drawn_points: list[tuple[int, int]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            point = points_by_frame.get(frame_index)
            if point is not None:
                drawn_points.append((int(round(point.x)), int(round(point.y))))

            if len(drawn_points) > 1:
                cv2.polylines(frame, [_points_to_array(drawn_points)], False, TRACE_COLOR, 4)
            if drawn_points:
                cv2.circle(frame, drawn_points[-1], 8, TRACE_COLOR, -1)

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    _transcode_with_ffmpeg(temp_path, output_path)
    temp_path.unlink(missing_ok=True)


def _points_to_array(points: list[tuple[int, int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


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
            result.stderr.strip() or "ffmpeg failed while rendering the tracer video.",
        )
