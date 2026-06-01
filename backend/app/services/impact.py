import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImpactEstimate:
    frame_index: int
    source: str
    confidence: float


def estimate_impact_frame(video_path: Path, fps: float) -> ImpactEstimate | None:
    audio_estimate = _estimate_audio_impact(video_path, fps)
    if audio_estimate is not None:
        return audio_estimate
    return _estimate_visual_impact(video_path)


def _estimate_audio_impact(video_path: Path, fps: float) -> ImpactEstimate | None:
    sample_rate = 16_000
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    if len(samples) < sample_rate // 2:
        return None

    frame_size = max(1, int(round(sample_rate / max(fps, 1.0))))
    usable_samples = samples[: len(samples) - len(samples) % frame_size]
    if len(usable_samples) == 0:
        return None

    envelopes = np.mean(np.abs(usable_samples.reshape(-1, frame_size)), axis=1)
    if len(envelopes) < 5:
        return None

    baseline = float(np.median(envelopes))
    mad = float(np.median(np.abs(envelopes - baseline)))
    peak_index = int(np.argmax(envelopes))
    peak = float(envelopes[peak_index])
    threshold = baseline + max(6.0 * mad, 250.0)
    if peak < threshold:
        return None

    confidence = min(1.0, (peak - baseline) / max(threshold - baseline, 1.0) / 3.0)
    return ImpactEstimate(frame_index=peak_index, source="audio", confidence=confidence)


def _estimate_visual_impact(video_path: Path) -> ImpactEstimate | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    previous_gray: np.ndarray | None = None
    scores: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            if previous_gray is not None:
                diff = cv2.absdiff(previous_gray, gray)
                global_motion = float(np.median(diff))
                residual = np.maximum(diff.astype(np.float32) - global_motion, 0.0)
                lower_half = residual[residual.shape[0] // 2 :, :]
                scores.append(float(np.percentile(lower_half, 95)))
            previous_gray = gray
    finally:
        capture.release()

    if len(scores) < 5:
        return None

    baseline = float(np.median(scores))
    mad = float(np.median(np.abs(np.array(scores) - baseline)))
    peak_index = int(np.argmax(scores)) + 1
    peak = float(scores[peak_index - 1])
    threshold = baseline + max(4.0 * mad, 8.0)
    if peak < threshold:
        return None

    confidence = min(1.0, (peak - baseline) / max(threshold - baseline, 1.0) / 3.0)
    return ImpactEstimate(frame_index=peak_index, source="visual_motion", confidence=confidence)
