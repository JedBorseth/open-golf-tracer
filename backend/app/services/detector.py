from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from app.services.pipeline_errors import PipelineError


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    label: str

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x)),
            int(round(self.y)),
            int(round(self.x + self.width)),
            int(round(self.y + self.height)),
        )


class GolfBallDetector:
    def __init__(self, model_path: Path, device: str, confidence: float) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence = confidence

    @cached_property
    def model(self):
        if not self.model_path.exists():
            raise PipelineError(
                "model_not_found",
                f"YOLO model weights were not found at {self.model_path}.",
            )

        from ultralytics import YOLO

        return YOLO(str(self.model_path))

    def detect_image(self, image_path: Path) -> list[Detection]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise PipelineError("invalid_image", f"Could not read image: {image_path}")
        return self.detect_frame(image)

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.confidence,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names or {}
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=confidence,
                    class_id=class_id,
                    label=str(names.get(class_id, class_id)),
                )
            )

        return sorted(detections, key=lambda item: item.confidence, reverse=True)


def get_device_info(preferred_device: str) -> dict[str, object]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        return {
            "preferred": preferred_device,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except Exception as error:  # noqa: BLE001 - health checks should report import failures.
        return {
            "preferred": preferred_device,
            "cuda_available": False,
            "error": str(error),
        }
