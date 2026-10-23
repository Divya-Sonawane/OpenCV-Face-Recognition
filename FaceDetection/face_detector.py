from dataclasses import dataclass

import numpy as np
from deepface import DeepFace


@dataclass(frozen=True)
class DetectedFace:
    box: tuple[int, int, int, int]
    crop: np.ndarray
    confidence: float


class FaceDetector:
    """Detect and align faces with DeepFace's RetinaFace backend."""

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR color image")

        try:
            detections = DeepFace.extract_faces(
                img_path=image,
                detector_backend="retinaface",
                enforce_detection=True,
                align=True,
                color_face="bgr",
                normalize_face=False,
            )
        except ValueError as error:
            if "face could not be detected" in str(error).lower():
                return []
            raise

        height, width = image.shape[:2]
        faces: list[DetectedFace] = []
        for detection in detections:
            area = detection["facial_area"]
            x = max(0, int(area["x"]))
            y = max(0, int(area["y"]))
            right = min(width, x + int(area["w"]))
            bottom = min(height, y + int(area["h"]))
            crop = detection["face"]

            if right <= x or bottom <= y or crop.size == 0:
                continue

            if crop.dtype != np.uint8:
                if np.issubdtype(crop.dtype, np.floating) and crop.max() <= 1.0:
                    crop = crop * 255.0
                crop = np.clip(crop, 0, 255).astype(np.uint8)

            confidence = float(detection.get("confidence", 0.0))
            faces.append(
                DetectedFace(
                    box=(x, y, right - x, bottom - y),
                    crop=np.ascontiguousarray(crop),
                    confidence=confidence,
                )
            )

        return faces
