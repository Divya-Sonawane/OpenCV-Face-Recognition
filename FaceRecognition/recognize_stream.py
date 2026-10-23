import argparse
import logging
import pickle
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FaceDetection.face_detector import DetectedFace, FaceDetector
from FaceRecognition.embedding_extractor import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    create_embedding,
)

DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "database" / "embeddings.pkl"
LOGGER = logging.getLogger("recognize_faces")


def load_registry(registry_path: Path) -> dict[str, list[np.ndarray]]:
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Embedding registry not found: {registry_path}. "
            "Run FaceDetection/register_via_photos.py first."
        )

    with registry_path.open("rb") as registry_file:
        payload: Any = pickle.load(registry_file)

    if not isinstance(payload, dict):
        raise ValueError("Invalid embedding registry: expected a dictionary")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported embedding registry schema version")
    if payload.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Registry uses {payload.get('model_name')!r}; expected {MODEL_NAME!r}"
        )
    if payload.get("embedding_dimension") != EMBEDDING_DIMENSION:
        raise ValueError("Registry embedding dimension does not match the model")

    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("Invalid embedding registry: no employee profiles")

    profiles: dict[str, list[np.ndarray]] = {}
    for employee_name, templates in raw_profiles.items():
        if not isinstance(employee_name, str) or not isinstance(templates, list):
            raise ValueError("Invalid employee entry in embedding registry")

        vectors: list[np.ndarray] = []
        for template in templates:
            if not isinstance(template, dict) or "embedding" not in template:
                raise ValueError(f"Invalid embedding template for {employee_name!r}")
            vector = np.asarray(template["embedding"], dtype=np.float32)
            if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid embedding vector for {employee_name!r}")
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise ValueError(f"Zero-length embedding vector for {employee_name!r}")
            vectors.append(vector / norm)

        if vectors:
            profiles[employee_name] = vectors

    if not profiles:
        raise ValueError("Invalid embedding registry: no usable face templates")
    return profiles


def match_embedding(
    query_embedding: np.ndarray,
    profiles: dict[str, list[np.ndarray]],
    max_distance: float,
) -> tuple[str | None, float]:
    """Match against every enrolled view and use the closest cosine distance."""
    best_name: str | None = None
    best_distance = float("inf")

    for employee_name, templates in profiles.items():
        for template in templates:
            similarity = float(np.dot(query_embedding, template))
            distance = 1.0 - float(np.clip(similarity, -1.0, 1.0))
            if distance < best_distance:
                best_name = employee_name
                best_distance = distance

    if best_distance > max_distance:
        return None, best_distance
    return best_name, best_distance


def draw_face(
    frame: np.ndarray,
    face: DetectedFace,
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = face.box
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def run_camera(camera_index: int, registry_path: Path, max_distance: float) -> None:
    if not 0.0 <= max_distance <= 2.0:
        raise ValueError("--max-distance must be between 0 and 2")

    profiles = load_registry(registry_path)
    detector = FaceDetector()
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {camera_index}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError(f"Could not read a frame from camera {camera_index}")

            frame = cv2.flip(frame, 1)
            for face in detector.detect(frame):
                embedding = create_embedding(face.crop)
                name, distance = match_embedding(embedding, profiles, max_distance)
                if name is None:
                    draw_face(frame, face, f"Unknown ({distance:.2f})", (0, 0, 255))
                else:
                    draw_face(frame, face, f"{name} ({distance:.2f})", (0, 200, 0))

            cv2.imshow("Face recognition - press ESC to exit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize multiple faces in a live camera stream."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help=f"Embedding registry (default: {DEFAULT_REGISTRY_PATH})",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.30,
        help=(
            "Maximum cosine distance for a match (default: 0.30). "
            "Calibrate with representative validation images."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_camera(args.camera, args.registry, args.max_distance)


if __name__ == "__main__":
    main()
