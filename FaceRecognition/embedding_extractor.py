"""FaceNet512 embedding generation for aligned face crops."""

import numpy as np
from deepface import DeepFace


MODEL_NAME = "Facenet512"
EMBEDDING_DIMENSION = 512


def create_embedding(face_crop: np.ndarray) -> np.ndarray:
    """Create a normalized FaceNet512 embedding from one aligned BGR face crop."""
    if face_crop.size == 0 or face_crop.ndim != 3 or face_crop.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR face crop")

    representations = DeepFace.represent(
        img_path=face_crop,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        normalization="Facenet",
    )
    if len(representations) != 1:
        raise RuntimeError(
            f"Expected one embedding for a detected face, got {len(representations)}"
        )

    embedding = np.asarray(representations[0]["embedding"], dtype=np.float32)
    if embedding.shape != (EMBEDDING_DIMENSION,):
        raise RuntimeError(
            f"{MODEL_NAME} returned {embedding.size} values; "
            f"expected {EMBEDDING_DIMENSION}"
        )
    if not np.isfinite(embedding).all():
        raise RuntimeError(f"{MODEL_NAME} returned a non-finite embedding")

    norm = float(np.linalg.norm(embedding))
    if norm == 0:
        raise RuntimeError(f"{MODEL_NAME} returned a zero-length embedding")

    return embedding / norm
