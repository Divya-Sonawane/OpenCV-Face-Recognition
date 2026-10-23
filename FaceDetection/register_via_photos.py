import argparse
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FaceDetection.face_detector import FaceDetector
from FaceRecognition.embedding_extractor import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    create_embedding,
)

DATABASE_DIR = PROJECT_ROOT / "database"
DEFAULT_IMAGES_DIR = DATABASE_DIR / "raw_images"
DEFAULT_REGISTRY_PATH = DATABASE_DIR / "embeddings.pkl"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOGGER = logging.getLogger("create_embeddings")


def build_registry(images_dir: Path) -> dict[str, list[dict[str, object]]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Raw image directory does not exist: {images_dir}")

    detector = FaceDetector()
    registry: dict[str, list[dict[str, object]]] = {}
    for employee_dir in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        employee_name = employee_dir.name.strip()
        if not employee_name:
            LOGGER.warning("Skipping employee folder with an empty name: %s", employee_dir)
            continue

        templates: list[dict[str, object]] = []
        image_paths = sorted(
            path
            for path in employee_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not image_paths:
            LOGGER.warning("No supported images found for %s", employee_name)
            continue

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("Skipping unreadable image: %s", image_path)
                continue

            faces = detector.detect(image)
            if len(faces) != 1:
                LOGGER.warning(
                    "Skipping %s: expected exactly one face, detected %d",
                    image_path,
                    len(faces),
                )
                continue

            embedding = create_embedding(faces[0].crop)
            templates.append(
                {
                    "source": image_path.relative_to(images_dir).as_posix(),
                    "embedding": embedding.tolist(),
                }
            )
            LOGGER.info("Registered %s from %s", employee_name, image_path.name)

        if templates:
            registry[employee_name] = templates

    if not registry:
        raise RuntimeError(
            f"No face templates were created from images in {images_dir}"
        )
    return registry


def write_registry(registry_path: Path, registry: dict[str, list[dict[str, object]]]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "profiles": registry,
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=registry_path.parent,
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            pickle.dump(payload, temporary_file, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, registry_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register FaceNet512 embeddings from employee photos."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Employee image root (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help=f"Registry output path (default: {DEFAULT_REGISTRY_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    registry = build_registry(args.images)
    write_registry(args.output, registry)
    LOGGER.info(
        "Saved %d templates for %d employees to %s",
        sum(len(templates) for templates in registry.values()),
        len(registry),
        args.output,
    )


if __name__ == "__main__":
    main()
