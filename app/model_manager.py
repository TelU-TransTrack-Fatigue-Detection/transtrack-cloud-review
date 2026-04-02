import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_MEDIAPIPE_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker"
    "/face_landmarker/float16/1/face_landmarker.task"
)
_MEDIAPIPE_PATH = Path("models/mediapipe/face_landmarker.task")


def ensure_mediapipe():
    if _MEDIAPIPE_PATH.exists():
        return
    _MEDIAPIPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe face_landmarker model...")
    urllib.request.urlretrieve(_MEDIAPIPE_URL, _MEDIAPIPE_PATH)
    logger.info("MediaPipe model ready: %s", _MEDIAPIPE_PATH)


def ensure_classifier(model_path: Path, download_url: str = ""):
    if model_path.exists():
        return
    if not download_url:
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            "Set MODEL_DOWNLOAD_URL in .env to enable auto-download."
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading classifier weights from %s ...", download_url)
    urllib.request.urlretrieve(download_url, model_path)
    logger.info("Classifier weights ready: %s", model_path)


def ensure_all_models(model_path: Path, download_url: str = ""):
    ensure_mediapipe()
    ensure_classifier(model_path, download_url)
