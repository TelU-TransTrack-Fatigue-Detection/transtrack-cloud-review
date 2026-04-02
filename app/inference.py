import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from .config import settings
from .pipeline import predict as pipeline_predict

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


def _read_video_properties(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "total_frames": 0,
            "fps": 0.0,
            "duration_sec": 0.0,
            "resolution": "0x0",
            "first_frame_read": False,
            "error": f"cv2 could not open: {video_path}",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = round(total_frames / fps, 3) if fps > 0 else 0.0

    ret, first_frame = cap.read()
    has_first_frame = ret and first_frame is not None
    cap.release()

    return {
        "total_frames": total_frames,
        "fps": round(fps, 3),
        "duration_sec": duration_sec,
        "resolution": f"{width}x{height}",
        "first_frame_read": has_first_frame,
    }


def _run_pipeline(video_path: str) -> dict:
    return pipeline_predict(video_path, Path(settings.MODEL_PATH), settings.MODEL_NAME)


async def run_inference(video_path: str, alarm: str) -> dict:
    loop = asyncio.get_event_loop()

    video_info = await loop.run_in_executor(_executor, _read_video_properties, video_path)

    if "error" in video_info:
        logger.warning("Cannot read video %s: %s", video_path, video_info["error"])
        return {
            "video_url_after_process": video_path,
            "confidence_level": 0,
            "review_result": False,
            "other": {
                "model": settings.MODEL_NAME,
                "alarm_type": alarm,
                "video_info": video_info,
            },
        }

    prediction = await loop.run_in_executor(_executor, _run_pipeline, video_path)

    confidence_level = int(prediction["confidence"] * 100)
    review_result = prediction["label"] != "normal"

    return {
        "video_url_after_process": video_path,
        "confidence_level": confidence_level,
        "review_result": review_result,
        "other": {
            "model": settings.MODEL_NAME,
            "alarm_type": alarm,
            "label": prediction["label"],
            "class_id": prediction["class_id"],
            "video_info": video_info,
        },
    }
