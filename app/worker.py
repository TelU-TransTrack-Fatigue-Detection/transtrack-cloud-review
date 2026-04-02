import json
import logging
import os
import time
from pathlib import Path

import httpx
import torch
import torch.nn.functional as F
from celery import Celery
from celery.signals import worker_process_init

from .config import settings

logger = logging.getLogger(__name__)

celery_app = Celery("transtrack", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    worker_concurrency=settings.WORKER_CONCURRENCY or os.cpu_count(),
    broker_transport_options={"visibility_timeout": 3600},
)

_model  = None
_device = None


@worker_process_init.connect
def _init_worker(**kwargs):
    global _model, _device
    from .model_manager import ensure_all_models
    from .pipeline import _load_model

    ensure_all_models(Path(settings.MODEL_PATH), settings.MODEL_DOWNLOAD_URL)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = _load_model(Path(settings.MODEL_PATH), settings.MODEL_NAME, _device)
    logger.info("Worker process ready — model on %s", _device)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5, name="process_alarm")
def process_alarm_task(self, payload_dict: dict):
    from .pipeline import _extract, _prepare, CLASS_NAMES
    from .schemas import IncomingAlarm, ReviewResult

    payload    = IncomingAlarm(**payload_dict)
    start      = time.monotonic()
    video_path = None

    try:
        tmp_dir    = Path(settings.RECORDS_DIR) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        video_path = tmp_dir / f"{payload.id}.mp4"

        with httpx.Client(timeout=settings.VIDEO_DOWNLOAD_TIMEOUT) as client:
            with client.stream("GET", payload.dms_video_url) as resp:
                resp.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in resp.iter_bytes(1024 * 64):
                        f.write(chunk)

        device = _device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = _model
        if model is None:
            from .pipeline import _load_model
            model = _load_model(Path(settings.MODEL_PATH), settings.MODEL_NAME, device)

        features = _extract(video_path)
        tensor   = _prepare(features).to(device)

        with torch.no_grad():
            probs     = F.softmax(model(tensor), dim=-1)
            conf, cls = torch.max(probs, dim=-1)

        label    = CLASS_NAMES[cls.item()]
        conf_val = round(conf.item(), 4)
        duration = int((time.monotonic() - start) * 1000)

        result = ReviewResult(
            id=payload.id,
            imei=payload.imei,
            time=payload.time,
            alarm=payload.alarm,
            dms_video_url=payload.dms_video_url,
            dms_video_url_after_proccess=str(video_path),
            confidence_level=int(conf_val * 100),
            review_result=(label != "normal"),
            process_duration=duration,
            other={"model": settings.MODEL_NAME, "label": label, "class_id": cls.item()},
        )

        records_dir = Path(settings.RECORDS_DIR)
        records_dir.mkdir(parents=True, exist_ok=True)
        ts    = int(time.time() * 1000)
        fname = f"{payload.id}_{payload.imei}_{ts}.json"
        (records_dir / fname).write_text(
            json.dumps({"original": payload.model_dump(), "result": result.model_dump()})
        )

        for attempt in range(3):
            try:
                with httpx.Client(timeout=settings.HTTPX_TIMEOUT) as cb_client:
                    r = cb_client.post(settings.CALLBACK_URL, json=result.model_dump())
                    r.raise_for_status()
                break
            except Exception as exc:
                if attempt == 2:
                    logger.error("Callback failed after 3 attempts: %s", exc)
                else:
                    time.sleep(2 ** attempt)

        logger.info("Done id=%s label=%s conf=%.4f duration_ms=%d",
                    payload.id, label, conf_val, duration)

    except Exception as exc:
        logger.error("Task failed alarm=%s: %s", payload.id, exc)
        raise self.retry(exc=exc)

    finally:
        if video_path and not settings.KEEP_TMP_VIDEOS:
            Path(video_path).unlink(missing_ok=True)
