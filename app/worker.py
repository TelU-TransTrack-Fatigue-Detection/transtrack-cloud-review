import json
import logging
import os
import time
from datetime import datetime, timezone
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

# --- heuristic thresholds ---
_NIGHT_HOURS            = range(0, 6)       # 00:00–05:59 local
_NIGHT_CONF_FLOOR       = 0.75              # "normal" below this at night → review
_HISTORY_WINDOW_SEC     = 3600             # sliding window for driver history
_HISTORY_ESCALATE_COUNT = 2               # fatigue events in window before escalation
_DRIVER_KEY             = "driver_fatigue:{imei}"

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


# ---------------------------------------------------------------------------
# Driver history helpers (Redis)
# ---------------------------------------------------------------------------

def _redis_client():
    import redis as redis_lib
    return redis_lib.from_url(settings.REDIS_URL)


def _record_driver_fatigue(imei: str):
    try:
        key = _DRIVER_KEY.format(imei=imei)
        r   = _redis_client()
        r.rpush(key, int(time.time()))
        r.expire(key, _HISTORY_WINDOW_SEC * 2)
    except Exception as exc:
        logger.warning("Could not record driver history imei=%s: %s", imei, exc)


def _recent_fatigue_count(imei: str) -> int:
    try:
        key    = _DRIVER_KEY.format(imei=imei)
        r      = _redis_client()
        events = r.lrange(key, 0, -1)
        cutoff = int(time.time()) - _HISTORY_WINDOW_SEC
        return sum(1 for ts in events if int(ts) >= cutoff)
    except Exception as exc:
        logger.warning("Could not read driver history imei=%s: %s", imei, exc)
        return 0


# ---------------------------------------------------------------------------
# Heuristics post-processing
# ---------------------------------------------------------------------------

def _apply_heuristics(label: str, conf: float, imei: str, alarm_time_str: str) -> tuple[bool, dict]:
    """
    Returns (review_result, heuristic_meta).

    Rules applied on top of raw model output:
      1. Night-hour rule  — if model says "normal" with low confidence during 00:00–05:59,
                            flag for review (eyes_closed events peak in this window).
      2. Driver-history   — if same IMEI has had >= N fatigue events in the last hour,
                            treat a borderline "normal" as a review trigger.
    """
    night_triggered   = False
    history_triggered = False

    # parse hour from alarm_time (ISO-8601 or HH:MM:SS prefix)
    try:
        hour = datetime.fromisoformat(alarm_time_str).hour
    except Exception:
        try:
            hour = int(alarm_time_str.split("T")[-1].split(":")[0])
        except Exception:
            hour = -1

    if label == "normal" and hour in _NIGHT_HOURS and conf < _NIGHT_CONF_FLOOR:
        night_triggered = True
        logger.info("Night-hour rule triggered imei=%s hour=%d conf=%.3f", imei, hour, conf)

    recent = _recent_fatigue_count(imei)
    if label == "normal" and recent >= _HISTORY_ESCALATE_COUNT:
        history_triggered = True
        logger.info("History escalation triggered imei=%s recent_events=%d", imei, recent)

    review_result = (label != "normal") or night_triggered or history_triggered
    return review_result, {
        "night_rule_triggered":    night_triggered,
        "history_rule_triggered":  history_triggered,
        "recent_fatigue_count":    recent,
        "alarm_hour":              hour if hour >= 0 else None,
    }


# ---------------------------------------------------------------------------
# Error record
# ---------------------------------------------------------------------------

def _write_error_record(payload_dict: dict, stage: str, exc: Exception):
    try:
        records_dir = Path(settings.RECORDS_DIR)
        records_dir.mkdir(parents=True, exist_ok=True)
        ts    = int(time.time() * 1000)
        fname = f"ERR_{payload_dict.get('id', 'unknown')}_{ts}.json"
        (records_dir / fname).write_text(json.dumps({
            "status":    "error",
            "stage":     stage,
            "error":     str(exc),
            "error_type": type(exc).__name__,
            "payload":   payload_dict,
            "ts":        ts,
        }))
    except Exception as write_exc:
        logger.warning("Could not write error record: %s", write_exc)


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=5, name="process_alarm")
def process_alarm_task(self, payload_dict: dict):
    from .pipeline import _extract, _prepare, CLASS_NAMES
    from .schemas import IncomingAlarm, ReviewResult

    payload    = IncomingAlarm(**payload_dict)
    start      = time.monotonic()
    video_path = None
    stage      = "init"

    try:
        tmp_dir = Path(settings.RECORDS_DIR) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        video_path = tmp_dir / f"{payload.id}.mp4"

        stage = "download"
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

        stage    = "extract"
        features = _extract(video_path)

        stage  = "infer"
        tensor = _prepare(features).to(device)
        with torch.no_grad():
            probs      = F.softmax(model(tensor), dim=-1)
            conf, cls  = torch.max(probs, dim=-1)

        label    = CLASS_NAMES[cls.item()]
        conf_val = round(conf.item(), 4)
        duration = int((time.monotonic() - start) * 1000)

        review_result, heuristic_meta = _apply_heuristics(
            label, conf_val, payload.imei, payload.time
        )

        if label != "normal":
            _record_driver_fatigue(payload.imei)

        result = ReviewResult(
            id=payload.id,
            imei=payload.imei,
            time=payload.time,
            alarm=payload.alarm,
            dms_video_url=payload.dms_video_url,
            dms_video_url_after_proccess="-",
            confidence_level=int(conf_val * 100),
            review_result=review_result,
            process_duration=duration,
            other={
                "model":    settings.MODEL_NAME,
                "label":    label,
                "class_id": cls.item(),
                **heuristic_meta,
            },
        )

        records_dir = Path(settings.RECORDS_DIR)
        records_dir.mkdir(parents=True, exist_ok=True)
        ts    = int(time.time() * 1000)
        fname = f"{payload.id}_{payload.imei}_{ts}.json"
        (records_dir / fname).write_text(
            json.dumps({"original": payload.model_dump(), "result": result.model_dump()})
        )

        stage    = "callback"
        last_exc = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=settings.HTTPX_TIMEOUT) as cb_client:
                    r = cb_client.post(settings.CALLBACK_URL, json=result.model_dump())
                    r.raise_for_status()
                    logger.info("Callback delivered id=%s (attempt %d)", payload.id, attempt + 1)
                    last_exc = None
                    break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if last_exc:
            logger.error("Callback failed after 3 attempts id=%s: %s", payload.id, last_exc)
            _write_error_record(payload_dict, "callback", last_exc)

        logger.info("Done id=%s label=%s conf=%.4f review=%s duration_ms=%d night=%s history=%s",
                    payload.id, label, conf_val, review_result, duration,
                    heuristic_meta["night_rule_triggered"], heuristic_meta["history_rule_triggered"])

    except Exception as exc:
        logger.error("Task failed id=%s stage=%s error=%s: %s",
                     payload.id, stage, type(exc).__name__, exc, exc_info=True)
        _write_error_record(payload_dict, stage, exc)
        raise self.retry(exc=exc)

    finally:
        if video_path and not settings.KEEP_TMP_VIDEOS:
            Path(video_path).unlink(missing_ok=True)
