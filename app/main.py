import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from .config import settings
from .model_manager import ensure_all_models
from .schemas import IncomingAlarm
from .worker import process_alarm_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

async def _cleanup_loop():
    while True:
        await asyncio.sleep(60)
        now         = time.time()
        records_dir = Path(settings.RECORDS_DIR)

        for f in records_dir.glob("*.json"):
            try:
                if now - f.stat().st_mtime > settings.RECORD_TTL_SECONDS:
                    f.unlink()
                    logger.info("Expired record deleted: %s", f.name)
            except Exception:
                pass

        for f in (records_dir / "tmp").glob("*.mp4"):
            try:
                if now - f.stat().st_mtime > settings.VIDEO_TTL_SECONDS:
                    f.unlink()
                    logger.info("Expired video deleted: %s", f.name)
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.RECORDS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.RECORDS_DIR, "tmp").mkdir(parents=True, exist_ok=True)
    try:
        ensure_all_models(Path(settings.MODEL_PATH), settings.MODEL_DOWNLOAD_URL)
    except FileNotFoundError as e:
        logger.warning("Model not ready at startup: %s", e)
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="TransTRACK Cloud Review Engine", lifespan=lifespan)


def _queue_depth() -> int:
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.REDIS_URL)
        return r.llen("celery")
    except Exception:
        return 0


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "ok", "queue_depth": _queue_depth()}


@app.get("/videos/{alarm_id}", status_code=status.HTTP_200_OK)
async def get_video(alarm_id: str):
    """Serve processed video for TransTrack review (available for 45 min after processing)."""
    video_path = Path(settings.RECORDS_DIR) / "tmp" / f"{alarm_id}.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or already expired",
        )
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{alarm_id}.mp4",
    )


@app.post("/review", status_code=status.HTTP_202_ACCEPTED)
async def review(payload: IncomingAlarm):
    if _queue_depth() >= settings.MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue is full, try again later",
        )
    process_alarm_task.delay(payload.model_dump())
    logger.info("Queued alarm id=%s alarm=%s", payload.id, payload.alarm)
    return {"status": "queued", "id": payload.id}
