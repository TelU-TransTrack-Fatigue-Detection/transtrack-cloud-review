import asyncio
import json
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient, ASGITransport

from app.main import app
from app.config import settings


ALWAYS_TRUE_INFERENCE = {
    "video_url_after_process": "/tmp/fake.mp4",
    "confidence_level": 100,
    "review_result": True,
    "other": {"model": "stub", "alarm_type": "eyes_closed"},
}

VALID_PAYLOAD = {
    "id": "1770249606650000020867395078806052",
    "imei": "867395078806052",
    "time": "05/02/2026 00:00:06",
    "alarm": "eyes_closed",
    "dms_video_url": "https://mdvr.transtrack.id/fake_video.mp4",
}

VALID_HEADERS = {"Authorization": f"Bearer {settings.API_KEY}"}


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path / "records"))
    with patch("app.main.process_alarm_task") as mock_task:
        mock_task.delay.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


@pytest_asyncio.fixture
async def client_with_mocks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path / "records"))

    with (
        patch("app.main.download_video", new_callable=AsyncMock) as mock_dl,
        patch("app.main.run_inference", new_callable=AsyncMock) as mock_infer,
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        mock_dl.return_value = "/tmp/fake.mp4"
        mock_infer.return_value = ALWAYS_TRUE_INFERENCE
        mock_post.return_value = None

        async def _run_process_alarm(payload_dict):
            from app.main import process_alarm
            from app.schemas import IncomingAlarm
            await process_alarm(IncomingAlarm(**payload_dict))

        with patch("app.main.process_alarm_task") as mock_task:
            mock_task.delay.side_effect = lambda pd: asyncio.create_task(
                _run_process_alarm(pd)
            )
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                yield ac, mock_dl, mock_infer, mock_post


@pytest.fixture
def records_dir(tmp_path):
    d = tmp_path / "records"
    d.mkdir()
    return d
