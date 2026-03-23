import asyncio
import json
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient, ASGITransport

from main import app
from config import settings


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
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def client_with_mocks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path / "records"))

    with (
        patch("main.download_video", new_callable=AsyncMock) as mock_dl,
        patch("main.run_inference", new_callable=AsyncMock) as mock_infer,
        patch("main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        mock_dl.return_value = "/tmp/fake.mp4"
        mock_infer.return_value = ALWAYS_TRUE_INFERENCE
        mock_post.return_value = None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac, mock_dl, mock_infer, mock_post


@pytest.fixture
def records_dir(tmp_path):
    d = tmp_path / "records"
    d.mkdir()
    return d
