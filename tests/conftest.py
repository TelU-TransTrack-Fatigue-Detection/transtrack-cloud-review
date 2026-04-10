import numpy as np
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from httpx import AsyncClient, ASGITransport

from app.main import app
from app.config import settings


VALID_PAYLOAD = {
    "id": "1770249606650000020867395078806052",
    "imei": "867395078806052",
    "time": "05/02/2026 00:00:06",
    "alarm": "eyes_closed",
    "dms_video_url": "https://mdvr.transtrack.id/fake_video.mp4",
}


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path / "records"))
    with patch("app.main.process_alarm_task") as mock_task:
        mock_task.delay.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


@pytest.fixture
def records_dir(tmp_path):
    d = tmp_path / "records"
    d.mkdir()
    return d


@pytest.fixture
def synthetic_video(tmp_path):
    import cv2
    path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for _ in range(90):
        out.write(np.zeros((240, 320, 3), dtype=np.uint8))
    out.release()
    return path


@pytest.fixture
def black_video(tmp_path):
    import cv2
    path = tmp_path / "black_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))
    for _ in range(30):
        out.write(np.zeros((240, 320, 3), dtype=np.uint8))
    out.release()
    return path


def make_mock_landmark(x=0.5, y=0.5, z=0.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def make_face_landmarks(n=478):
    lms = [make_mock_landmark(0.5, 0.5, 0.0) for _ in range(n)]
    eye_indices = [33, 159, 158, 133, 153, 144, 263, 386, 385, 362, 380, 373]
    for i, idx in enumerate(eye_indices):
        lms[idx] = make_mock_landmark(0.4 + i * 0.01, 0.5 + (i % 2) * 0.02)
    mouth_indices = [13, 14, 61, 291]
    for i, idx in enumerate(mouth_indices):
        lms[idx] = make_mock_landmark(0.5, 0.4 + i * 0.05)
    return lms
