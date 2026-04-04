"""Tests for app.worker — Celery task structure, retry logic, and task registration."""
import json
import time
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from app.config import settings
from app.schemas import IncomingAlarm


VALID_PAYLOAD = {
    "id": "worker-test-001",
    "imei": "867395078806052",
    "time": "05/02/2026 00:00:06",
    "alarm": "eyes_closed",
    "dms_video_url": "https://example.com/video.mp4",
}


# ─── Celery App Config ────────────────────────────────────────────────────────

def test_celery_app_is_configured():
    from app.worker import celery_app
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_process_alarm_task_is_registered():
    from app.worker import celery_app
    assert "process_alarm" in celery_app.tasks


def test_celery_app_uses_redis_broker():
    from app.worker import celery_app
    assert "redis" in celery_app.conf.broker_url


# ─── Task Result: Success Path ────────────────────────────────────────────────

def _make_task_mocks(tmp_path):
    """Set up all mocks needed for a successful worker task run."""
    fake_video = tmp_path / "worker-test-001.mp4"
    fake_video.write_bytes(b"fake video content")

    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_bytes = MagicMock(return_value=iter([b"chunk1", b"chunk2"]))

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.stream = MagicMock(return_value=mock_response)

    return fake_video, mock_client


def test_worker_task_saves_json_record(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "KEEP_TMP_VIDEOS", True)

    from app.worker import process_alarm_task
    from app.pipeline import MultiScaleTCN, SEQUENCE_LENGTH, CLASS_NAMES

    dummy_model = MultiScaleTCN(num_classes=3)
    dummy_model.eval()

    fake_features = np.zeros((SEQUENCE_LENGTH, 8), dtype=np.float32)
    fake_logits = torch.zeros(1, 3)

    mock_cb_response = MagicMock()
    mock_cb_response.raise_for_status = MagicMock()
    mock_cb_client = MagicMock()
    mock_cb_client.__enter__ = MagicMock(return_value=mock_cb_client)
    mock_cb_client.__exit__ = MagicMock(return_value=False)
    mock_cb_client.post = MagicMock(return_value=mock_cb_response)

    mock_dl_response = MagicMock()
    mock_dl_response.__enter__ = MagicMock(return_value=mock_dl_response)
    mock_dl_response.__exit__ = MagicMock(return_value=False)
    mock_dl_response.raise_for_status = MagicMock()
    mock_dl_response.iter_bytes = MagicMock(return_value=iter([b""]))

    mock_dl_client = MagicMock()
    mock_dl_client.__enter__ = MagicMock(return_value=mock_dl_client)
    mock_dl_client.__exit__ = MagicMock(return_value=False)
    mock_dl_client.stream = MagicMock(return_value=mock_dl_response)

    with (
        patch("app.worker._model", dummy_model),
        patch("app.worker._device", torch.device("cpu")),
        # Change app.worker to app.pipeline for these two:
        patch("app.pipeline._extract", return_value=fake_features),
        patch("app.pipeline._prepare", return_value=torch.zeros(1, 8, SEQUENCE_LENGTH)),
        patch("app.worker.httpx.Client", side_effect=[mock_dl_client, mock_cb_client]),
    ):
        process_alarm_task(VALID_PAYLOAD)

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    record = json.loads(json_files[0].read_text())
    assert record["original"]["id"] == VALID_PAYLOAD["id"]


def test_worker_task_callback_called(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "KEEP_TMP_VIDEOS", True)

    from app.worker import process_alarm_task
    from app.pipeline import MultiScaleTCN, SEQUENCE_LENGTH

    dummy_model = MultiScaleTCN(num_classes=3)
    dummy_model.eval()

    mock_cb_response = MagicMock()
    mock_cb_response.raise_for_status = MagicMock()
    mock_cb_client = MagicMock()
    mock_cb_client.__enter__ = MagicMock(return_value=mock_cb_client)
    mock_cb_client.__exit__ = MagicMock(return_value=False)
    mock_cb_client.post = MagicMock(return_value=mock_cb_response)

    mock_dl_response = MagicMock()
    mock_dl_response.__enter__ = MagicMock(return_value=mock_dl_response)
    mock_dl_response.__exit__ = MagicMock(return_value=False)
    mock_dl_response.raise_for_status = MagicMock()
    mock_dl_response.iter_bytes = MagicMock(return_value=iter([b""]))

    mock_dl_client = MagicMock()
    mock_dl_client.__enter__ = MagicMock(return_value=mock_dl_client)
    mock_dl_client.__exit__ = MagicMock(return_value=False)
    mock_dl_client.stream = MagicMock(return_value=mock_dl_response)

    with (
        patch("app.worker._model", dummy_model),
        patch("app.worker._device", torch.device("cpu")),
        # Change app.worker to app.pipeline for these two:
        patch("app.pipeline._extract", return_value=np.zeros((SEQUENCE_LENGTH, 8), dtype=np.float32)),
        patch("app.pipeline._prepare", return_value=torch.zeros(1, 8, SEQUENCE_LENGTH)),
        patch("app.worker.httpx.Client", side_effect=[mock_dl_client, mock_cb_client]),
    ):
        process_alarm_task(VALID_PAYLOAD)

    mock_cb_client.post.assert_called_once()
    post_url = mock_cb_client.post.call_args[0][0]
    assert post_url == settings.CALLBACK_URL


def test_worker_task_result_label_is_valid_class(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "KEEP_TMP_VIDEOS", True)

    from app.worker import process_alarm_task
    from app.pipeline import MultiScaleTCN, SEQUENCE_LENGTH, CLASS_NAMES

    dummy_model = MultiScaleTCN(num_classes=3)
    dummy_model.eval()

    callback_payload = {}

    mock_cb_response = MagicMock()
    mock_cb_response.raise_for_status = MagicMock()
    mock_cb_client = MagicMock()
    mock_cb_client.__enter__ = MagicMock(return_value=mock_cb_client)
    mock_cb_client.__exit__ = MagicMock(return_value=False)

    def capture_post(url, **kwargs):
        callback_payload.update(kwargs.get("json", {}))
        return mock_cb_response

    mock_cb_client.post = MagicMock(side_effect=capture_post)

    mock_dl_response = MagicMock()
    mock_dl_response.__enter__ = MagicMock(return_value=mock_dl_response)
    mock_dl_response.__exit__ = MagicMock(return_value=False)
    mock_dl_response.raise_for_status = MagicMock()
    mock_dl_response.iter_bytes = MagicMock(return_value=iter([b""]))

    mock_dl_client = MagicMock()
    mock_dl_client.__enter__ = MagicMock(return_value=mock_dl_client)
    mock_dl_client.__exit__ = MagicMock(return_value=False)
    mock_dl_client.stream = MagicMock(return_value=mock_dl_response)

    with (
        patch("app.worker._model", dummy_model),
        patch("app.worker._device", torch.device("cpu")),
        # Change app.worker to app.pipeline for these two:
        patch("app.pipeline._extract", return_value=np.zeros((SEQUENCE_LENGTH, 8), dtype=np.float32)),
        patch("app.pipeline._prepare", return_value=torch.zeros(1, 8, SEQUENCE_LENGTH)),
        patch("app.worker.httpx.Client", side_effect=[mock_dl_client, mock_cb_client]),
    ):
        process_alarm_task(VALID_PAYLOAD)

    label = callback_payload.get("other", {}).get("label", "")
    assert label in CLASS_NAMES


# ─── Retry on Failure ────────────────────────────────────────────────────────

def test_worker_task_retries_on_download_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))

    from app.worker import process_alarm_task

    mock_dl_client = MagicMock()
    mock_dl_client.__enter__ = MagicMock(return_value=mock_dl_client)
    mock_dl_client.__exit__ = MagicMock(return_value=False)
    mock_dl_client.stream = MagicMock(side_effect=Exception("connection error"))

    with (
        patch("app.worker._model", MagicMock()),
        patch("app.worker._device", MagicMock()),
        patch("app.pipeline._extract"),
        patch("app.pipeline._prepare"),
        patch("app.worker.httpx.Client", return_value=mock_dl_client),
        patch.object(process_alarm_task, "retry", side_effect=Exception("retry")) as mock_retry
    ):
        with pytest.raises(Exception, match="retry"):
            process_alarm_task(VALID_PAYLOAD)
            
        mock_retry.assert_called_once()
# ─── Video Cleanup ───────────────────────────────────────────────────────────

def test_worker_deletes_tmp_video_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "KEEP_TMP_VIDEOS", False)

    from app.worker import process_alarm_task
    from app.pipeline import MultiScaleTCN, SEQUENCE_LENGTH

    dummy_model = MultiScaleTCN(num_classes=3)
    dummy_model.eval()

    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    video_path = tmp_dir / f"{VALID_PAYLOAD['id']}.mp4"

    mock_cb_response = MagicMock()
    mock_cb_response.raise_for_status = MagicMock()
    mock_cb_client = MagicMock()
    mock_cb_client.__enter__ = MagicMock(return_value=mock_cb_client)
    mock_cb_client.__exit__ = MagicMock(return_value=False)
    mock_cb_client.post = MagicMock(return_value=mock_cb_response)

    mock_dl_response = MagicMock()
    mock_dl_response.__enter__ = MagicMock(return_value=mock_dl_response)
    mock_dl_response.__exit__ = MagicMock(return_value=False)
    mock_dl_response.raise_for_status = MagicMock()

    def write_fake_video(chunk_size=None):
        video_path.write_bytes(b"fake")
        return iter([b""])

    mock_dl_response.iter_bytes = MagicMock(side_effect=write_fake_video)

    mock_dl_client = MagicMock()
    mock_dl_client.__enter__ = MagicMock(return_value=mock_dl_client)
    mock_dl_client.__exit__ = MagicMock(return_value=False)
    mock_dl_client.stream = MagicMock(return_value=mock_dl_response)

    with (
        patch("app.worker._model", dummy_model),
        patch("app.worker._device", torch.device("cpu")),
        patch("app.pipeline._extract", return_value=np.zeros((SEQUENCE_LENGTH, 8), dtype=np.float32)),
        patch("app.pipeline._prepare", return_value=torch.zeros(1, 8, SEQUENCE_LENGTH)),
        patch("app.worker.httpx.Client", side_effect=[mock_dl_client, mock_cb_client]),
    ):
        process_alarm_task(VALID_PAYLOAD)

    assert not video_path.exists()