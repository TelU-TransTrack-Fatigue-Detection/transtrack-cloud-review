import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.main import process_alarm
from app.schemas import IncomingAlarm
from conftest import VALID_PAYLOAD, ALWAYS_TRUE_INFERENCE


def _payload():
    return IncomingAlarm(**VALID_PAYLOAD)


def _mocks(tmp_path=None):
    patches = [
        patch("app.main.download_video", new_callable=AsyncMock),
        patch("app.main.run_inference", new_callable=AsyncMock),
        patch("app.main.post_result", new_callable=AsyncMock),
    ]
    return patches


@pytest.mark.asyncio
async def test_pipeline_calls_inference_with_video_and_alarm():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE) as mock_infer,
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())
    mock_infer.assert_called_once_with("/tmp/fake.mp4", VALID_PAYLOAD["alarm"])


@pytest.mark.asyncio
async def test_pipeline_downloads_video_from_payload_url():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4") as mock_dl,
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())
    mock_dl.assert_called_once_with(VALID_PAYLOAD["dms_video_url"], VALID_PAYLOAD["id"])


@pytest.mark.asyncio
async def test_pipeline_posts_callback_once():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_callback_review_result_and_confidence():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.review_result is True
    assert result.confidence_level == 100


@pytest.mark.asyncio
async def test_pipeline_posts_callback_with_review_true():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    assert mock_post.called
    result = mock_post.call_args[0][0]
    assert result.review_result is True
    assert result.confidence_level == 100


@pytest.mark.asyncio
async def test_pipeline_callback_contains_original_fields():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.id    == VALID_PAYLOAD["id"]
    assert result.imei  == VALID_PAYLOAD["imei"]
    assert result.alarm == VALID_PAYLOAD["alarm"]
    assert result.time  == VALID_PAYLOAD["time"]
    assert result.dms_video_url == VALID_PAYLOAD["dms_video_url"]


@pytest.mark.asyncio
async def test_pipeline_saves_record_to_disk(tmp_path, monkeypatch):
    from app.config import settings
    records_path = tmp_path / "records"
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_path))
    records_path.mkdir()

    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())

    json_files = list(records_path.glob("*.json"))
    assert len(json_files) == 1
    record = json.loads(json_files[0].read_text())
    assert record["original"]["id"] == VALID_PAYLOAD["id"]
    assert record["result"]["review_result"] is True
    assert record["result"]["confidence_level"] == 100


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_download_failure():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, side_effect=Exception("network error")),
        patch("app.main.run_inference", new_callable=AsyncMock) as mock_infer,
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    mock_infer.assert_not_called()
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_inference_failure():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, side_effect=Exception("model error")),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_callback_failure():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock, side_effect=RuntimeError("callback failed")),
    ):
        await process_alarm(_payload())


@pytest.mark.asyncio
async def test_pipeline_process_duration_is_positive():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.process_duration >= 0


@pytest.mark.asyncio
async def test_pipeline_process_duration_is_non_negative():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.process_duration >= 0


@pytest.mark.asyncio
async def test_pipeline_review_result_false_when_inference_normal():
    normal_inference = {
        "video_url_after_process": "/tmp/fake.mp4",
        "confidence_level": 92,
        "review_result": False,
        "other": {"model": "stub", "label": "normal", "alarm_type": "eyes_closed"},
    }
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=normal_inference),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.review_result is False


@pytest.mark.asyncio
async def test_pipeline_dms_video_url_after_proccess_set():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/abc.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value={
            **ALWAYS_TRUE_INFERENCE, "video_url_after_process": "/tmp/abc.mp4"
        }),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.dms_video_url_after_proccess == "/tmp/abc.mp4"


@pytest.mark.asyncio
async def test_pipeline_record_json_structure(tmp_path, monkeypatch):
    from app.config import settings
    records_path = tmp_path / "records"
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_path))
    records_path.mkdir()
 
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())
 
    record = json.loads(list(records_path.glob("*.json"))[0].read_text())
    assert record["original"]["id"] == VALID_PAYLOAD["id"]
    assert record["result"]["review_result"] is True
    assert record["result"]["confidence_level"] == 100


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_save_failure(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path / "records"))
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.save_record", new_callable=AsyncMock, side_effect=IOError("disk full")),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())


@pytest.mark.asyncio
async def test_pipeline_deletes_tmp_video_by_default(tmp_path, monkeypatch):
    from app.config import settings
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"fake")
    monkeypatch.setattr(settings, "KEEP_TMP_VIDEOS", False)
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path / "records"))
    (tmp_path / "records").mkdir()
 
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value=str(fake_video)),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value={
            **ALWAYS_TRUE_INFERENCE, "video_url_after_process": str(fake_video)
        }),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())
 
    assert not fake_video.exists()


@pytest.mark.asyncio
async def test_pipeline_keeps_tmp_video_when_flag_set(tmp_path, monkeypatch):
    from app.config import settings
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"fake")
    monkeypatch.setattr(settings, "KEEP_TMP_VIDEOS", True)
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path / "records"))
    (tmp_path / "records").mkdir()
 
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value=str(fake_video)),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value={
            **ALWAYS_TRUE_INFERENCE, "video_url_after_process": str(fake_video)
        }),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())
 
    assert fake_video.exists()