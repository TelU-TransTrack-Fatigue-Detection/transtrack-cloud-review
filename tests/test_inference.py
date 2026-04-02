import pytest
from unittest.mock import patch

from app.inference import run_inference

FAKE_VIDEO_INFO = {
    "total_frames": 300,
    "fps": 30.0,
    "duration_sec": 10.0,
    "resolution": "1280x720",
    "first_frame_read": True,
}

FAKE_DROWSY = {"label": "drowsy", "class_id": 1, "confidence": 0.85}
FAKE_AWAKE  = {"label": "awake",  "class_id": 0, "confidence": 0.92}


def _mock_video():
    return patch("app.inference._read_video_properties", return_value=FAKE_VIDEO_INFO)


def _mock_pipeline(result=None):
    return patch("app.inference._run_pipeline", return_value=result or FAKE_DROWSY)


@pytest.mark.asyncio
async def test_run_inference_returns_required_keys():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "eyes_closed")

    assert "video_url_after_process" in result
    assert "confidence_level" in result
    assert "review_result" in result
    assert "other" in result


@pytest.mark.asyncio
async def test_run_inference_confidence_level_in_range():
    with _mock_video(), _mock_pipeline({"label": "drowsy", "class_id": 1, "confidence": 0.75}):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["confidence_level"] == 75
    assert 0 <= result["confidence_level"] <= 100


@pytest.mark.asyncio
async def test_run_inference_review_result_is_bool():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert isinstance(result["review_result"], bool)


@pytest.mark.asyncio
async def test_run_inference_passes_alarm_type_to_other():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "yawning")
    assert result["other"]["alarm_type"] == "yawning"


@pytest.mark.asyncio
async def test_run_inference_video_info_included_in_other():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["other"]["video_info"] == FAKE_VIDEO_INFO


@pytest.mark.asyncio
async def test_run_inference_review_result_true_for_drowsy():
    with _mock_video(), _mock_pipeline(FAKE_DROWSY):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is True
    assert result["other"]["label"] == "drowsy"


@pytest.mark.asyncio
async def test_run_inference_review_result_false_for_awake():
    with _mock_video(), _mock_pipeline(FAKE_AWAKE):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is False
    assert result["other"]["label"] == "awake"


@pytest.mark.asyncio
async def test_run_inference_graceful_on_unreadable_video():
    result = await run_inference("/tmp/nonexistent.mp4", "eyes_closed")

    assert "error" in result["other"]["video_info"]
    assert result["other"]["video_info"]["total_frames"] == 0
    assert result["confidence_level"] == 0
    assert result["review_result"] is False
