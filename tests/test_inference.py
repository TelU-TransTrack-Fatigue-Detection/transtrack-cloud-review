import pytest
from unittest.mock import patch, AsyncMock

from app.inference import run_inference

FAKE_VIDEO_INFO = {
    "total_frames": 300,
    "fps": 30.0,
    "duration_sec": 10.0,
    "resolution": "1280x720",
    "first_frame_read": True,
}


def _mock_video():
    return patch("app.inference._read_video_properties", return_value=FAKE_VIDEO_INFO)


@pytest.mark.asyncio
async def test_run_inference_returns_required_keys():
    with _mock_video(), patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")

    assert "video_url_after_process" in result
    assert "confidence_level" in result
    assert "review_result" in result
    assert "other" in result


@pytest.mark.asyncio
async def test_run_inference_confidence_level_in_range():
    with _mock_video(), patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        for _ in range(20):
            result = await run_inference("/tmp/video.mp4", "eyes_closed")
            assert 0 <= result["confidence_level"] <= 100


@pytest.mark.asyncio
async def test_run_inference_review_result_is_bool():
    with _mock_video(), patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert isinstance(result["review_result"], bool)


@pytest.mark.asyncio
async def test_run_inference_passes_alarm_type_to_other():
    with _mock_video(), patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        result = await run_inference("/tmp/video.mp4", "yawning")
    assert result["other"]["alarm_type"] == "yawning"


@pytest.mark.asyncio
async def test_run_inference_video_info_included_in_other():
    with _mock_video(), patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["other"]["video_info"] == FAKE_VIDEO_INFO


@pytest.mark.asyncio
async def test_run_inference_always_true_mock():
    async def always_true(video_path, alarm):
        return {
            "video_url_after_process": video_path,
            "confidence_level": 100,
            "review_result": True,
            "other": {"alarm_type": alarm},
        }

    with patch("app.inference.run_inference", side_effect=always_true):
        from app.inference import run_inference as patched
        result = await patched("/tmp/video.mp4", "eyes_closed")
        assert result["review_result"] is True
        assert result["confidence_level"] == 100


@pytest.mark.asyncio
async def test_run_inference_review_result_true_when_confidence_high():
    with _mock_video(), patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        with patch("app.inference.random.randint", return_value=95):
            result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is True


@pytest.mark.asyncio
async def test_run_inference_review_result_false_when_confidence_low():
    with _mock_video(), patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        with patch("app.inference.random.randint", return_value=40):
            result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is False


@pytest.mark.asyncio
async def test_run_inference_graceful_on_unreadable_video():
    with patch("app.inference.asyncio.sleep", new_callable=AsyncMock):
        result = await run_inference("/tmp/nonexistent.mp4", "eyes_closed")

    assert "error" in result["other"]["video_info"]
    assert result["other"]["video_info"]["total_frames"] == 0
    assert isinstance(result["confidence_level"], int)
    assert isinstance(result["review_result"], bool)
