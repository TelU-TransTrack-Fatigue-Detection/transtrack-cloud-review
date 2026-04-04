import pytest
from unittest.mock import patch

from app.inference import run_inference, _read_video_properties

FAKE_VIDEO_INFO = {
    "total_frames": 300,
    "fps": 30.0,
    "duration_sec": 10.0,
    "resolution": "1280x720",
    "first_frame_read": True,
}

FAKE_DROWSY  = {"label": "yawning",     "class_id": 2, "confidence": 0.85}
FAKE_AWAKE   = {"label": "normal",      "class_id": 1, "confidence": 0.92}
FAKE_ASLEEP  = {"label": "eyes_closed", "class_id": 0, "confidence": 0.91}


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
async def test_run_inference_review_result_true_for_yawning():
    with _mock_video(), _mock_pipeline(FAKE_DROWSY):
        result = await run_inference("/tmp/video.mp4", "yawning")
    assert result["review_result"] is True
    assert result["other"]["label"] == "yawning"


@pytest.mark.asyncio
async def test_run_inference_review_result_true_for_eyes_closed():
    with _mock_video(), _mock_pipeline(FAKE_ASLEEP):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is True
    assert result["other"]["label"] == "eyes_closed"


@pytest.mark.asyncio
async def test_run_inference_review_result_false_for_normal():
    with _mock_video(), _mock_pipeline(FAKE_AWAKE):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is False
    assert result["other"]["label"] == "normal"


@pytest.mark.asyncio
async def test_run_inference_graceful_on_unreadable_video():
    result = await run_inference("/tmp/nonexistent.mp4", "eyes_closed")

    assert "error" in result["other"]["video_info"]
    assert result["other"]["video_info"]["total_frames"] == 0
    assert result["confidence_level"] == 0
    assert result["review_result"] is False

@pytest.mark.asyncio
async def test_run_inference_video_url_matches_path():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["video_url_after_process"] == "/tmp/video.mp4"


@pytest.mark.asyncio
async def test_run_inference_confidence_level_is_int():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert isinstance(result["confidence_level"], int)


@pytest.mark.asyncio
async def test_run_inference_confidence_level_0_to_100():
    with _mock_video(), _mock_pipeline({"label": "yawning", "class_id": 2, "confidence": 0.75}):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["confidence_level"] == 75


@pytest.mark.asyncio
async def test_run_inference_confidence_level_rounds_down():
    with _mock_video(), _mock_pipeline({"label": "normal", "class_id": 1, "confidence": 0.999}):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["confidence_level"] == 99


@pytest.mark.asyncio
async def test_run_inference_confidence_level_zero():
    with _mock_video(), _mock_pipeline({"label": "normal", "class_id": 1, "confidence": 0.0}):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["confidence_level"] == 0

@pytest.mark.asyncio
async def test_review_result_true_for_yawning():
    with _mock_video(), _mock_pipeline(FAKE_DROWSY):
        result = await run_inference("/tmp/video.mp4", "yawning")
    assert result["review_result"] is True
    assert result["other"]["label"] == "yawning"


@pytest.mark.asyncio
async def test_review_result_true_for_eyes_closed():
    with _mock_video(), _mock_pipeline(FAKE_ASLEEP):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is True
 
 
@pytest.mark.asyncio
async def test_review_result_false_for_normal():
    with _mock_video(), _mock_pipeline(FAKE_AWAKE):
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["review_result"] is False
    assert result["other"]["label"] == "normal"


@pytest.mark.asyncio
async def test_run_inference_video_info_in_other():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    assert result["other"]["video_info"] == FAKE_VIDEO_INFO


@pytest.mark.asyncio
async def test_run_inference_model_name_in_other():
    with _mock_video(), _mock_pipeline():
        result = await run_inference("/tmp/video.mp4", "eyes_closed")
    from app.config import settings
    assert result["other"]["model"] == settings.MODEL_NAME


@pytest.mark.asyncio
async def test_run_inference_graceful_skips_pipeline_for_bad_video():
    with patch("app.inference._run_pipeline") as mock_pipeline:
        await run_inference("/tmp/nonexistent.mp4", "eyes_closed")
    mock_pipeline.assert_not_called()


def test_read_video_properties_missing_file():
    props = _read_video_properties("/tmp/no_such_file_xyz.mp4")
    assert "error" in props
    assert props["total_frames"] == 0
    assert props["first_frame_read"] is False


def test_read_video_properties_valid_file(synthetic_video):
    props = _read_video_properties(str(synthetic_video))
    assert props["total_frames"] > 0
    assert props["fps"] > 0
    assert props["duration_sec"] > 0
    assert "x" in props["resolution"]
    assert "error" not in props
 
 
def test_read_video_properties_resolution_format(synthetic_video):
    props = _read_video_properties(str(synthetic_video))
    w, h = props["resolution"].split("x")
    assert int(w) > 0 and int(h) > 0