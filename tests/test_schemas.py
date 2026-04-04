"""Tests for app.schemas — Pydantic model validation and serialization."""
import pytest
from pydantic import ValidationError

from app.schemas import IncomingAlarm, ReviewResult


# ─── IncomingAlarm ────────────────────────────────────────────────────────────

VALID_ALARM = {
    "id": "alarm-001",
    "imei": "867395078806052",
    "time": "05/02/2026 00:00:06",
    "alarm": "eyes_closed",
    "dms_video_url": "https://example.com/video.mp4",
}


def test_incoming_alarm_valid():
    alarm = IncomingAlarm(**VALID_ALARM)
    assert alarm.id == "alarm-001"
    assert alarm.imei == "867395078806052"
    assert alarm.alarm == "eyes_closed"
    assert alarm.dms_video_url == "https://example.com/video.mp4"


def test_incoming_alarm_missing_id_raises():
    with pytest.raises(ValidationError):
        IncomingAlarm(**{k: v for k, v in VALID_ALARM.items() if k != "id"})


def test_incoming_alarm_missing_imei_raises():
    with pytest.raises(ValidationError):
        IncomingAlarm(**{k: v for k, v in VALID_ALARM.items() if k != "imei"})


def test_incoming_alarm_missing_time_raises():
    with pytest.raises(ValidationError):
        IncomingAlarm(**{k: v for k, v in VALID_ALARM.items() if k != "time"})


def test_incoming_alarm_missing_alarm_raises():
    with pytest.raises(ValidationError):
        IncomingAlarm(**{k: v for k, v in VALID_ALARM.items() if k != "alarm"})


def test_incoming_alarm_missing_dms_video_url_raises():
    with pytest.raises(ValidationError):
        IncomingAlarm(**{k: v for k, v in VALID_ALARM.items() if k != "dms_video_url"})


def test_incoming_alarm_extra_fields_ignored():
    alarm = IncomingAlarm(**VALID_ALARM, extra_field="ignored")
    assert not hasattr(alarm, "extra_field")


def test_incoming_alarm_model_dump_roundtrip():
    alarm = IncomingAlarm(**VALID_ALARM)
    d = alarm.model_dump()
    assert d["id"] == VALID_ALARM["id"]
    assert d["dms_video_url"] == VALID_ALARM["dms_video_url"]


def test_incoming_alarm_all_fields_are_strings():
    alarm = IncomingAlarm(**VALID_ALARM)
    assert isinstance(alarm.id, str)
    assert isinstance(alarm.imei, str)
    assert isinstance(alarm.time, str)
    assert isinstance(alarm.alarm, str)
    assert isinstance(alarm.dms_video_url, str)


# ─── ReviewResult ─────────────────────────────────────────────────────────────

VALID_RESULT = {
    "id": "alarm-001",
    "imei": "867395078806052",
    "time": "05/02/2026 00:00:06",
    "alarm": "eyes_closed",
    "dms_video_url": "https://example.com/video.mp4",
    "dms_video_url_after_proccess": "/tmp/alarm-001.mp4",
    "confidence_level": 85,
    "review_result": True,
    "process_duration": 1200,
    "other": {"model": "MultiScaleTCN"},
}


def test_review_result_valid():
    result = ReviewResult(**VALID_RESULT)
    assert result.id == "alarm-001"
    assert result.confidence_level == 85
    assert result.review_result is True
    assert result.process_duration == 1200


def test_review_result_missing_confidence_level_raises():
    with pytest.raises(ValidationError):
        ReviewResult(**{k: v for k, v in VALID_RESULT.items() if k != "confidence_level"})


def test_review_result_missing_review_result_raises():
    with pytest.raises(ValidationError):
        ReviewResult(**{k: v for k, v in VALID_RESULT.items() if k != "review_result"})


def test_review_result_missing_process_duration_raises():
    with pytest.raises(ValidationError):
        ReviewResult(**{k: v for k, v in VALID_RESULT.items() if k != "process_duration"})


def test_review_result_missing_other_raises():
    with pytest.raises(ValidationError):
        ReviewResult(**{k: v for k, v in VALID_RESULT.items() if k != "other"})


def test_review_result_typo_field_name_preserved():
    # API contract has "dms_video_url_after_proccess" (double-c typo intentional)
    result = ReviewResult(**VALID_RESULT)
    d = result.model_dump()
    assert "dms_video_url_after_proccess" in d
    assert "dms_video_url_after_process" not in d


def test_review_result_review_result_false():
    result = ReviewResult(**{**VALID_RESULT, "review_result": False})
    assert result.review_result is False


def test_review_result_confidence_level_zero():
    result = ReviewResult(**{**VALID_RESULT, "confidence_level": 0})
    assert result.confidence_level == 0


def test_review_result_confidence_level_100():
    result = ReviewResult(**{**VALID_RESULT, "confidence_level": 100})
    assert result.confidence_level == 100


def test_review_result_other_can_be_empty_dict():
    result = ReviewResult(**{**VALID_RESULT, "other": {}})
    assert result.other == {}


def test_review_result_other_can_be_nested():
    nested = {"model": "TCN", "video_info": {"fps": 30.0, "frames": 300}}
    result = ReviewResult(**{**VALID_RESULT, "other": nested})
    assert result.other["video_info"]["fps"] == 30.0


def test_review_result_model_dump_has_all_fields():
    result = ReviewResult(**VALID_RESULT)
    d = result.model_dump()
    expected = {
        "id", "imei", "time", "alarm", "dms_video_url",
        "dms_video_url_after_proccess", "confidence_level",
        "review_result", "process_duration", "other"
    }
    assert expected == set(d.keys())