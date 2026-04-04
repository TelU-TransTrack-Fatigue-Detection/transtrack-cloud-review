import asyncio
import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from app.storage import save_record
from app.config import settings


SAMPLE_DATA = {
    "original": {"id": "abc123", "imei": "867395078806052"},
    "result": {"review_result": True, "confidence_level": 100},
}


@pytest.mark.asyncio
async def test_save_record_creates_file(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    assert path.exists()


@pytest.mark.asyncio
async def test_save_record_filename_contains_id_and_imei(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    assert "abc123" in path.name
    assert "867395" in path.name


@pytest.mark.asyncio
async def test_save_record_content_is_valid_json(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content == SAMPLE_DATA


@pytest.mark.asyncio
async def test_save_record_multiple_alarms_produce_separate_files(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    await save_record("id1", "imei1", SAMPLE_DATA)
    await asyncio.sleep(0.01)
    await save_record("id2", "imei2", SAMPLE_DATA)
    files = list(records_dir.glob("*.json"))
    assert len(files) == 2


@pytest.mark.asyncio
async def test_cleanup_removes_expired_records(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)

    old_time = time.time() - (31 * 60)
    import os
    os.utime(path, (old_time, old_time))

    _RECORD_TTL_SECONDS = 30 * 60
    now = time.time()
    for f in records_dir.glob("*.json"):
        if now - f.stat().st_mtime > _RECORD_TTL_SECONDS:
            f.unlink()

    assert not path.exists()


@pytest.mark.asyncio
async def test_cleanup_keeps_fresh_records(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)

    _RECORD_TTL_SECONDS = 30 * 60
    now = time.time()
    for f in records_dir.glob("*.json"):
        if now - f.stat().st_mtime > _RECORD_TTL_SECONDS:
            f.unlink()

    assert path.exists()


@pytest.mark.asyncio
async def test_save_record_returns_path_object(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    assert isinstance(path, Path)


@pytest.mark.asyncio
async def test_save_record_file_is_not_empty(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    assert path.stat().st_size > 0


@pytest.mark.asyncio
async def test_save_record_filename_contains_alarm_id(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    assert "abc123" in path.name


@pytest.mark.asyncio
async def test_save_record_filename_contains_imei(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    assert "867395" in path.name


@pytest.mark.asyncio
async def test_save_record_filename_has_json_extension(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    assert path.suffix == ".json"


@pytest.mark.asyncio
async def test_save_record_filename_contains_timestamp(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    before = int(time.time() * 1000) - 100
    path = await save_record("abc123", "867395", SAMPLE_DATA)
    after = int(time.time() * 1000) + 100
    # Extract timestamp from filename (last numeric part before .json)
    parts = path.stem.split("_")
    ts = int(parts[-1])
    assert before <= ts <= after


@pytest.mark.asyncio
async def test_save_record_content_roundtrips_nested_data(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    nested = {
        "original": {"id": "x", "nested": {"deep": [1, 2, 3]}},
        "result": {"review_result": False, "other": {"model": "TCN"}},
    }
    path = await save_record("x", "imei", nested)
    assert json.loads(path.read_text()) == nested


@pytest.mark.asyncio
async def test_save_record_content_has_utf8_encoding(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    data = {"text": "emoji 🚗 unicode ñ"}
    path = await save_record("u1", "imei1", data)
    assert "🚗" in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_save_record_concurrent_writes_produce_separate_files(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    await asyncio.gather(
        save_record("id1", "imei1", SAMPLE_DATA),
        save_record("id2", "imei2", SAMPLE_DATA),
        save_record("id3", "imei3", SAMPLE_DATA),
    )
    files = list(records_dir.glob("*.json"))
    assert len(files) == 3


@pytest.mark.asyncio
async def test_cleanup_removes_only_expired_leaves_fresh(records_dir, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_dir))
    old_path = await save_record("old", "imei1", SAMPLE_DATA)
    await asyncio.sleep(0.01)
    fresh_path = await save_record("fresh", "imei2", SAMPLE_DATA)
 
    os.utime(old_path, (time.time() - 31 * 60, time.time() - 31 * 60))
 
    _TTL = 30 * 60
    now = time.time()
    for f in records_dir.glob("*.json"):
        if now - f.stat().st_mtime > _TTL:
            f.unlink()
 
    assert not old_path.exists()
    assert fresh_path.exists()
 