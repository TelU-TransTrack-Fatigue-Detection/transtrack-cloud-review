import pytest
from unittest.mock import AsyncMock, patch

from conftest import VALID_PAYLOAD, VALID_HEADERS


@pytest.mark.asyncio
async def test_review_returns_202(client_with_mocks):
    client, *_ = client_with_mocks
    resp = await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_review_returns_queued_status(client_with_mocks):
    client, *_ = client_with_mocks
    resp = await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    body = resp.json()
    assert body["status"] == "queued"
    assert body["id"] == VALID_PAYLOAD["id"]


@pytest.mark.asyncio
async def test_missing_auth_returns_401(client):
    resp = await client.post("/review", json=VALID_PAYLOAD)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_api_key_returns_401(client):
    resp = await client.post(
        "/review", json=VALID_PAYLOAD, headers={"Authorization": "Bearer wrongkey"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_missing_required_field_returns_422(client_with_mocks):
    client, *_ = client_with_mocks
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "dms_video_url"}
    resp = await client.post("/review", json=payload, headers=VALID_HEADERS)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_empty_body_returns_422(client_with_mocks):
    client, *_ = client_with_mocks
    resp = await client.post("/review", json={}, headers=VALID_HEADERS)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_extra_fields_are_accepted(client_with_mocks):
    client, *_ = client_with_mocks
    payload = {**VALID_PAYLOAD, "unexpected_field": "value"}
    resp = await client.post("/review", json=payload, headers=VALID_HEADERS)
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_get_method_not_allowed(client):
    resp = await client.get("/review", headers=VALID_HEADERS)
    assert resp.status_code == 405
