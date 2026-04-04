# TransTrack Cloud Review Engine

FastAPI service that receives DMS alarm events, runs fatigue detection (MediaPipe + MultiScaleTCN), and POSTs the verdict back via callback.

```
TransTrack Platform
      │  POST /review  (alarm event + signed video URL)
      ▼
Cloud Review Engine
      ├─ Download MP4 → Extract landmarks at 10 FPS via MediaPipe
      ├─ Compute EAR, MAR, head pose per frame → (8, 200) feature sequence
      └─ MultiScaleTCN → awake / drowsy / asleep → POST to callback URL
```

## Setup

Copy and fill in `.env`:
```env
API_KEY=changeme
CALLBACK_URL=https://platform-integrator.transtrack.co/cloud-result
RECORDS_DIR=./records
VIDEO_DOWNLOAD_TIMEOUT=60
HTTPX_TIMEOUT=30
KEEP_TMP_VIDEOS=False

TRANSTRACK_BASE_URL=https://api-platform-integrator.transtrack.co/api/v1/vss
TRANSTRACK_USERNAME=your_username
TRANSTRACK_PASSWORD=your_password

MODEL_PATH=models/classifier/best_val_f1.pth
MODEL_NAME=MultiScaleTCN
REDIS_URL=redis://localhost:6379/0
WORKER_CONCURRENCY=2
```

## Running with Docker (recommended)

```bash
docker compose up --build
```

This starts 3 services: `redis`, `api` (port 8000), and `worker`.

To run in the background:
```bash
docker compose up -d
```

## Running Locally

```bash
conda create -n transtrack_review python=3.11
conda activate transtrack_review

pip install -r requirements.txt
pip install -r requirements-torch-gpu.txt   # GPU (CUDA 12.1)
# pip install -r requirements-torch-cpu.txt # CPU only
pip install -r requirements-dev.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /review`

**Headers:** `Authorization: Bearer <API_KEY>`

**Body:**
```json
{
  "id": "alarm-001",
  "imei": "123456789",
  "time": "2026-02-01T03:00:00Z",
  "alarm": "eyes_closed",
  "dms_video_url": "https://..."
}
```

**Response:** `202 Accepted` — result POSTed asynchronously to `CALLBACK_URL`.

## Tests

```bash
pytest -v --ignore=tests/test_integration.py
```
