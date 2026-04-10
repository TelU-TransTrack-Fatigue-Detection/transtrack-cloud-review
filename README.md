# TransTrack Cloud Review Engine

FastAPI service that receives DMS alarm events, runs fatigue detection (MediaPipe + MultiScaleTCN), and POSTs the verdict back via callback.

```
TransTrack Platform
      │  POST /review  (alarm event + signed video URL)
      ▼
Cloud Review Engine
      ├─ Download MP4 → Extract landmarks at 10 FPS via MediaPipe
      ├─ Compute EAR, MAR, head pose per frame → (8, 200) feature sequence
      ├─ MultiScaleTCN → eyes_closed / normal / yawning
      ├─ Night-hour heuristic  (00:00–05:59, conf < 0.75 → escalate)
      ├─ Driver history rule   (≥2 fatigue events in last hour → escalate)
      └─ POST verdict to callback URL
```

## Setup

Copy `.env.example` to `.env` and fill in values:
```env
CALLBACK_URL=https://platform-integrator.transtrack.co/cloud-result
MODEL_PATH=models/classifier/best_val_f1.pth
MODEL_NAME=MultiScaleTCN
REDIS_URL=redis://localhost:6379/0
WORKER_CONCURRENCY=2
```

## Docker (recommended)

```bash
docker compose up --build
```

Starts 3 services: `redis`, `api` (port 8000), `worker`.

**Local dev with mock servers:**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Adds: `mock-callback` (port 8001), `mock-video` (port 8002), `flower` (port 5555).

## Local (Windows)

```bash
conda activate transtrack_test
powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1
```

Launches all services (API, worker, mock callback, mock video) in separate windows.

**Test alarm:**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/review `
  -ContentType "application/json" `
  -Body '{"id":"TEST-001","imei":"123456","time":"2025-01-10T02:00:00Z","alarm":"fatigue","dms_video_url":"http://localhost:8002/videos/test.mp4"}'
```

**Stop everything:**
```bash
powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1 -Stop
```

## API

### `POST /review`

**Body:**
```json
{
  "id": "alarm-001",
  "imei": "123456789",
  "time": "2026-02-01T03:00:00Z",
  "alarm": "fatigue",
  "dms_video_url": "https://..."
}
```

**Response:** `202 Accepted` — result POSTed asynchronously to `CALLBACK_URL`.

**Callback payload:**
```json
{
  "id": "alarm-001",
  "imei": "123456789",
  "confidence_level": 85,
  "review_result": true,
  "process_duration": 3200,
  "other": {
    "label": "eyes_closed",
    "night_rule_triggered": false,
    "history_rule_triggered": false
  }
}
```

### `GET /health`

Returns queue depth and service status.

## Batch Inference

Process a list of video URLs offline — no API, no queue, no Redis needed.

**Input CSV** (`alarms.csv`):
```
id,imei,time,alarm,video_url
001,867395078806052,2025-01-10T02:00:00Z,fatigue,http://server/video1.mp4
002,867395078806052,2025-01-10T14:00:00Z,fatigue,http://server/video2.mp4
```

**Run:**
```bash
python scripts/batch_video_infer.py --input alarms.csv
```

Output written to `output/batch_<timestamp>/`:
- `results.csv` — label, confidence, review_result, error per row
- `summary.json` — totals, class counts, error breakdown

## Tests

```bash
pytest tests/ -v
```

## Error Records

Failed alarms are written to `records/ERR_<id>_<ts>.json` with the failure stage (`download`, `extract`, `infer`, or `callback`) and error message.
