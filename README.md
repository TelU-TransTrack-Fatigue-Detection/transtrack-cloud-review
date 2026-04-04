# TransTrack Cloud Review Engine

A FastAPI service that receives DMS (Driver Monitoring System) alarm events from the TransTrack fleet platform, downloads the flagged MP4 video, runs fatigue detection inference using MediaPipe + MultiScaleTCN, and POSTs the verdict back via callback.

## How It Works

```
TransTrack Platform
      │
      │  POST /review  (alarm event + signed video URL)
      ▼
Cloud Review Engine
      │
      ├─ Download MP4 from signed MDVR URL
      ├─ Extract landmarks at 10 FPS via MediaPipe FaceLandmarker
      ├─ Compute EAR, MAR, head pose (pitch/yaw/roll) per frame
      ├─ Build (8, 200) feature sequence
      ├─ Run MultiScaleTCN classifier → awake / drowsy / asleep
      │
      └─ POST result back to callback URL
```

Records are stored locally and auto-purged after 30 minutes.

## Project Structure

```
app/
  main.py         FastAPI app, /review endpoint, background pipeline
  pipeline.py     MediaPipe → EAR/MAR/headpose → MultiScaleTCN inference
  inference.py    Async wrapper around pipeline.predict()
  config.py       Settings (loaded from .env)
  schemas.py      Pydantic v2 request/response models
  notifier.py     POST result to callback URL with retry
  storage.py      Save JSON record to disk

scripts/
  batch_infer.py      Download 5 real videos and run inference, print labels
  check_model.py      Verify all model weights load and forward-pass correctly
  fetch_test_url.py   Get a fresh signed video URL from TransTrack API
  download_test_video.py  Download one video and inspect with cv2

models/
  mediapipe/          face_landmarker.task (auto-downloaded on first run)
  classifier/
    best_val_f1.pth   Best checkpoint by validation F1 (default)
    best_val_loss.pth Best checkpoint by validation loss
    latest.pth        Latest epoch checkpoint
    settings.yaml     Training config (sequence_length=200, landmark_fps=10)

tests/              Full pytest suite (38 tests)
records/            Temporary storage — auto-purged, gitignored
output/             Runner Excel output — gitignored
```

## Model

- **Architecture**: MultiScaleTCN — multi-scale stems (k3/k5/k7) + dilated TCN blocks + SE attention + dual avg/max pooling
- **Input**: (batch, 8, 200) — 8 landmark features × 200 frames at 10 FPS = 20 seconds
- **Output**: 3 classes — `awake`, `drowsy`, `asleep`
- **Features**: left EAR, right EAR, MAR, pitch, yaw, roll, nose tip x, nose tip y
- **Best checkpoint**: epoch 44, Val F1 = 0.6997

## Setup

**Requirements**: Python 3.11, CUDA-compatible GPU recommended (tested on RTX 4050)

**Quick setup (recommended):**
```bash
# Windows
setup_env.bat

# Linux / Mac
bash setup_env.sh
```

The script creates a new conda env `transtrack_review`, asks if you have a GPU, and installs everything automatically.

**Manual setup:**
```bash
conda create -n transtrack_review python=3.11
conda activate transtrack_review

pip install -r requirements.txt
pip install -r requirements-torch-gpu.txt   # GPU (CUDA 12.1)
# pip install -r requirements-torch-cpu.txt # CPU only
pip install -r requirements-dev.txt         # for running tests
```

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
```

## Running

**Start the API server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Verify model weights:**
```bash
python scripts/check_model.py
```

**Run batch inference on 5 real videos:**
```bash
python scripts/batch_infer.py
```

**Run tests:**
```bash
pytest -v --ignore=tests/test_integration.py
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

**Response:** `202 Accepted`
```json
{ "status": "queued", "id": "alarm-001" }
```

The result is sent asynchronously to `CALLBACK_URL` after inference completes.
