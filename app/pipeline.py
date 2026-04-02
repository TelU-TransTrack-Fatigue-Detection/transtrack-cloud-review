from __future__ import annotations

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.image import Image as MpImage

_MEDIAPIPE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker"
    "/face_landmarker/float16/1/face_landmarker.task"
)
_MEDIAPIPE_MODEL_PATH = Path("models/mediapipe/face_landmarker.task")

_RIGHT_EYE   = [33, 159, 158, 133, 153, 144]
_LEFT_EYE    = [263, 386, 385, 362, 380, 373]
_MOUTH       = [13, 14, 61, 291]
_PNP_INDICES = [1, 152, 33, 263, 61, 291]
_MODEL_3D    = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float32)

SEQUENCE_LENGTH = 200
LANDMARK_FPS    = 10.0
CLASS_NAMES     = ["eyes_closed", "normal", "yawning"]
_MASK_WINDOW    = 5
_MASK_THRESH    = 1e-4


def _ensure_mediapipe_model() -> str:
    if _MEDIAPIPE_MODEL_PATH.exists():
        return str(_MEDIAPIPE_MODEL_PATH)
    _MEDIAPIPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(_MEDIAPIPE_MODEL_URL, _MEDIAPIPE_MODEL_PATH)
    return str(_MEDIAPIPE_MODEL_PATH)


def _ear(lm, indices) -> float:
    p  = [lm[i] for i in indices]
    v1 = np.sqrt((p[1].x - p[5].x) ** 2 + (p[1].y - p[5].y) ** 2)
    v2 = np.sqrt((p[2].x - p[4].x) ** 2 + (p[2].y - p[4].y) ** 2)
    h  = np.sqrt((p[0].x - p[3].x) ** 2 + (p[0].y - p[3].y) ** 2)
    return float((v1 + v2) / (2.0 * h)) if h > 0 else np.nan


def _mar(lm, indices) -> float:
    p = [lm[i] for i in indices]
    v = np.sqrt((p[0].x - p[1].x) ** 2 + (p[0].y - p[1].y) ** 2)
    h = np.sqrt((p[2].x - p[3].x) ** 2 + (p[2].y - p[3].y) ** 2)
    return float(v / h) if h > 0 else np.nan


def _head_pose(lm, w: int, h: int):
    pts_2d = np.array([[lm[i].x * w, lm[i].y * h] for i in _PNP_INDICES], dtype=np.float32)
    focal  = float(w)
    cam    = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32)
    dist   = np.zeros((4, 1), dtype=np.float32)
    ok, rv, _ = cv2.solvePnP(_MODEL_3D, pts_2d, cam, dist)
    if not ok:
        return np.nan, np.nan, np.nan
    rmat, _ = cv2.Rodrigues(rv)
    proj    = np.hstack((rmat, np.zeros((3, 1), dtype=np.float32)))
    angles  = cv2.decomposeProjectionMatrix(proj)[6]
    return float(angles[0, 0]), float(angles[1, 0]), float(angles[2, 0])


def _is_masked(mar_hist: List[float]) -> bool:
    if len(mar_hist) < _MASK_WINDOW:
        return False
    recent = np.array(mar_hist[-_MASK_WINDOW:], dtype=np.float32)
    if np.any(np.isnan(recent)):
        return False
    return bool(np.var(recent) < _MASK_THRESH)


def _enforce_continuity(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    idx = np.where(~np.isnan(out))[0]
    if len(idx) < 2:
        return out
    for i in range(1, len(idx)):
        c, p   = idx[i], idx[i - 1]
        cands  = np.array([out[c], out[c] + 180, out[c] - 180, out[c] + 360, out[c] - 360])
        out[c] = cands[np.argmin(np.abs(cands - out[p]))]
    return out


def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        arr[~valid] = 0.0
        return arr
    x       = np.arange(len(arr))
    arr[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    return arr


def _extract(video_path: Path) -> np.ndarray:
    model_path = _ensure_mediapipe_model()
    options    = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    cap      = cv2.VideoCapture(str(video_path))
    rows: List[List[float]] = []
    mar_hist: List[float]   = []

    original_fps   = cap.get(cv2.CAP_PROP_FPS) or LANDMARK_FPS
    frame_interval = max(1, round(original_fps / LANDMARK_FPS))

    stderr_fd = os.dup(2)
    devnull   = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        with FaceLandmarker.create_from_options(options) as lmk:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if (frame_idx - 1) % frame_interval != 0:
                    continue
                fh, fw = frame.shape[:2]
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = lmk.detect(MpImage(image_format=mp.ImageFormat.SRGB, data=rgb))

                if not result.face_landmarks:
                    rows.append([np.nan] * 8)
                    continue

                lm    = result.face_landmarks[0]
                ear_l = _ear(lm, _LEFT_EYE)
                ear_r = _ear(lm, _RIGHT_EYE)
                mar   = _mar(lm, _MOUTH)

                masked = _is_masked(mar_hist)
                if not masked and not np.isnan(mar):
                    mar_hist.append(mar)

                pitch, yaw, roll = _head_pose(lm, fw, fh)
                rows.append([
                    ear_l, ear_r,
                    np.nan if masked else mar,
                    pitch, yaw, roll,
                    lm[1].x, lm[1].y,
                ])
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        cap.release()

    if not rows:
        return np.zeros((SEQUENCE_LENGTH, 8), dtype=np.float32)

    arr = np.array(rows, dtype=np.float32)
    for i in (3, 4, 5):
        arr[:, i] = _enforce_continuity(arr[:, i])
    for i in range(8):
        arr[:, i] = _interpolate_nan(arr[:, i])
    return arr


def _prepare(features: np.ndarray) -> torch.Tensor:
    T = features.shape[0]
    if T >= SEQUENCE_LENGTH:
        features = features[:SEQUENCE_LENGTH]
    else:
        features = np.concatenate([features, np.zeros((SEQUENCE_LENGTH - T, features.shape[1]), dtype=np.float32)])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(features).float().T.unsqueeze(0)


class _SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1).expand_as(x)


class _DilatedResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding  = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se  = _SEBlock(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.conv(x)) + x)


class MultiScaleTCN(nn.Module):
    def __init__(self, num_classes: int = 3, input_channels: int = 8,
                 seq_length: int = 100, dropout: float = 0.3):
        super().__init__()
        stem_ch  = 64
        proj_ch  = 256

        self.stem_k3 = nn.Sequential(
            nn.Conv1d(input_channels, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(stem_ch), nn.SiLU(),
        )
        self.stem_k5 = nn.Sequential(
            nn.Conv1d(input_channels, stem_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(stem_ch), nn.SiLU(),
        )
        self.stem_k7 = nn.Sequential(
            nn.Conv1d(input_channels, stem_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch), nn.SiLU(),
        )
        self.stem_proj = nn.Sequential(
            nn.Conv1d(stem_ch * 3, proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(proj_ch), nn.SiLU(), nn.Dropout(dropout),
        )
        self.tcn_blocks = nn.ModuleList([
            _DilatedResBlock(proj_ch, dilation=1, dropout=dropout),
            _DilatedResBlock(proj_ch, dilation=2, dropout=dropout),
            _DilatedResBlock(proj_ch, dilation=4, dropout=dropout),
            _DilatedResBlock(proj_ch, dilation=8, dropout=dropout),
        ])
        self.expand = nn.Sequential(
            nn.Conv1d(proj_ch, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512), nn.SiLU(),
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(512 * 2, 512), nn.SiLU(), nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.stem_k3(x), self.stem_k5(x), self.stem_k7(x)], dim=1)
        x = self.stem_proj(x)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.expand(x)
        return self.feature_proj(torch.cat([x.mean(dim=2), x.max(dim=2)[0]], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self._features(x))


_MODEL_REGISTRY = {
    "MultiScaleTCN": MultiScaleTCN,
}

_model_cache: dict = {}


def _load_model(model_path: Path, model_name: str, device: torch.device) -> nn.Module:
    cache_key = str(model_path)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")

    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    if any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}

    model = _MODEL_REGISTRY[model_name](num_classes=len(CLASS_NAMES))
    model.load_state_dict(state)
    model.to(device).eval()
    _model_cache[cache_key] = model
    return model


def predict(video_path: str | Path, model_path: str | Path,
            model_name: str = "MultiScaleTCN") -> dict:
    video_path = Path(video_path)
    model_path = Path(model_path)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = _extract(video_path)
    tensor   = _prepare(features).to(device)
    model    = _load_model(model_path, model_name, device)

    with torch.no_grad():
        probs     = F.softmax(model(tensor), dim=-1)
        conf, cls = torch.max(probs, dim=-1)

    return {
        "label":      CLASS_NAMES[cls.item()],
        "class_id":   cls.item(),
        "confidence": round(conf.item(), 4),
    }
