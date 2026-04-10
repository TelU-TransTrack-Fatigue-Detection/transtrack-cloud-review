import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

OUTPUT_DIR      = Path("output")
CLASS_NAMES     = ["eyes_closed", "normal", "yawning"]
SEQUENCE_LENGTH = 200
NPZ_CHANNELS    = ["ear_l", "ear_r", "mar", "pitch", "yaw", "roll", "nose_x", "nose_y"]
VIDEO_EXTS      = {".mp4", ".avi", ".mov", ".mkv"}


class _SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1).expand_as(x)


class _DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se  = _SEBlock(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.se(self.conv(x)) + x)


class MultiScaleTCN(nn.Module):
    def __init__(self, num_classes=3, input_channels=8, dropout=0.3):
        super().__init__()
        stem_ch, proj_ch = 64, 256
        self.stem_k3 = nn.Sequential(nn.Conv1d(input_channels, stem_ch, 3, padding=1, bias=False), nn.BatchNorm1d(stem_ch), nn.SiLU())
        self.stem_k5 = nn.Sequential(nn.Conv1d(input_channels, stem_ch, 5, padding=2, bias=False), nn.BatchNorm1d(stem_ch), nn.SiLU())
        self.stem_k7 = nn.Sequential(nn.Conv1d(input_channels, stem_ch, 7, padding=3, bias=False), nn.BatchNorm1d(stem_ch), nn.SiLU())
        self.stem_proj = nn.Sequential(nn.Conv1d(stem_ch * 3, proj_ch, 1, bias=False), nn.BatchNorm1d(proj_ch), nn.SiLU(), nn.Dropout(dropout))
        self.tcn_blocks = nn.ModuleList([
            _DilatedResBlock(proj_ch, dilation=1, dropout=dropout),
            _DilatedResBlock(proj_ch, dilation=2, dropout=dropout),
            _DilatedResBlock(proj_ch, dilation=4, dropout=dropout),
            _DilatedResBlock(proj_ch, dilation=8, dropout=dropout),
        ])
        self.expand       = nn.Sequential(nn.Conv1d(proj_ch, 512, 1, bias=False), nn.BatchNorm1d(512), nn.SiLU())
        self.feature_proj = nn.Sequential(nn.Linear(512 * 2, 512), nn.SiLU(), nn.Dropout(dropout))
        self.fc           = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        x = torch.cat([self.stem_k3(x), self.stem_k5(x), self.stem_k7(x)], dim=1)
        x = self.stem_proj(x)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.expand(x)
        x = self.feature_proj(torch.cat([x.mean(dim=2), x.max(dim=2)[0]], dim=1))
        return self.fc(x)


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    model = MultiScaleTCN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(state)
    return model.to(device).eval()


def load_npz(path: Path, device: torch.device) -> torch.Tensor:
    d = np.load(path)
    cols = [d[k] if k in d else np.zeros(SEQUENCE_LENGTH, dtype=np.float32) for k in NPZ_CHANNELS]
    arr  = np.stack(cols, axis=1).astype(np.float32)
    if arr.shape[0] >= SEQUENCE_LENGTH:
        arr = arr[:SEQUENCE_LENGTH]
    else:
        arr = np.concatenate([arr, np.zeros((SEQUENCE_LENGTH - arr.shape[0], 8), dtype=np.float32)])
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr).T.unsqueeze(0).to(device)


def load_video_as_tensor(path: Path, device: torch.device) -> torch.Tensor:
    import os
    os.environ.setdefault("GLOG_minloglevel", "2")
    from app.pipeline import _extract, _prepare
    features = _extract(path)
    arr = _prepare(features).squeeze(0).T.numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr).T.unsqueeze(0).to(device)


def collect_samples(dataset_dir: Path):
    """
    Supports both flat and nested dataset structures:
      flat:   dataset/{label}/*.npz
      nested: dataset/{date}/{fleet}/VIDEO/{label}/*.npz
    Recursively finds any folder whose name matches a class label.
    """
    samples = []
    seen    = set()
    for label_dir in dataset_dir.rglob("*"):
        if not label_dir.is_dir() or label_dir.name not in CLASS_NAMES:
            continue
        for f in sorted(label_dir.iterdir()):
            if f.suffix.lower() in {".npz"} | VIDEO_EXTS and f not in seen:
                seen.add(f)
                samples.append((f, label_dir.name))
    return sorted(samples, key=lambda x: (x[1], x[0]))


def compute_metrics(results):
    valid   = [r for r in results if r["pred"] != "error"]
    correct = sum(1 for r in valid if r["correct"])
    per_class = {}
    for cls in CLASS_NAMES:
        tp = sum(1 for r in valid if r["true"] == cls and r["pred"] == cls)
        fp = sum(1 for r in valid if r["true"] != cls and r["pred"] == cls)
        fn = sum(1 for r in valid if r["true"] == cls and r["pred"] != cls)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
    matrix = {cls: {n: 0 for n in CLASS_NAMES} for cls in CLASS_NAMES}
    for r in valid:
        matrix[r["true"]][r["pred"]] += 1
    return {
        "total": len(valid), "correct": correct, "errors": len(results) - len(valid),
        "accuracy": round(correct / len(valid), 4) if valid else 0.0,
        "per_class": per_class, "confusion_matrix": matrix,
    }


def print_summary(metrics):
    print(f"\nAccuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.4f}")
    if metrics["errors"]:
        print(f"Errors:   {metrics['errors']}")
    print(f"\n{'Class':<15} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 52)
    for cls, m in metrics["per_class"].items():
        print(f"{cls:<15} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':>15}" + "".join(f"{c:>15}" for c in CLASS_NAMES))
    for true_cls in CLASS_NAMES:
        print(f"{true_cls:>15}" + "".join(f"{metrics['confusion_matrix'][true_cls][p]:>15}" for p in CLASS_NAMES))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch inference on NPZ or video dataset folders")
    parser.add_argument("--dataset", default="dataset-260205", help="Root dataset dir (label subfolders)")
    parser.add_argument("--model",   default="models/classifier/best_val_f1.pth", help="Model checkpoint path")
    args = parser.parse_args()

    model_path  = Path(args.model)
    dataset_dir = Path(args.dataset)

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    if not dataset_dir.exists():
        sys.exit(f"Dataset not found: {dataset_dir}")

    samples = collect_samples(dataset_dir)
    if not samples:
        sys.exit(f"No files found under {dataset_dir}")

    print(f"Found {len(samples)} samples — {dataset_dir}")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(model_path, device)
    results = []

    for file_path, true_label in tqdm(samples, desc="Inferring", unit="file"):
        try:
            if file_path.suffix.lower() == ".npz":
                tensor = load_npz(file_path, device)
            else:
                tensor = load_video_as_tensor(file_path, device)
            with torch.no_grad():
                probs      = F.softmax(model(tensor), dim=-1)
                conf, cls  = torch.max(probs, dim=-1)
            pred_label = CLASS_NAMES[cls.item()]
            results.append({"file": str(file_path), "true": true_label, "pred": pred_label,
                            "confidence": round(conf.item(), 4), "correct": pred_label == true_label})
        except Exception as e:
            results.append({"file": str(file_path), "true": true_label, "pred": "error",
                            "confidence": 0.0, "correct": False, "error": str(e)})

    metrics = compute_metrics(results)
    print_summary(metrics)

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = OUTPUT_DIR / f"eval_{dataset_dir.name}_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "true", "pred", "confidence", "correct"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "error"} for r in results)

    metrics_path = OUTPUT_DIR / f"eval_{dataset_dir.name}_{ts}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nCSV     : {csv_path}")
    print(f"Metrics : {metrics_path}")
