import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
from datetime import datetime
from tqdm import tqdm
from app.config import settings
from app.pipeline import predict, CLASS_NAMES

DATASET_DIR   = Path("data-test-other/test/VIDEO")
OUTPUT_DIR    = Path("output")
MODEL_PATH    = Path(settings.MODEL_PATH)
MODEL_NAME    = settings.MODEL_NAME
MAX_PER_CLASS = None

LABEL_MAP = {
    "normal":      "normal",
    "yawning":     "yawning",
    "eyes_closed": "eyes_closed",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_samples():
    samples = []
    for ds_label, model_label in LABEL_MAP.items():
        folder = DATASET_DIR / ds_label
        if not folder.exists():
            continue
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        if MAX_PER_CLASS is not None:
            files = files[:MAX_PER_CLASS]
        for f in files:
            samples.append((f, ds_label, model_label))
    return samples


def compute_metrics(results):
    valid = [r for r in results if r["pred"] != "error"]
    correct = sum(1 for r in valid if r["correct"])
    accuracy = correct / len(valid) if valid else 0.0

    per_class = {}
    for cls in CLASS_NAMES:
        tp = sum(1 for r in valid if r["true_model"] == cls and r["pred"] == cls)
        fp = sum(1 for r in valid if r["true_model"] != cls and r["pred"] == cls)
        fn = sum(1 for r in valid if r["true_model"] == cls and r["pred"] != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall, "f1": f1}

    matrix = {}
    for r in valid:
        matrix.setdefault(r["true_model"], {n: 0 for n in CLASS_NAMES})
        matrix[r["true_model"]][r["pred"]] += 1

    return {
        "total": len(valid),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found: {MODEL_PATH}")

    samples = collect_samples()
    if not samples:
        sys.exit(f"No videos found under {DATASET_DIR}")

    results = []

    for video_path, ds_label, true_model_label in tqdm(samples, desc="Evaluating", unit="video"):
        try:
            pred = predict(video_path, MODEL_PATH, MODEL_NAME)
            results.append({
                "file":       str(video_path),
                "ds_label":   ds_label,
                "true_model": true_model_label,
                "pred":       pred["label"],
                "confidence": pred["confidence"],
                "correct":    pred["label"] == true_model_label,
            })
        except Exception as e:
            results.append({
                "file":       str(video_path),
                "ds_label":   ds_label,
                "true_model": true_model_label,
                "pred":       "error",
                "confidence": 0.0,
                "correct":    False,
                "error":      str(e),
            })

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = OUTPUT_DIR / f"eval_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "ds_label", "true_model", "pred", "confidence", "correct"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "error"} for r in results)

    metrics = compute_metrics(results)
    metrics_path = OUTPUT_DIR / f"eval_{ts}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    tqdm.write(f"Done. {metrics['correct']}/{metrics['total']} correct — accuracy {metrics['accuracy']:.4f}")
    tqdm.write(f"Results : {csv_path}")
    tqdm.write(f"Metrics : {metrics_path}")
