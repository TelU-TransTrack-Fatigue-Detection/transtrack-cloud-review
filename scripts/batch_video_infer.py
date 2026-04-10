"""
Batch video inference — full pipeline without API/queue.

Downloads each video, runs mediapipe landmark extraction,
feeds features into the classifier, applies night-hour heuristic,
and writes results to a CSV + JSON summary.

Usage:
    python scripts/batch_video_infer.py --input alarms.csv

Input CSV columns (required):
    id, imei, time, alarm, video_url

Optional columns (passed through to output):
    any extra columns are preserved as-is

Output (written to output/batch_<timestamp>/):
    results.csv   — one row per alarm with label, confidence, review_result, error
    summary.json  — aggregate stats (total, reviewed, errors, per-class counts)

Options:
    --input      PATH    Input CSV file (required)
    --model      PATH    Model weights (default: models/classifier/best_val_f1.pth)
    --output-dir PATH    Output directory (default: output/)
    --keep-videos        Don't delete downloaded videos after processing
    --workers    N       Parallel download threads (default: 4; inference always sequential)
    --timeout    N       Per-video download timeout in seconds (default: 60)
"""

import argparse
import csv
import json
import logging
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("batch_infer")

# ---------------------------------------------------------------------------
# Constants — must match worker.py
# ---------------------------------------------------------------------------
_NIGHT_HOURS      = range(0, 6)
_NIGHT_CONF_FLOOR = 0.75
REQUIRED_COLS     = {"id", "imei", "time", "alarm", "video_url"}


# ---------------------------------------------------------------------------
# Night-hour heuristic (no Redis needed — history rule skipped in batch mode)
# ---------------------------------------------------------------------------

def _apply_night_heuristic(label: str, conf: float, alarm_time_str: str) -> tuple[bool, bool]:
    """Returns (review_result, night_triggered)."""
    try:
        hour = datetime.fromisoformat(alarm_time_str).hour
    except Exception:
        try:
            hour = int(alarm_time_str.split("T")[-1].split(":")[0])
        except Exception:
            hour = -1

    night_triggered = (
        label == "normal"
        and hour in _NIGHT_HOURS
        and conf < _NIGHT_CONF_FLOOR
    )
    review_result = (label != "normal") or night_triggered
    return review_result, night_triggered


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(video_url: str, dest: Path, timeout: int) -> None:
    with httpx.Client(timeout=timeout) as client:
        with client.stream("GET", video_url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(64 * 1024):
                    f.write(chunk)


# ---------------------------------------------------------------------------
# Process one row
# ---------------------------------------------------------------------------

def process_row(row: dict, model, device, tmp_dir: Path,
                keep_videos: bool, timeout: int) -> dict:
    alarm_id  = row["id"]
    video_url = row["video_url"]
    video_path = tmp_dir / f"{alarm_id}.mp4"

    result = {
        **row,
        "label":          None,
        "confidence":     None,
        "review_result":  None,
        "night_triggered": None,
        "duration_ms":    None,
        "error":          None,
        "error_stage":    None,
    }

    start = time.monotonic()
    stage = "download"

    try:
        _download(video_url, video_path, timeout)

        stage = "extract"
        # Import here so mediapipe stderr suppression takes effect
        from app.pipeline import _extract, _prepare, CLASS_NAMES

        features = _extract(video_path)

        stage  = "infer"
        tensor = _prepare(features).to(device)
        with torch.no_grad():
            probs      = F.softmax(model(tensor), dim=-1)
            conf, cls  = torch.max(probs, dim=-1)

        label    = CLASS_NAMES[cls.item()]
        conf_val = round(conf.item(), 4)

        review_result, night_triggered = _apply_night_heuristic(
            label, conf_val, row.get("time", "")
        )

        result.update({
            "label":           label,
            "confidence":      conf_val,
            "review_result":   review_result,
            "night_triggered": night_triggered,
            "duration_ms":     int((time.monotonic() - start) * 1000),
        })

    except Exception as exc:
        result.update({
            "error":       str(exc),
            "error_stage": stage,
            "duration_ms": int((time.monotonic() - start) * 1000),
        })
        logger.warning("FAIL id=%s stage=%s error=%s", alarm_id, stage, exc)

    finally:
        if not keep_videos and video_path.exists():
            video_path.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch video inference")
    parser.add_argument("--input",      required=True,  help="Input CSV file")
    parser.add_argument("--model",      default="models/classifier/best_val_f1.pth")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--timeout",    type=int, default=60)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Read CSV
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
        cols   = set(reader.fieldnames or [])

    missing = REQUIRED_COLS - cols
    if missing:
        logger.error("Input CSV missing required columns: %s", missing)
        sys.exit(1)

    logger.info("Loaded %d rows from %s", len(rows), input_path)

    # Load model once
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    from app.pipeline import _load_model, CLASS_NAMES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from %s on %s", model_path, device)
    model = _load_model(model_path, "MultiScaleTCN", device)
    logger.info("Model ready — classes: %s", CLASS_NAMES)

    # Output dir
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"batch_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "tmp_videos"
    tmp_dir.mkdir(exist_ok=True)

    # Process rows sequentially (inference is GPU-bound, parallelism doesn't help)
    results = []
    for row in tqdm(rows, desc="Processing", unit="video"):
        r = process_row(row, model, device, tmp_dir, args.keep_videos, args.timeout)
        results.append(r)

    # Clean up tmp dir if empty
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # Write results CSV
    out_csv = output_dir / "results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    logger.info("Results written to %s", out_csv)

    # Summary
    total        = len(results)
    errors       = [r for r in results if r["error"]]
    ok           = [r for r in results if not r["error"]]
    reviewed     = [r for r in ok if r["review_result"]]
    night_hits   = [r for r in ok if r["night_triggered"]]
    class_counts = {}
    for r in ok:
        class_counts[r["label"]] = class_counts.get(r["label"], 0) + 1

    summary = {
        "input":          str(input_path),
        "model":          str(model_path),
        "total":          total,
        "processed":      len(ok),
        "errors":         len(errors),
        "reviewed":       len(reviewed),
        "night_triggered": len(night_hits),
        "class_counts":   class_counts,
        "error_stages":   {},
        "run_at":         ts,
    }
    for r in errors:
        s = r["error_stage"] or "unknown"
        summary["error_stages"][s] = summary["error_stages"].get(s, 0) + 1

    out_json = output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Print summary to console
    print("\n" + "=" * 50)
    print(f"  Total alarms : {total}")
    print(f"  Processed    : {len(ok)}")
    print(f"  Errors       : {len(errors)}")
    print(f"  For review   : {len(reviewed)}  ({len(reviewed)/max(len(ok),1)*100:.1f}%)")
    print(f"  Night-hour   : {len(night_hits)}")
    print(f"  Class counts : {class_counts}")
    if errors:
        print(f"  Error stages : {summary['error_stages']}")
    print(f"\n  Output: {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
