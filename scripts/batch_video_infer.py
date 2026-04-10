"""
Batch video inference — full pipeline without API/queue.

Each worker process handles the full pipeline independently:
    download → mediapipe extract → classifier infer → result

This saturates all CPU cores. On a 4-vCPU machine, use --workers 4.
Each worker loads its own model copy (cached per process).

Usage:
    python scripts/batch_video_infer.py --input alarms.csv
    python scripts/batch_video_infer.py --input alarms.csv --workers 4
    python scripts/batch_video_infer.py --input alarms.csv --device cuda  # if GPU available

Input CSV columns (required):
    id, imei, time, alarm, video_url

Output (written to output/batch_<timestamp>/):
    results.csv   — one row per alarm with label, confidence, review_result, error
    summary.json  — aggregate stats

Options:
    --input      PATH    Input CSV file (required)
    --model      PATH    Model weights (default: models/classifier/best_val_f1.pth)
    --output-dir PATH    Output directory (default: output/)
    --device     STR     cpu (default) or cuda
    --workers    N       Parallel worker processes (default: all CPU cores)
    --keep-videos        Don't delete downloaded videos after processing
    --timeout    N       Per-video download timeout in seconds (default: 60)
"""

import argparse
import csv
import json
import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults — override via .env or environment variables
# ---------------------------------------------------------------------------
# On a CPU-only machine (e.g. 4-vCPU VM):
#   BATCH_WORKERS=4   BATCH_DEVICE=cpu
#
# After upgrading to a GPU machine:
#   BATCH_DEVICE=cuda   BATCH_WORKERS=2  (GPU handles parallelism internally)
#
# After adding more RAM / vCPUs:
#   BATCH_WORKERS=8  (or however many cores you have)
# ---------------------------------------------------------------------------
_DEFAULT_WORKERS = int(os.getenv("BATCH_WORKERS", os.cpu_count() or 4))
_DEFAULT_DEVICE  = os.getenv("BATCH_DEVICE", "cpu")
_DEFAULT_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("batch_infer")

REQUIRED_COLS = {"id", "imei", "time", "alarm", "video_url"}


# ---------------------------------------------------------------------------
# Worker — runs in its own process
# Each process loads the model once (cached via pipeline._model_cache)
# ---------------------------------------------------------------------------

def _run_one(task: dict) -> dict:
    """Full pipeline for one alarm. Runs inside a worker process."""
    row        = task["row"]
    tmp_dir    = Path(task["tmp_dir"])
    model_path = Path(task["model_path"])
    device_str = task["device"]
    keep       = task["keep_videos"]
    timeout    = task["timeout"]

    video_path = tmp_dir / f"{row['id']}.mp4"
    result = {
        **row,
        "label":         None,
        "confidence":    None,
        "review_result": None,
        "duration_ms":   None,
        "error":         None,
        "error_stage":   None,
    }
    start = time.monotonic()
    stage = "download"

    try:
        # 1. Download
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", row["video_url"]) as resp:
                resp.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in resp.iter_bytes(64 * 1024):
                        f.write(chunk)

        # 2. Extract landmarks (mediapipe)
        stage = "extract"
        from app.pipeline import _extract, _prepare, _load_model, CLASS_NAMES
        features = _extract(video_path)

        # 3. Infer (model cached per process)
        stage  = "infer"
        device = torch.device(device_str)
        model  = _load_model(model_path, "MultiScaleTCN", device)
        tensor = _prepare(features).to(device)

        with torch.no_grad():
            probs     = F.softmax(model(tensor), dim=-1)
            conf, cls = torch.max(probs, dim=-1)

        label    = CLASS_NAMES[cls.item()]
        conf_val = round(conf.item(), 4)

        result.update({
            "label":         label,
            "confidence":    conf_val,
            "review_result": label != "normal",
            "duration_ms":   int((time.monotonic() - start) * 1000),
        })

    except Exception as exc:
        result.update({
            "error":       str(exc),
            "error_stage": stage,
            "duration_ms": int((time.monotonic() - start) * 1000),
        })

    finally:
        if not keep and video_path.exists():
            video_path.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch video inference")
    parser.add_argument("--input",       required=True)
    parser.add_argument("--model",       default="models/classifier/best_val_f1.pth")
    parser.add_argument("--output-dir",  default="output")
    parser.add_argument("--device",      default=_DEFAULT_DEVICE, choices=["cpu", "cuda"],
                        help=f"Inference device (default: {_DEFAULT_DEVICE}, env: BATCH_DEVICE)")
    parser.add_argument("--workers",     type=int, default=_DEFAULT_WORKERS,
                        help=f"Parallel worker processes (default: {_DEFAULT_WORKERS}, env: BATCH_WORKERS)")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--timeout",     type=int, default=_DEFAULT_TIMEOUT,
                        help=f"Download timeout in seconds (default: {_DEFAULT_TIMEOUT}, env: BATCH_TIMEOUT)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        args.device = "cpu"

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
        cols   = set(reader.fieldnames or [])

    missing = REQUIRED_COLS - cols
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"batch_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp_videos"
    tmp_dir.mkdir(exist_ok=True)

    logger.info(
        "Starting batch — rows=%d workers=%d device=%s",
        len(rows), args.workers, args.device
    )

    tasks = [
        {
            "row":        row,
            "tmp_dir":    str(tmp_dir),
            "model_path": str(model_path),
            "device":     args.device,
            "keep_videos": args.keep_videos,
            "timeout":    args.timeout,
        }
        for row in rows
    ]

    results = []

    # spawn context works correctly on both Linux and Windows
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_one, tasks),
            total=len(tasks),
            desc="Processing",
            unit="video",
        ):
            results.append(result)

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

    # Summary
    total      = len(results)
    errors       = [r for r in results if r["error"]]
    ok           = [r for r in results if not r["error"]]
    reviewed     = [r for r in ok if r["review_result"]]
    class_counts: dict = {}
    for r in ok:
        class_counts[r["label"]] = class_counts.get(r["label"], 0) + 1

    error_stages: dict = {}
    for r in errors:
        s = r["error_stage"] or "unknown"
        error_stages[s] = error_stages.get(s, 0) + 1

    summary = {
        "input": str(input_path), "model": str(model_path),
        "total": total, "processed": len(ok), "errors": len(errors),
        "reviewed": len(reviewed),
        "class_counts": class_counts, "error_stages": error_stages,
        "workers": args.workers, "device": args.device, "run_at": ts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 50)
    print(f"  Total      : {total}")
    print(f"  Processed  : {len(ok)}")
    print(f"  Errors     : {len(errors)}  {error_stages if errors else ''}")
    print(f"  For review : {len(reviewed)}  ({len(reviewed)/max(len(ok),1)*100:.1f}%)")
    print(f"  Classes    : {class_counts}")
    print(f"\n  Output: {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
