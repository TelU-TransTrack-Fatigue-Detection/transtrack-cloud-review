import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import requests
from app.config import settings
from app.pipeline import predict

TARGET     = 5
OUT_DIR    = Path("records/batch")
DATE_START = "2026-02-01%2003:00:00"
DATE_END   = "2026-02-01%2005:00:00"
MODEL_PATH = Path(settings.MODEL_PATH)
MODEL_NAME = settings.MODEL_NAME


def login() -> tuple[str, str]:
    resp = requests.post(
        f"{settings.TRANSTRACK_BASE_URL}/auth",
        json={"username": settings.TRANSTRACK_USERNAME, "password": settings.TRANSTRACK_PASSWORD},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Login {resp.status_code}: {resp.text}")
    data = resp.json().get("data", {})
    token, pid = data.get("token"), data.get("pid")
    if not token or not pid:
        raise RuntimeError(f"Login failed — no token/pid: {resp.json()}")
    print(f"[login] OK — pid={pid}\n")
    return token, pid


def fetch_events(token: str, pid: str, page_size: int = 20) -> list[dict]:
    session = requests.Session()
    session.headers.update({"X-Token": token, "X-User": pid, "Authorization": f"Bearer {token}"})
    url = (
        f"{settings.TRANSTRACK_BASE_URL}/events"
        f"?order=desc&sort=device_time"
        f"&filter_columns=alarm_type&filter_value=121,122"
        f"&page=1&page_size={page_size}"
        f"&range_date_columns=device_time"
        f"&range_date_start={DATE_START}"
        f"&range_date_end={DATE_END}"
    )
    resp = session.get(url, timeout=180)
    resp.raise_for_status()
    return resp.json().get("data", {}).get("list", [])


def collect_urls(events: list[dict]) -> list[tuple[str, str]]:
    results = []
    for event in events:
        eid = event.get("identity", "?")
        alarm_type = event.get("alarm_type", "?")
        for af in event.get("alarm_file", []):
            url = af.get("downUrl", "").replace("&amp;", "&")
            if url:
                results.append((eid, alarm_type, url))
                break
    return results


def download(url: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=settings.VIDEO_DOWNLOAD_TIMEOUT) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                    f.write(chunk)
    return path.stat().st_size


if __name__ == "__main__":
    if not settings.TRANSTRACK_BASE_URL or not settings.TRANSTRACK_USERNAME:
        print("Fill in TRANSTRACK_BASE_URL, TRANSTRACK_USERNAME, TRANSTRACK_PASSWORD in .env")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    token, pid = login()

    print(f"Fetching events...")
    events = fetch_events(token, pid, page_size=20)
    candidates = collect_urls(events)
    print(f"Found {len(candidates)} downloadable events, targeting {TARGET}\n")

    if not candidates:
        print("No events with downloadable URLs found.")
        sys.exit(1)

    results = []
    for i, (eid, alarm_type, url) in enumerate(candidates[:TARGET], start=1):
        print(f"[{i}/{TARGET}] Event {eid}  alarm_type={alarm_type}")

        video_path = OUT_DIR / f"video_{i:02d}_event{eid}.mp4"
        try:
            size = download(url, video_path)
            print(f"         Downloaded {size/1024:.1f} KB → {video_path.name}")
        except Exception as e:
            print(f"         Download FAILED: {e}")
            results.append({"event": eid, "alarm_type": alarm_type, "error": str(e)})
            continue

        try:
            pred = predict(video_path, MODEL_PATH, MODEL_NAME)
            print(f"         Label      : {pred['label'].upper()}")
            print(f"         Confidence : {pred['confidence']:.4f} ({int(pred['confidence']*100)}%)")
            results.append({
                "event":      eid,
                "alarm_type": alarm_type,
                "file":       str(video_path),
                "label":      pred["label"],
                "class_id":   pred["class_id"],
                "confidence": pred["confidence"],
            })
        except Exception as e:
            print(f"         Inference FAILED: {e}")
            results.append({"event": eid, "alarm_type": alarm_type, "file": str(video_path), "error": str(e)})

        print()

    print("=" * 55)
    print(f"{'#':<4} {'Event':<12} {'AlarmType':<12} {'Label':<10} {'Conf':>6}")
    print("-" * 55)
    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"{i:<4} {str(r['event']):<12} {str(r['alarm_type']):<12} {'ERROR':<10} {'—':>6}")
        else:
            print(f"{i:<4} {str(r['event']):<12} {str(r['alarm_type']):<12} {r['label']:<10} {r['confidence']:>6.4f}")
    print("=" * 55)
    print(f"\nVideos saved in: {OUT_DIR.resolve()}")
