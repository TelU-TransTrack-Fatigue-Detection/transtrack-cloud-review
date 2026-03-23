import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import cv2
import requests
from config import settings


def login() -> tuple[str, str]:
    resp = requests.post(
        f"{settings.TRANSTRACK_BASE_URL}/auth",
        json={"username": settings.TRANSTRACK_USERNAME, "password": settings.TRANSTRACK_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    token, pid = data.get("token"), data.get("pid")
    if not token or not pid:
        raise RuntimeError(f"Login failed: {resp.json()}")
    print(f"[1/4] Logged in")
    return token, pid


def fetch_video_url(token: str, pid: str) -> tuple[str, dict]:
    session = requests.Session()
    session.headers.update({
        "X-Token": token,
        "X-User": pid,
        "Authorization": f"Bearer {token}",
    })
    url = (
        f"{settings.TRANSTRACK_BASE_URL}/events"
        f"?order=desc&sort=device_time"
        f"&filter_columns=alarm_type&filter_value=121,122"
        f"&page=1&page_size=5"
        f"&range_date_columns=device_time"
        f"&range_date_start=2026-02-01%2003:00:00"
        f"&range_date_end=2026-02-01%2005:00:00"
    )
    resp = session.get(url, timeout=180)
    resp.raise_for_status()
    events = resp.json().get("data", {}).get("list", [])
    if not events:
        raise RuntimeError("No events found")
    for event in events:
        for alarm_file in event.get("alarm_file", []):
            down_url = alarm_file.get("downUrl", "").replace("&amp;", "&")
            if down_url:
                print(f"[2/4] Got video URL from event id={event.get('identity')}")
                return down_url, event
    raise RuntimeError("No downloadable URL in events")


def download_video(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] Downloading to {output_path} ...")
    with httpx.Client(timeout=settings.VIDEO_DOWNLOAD_TIMEOUT) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                    f.write(chunk)
    size_kb = output_path.stat().st_size / 1024
    print(f"      Saved — {size_kb:.1f} KB")
    return output_path


def inspect_video(path: Path):
    print(f"[4/4] Inspecting video with cv2 ...")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"      cv2 could not open the file — likely not a valid MP4")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = round(total_frames / fps, 2) if fps > 0 else 0
    ret, frame  = cap.read()
    cap.release()

    print(f"      Resolution   : {width}x{height}")
    print(f"      FPS          : {fps}")
    print(f"      Total frames : {total_frames}")
    print(f"      Duration     : {duration}s")
    print(f"      First frame  : {'OK' if ret else 'FAILED'}")


if __name__ == "__main__":
    if not settings.TRANSTRACK_BASE_URL or not settings.TRANSTRACK_USERNAME:
        print("Fill in TRANSTRACK_BASE_URL, TRANSTRACK_USERNAME, TRANSTRACK_PASSWORD in .env first")
        sys.exit(1)

    token, pid = login()
    video_url, event = fetch_video_url(token, pid)
    output_path = Path("records/test_download/test_video.mp4")
    download_video(video_url, output_path)
    inspect_video(output_path)
    print(f"\nDone. Video saved at: {output_path.resolve()}")
