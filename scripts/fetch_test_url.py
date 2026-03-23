import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from app.config import settings


def login() -> tuple[str, str]:
    url = f"{settings.TRANSTRACK_BASE_URL}/auth"
    resp = requests.post(
        url,
        json={"username": settings.TRANSTRACK_USERNAME, "password": settings.TRANSTRACK_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    token = data.get("token")
    pid = data.get("pid")
    if not token or not pid:
        raise RuntimeError(f"Login failed: {resp.json()}")
    print(f"Logged in — pid={pid}")
    return token, pid


def fetch_one_url(token: str, pid: str) -> str:
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
        raise RuntimeError("No events returned")

    for event in events:
        alarm_files = event.get("alarm_file", [])
        if alarm_files:
            down_url = alarm_files[0].get("downUrl", "")
            if down_url:
                return down_url.replace("&amp;", "&")

    raise RuntimeError("No downloadable URL found in events")


if __name__ == "__main__":
    if not settings.TRANSTRACK_BASE_URL or not settings.TRANSTRACK_USERNAME:
        print("Fill in TRANSTRACK_BASE_URL, TRANSTRACK_USERNAME, TRANSTRACK_PASSWORD in .env first")
        exit(1)

    token, pid = login()
    video_url = fetch_one_url(token, pid)

    print(f"\nFresh video URL:\n{video_url}")
    print(f"\nRun integration test with:")
    print(f"set TEST_VIDEO_URL={video_url}")
    print(f"pytest tests/ -v -m integration -s")
