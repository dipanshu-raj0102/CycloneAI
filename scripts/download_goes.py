import os
import time
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
from tqdm import tqdm

# --- CONFIGURATION ---
START_DATE = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
END_DATE   = datetime(2023, 12, 31, 18, 0, tzinfo=timezone.utc)
INTERVAL_HOURS = 6
OUTPUT_DIR = "./../data//goes16_dataset"

# Tuned for 200 Mbps (max 18 concurrent sockets to prevent router NAT stalls)
FILE_WORKERS = 32
CHUNK_WORKERS = 32
CHUNK_THRESHOLD = 12 * 1024 * 1024  # 12 MB
BUFFER_SIZE = 1024 * 1024           # 1 MB
MAX_CHUNK_RETRIES = 5

# NOAA AWS S3 Public HTTPS Endpoint (Use "noaa-goes19" for dates after April 4, 2025)
GOES_BUCKET = "noaa-goes16"
BASE_URL = f"https://{GOES_BUCKET}.s3.amazonaws.com"
PRODUCT = "ABI-L2-CMIPF"

CHANNELS = {
    "WV":  "C08",  # 6.2 µm  (~15-25 MB)
    "IR":  "C13",  # 10.3 µm (~15-25 MB)
}

# Enforce strict connect & read timeouts so stalled sockets fail fast and retry
HTTP = urllib3.PoolManager(
    num_pools=2,
    maxsize=FILE_WORKERS * CHUNK_WORKERS + 4,
    block=True,
    timeout=urllib3.Timeout(connect=10.0, read=15.0),
    retries=urllib3.Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504]),
)


class BandwidthTracker:
    def __init__(self):
        self.total_bytes = 0
        self.start_time = time.perf_counter()
        self.lock = threading.Lock()

    def record(self, n_bytes: int):
        with self.lock:
            self.total_bytes += n_bytes

    def current_mbps(self) -> float:
        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        return (self.total_bytes * 8.0) / (elapsed * 1_000_000)


TRACKER = BandwidthTracker()


def generate_timestamps(start: datetime, end: datetime, step_hours: int):
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=step_hours)


def list_keys_for_timestamp(dt: datetime):
    prefix = f"{PRODUCT}/{dt.strftime('%Y')}/{dt.strftime('%j')}/{dt.strftime('%H')}/"
    url = f"{BASE_URL}/?list-type=2&prefix={prefix}"

    resp = HTTP.request("GET", url)
    if resp.status != 200:
        return []

    root = ET.fromstring(resp.data)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    objects = [
        (c.find("s3:Key", ns).text, int(c.find("s3:Size", ns).text))
        for c in root.findall("s3:Contents", ns)
    ]

    tasks = []
    for band_label, channel_id in CHANNELS.items():
        band_objs = sorted([obj for obj in objects if f"{channel_id}_" in obj[0]], key=lambda x: x[0])
        if band_objs:
            target_key, target_size = band_objs[0]
            local_path = os.path.join(
                OUTPUT_DIR, dt.strftime("%Y%m%d_%H%M"), band_label, os.path.basename(target_key)
            )
            tasks.append((target_key, target_size, local_path))

    return tasks


def _download_range_with_retry(url: str, start: int, end: int, fd: int, write_lock: threading.Lock, pbar: tqdm):
    """Downloads a byte range with automatic mid-stream stall recovery."""
    offset = start
    for attempt in range(1, MAX_CHUNK_RETRIES + 1):
        headers = {"Range": f"bytes={offset}-{end}"}
        resp = None
        try:
            resp = HTTP.request("GET", url, headers=headers, preload_content=False)
            if resp.status not in (200, 206):
                raise urllib3.exceptions.HTTPError(f"HTTP {resp.status}")

            for chunk in resp.stream(BUFFER_SIZE):
                n = len(chunk)
                if hasattr(os, "pwrite"):
                    os.pwrite(fd, chunk, offset)
                else:
                    with write_lock:
                        os.lseek(fd, offset, os.SEEK_SET)
                        os.write(fd, chunk)
                offset += n
                TRACKER.record(n)
                pbar.update(n)

            resp.release_conn()
            return  # Range completed cleanly
        except Exception:
            if resp is not None:
                resp.close()  # Discard stalled/broken socket instead of pooling it
            if attempt == MAX_CHUNK_RETRIES:
                raise
            time.sleep(0.5 * attempt)


def download_file_fast(key: str, size: int, local_path: str, pbar: tqdm):
    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        pbar.update(size)
        return False

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    part_path = f"{local_path}.part"
    url = f"{BASE_URL}/{key}"

    if size >= CHUNK_THRESHOLD and CHUNK_WORKERS > 1:
        fd = os.open(part_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        os.ftruncate(fd, size)
        write_lock = threading.Lock()
        part_size = size // CHUNK_WORKERS

        ranges = [
            (i * part_size, size - 1 if i == CHUNK_WORKERS - 1 else (i + 1) * part_size - 1)
            for i in range(CHUNK_WORKERS)
        ]

        try:
            with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as chunk_pool:
                futures = [
                    chunk_pool.submit(_download_range_with_retry, url, r_start, r_end, fd, write_lock, pbar)
                    for r_start, r_end in ranges
                ]
                for f in as_completed(futures):
                    f.result()
        finally:
            os.close(fd)
    else:
        for attempt in range(1, MAX_CHUNK_RETRIES + 1):
            resp = None
            bytes_written = 0
            try:
                resp = HTTP.request("GET", url, preload_content=False)
                with open(part_path, "wb", buffering=BUFFER_SIZE) as f:
                    for chunk in resp.stream(BUFFER_SIZE):
                        f.write(chunk)
                        n = len(chunk)
                        bytes_written += n
                        TRACKER.record(n)
                        pbar.update(n)
                resp.release_conn()
                break
            except Exception:
                if resp is not None:
                    resp.close()
                pbar.update(-bytes_written)
                if attempt == MAX_CHUNK_RETRIES:
                    raise
                time.sleep(0.5 * attempt)

    os.replace(part_path, local_path)
    return True


def main():
    timestamps = list(generate_timestamps(START_DATE, END_DATE, INTERVAL_HOURS))
    print(f"Indexing GOES-16 ABI files across {len(timestamps)} timestamps...")

    download_queue = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(list_keys_for_timestamp, dt) for dt in timestamps]
        for future in as_completed(futures):
            download_queue.extend(future.result())

    if not download_queue:
        print("No matching files found for the specified date range.")
        return

    total_bytes = sum(item[1] for item in download_queue)
    print(f"Queued {len(download_queue)} files ({total_bytes / (1024**3):.2f} GB) | Workers: {FILE_WORKERS}x{CHUNK_WORKERS}")

    downloaded = 0
    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="GOES-16") as pbar:
        with ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
            futures = [
                executor.submit(download_file_fast, key, size, path, pbar)
                for key, size, path in download_queue
            ]
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
                pbar.set_postfix(speed=f"{TRACKER.current_mbps():.1f} Mbps")

    print(
        f"Done. Downloaded: {downloaded} | Skipped: {len(download_queue) - downloaded} | "
        f"Avg Speed: {TRACKER.current_mbps():.1f} Mbps"
    )


if __name__ == "__main__":
    main()
