"""
Phase 6 helper - download "stranger" face photos for the evaluation.

The evaluation needs photos of people who are NOT enrolled, to check that the
system correctly answers "Unknown" for strangers.

The classic LFW face dataset's servers are currently unreachable, so this
script uses thispersondoesnotexist.com, which generates a unique, realistic
(AI-made) face each time. Every downloaded face is a distinct "person" the
system has never seen - exactly what a stranger / impostor test needs.

Images are saved to:  data/test/Unknown/

Run it with:
    python src/download_strangers.py
    python src/download_strangers.py --count 30
"""

import argparse
import hashlib
import os
import time
import urllib.request

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
UNKNOWN_DIR = os.path.join(PROJECT_ROOT, "data", "test", "Unknown")

SOURCE_URL = "https://thispersondoesnotexist.com/"
# A browser-like User-Agent - some servers reject the default Python one.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_one(timeout=30):
    """Download one face image; return its bytes, or None on failure."""
    try:
        request = urllib.request.Request(SOURCE_URL, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as e:
        print(f"  request failed: {e}")
        return None


def download(count, delay=2.0):
    """Download `count` unique stranger faces into data/test/Unknown/."""
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    print(f"Downloading {count} stranger faces into {UNKNOWN_DIR}\n")

    seen_hashes = set()         # to skip images we already have
    saved = 0
    attempts = 0
    max_attempts = count * 5    # give up eventually if the site misbehaves

    while saved < count and attempts < max_attempts:
        attempts += 1
        data = fetch_one()
        if data is None:
            time.sleep(delay)
            continue

        # The site updates its image periodically; quick requests can return
        # the same picture. Skip duplicates by hashing the image bytes.
        digest = hashlib.md5(data).hexdigest()
        if digest in seen_hashes:
            print("  duplicate image - waiting for a fresh one")
            time.sleep(delay + 1.0)
            continue

        seen_hashes.add(digest)
        saved += 1
        path = os.path.join(UNKNOWN_DIR, f"stranger_{saved:02d}.jpg")
        with open(path, "wb") as f:
            f.write(data)
        print(f"  saved {saved}/{count}  ->  {os.path.basename(path)}")
        time.sleep(delay)

    print(f"\nDone. {saved} stranger photo(s) saved to {UNKNOWN_DIR}")
    if saved < count:
        print("(fewer than requested - the site may be rate-limiting; "
              "run again to add more)")


def main():
    parser = argparse.ArgumentParser(
        description="Download stranger faces for the Phase 6 evaluation.")
    parser.add_argument("--count", type=int, default=25,
                        help="how many stranger faces to download (default 25)")
    args = parser.parse_args()
    download(args.count)


if __name__ == "__main__":
    main()
