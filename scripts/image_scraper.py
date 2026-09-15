#!/usr/bin/env python3
"""
image_scraper.py

Scrape images from a website in batches.

Features:
- Crawls a starting URL (optionally following internal links up to a depth)
- Extracts <img> src/srcset and CSS background-image URLs
- Downloads images concurrently in configurable batch sizes
- Skips duplicates (by URL and by content hash)
- Resumable: keeps a manifest so re-running won't re-download existing files
- Respects a simple rate limit between batches

Usage:
    python image_scraper.py https://example.com --out ./images --batch-size 10

    # Crawl internal pages up to depth 2, only take images >= 200x200 (if size known)
    python image_scraper.py https://example.com --out ./images --depth 2

Requirements:
    pip install requests beautifulsoup4 --break-system-packages
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.parse as urlparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ImageScraperBot/1.0; "
        "+https://example.com/bot-info)"
    )
}


def is_valid_url(url: str) -> bool:
    parsed = urlparse.urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)


def normalize_url(base: str, link: str) -> str:
    return urlparse.urljoin(base, link.strip())


def same_domain(url: str, domain: str) -> bool:
    return urlparse.urlparse(url).netloc == domain


def extract_image_urls(html: str, page_url: str) -> set:
    """Pull image URLs from <img> tags (src, data-src, srcset) and inline CSS background-image."""
    soup = BeautifulSoup(html, "html.parser")
    urls = set()

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src"):
            val = img.get(attr)
            if val:
                urls.add(normalize_url(page_url, val))
        srcset = img.get("srcset")
        if srcset:
            # srcset format: "url1 1x, url2 2x, ..."
            for part in srcset.split(","):
                candidate = part.strip().split(" ")[0]
                if candidate:
                    urls.add(normalize_url(page_url, candidate))

    # inline style background-image: url(...)
    for tag in soup.find_all(style=True):
        style = tag["style"]
        if "background-image" in style or "background:" in style:
            import re

            for match in re.findall(r'url\(["\']?(.*?)["\']?\)', style):
                if match:
                    urls.add(normalize_url(page_url, match))

    return urls


def extract_page_links(html: str, page_url: str, domain: str) -> set:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        full = normalize_url(page_url, a["href"])
        full = full.split("#")[0]  # drop fragments
        if is_valid_url(full) and same_domain(full, domain):
            links.add(full)
    return links


def crawl(start_url: str, depth: int, session: requests.Session, delay: float):
    """Breadth-first crawl collecting image URLs from each visited page."""
    domain = urlparse.urlparse(start_url).netloc
    visited = set()
    to_visit = {start_url}
    all_image_urls = set()

    for current_depth in range(depth + 1):
        next_level = set()
        for url in to_visit:
            if url in visited:
                continue
            visited.add(url)
            try:
                resp = session.get(url, headers=DEFAULT_HEADERS, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  [skip] {url} -> {e}")
                continue

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            html = resp.text
            imgs = extract_image_urls(html, url)
            all_image_urls.update(imgs)
            print(f"  [page] {url} -> {len(imgs)} image(s) found (total {len(all_image_urls)})")

            if current_depth < depth:
                next_level.update(extract_page_links(html, url, domain))

            time.sleep(delay)

        to_visit = next_level - visited

    return all_image_urls


def guess_filename(url: str) -> str:
    parsed = urlparse.urlparse(url)
    name = os.path.basename(parsed.path)
    if not name:
        name = hashlib.sha1(url.encode()).hexdigest()[:16]
    # strip query-string style junk from filename, keep extension if present
    if "." not in name:
        name += ".jpg"
    return name


def download_one(url: str, out_dir: Path, session: requests.Session, min_bytes: int):
    """Download a single image. Returns (url, status, path_or_error)."""
    try:
        resp = session.get(url, headers=DEFAULT_HEADERS, timeout=20, stream=True)
        resp.raise_for_status()
        content = resp.content

        if len(content) < min_bytes:
            return (url, "too_small", None)

        content_hash = hashlib.sha256(content).hexdigest()
        filename = guess_filename(url)
        base, ext = os.path.splitext(filename)
        # avoid collisions: prefix with short hash
        final_name = f"{base}_{content_hash[:10]}{ext}"
        dest = out_dir / final_name

        if dest.exists():
            return (url, "exists", str(dest))

        with open(dest, "wb") as f:
            f.write(content)

        return (url, "ok", str(dest))

    except requests.RequestException as e:
        return (url, "error", str(e))


def batched(iterable, batch_size):
    items = list(iterable)
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest_path: Path, manifest: dict):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Scrape images from a website in batches.")
    parser.add_argument("url", help="Starting URL to scrape")
    parser.add_argument("--out", default="./scraped_images", help="Output directory")
    parser.add_argument("--depth", type=int, default=0, help="Link-crawl depth (0 = only the given page)")
    parser.add_argument("--batch-size", type=int, default=10, help="Concurrent downloads per batch")
    parser.add_argument("--batch-delay", type=float, default=1.0, help="Seconds to wait between batches")
    parser.add_argument("--crawl-delay", type=float, default=0.5, help="Seconds to wait between page fetches while crawling")
    parser.add_argument("--min-bytes", type=int, default=1024, help="Skip downloaded files smaller than this (filters tracking pixels/icons)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to download total")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore system/env proxy settings and connect directly")
    args = parser.parse_args()

    if not is_valid_url(args.url):
        print(f"Invalid URL: {args.url}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.json"
    manifest = load_manifest(manifest_path)

    session = requests.Session()
    if args.no_proxy:
        session.trust_env = False  # ignore HTTP_PROXY/HTTPS_PROXY and OS proxy config

    print(f"Crawling {args.url} (depth={args.depth}) ...")
    image_urls = crawl(args.url, args.depth, session, args.crawl_delay)
    print(f"\nFound {len(image_urls)} unique image URLs total.")

    # skip already-downloaded (by URL) per manifest
    pending = [u for u in image_urls if manifest.get(u, {}).get("status") != "ok"]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(pending)} pending download(s). Downloading in batches of {args.batch_size}...\n")

    total_ok, total_skip, total_err = 0, 0, 0

    for batch_num, batch in enumerate(batched(pending, args.batch_size), start=1):
        print(f"--- Batch {batch_num} ({len(batch)} images) ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {
                executor.submit(download_one, url, out_dir, session, args.min_bytes): url
                for url in batch
            }
            for future in concurrent.futures.as_completed(futures):
                url, status, info = future.result()
                manifest[url] = {"status": status, "path": info}
                if status == "ok":
                    total_ok += 1
                    print(f"  [ok]     {url} -> {info}")
                elif status == "exists":
                    total_skip += 1
                    print(f"  [exists] {url}")
                elif status == "too_small":
                    total_skip += 1
                    print(f"  [small]  {url}")
                else:
                    total_err += 1
                    print(f"  [error]  {url} -> {info}")

        save_manifest(manifest_path, manifest)
        time.sleep(args.batch_delay)

    print("\nDone.")
    print(f"  Downloaded: {total_ok}")
    print(f"  Skipped:    {total_skip}")
    print(f"  Errors:     {total_err}")
    print(f"  Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()