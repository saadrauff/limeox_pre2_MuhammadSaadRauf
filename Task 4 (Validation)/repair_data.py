
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import re
import time
import json
import html
import threading

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR.parent
    / "Task 3 (Preprocessing)"
    / "final_preprocessed_articles.csv"
)

OUTPUT_DIR = BASE_DIR / "repair_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "repaired_articles.csv"
LOG_FILE = OUTPUT_DIR / "repair_log.csv"
SUMMARY_FILE = OUTPUT_DIR / "repair_summary.json"

MIN_CONTENT_LENGTH = 200
MIN_TITLE_LENGTH = 20

TIMEOUT_SECONDS = 8
MAX_RETRIES = 3
MAX_WORKERS = 8
SAVE_EVERY = 25

# No runtime limit: process all pending candidates.
ERROR_REGEX = re.compile(
    r"access denied|enable javascript|robot check|"
    r"403 forbidden|verify you are human|"
    r"checking your browser|captcha|"
    r"attention required|request blocked",
    re.I,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

print_lock = threading.Lock()


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_string(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def looks_like_error_page(text):
    text = clean_text(text)

    return len(text) < 3000 and bool(ERROR_REGEX.search(text))


def is_valid_http_url(url):
    if not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
        )
    except ValueError:
        return False


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_json_ld(soup):
    """Extract articleBody and headline from JSON-LD metadata."""
    bodies = []
    title = ""

    scripts = soup.find_all(
        "script",
        attrs={"type": re.compile(r"application/ld\+json", re.I)}
    )

    for script in scripts:
        raw = script.string or script.get_text()

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        stack = data if isinstance(data, list) else [data]

        while stack:
            obj = stack.pop()

            if isinstance(obj, list):
                stack.extend(obj)
                continue

            if not isinstance(obj, dict):
                continue

            graph = obj.get("@graph")
            if isinstance(graph, list):
                stack.extend(graph)

            body = obj.get("articleBody")
            if isinstance(body, str):
                body = clean_text(body)
                if body:
                    bodies.append(body)

            if not title:
                headline = obj.get("headline") or obj.get("name")
                if isinstance(headline, str):
                    title = clean_text(headline)

    best_body = max(bodies, key=len) if bodies else ""
    return title, best_body


def extract_article_data(soup):
    """Try structured data, metadata and common article selectors."""
    title = ""

    metadata_selectors = [
        ("meta", {"property": "og:title"}),
        ("meta", {"name": "twitter:title"}),
        ("meta", {"name": "title"}),
    ]

    for tag_name, attrs in metadata_selectors:
        tag = soup.find(tag_name, attrs=attrs)

        if tag and tag.get("content"):
            title = clean_text(tag["content"])
            if title:
                break

    json_title, json_body = extract_json_ld(soup)

    if not title:
        title = json_title

    if not title:
        h1 = soup.find("h1")
        if h1:
            title = clean_text(h1.get_text(" ", strip=True))

    if not title and soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))

    selectors = [
        "article",
        '[itemprop="articleBody"]',
        ".article-body",
        ".article-content",
        ".story-body",
        ".story-content",
        ".post-content",
        ".entry-content",
        ".article__body",
        ".article__content",
        "main",
    ]

    candidates = []

    if json_body:
        candidates.append(json_body)

    for selector in selectors:
        for element in soup.select(selector):
            paragraphs = element.find_all("p")

            if paragraphs:
                text = clean_text(
                    " ".join(
                        p.get_text(" ", strip=True)
                        for p in paragraphs
                    )
                )
            else:
                text = clean_text(element.get_text(" ", strip=True))

            if text:
                candidates.append(text)

    if not candidates:
        for tag in soup([
            "script", "style", "nav", "footer", "header",
            "aside", "form", "noscript", "svg"
        ]):
            tag.decompose()

        text = clean_text(
            " ".join(
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
            )
        )

        if text:
            candidates.append(text)

    content = max(candidates, key=len) if candidates else ""

    return title, content


# ============================================================
# FETCH ARTICLE
# ============================================================

def fetch_article(idx, url):
    result = {
        "row_index": int(idx),
        "status": "",
        "title": None,
        "content": None,
        "http_status": None,
        "final_url": None,
    }

    if not isinstance(url, str) or not url.strip():
        result["status"] = "MISSING_URL"
        return result

    url = url.strip()

    if not is_valid_http_url(url):
        result["status"] = "INVALID_URL"
        return result

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT_SECONDS,
                allow_redirects=True,
            )

            result["http_status"] = response.status_code
            result["final_url"] = response.url

            if response.status_code in (403, 404, 410, 429):
                result["status"] = f"HTTP_{response.status_code}"
                return result

            if response.status_code >= 500:
                result["status"] = f"HTTP_{response.status_code}"

                if attempt < MAX_RETRIES:
                    time.sleep(attempt)
                    continue

                return result

            if response.status_code != 200:
                result["status"] = f"HTTP_{response.status_code}"
                return result

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "html" not in content_type:
                result["status"] = "NOT_HTML"
                return result

            soup = BeautifulSoup(response.text, "html.parser")

            page_text = clean_text(soup.get_text(" ", strip=True))

            if looks_like_error_page(page_text):
                result["status"] = "POSSIBLE_BLOCK_PAGE"
                return result

            title, content = extract_article_data(soup)

            result["title"] = title or None
            result["content"] = content or None

            if not content:
                result["status"] = "NO_TEXT_EXTRACTED"
            elif looks_like_error_page(content):
                result["status"] = "POSSIBLE_BLOCK_PAGE"
                result["content"] = None
            elif len(content) < MIN_CONTENT_LENGTH:
                result["status"] = "CONTENT_STILL_TOO_SHORT"
            else:
                result["status"] = "FETCH_OK"

            return result

        except requests.Timeout:
            result["status"] = "REQUEST_FAILED: Timeout"

        except requests.ConnectionError:
            result["status"] = "REQUEST_FAILED: ConnectionError"

        except requests.RequestException as exc:
            result["status"] = (
                f"REQUEST_FAILED: {type(exc).__name__}"
            )

        except Exception as exc:
            result["status"] = (
                f"UNEXPECTED_ERROR: {type(exc).__name__}"
            )
            return result

        if attempt < MAX_RETRIES:
            time.sleep(attempt)

    return result


# ============================================================
# SAVE FILES
# ============================================================

def save_outputs(df, logs):
    """Write temporary files, then replace the previous outputs."""
    temp_output = OUTPUT_FILE.with_suffix(".tmp.csv")
    temp_log = LOG_FILE.with_suffix(".tmp.csv")

    df.to_csv(temp_output, index=False, encoding="utf-8-sig")

    pd.DataFrame(logs).to_csv(
        temp_log, index=False, encoding="utf-8-sig"
    )

    temp_output.replace(OUTPUT_FILE)
    temp_log.replace(LOG_FILE)


# ============================================================
# MAIN REPAIR PROCESS
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Resume from the existing repaired output when available.
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE, low_memory=False)
        print("Existing repaired output loaded.")
    else:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        print("Original preprocessed input loaded.")

    required = {"article_url", "content", "title"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    if LOG_FILE.exists():
        logs = (
            pd.read_csv(LOG_FILE, low_memory=False)
            .fillna("")
            .to_dict("records")
        )
    else:
        logs = []

    # Only successful repairs are skipped on future runs.
    successful_rows = set()

    for item in logs:
        try:
            idx = int(item["row_index"])
            if str(item.get("status", "")) == "REPAIRED":
                successful_rows.add(idx)
        except (KeyError, ValueError, TypeError):
            continue

    content_series = df["content"].fillna("").astype(str)
    title_series = df["title"].fillna("").astype(str)

    content_lengths = content_series.str.strip().str.len()
    title_lengths = title_series.str.strip().str.len()

    candidate_mask = (
        (content_lengths < MIN_CONTENT_LENGTH)
        | content_series.map(looks_like_error_page)
        | (title_lengths < MIN_TITLE_LENGTH)
    )

    candidates = df.index[candidate_mask].tolist()

    pending = [
        idx for idx in candidates
        if int(idx) not in successful_rows
    ]

    print(f"Total dataset rows: {len(df):,}")
    print(f"Repair candidates: {len(candidates):,}")
    print(f"Previously repaired rows skipped: {len(successful_rows):,}")
    print(f"Rows to attempt this run: {len(pending):,}")
    print("Runtime limit: None")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("\nRepair started...\n")

    start = time.monotonic()

    attempted_this_run = 0
    repaired_this_run = 0
    title_updates = 0
    content_updates = 0
    next_position = 0

    # No deadline condition: process every pending candidate.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while next_position < len(pending):
            batch = pending[
                next_position:next_position + MAX_WORKERS * 2
            ]
            next_position += len(batch)

            future_map = {
                executor.submit(
                    fetch_article,
                    idx,
                    safe_string(df.at[idx, "article_url"]),
                ): idx
                for idx in batch
            }

            for future in as_completed(future_map):
                idx = future_map[future]

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "row_index": int(idx),
                        "status": (
                            f"UNEXPECTED_ERROR: {type(exc).__name__}"
                        ),
                        "title": None,
                        "content": None,
                        "http_status": None,
                        "final_url": None,
                    }

                old_title = safe_string(df.at[idx, "title"])
                old_content = safe_string(df.at[idx, "content"])

                fetched_title = safe_string(result.get("title"))
                fetched_content = safe_string(result.get("content"))

                title_changed = False
                content_changed = False

                # Update title only if it is usable and improves
                # the existing title.
                if (
                    fetched_title
                    and len(fetched_title) >= MIN_TITLE_LENGTH
                    and len(fetched_title) > len(old_title)
                    and len(fetched_title) <= 500
                ):
                    df.at[idx, "title"] = fetched_title
                    title_changed = True
                    title_updates += 1

                # Replace content only with a longer, usable body.
                if (
                    fetched_content
                    and len(fetched_content) >= MIN_CONTENT_LENGTH
                    and len(fetched_content) > len(old_content)
                    and not looks_like_error_page(fetched_content)
                ):
                    df.at[idx, "content"] = fetched_content
                    content_changed = True
                    content_updates += 1

                did_repair = title_changed or content_changed

                if did_repair:
                    status = "REPAIRED"
                    repaired_this_run += 1
                    successful_rows.add(int(idx))
                else:
                    status = result.get("status", "UNKNOWN")

                logs.append({
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "row_index": int(idx),
                    "article_url": safe_string(
                        df.at[idx, "article_url"]
                    ),
                    "status": status,
                    "fetch_status": result.get("status", ""),
                    "http_status": result.get("http_status"),
                    "final_url": result.get("final_url"),
                    "original_title_length": len(old_title),
                    "final_title_length": len(
                        safe_string(df.at[idx, "title"])
                    ),
                    "original_content_length": len(old_content),
                    "final_content_length": len(
                        safe_string(df.at[idx, "content"])
                    ),
                    "title_replaced": title_changed,
                    "content_replaced": content_changed,
                })

                attempted_this_run += 1

                if attempted_this_run % SAVE_EVERY == 0:
                    save_outputs(df, logs)

                    elapsed = (time.monotonic() - start) / 60

                    with print_lock:
                        print(
                            f"Progress: {attempted_this_run:,}/"
                            f"{len(pending):,} | "
                            f"repaired: {repaired_this_run:,} | "
                            f"content updates: {content_updates:,} | "
                            f"title updates: {title_updates:,} | "
                            f"elapsed: {elapsed:.1f} min"
                        )

    # Save final progress after all pending candidates are attempted.
    save_outputs(df, logs)

    final_content = (
        df["content"].fillna("").astype(str).str.strip()
    )
    final_title = (
        df["title"].fillna("").astype(str).str.strip()
    )

    remaining_short_content = int(
        (final_content.str.len() < MIN_CONTENT_LENGTH).sum()
    )

    remaining_short_titles = int(
        (final_title.str.len() < MIN_TITLE_LENGTH).sum()
    )

    remaining_error_pages = int(
        final_content.map(looks_like_error_page).sum()
    )

    summary = {
        "input_rows": int(len(df)),
        "repair_candidates": int(len(candidates)),
        "previously_repaired_rows_skipped": int(
            len(successful_rows) - repaired_this_run
        ),
        "attempted_this_run": int(attempted_this_run),
        "repaired_this_run": int(repaired_this_run),
        "content_updates_this_run": int(content_updates),
        "title_updates_this_run": int(title_updates),
        "candidates_not_attempted_this_run": int(
            max(0, len(pending) - attempted_this_run)
        ),
        "short_content_remaining": remaining_short_content,
        "short_titles_remaining": remaining_short_titles,
        "possible_error_pages_remaining": remaining_error_pages,
        "original_input_modified": False,
        "runtime_limit_minutes": None,
        "output_file": str(OUTPUT_FILE),
        "repair_log": str(LOG_FILE),
        "elapsed_minutes": round(
            (time.monotonic() - start) / 60, 2
        ),
        "note": (
            "No runtime limit is configured. All pending candidates "
            "are attempted unless the process is interrupted. "
            "Failed attempts can be retried on later runs. "
            "Rows marked REPAIRED are skipped on later runs. "
            "A successful HTTP response does not guarantee "
            "that the page contains the full article."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nRepair processing completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
