import csv
import json
import re
import time as sleep_time
import html
import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

TARGET_ARTICLES = 1000
OUTPUT_FILE = "articles excel file.csv"
REQUEST_TIMEOUT = 20
REQUEST_DELAY_SECONDS = 0.35
USE_SELENIUM_FALLBACK = True

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Ten sources. Feed URLs are used first; sitemaps are a fallback/discovery route.
SOURCES = [
    {
        "name": "OpenAI",
        "feed": "https://openai.com/news/rss.xml",
        "sitemap": "https://openai.com/sitemap.xml",
        "domain": "openai.com",
    },
    {
        "name": "Google DeepMind",
        "feed": "https://deepmind.google/blog/rss.xml",
        "sitemap": "https://deepmind.google/sitemap.xml",
        "domain": "deepmind.google",
    },
    {
        "name": "Hugging Face",
        "feed": "https://huggingface.co/blog/feed.xml",
        "sitemap": "https://huggingface.co/sitemap.xml",
        "domain": "huggingface.co",
    },
    {
        "name": "MIT Technology Review",
        "feed": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
        "sitemap": "https://www.technologyreview.com/sitemap.xml",
        "domain": "technologyreview.com",
    },
    {
        "name": "TechCrunch",
        "feed": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "sitemap": "https://techcrunch.com/sitemap.xml",
        "domain": "techcrunch.com",
    },
    {
        "name": "The Verge",
        "feed": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "sitemap": "https://www.theverge.com/sitemap.xml",
        "domain": "theverge.com",
    },
    {
        "name": "Ars Technica",
        "feed": "https://arstechnica.com/ai/feed/",
        "sitemap": "https://arstechnica.com/sitemap.xml",
        "domain": "arstechnica.com",
    },
    {
        "name": "WIRED",
        "feed": "https://www.wired.com/feed/tag/ai/latest/rss",
        "sitemap": "https://www.wired.com/sitemap.xml",
        "domain": "wired.com",
    },
    {
        "name": "VentureBeat",
        "feed": "https://venturebeat.com/category/ai/feed/",
        "sitemap": "https://venturebeat.com/sitemap_index.xml",
        "domain": "venturebeat.com",
    },
    {
        "name": "Google Research",
        "feed": "https://research.google/blog/rss/",
        "sitemap": "https://research.google/sitemap.xml",
        "domain": "research.google",
    },
]

session = requests.Session()
session.headers.update(HEADERS)
selenium_driver = None


def clean_text(value):
    """Remove HTML markup and normalize whitespace."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    soup = BeautifulSoup(str(value), "html.parser")
    return re.sub(r"\s+", " ", html.unescape(soup.get_text(" ", strip=True))).strip()


def normalize_url(raw_url, base_url=""):
    """
    Normalize common feed URL representations:
    - HTML entities (&amp;)
    - percent-encoding (%2F, %3A, etc.)
    - relative URLs
    - redirect/tracking URLs are resolved later with Requests
    - safely unwrap common base64 data only when it decodes to an HTTP(S) URL
    """
    if not raw_url:
        return ""
    url = html.unescape(str(raw_url).strip())
    url = url.replace("\\/", "/")
    url = unquote(url)

    # Some sources encode destination URLs in query parameters.
    try:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https") and parsed.query:
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            for key in ("url", "u", "target", "redirect", "redirect_url", "dest", "destination"):
                candidate = params.get(key, [""])[0]
                candidate = unquote(html.unescape(candidate))
                if candidate.startswith(("http://", "https://")):
                    url = candidate
                    break
    except Exception:
        pass

    # Decode base64 only if the result is clearly an HTTP(S) URL.
    if not url.startswith(("http://", "https://")):
        try:
            padded = url + ("=" * (-len(url) % 4))
            decoded = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", "ignore")
            if decoded.startswith(("http://", "https://")):
                url = decoded
        except Exception:
            pass

    if base_url:
        url = urljoin(base_url, url)
    if url.startswith("//"):
        url = "https:" + url
    if not url.startswith(("http://", "https://")):
        return ""
    return url


def resolve_url(url):
    """Follow HTTP redirects and return the final URL; keep original if request fails."""
    url = normalize_url(url)
    if not url:
        return ""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True)
        final_url = response.url
        response.close()
        if final_url and final_url.startswith(("http://", "https://")):
            return normalize_url(final_url)
    except requests.RequestException:
        pass
    return url


def parse_date(value):
    """Return (YYYY-MM-DD, HH:MM:SS) or ('', '')."""
    if not value:
        return "", ""
    value = str(value).strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except Exception:
            # Handle common date-only values.
            try:
                dt = datetime.strptime(value[:10], "%Y-%m-%d")
            except Exception:
                return "", ""
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def first_tag_text(element, names):
    for name in names:
        tag = element.find(name)
        if tag:
            value = tag.get("href") or tag.get("content") or tag.get_text(" ", strip=True)
            if value:
                return clean_text(value)
    return ""


def parse_feed(feed_url, source_name):
    """Parse RSS or Atom using BeautifulSoup XML parser."""
    entries = []
    try:
        response = session.get(feed_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
        nodes = soup.find_all("item")
        if not nodes:
            nodes = soup.find_all("entry")

        for node in nodes:
            title = first_tag_text(node, ["title"])
            link_tag = node.find("link")
            raw_link = ""
            if link_tag:
                raw_link = link_tag.get("href") or link_tag.get_text(strip=True)
            if not raw_link:
                guid = node.find("guid")
                raw_link = guid.get_text(strip=True) if guid else ""

            author = first_tag_text(node, ["author", "creator", "dc:creator", "name"])
            # Atom author is often <author><name>...</name></author>.
            author_tag = node.find("author")
            if author_tag and not author:
                author = first_tag_text(author_tag, ["name"])

            date_raw = first_tag_text(
                node,
                ["pubDate", "published", "updated", "dc:date", "date", "lastBuildDate"],
            )
            pub_date, pub_time = parse_date(date_raw)
            article_url = normalize_url(raw_link, feed_url)
            if title and article_url:
                entries.append({
                    "title": title,
                    "author": author,
                    "publication_date": pub_date,
                    "article_url": article_url,
                    "time": pub_time,
                    "source": source_name,
                })
        print(f"[FEED] {source_name}: {len(entries)} entries")
    except Exception as exc:
        print(f"[FEED ERROR] {source_name}: {exc}")
    return entries


def fetch_sitemap_urls(sitemap_url, domain, depth=0, max_depth=2):
    """Read sitemap XML and sitemap indexes, returning URL + lastmod pairs."""
    results = []
    if not sitemap_url or depth > max_depth:
        return results
    try:
        response = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")

        # Sitemap index: recursively visit a limited number of child sitemaps.
        sitemap_tags = soup.find_all("sitemap")
        if sitemap_tags:
            children = []
            for sm in sitemap_tags:
                loc = sm.find("loc")
                if loc and loc.get_text(strip=True):
                    children.append(loc.get_text(strip=True))
            # Prefer recent-looking article/content sitemaps and avoid crawling every sitemap.
            children = children[:40]
            for child in children:
                results.extend(fetch_sitemap_urls(child, domain, depth + 1, max_depth))
                if len(results) >= TARGET_ARTICLES * 3:
                    break
            return results

        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            if not loc:
                continue
            url = normalize_url(loc.get_text(strip=True), sitemap_url)
            if not url:
                continue
            host = urlparse(url).netloc.lower()
            if domain not in host:
                continue
            lastmod_tag = url_tag.find("lastmod")
            lastmod = lastmod_tag.get_text(strip=True) if lastmod_tag else ""
            results.append((url, lastmod))
            if len(results) >= TARGET_ARTICLES * 3:
                break
        print(f"[SITEMAP] {domain}: discovered {len(results)} URLs from {sitemap_url}")
    except Exception as exc:
        print(f"[SITEMAP ERROR] {domain}: {exc}")
    return results


def extract_jsonld(soup):
    """Return the first useful Article/NewsArticle JSON-LD object."""
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        try:
            data = json.loads(raw)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and isinstance(data.get("@graph"), list):
            candidates.extend(data["@graph"])
        for item in candidates:
            if not isinstance(item, dict):
                continue
            typ = item.get("@type", "")
            types = typ if isinstance(typ, list) else [typ]
            if any("article" in str(t).lower() or "blogposting" in str(t).lower() for t in types):
                return item
        if isinstance(data, dict):
            return data
    return {}


def meta_content(soup, *selectors):
    for selector in selectors:
        tag = soup.select_one(selector)
        if tag:
            val = tag.get("content") or tag.get_text(" ", strip=True)
            if val:
                return clean_text(val)
    return ""


def extract_article_with_requests(url, source_name):
    """Fetch article page and extract title, author and publication datetime."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        final_url = normalize_url(response.url)
        soup = BeautifulSoup(response.text, "html.parser")
        data = extract_jsonld(soup)

        title = clean_text(data.get("headline") or data.get("name") or "")
        if not title:
            title = meta_content(
                soup,
                'meta[property="og:title"]',
                'meta[name="twitter:title"]',
                'meta[name="title"]',
                "h1",
            )
        if not title and soup.title:
            title = clean_text(soup.title.get_text())

        author_value = data.get("author", "")
        if isinstance(author_value, list):
            author_value = ", ".join(
                clean_text(a.get("name", "")) if isinstance(a, dict) else clean_text(a)
                for a in author_value
            )
        elif isinstance(author_value, dict):
            author_value = author_value.get("name", "")
        author = clean_text(author_value) or meta_content(
            soup,
            'meta[name="author"]',
            'meta[property="article:author"]',
            'meta[name="parsely-author"]',
        )

        date_raw = (
            data.get("datePublished")
            or data.get("dateCreated")
            or data.get("dateModified")
            or meta_content(
                soup,
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]',
                'meta[name="publish-date"]',
                'meta[name="date"]',
                'time[datetime]',
            )
        )
        if isinstance(date_raw, dict):
            date_raw = date_raw.get("@value", "")
        pub_date, pub_time = parse_date(date_raw)

        # If structured data has no date, inspect <time datetime="...">.
        if not pub_date:
            time_tag = soup.find("time", datetime=True)
            if time_tag:
                pub_date, pub_time = parse_date(time_tag.get("datetime"))

        return {
            "title": title,
            "author": author,
            "publication_date": pub_date,
            "article_url": final_url or url,
            "time": pub_time,
            "source": source_name,
        }
    except Exception as exc:
        print(f"[ARTICLE ERROR] {url} -> {exc}")
        return None


def get_selenium_driver():
    """Create Selenium Chrome driver lazily; Selenium Manager handles the driver."""
    global selenium_driver
    if selenium_driver is not None:
        return selenium_driver
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1365,900")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={HEADERS['User-Agent']}")
        selenium_driver = webdriver.Chrome(options=options)
        return selenium_driver
    except Exception as exc:
        print(f"[SELENIUM] Could not start Chrome fallback: {exc}")
        return None


def selenium_fallback(url, source_name):
    """Try browser-rendered HTML only if Requests did not extract a title."""
    driver = get_selenium_driver()
    if driver is None:
        return None
    try:
        driver.get(url)
        sleep_time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        title = meta_content(soup, 'meta[property="og:title"]', "h1")
        if not title and soup.title:
            title = clean_text(soup.title.get_text())
        author = meta_content(
            soup, 'meta[name="author"]', 'meta[property="article:author"]'
        )
        date_raw = meta_content(
            soup,
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            'meta[name="date"]',
            "time[datetime]",
        )
        if not date_raw:
            time_tag = soup.find("time", datetime=True)
            date_raw = time_tag.get("datetime", "") if time_tag else ""
        pub_date, pub_time = parse_date(date_raw)
        return {
            "title": title,
            "author": author,
            "publication_date": pub_date,
            "article_url": normalize_url(driver.current_url) or url,
            "time": pub_time,
            "source": source_name,
        }
    except Exception as exc:
        print(f"[SELENIUM ERROR] {url}: {exc}")
        return None


def is_probably_article(url):
    """Filter out common non-article URLs discovered in sitemaps."""
    path = urlparse(url).path.lower()
    if not path or path in ("/", "/news/", "/blog/"):
        return False
    blocked = (
        "/tag/", "/category/", "/author/", "/search", "/page/",
        "/privacy", "/terms", "/about", "/contact", "/sitemap",
        ".jpg", ".jpeg", ".png", ".webp", ".pdf", ".xml",
    )
    return not any(part in path for part in blocked)


def sort_key(row):
    return (row.get("publication_date", ""), row.get("time", ""))


def main():
    all_candidates = []
    seen_urls = set()

    # Step 1: feed parsing. Feeds are the most reliable source for latest entries.
    for source in SOURCES:
        for row in parse_feed(source["feed"], source["name"]):
            url = normalize_url(row.get("article_url"))
            if url and url not in seen_urls:
                row["article_url"] = url
                all_candidates.append(row)
                seen_urls.add(url)
        sleep_time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nFeed candidates collected: {len(all_candidates)}")

    # Step 2: sitemap discovery. Helps exceed the small number of entries in some feeds.
    for source in SOURCES:
        if len(all_candidates) >= TARGET_ARTICLES * 2:
            break
        sitemap_rows = fetch_sitemap_urls(source["sitemap"], source["domain"])
        for url, lastmod in sitemap_rows:
            url = normalize_url(url)
            if not url or url in seen_urls or not is_probably_article(url):
                continue
            seen_urls.add(url)
            pub_date, pub_time = parse_date(lastmod)
            all_candidates.append({
                "title": "",
                "author": "",
                "publication_date": pub_date,
                "article_url": url,
                "time": pub_time,
                "source": source["name"],
            })
        sleep_time.sleep(REQUEST_DELAY_SECONDS)

    # Prefer known publication dates and sort newest-first.
    all_candidates.sort(key=sort_key, reverse=True)
    all_candidates = all_candidates[:TARGET_ARTICLES * 2]
    print(f"Unique candidate URLs to inspect: {len(all_candidates)}")

    # Step 3: scrape article pages to fill missing fields and verify title/URL.
    final_rows = []
    final_seen = set()
    for index, candidate in enumerate(all_candidates, start=1):
        if len(final_rows) >= TARGET_ARTICLES:
            break

        url = resolve_url(candidate.get("article_url", ""))
        if not url or url in final_seen or not is_probably_article(url):
            continue

        # Use feed/sitemap fields as initial values, then enrich with page metadata.
        row = extract_article_with_requests(url, candidate.get("source", "Unknown"))
        if (not row or not row.get("title")) and USE_SELENIUM_FALLBACK:
            row = selenium_fallback(url, candidate.get("source", "Unknown"))

        if not row:
            row = candidate.copy()
            row["article_url"] = url

        # Keep good values from feed/sitemap if the page does not expose them.
        for field in ("title", "author", "publication_date", "time", "source"):
            if not row.get(field) and candidate.get(field):
                row[field] = candidate[field]

        row["article_url"] = normalize_url(row.get("article_url") or url)
        row["title"] = clean_text(row.get("title"))
        row["author"] = clean_text(row.get("author"))
        row["publication_date"] = clean_text(row.get("publication_date"))
        row["time"] = clean_text(row.get("time"))

        # Reject rows with no usable title or URL to reduce junk/invalid output.
        if row["title"] and row["article_url"] and row["article_url"].startswith(("http://", "https://")):
            if row["article_url"] not in final_seen:
                final_rows.append(row)
                final_seen.add(row["article_url"])
                if len(final_rows) % 25 == 0:
                    print(f"Progress: {len(final_rows)}/{TARGET_ARTICLES} valid articles")
        if index % 10 == 0:
            sleep_time.sleep(REQUEST_DELAY_SECONDS)

    # Save partial results too, so progress is not lost if fewer than 1,000 are available.
    fields = ["title", "author", "publication_date", "article_url", "time", "source"]
    final_rows.sort(key=sort_key, reverse=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    if selenium_driver is not None:
        try:
            selenium_driver.quit()
        except Exception:
            pass

    print("\n========================================")
    print(f"Finished. Valid article rows saved: {len(final_rows)}")
    print(f"CSV file: {OUTPUT_FILE}")
    if len(final_rows) < TARGET_ARTICLES:
        print(
            f"NOTICE: Only {len(final_rows)} valid articles were available from the "
            "feeds/sitemaps/pages accessible in this run. RSS feeds often expose only "
            "their latest entries; some websites also block automated access."
        )
    print("========================================")


if __name__ == "__main__":
    main()
