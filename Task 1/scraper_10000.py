"""
Phases:
1. Discover article URLs from RSS/Atom feeds, robots.txt, sitemaps and homepage links.
2. Static scraping with Requests + BeautifulSoup for every candidate.
3. Dynamic fallback with Selenium for pages whose static extraction did not get a title.
4. Validate, deduplicate, sort newest-first, and save CSV output.


Important:
- Source sites may block requests, publish fewer accessible
  pages, require JavaScript/login, or change their HTML.
- Respect each website's Terms of Service, robots.txt, copyright rules and rate limits.
- Selenium is used only as a fallback; this does not mean every source is a dynamic website.
- The script does not bypass logins, paywalls, CAPTCHAs or access controls.
"""

import csv
import json
import re
import time
import html
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

# -------------------- SETTINGS --------------------
TARGET_ARTICLES = 10_000
OUTPUT_FILE = "articles.csv"
CHECKPOINT_FILE = "articles_checkpoint.csv"
REQUEST_TIMEOUT = 20
REQUEST_DELAY_SECONDS = 0.35
MAX_CANDIDATES_PER_SOURCE = 5_000
MAX_SITEMAPS_PER_SOURCE = 60
MAX_SITEMAP_DEPTH = 3
USE_SELENIUM_FALLBACK = True
SELENIUM_WAIT_SECONDS = 2
SAVE_CHECKPOINT_EVERY = 250

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# 100 technology, AI, science and research sources.
# Feed and sitemap locations are discovered automatically wherever possible.
SOURCE_HOMEPAGES = [
    ("OpenAI", "https://openai.com/"),
    ("Google DeepMind", "https://deepmind.google/"),
    ("Google Research", "https://research.google/"),
    ("Hugging Face", "https://huggingface.co/"),
    ("MIT Technology Review", "https://www.technologyreview.com/"),
    ("TechCrunch", "https://techcrunch.com/"),
    ("The Verge", "https://www.theverge.com/"),
    ("Ars Technica", "https://arstechnica.com/"),
    ("WIRED", "https://www.wired.com/"),
    ("VentureBeat", "https://venturebeat.com/"),
    ("Engadget", "https://www.engadget.com/"),
    ("ZDNET", "https://www.zdnet.com/"),
    ("CNET", "https://www.cnet.com/"),
    ("The Register", "https://www.theregister.com/"),
    ("Tom's Hardware", "https://www.tomshardware.com/"),
    ("TechRadar", "https://www.techradar.com/"),
    ("Android Authority", "https://www.androidauthority.com/"),
    ("9to5Google", "https://9to5google.com/"),
    ("9to5Mac", "https://9to5mac.com/"),
    ("MacRumors", "https://www.macrumors.com/"),
    ("Apple Machine Learning Research", "https://machinelearning.apple.com/"),
    ("Microsoft Research", "https://www.microsoft.com/en-us/research/"),
    ("Meta AI", "https://ai.meta.com/"),
    ("NVIDIA Blog", "https://blogs.nvidia.com/"),
    ("AMD Newsroom", "https://community.amd.com/t5/blogs/ct-p/amd-blogs"),
    ("Intel Newsroom", "https://www.intel.com/content/www/us/en/newsroom/home.html"),
    ("IBM Research", "https://research.ibm.com/blog"),
    ("Amazon Science", "https://www.amazon.science/"),
    ("AWS Machine Learning Blog", "https://aws.amazon.com/blogs/machine-learning/"),
    ("Microsoft Azure Blog", "https://azure.microsoft.com/en-us/blog/"),
    ("Google Cloud Blog", "https://cloud.google.com/blog"),
    ("Cloudflare Blog", "https://blog.cloudflare.com/"),
    ("GitHub Blog", "https://github.blog/"),
    ("GitLab Blog", "https://about.gitlab.com/blog/"),
    ("Stack Overflow Blog", "https://stackoverflow.blog/"),
    ("InfoQ", "https://www.infoq.com/"),
    ("DZone", "https://dzone.com/"),
    ("SitePoint", "https://www.sitepoint.com/"),
    ("Smashing Magazine", "https://www.smashingmagazine.com/"),
    ("CSS-Tricks", "https://css-tricks.com/"),
    ("Web.dev", "https://web.dev/"),
    ("Chrome Developers", "https://developer.chrome.com/"),
    ("Mozilla Hacks", "https://hacks.mozilla.org/"),
    ("Python Insider", "https://blog.python.org/"),
    ("Real Python", "https://realpython.com/"),
    ("PyTorch Blog", "https://pytorch.org/blog/"),
    ("TensorFlow Blog", "https://blog.tensorflow.org/"),
    ("Keras Blog", "https://keras.io/blog/"),
    (" scikit-learn Blog", "https://blog.scikit-learn.org/"),
    ("KDnuggets", "https://www.kdnuggets.com/"),
    ("Towards Data Science", "https://towardsdatascience.com/"),
    ("Analytics Vidhya", "https://www.analyticsvidhya.com/blog/"),
    ("DataCamp Blog", "https://www.datacamp.com/blog"),
    ("Machine Learning Mastery", "https://machinelearningmastery.com/blog/"),
    ("The Gradient", "https://thegradient.pub/"),
    ("Import AI", "https://jack-clark.net/"),
    ("BAIR Blog", "https://bair.berkeley.edu/blog/"),
    ("Berkeley AI Research", "https://bair.berkeley.edu/"),
    ("Stanford HAI", "https://hai.stanford.edu/news"),
    ("Stanford AI Lab", "https://ai.stanford.edu/blog/"),
    ("MIT CSAIL", "https://www.csail.mit.edu/news"),
    ("CMU School of Computer Science", "https://www.cs.cmu.edu/news"),
    ("Allen Institute for AI", "https://allenai.org/blog"),
    ("EleutherAI", "https://www.eleuther.ai/"),
    ("DeepLearning.AI", "https://www.deeplearning.ai/the-batch/"),
    ("AI21 Labs", "https://www.ai21.com/blog/"),
    ("Cohere", "https://cohere.com/blog"),
    ("Anthropic", "https://www.anthropic.com/news"),
    ("Mistral AI", "https://mistral.ai/news/"),
    ("Stability AI", "https://stability.ai/news"),
    ("Runway", "https://runwayml.com/research/"),
    ("Perplexity", "https://www.perplexity.ai/hub"),
    ("Scale AI", "https://scale.com/blog"),
    ("Databricks", "https://www.databricks.com/blog"),
    ("Snowflake", "https://www.snowflake.com/blog/"),
    ("MongoDB", "https://www.mongodb.com/blog"),
    ("PostgreSQL", "https://www.postgresql.org/about/news/"),
    ("Redis Blog", "https://redis.io/blog/"),
    ("Docker Blog", "https://www.docker.com/blog/"),
    ("Kubernetes Blog", "https://kubernetes.io/blog/"),
    ("HashiCorp", "https://www.hashicorp.com/blog"),
    ("Atlassian Blog", "https://www.atlassian.com/blog"),
    ("Notion Blog", "https://www.notion.so/blog"),
    ("Figma Blog", "https://www.figma.com/blog/"),
    ("Canva Newsroom", "https://www.canva.com/newsroom/"),
    ("Product Hunt", "https://www.producthunt.com/"),
    ("IEEE Spectrum", "https://spectrum.ieee.org/"),
    ("Nature News", "https://www.nature.com/news"),
    ("ScienceDaily", "https://www.sciencedaily.com/"),
    ("Phys.org", "https://phys.org/"),
    ("New Scientist", "https://www.newscientist.com/"),
    ("Scientific American", "https://www.scientificamerican.com/"),
    ("Quanta Magazine", "https://www.quantamagazine.org/"),
    ("The Conversation Technology", "https://theconversation.com/technology"),
    ("Reuters Technology", "https://www.reuters.com/technology/"),
    ("AP Technology", "https://apnews.com/hub/technology"),
    ("BBC Technology", "https://www.bbc.com/news/technology"),
    ("The Guardian Technology", "https://www.theguardian.com/technology"),
    ("Financial Times Technology", "https://www.ft.com/technology"),
    ("Forbes Technology", "https://www.forbes.com/technology/"),
    ("Axios AI", "https://www.axios.com/technology"),
    ("Semafor Technology", "https://www.semafor.com/technology"),
    ("Rest of World", "https://restofworld.org/"),
]

FIELDS = ["title", "author", "publication_date", "article_url", "time", "source"]
session = requests.Session()
session.headers.update(HEADERS)
selenium_driver = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    soup = BeautifulSoup(str(value), "html.parser")
    return re.sub(r"\s+", " ", html.unescape(soup.get_text(" ", strip=True))).strip()


def normalize_url(raw_url, base_url=""):
    if not raw_url:
        return ""
    url = html.unescape(str(raw_url).strip()).replace("\\/", "/")
    url = unquote(url)
    if base_url:
        url = urljoin(base_url, url)
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    # Remove fragments and common tracking parameters to reduce duplicates.
    query = parse_qs(parsed.query, keep_blank_values=True)
    query = {k: v for k, v in query.items()
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}}
    from urllib.parse import urlencode
    cleaned_query = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path or "/", "", cleaned_query, ""))


def parse_date(value):
    if not value:
        return "", ""
    value = str(value).strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except Exception:
            try:
                dt = datetime.strptime(value[:10], "%Y-%m-%d")
            except Exception:
                return "", ""
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def get_page(url):
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        logging.debug("Request failed %s: %s", url, exc)
        return None


def discover_feed_urls(homepage, soup):
    feeds = []
    for tag in soup.find_all("link", href=True):
        typ = (tag.get("type") or "").lower()
        rel = tag.get("rel") or []
        rel_text = " ".join(rel).lower() if isinstance(rel, list) else str(rel).lower()
        href = normalize_url(tag["href"], homepage)
        if href and ("alternate" in rel_text) and ("rss" in typ or "atom" in typ or "xml" in typ):
            feeds.append(href)
    # Common conventional feed paths, attempted only if discovered feeds are absent/few.
    if not feeds:
        feeds = [urljoin(homepage, p) for p in ("/feed/", "/feed", "/rss", "/rss.xml", "/feed.xml", "/atom.xml")]
    return list(dict.fromkeys(feeds))[:8]


def discover_sitemap_urls(homepage, soup):
    root = f"{urlparse(homepage).scheme}://{urlparse(homepage).netloc}"
    maps = []
    robots = get_page(urljoin(root, "/robots.txt"))
    if robots:
        for line in robots.text.splitlines():
            if line.lower().startswith("sitemap:"):
                candidate = normalize_url(line.split(":", 1)[1].strip())
                if candidate:
                    maps.append(candidate)
    # Try standard sitemap if robots.txt didn't declare one.
    maps.extend([urljoin(root, "/sitemap.xml"), urljoin(root, "/sitemap_index.xml")])
    return list(dict.fromkeys(maps))[:MAX_SITEMAPS_PER_SOURCE]


def source_domain(homepage):
    return urlparse(homepage).netloc.lower().removeprefix("www.")


def same_source(url, domain):
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return host == domain or host.endswith("." + domain)


def looks_like_article(url):
    parsed = urlparse(url)
    path = parsed.path.lower()
    if not path or path == "/":
        return False
    blocked = (
        "/tag/", "/category/", "/author/", "/search", "/page/", "/privacy",
        "/terms", "/about", "/contact", "/login", "/subscribe", "/newsletter",
        "/sitemap", "/feed", "/wp-json", ".jpg", ".jpeg", ".png", ".gif",
        ".webp", ".svg", ".pdf", ".xml", ".zip", ".mp4", ".mp3",
    )
    return not any(part in path for part in blocked)


def first_tag_text(element, names):
    for name in names:
        tag = element.find(name)
        if tag:
            value = tag.get("href") or tag.get("content") or tag.get_text(" ", strip=True)
            if value:
                return clean_text(value)
    return ""


def parse_feed(feed_url, source_name, domain):
    entries = []
    response = get_page(feed_url)
    if not response:
        return entries
    try:
        soup = BeautifulSoup(response.content, "xml")
        nodes = soup.find_all("item") or soup.find_all("entry")
        for node in nodes:
            title = first_tag_text(node, ["title"])
            link_tag = node.find("link")
            raw_link = ""
            if link_tag:
                raw_link = link_tag.get("href") or link_tag.get_text(strip=True)
            if not raw_link:
                guid = node.find("guid")
                raw_link = guid.get_text(strip=True) if guid else ""
            url = normalize_url(raw_link, feed_url)
            if not title or not url or not same_source(url, domain) or not looks_like_article(url):
                continue
            author = first_tag_text(node, ["author", "creator", "dc:creator", "name"])
            date_raw = first_tag_text(node, ["pubDate", "published", "updated", "dc:date", "date"])
            date, clock = parse_date(date_raw)
            entries.append({
                "title": title, "author": author, "publication_date": date,
                "article_url": url, "time": clock, "source": source_name,
            })
    except Exception as exc:
        logging.debug("Feed parse error %s: %s", feed_url, exc)
    return entries


def fetch_sitemap(sitemap_url, domain, source_name, depth=0, visited=None):
    if visited is None:
        visited = set()
    results = []
    if depth > MAX_SITEMAP_DEPTH or sitemap_url in visited:
        return results
    visited.add(sitemap_url)
    response = get_page(sitemap_url)
    if not response:
        return results
    try:
        soup = BeautifulSoup(response.content, "xml")
        children = [clean_text(x.get_text()) for x in soup.find_all("sitemap") if x.find("loc")]
        if children:
            for child in children[:MAX_SITEMAPS_PER_SOURCE]:
                results.extend(fetch_sitemap(normalize_url(child, sitemap_url), domain, source_name, depth + 1, visited))
                if len(results) >= MAX_CANDIDATES_PER_SOURCE:
                    break
        else:
            for node in soup.find_all("url"):
                loc = node.find("loc")
                if not loc:
                    continue
                url = normalize_url(loc.get_text(strip=True), sitemap_url)
                if not url or not same_source(url, domain) or not looks_like_article(url):
                    continue
                lastmod_tag = node.find("lastmod")
                lastmod = clean_text(lastmod_tag.get_text()) if lastmod_tag else ""
                date, clock = parse_date(lastmod)
                results.append({
                    "title": "", "author": "", "publication_date": date,
                    "article_url": url, "time": clock, "source": source_name,
                })
                if len(results) >= MAX_CANDIDATES_PER_SOURCE:
                    break
    except Exception as exc:
        logging.debug("Sitemap parse error %s: %s", sitemap_url, exc)
    return results


def discover_homepage_links(homepage, soup, source_name, domain):
    rows = []
    for a in soup.find_all("a", href=True):
        url = normalize_url(a["href"], homepage)
        if not url or not same_source(url, domain) or not looks_like_article(url):
            continue
        title = clean_text(a.get_text(" ", strip=True))
        if len(title) < 8:
            continue
        rows.append({
            "title": title, "author": "", "publication_date": "",
            "article_url": url, "time": "", "source": source_name,
        })
    return rows


def extract_jsonld(soup):
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
            value = tag.get("content") or tag.get("datetime") or tag.get_text(" ", strip=True)
            if value:
                return clean_text(value)
    return ""


def extract_static(url, source_name):
    response = get_page(url)
    if not response:
        return None
    try:
        final_url = normalize_url(response.url) or url
        soup = BeautifulSoup(response.text, "html.parser")
        data = extract_jsonld(soup)
        title = clean_text(data.get("headline") or data.get("name") or "")
        if not title:
            title = meta_content(soup, 'meta[property="og:title"]',
                                 'meta[name="twitter:title"]', 'meta[name="title"]', "h1")
        if not title and soup.title:
            title = clean_text(soup.title.get_text())
        author_value = data.get("author", "")
        if isinstance(author_value, list):
            author_value = ", ".join(clean_text(a.get("name", "") if isinstance(a, dict) else a)
                                     for a in author_value)
        elif isinstance(author_value, dict):
            author_value = author_value.get("name", "")
        author = clean_text(author_value) or meta_content(
            soup, 'meta[name="author"]', 'meta[property="article:author"]', 'meta[name="parsely-author"]')
        date_raw = (data.get("datePublished") or data.get("dateCreated") or
                    meta_content(soup, 'meta[property="article:published_time"]',
                                 'meta[name="pubdate"]', 'meta[name="publish-date"]',
                                 'meta[name="date"]', "time[datetime]"))
        if isinstance(date_raw, dict):
            date_raw = date_raw.get("@value", "")
        date, clock = parse_date(date_raw)
        if not date:
            time_tag = soup.find("time", datetime=True)
            if time_tag:
                date, clock = parse_date(time_tag.get("datetime"))
        return {
            "title": title, "author": author, "publication_date": date,
            "article_url": final_url, "time": clock, "source": source_name,
        }
    except Exception as exc:
        logging.debug("Static extraction failed %s: %s", url, exc)
        return None


def get_selenium_driver():
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
        logging.warning("Selenium/Chrome could not start: %s", exc)
        return None


def extract_dynamic(url, source_name):
    driver = get_selenium_driver()
    if driver is None:
        return None
    try:
        driver.get(url)
        time.sleep(SELENIUM_WAIT_SECONDS)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        title = meta_content(soup, 'meta[property="og:title"]', "h1")
        if not title and soup.title:
            title = clean_text(soup.title.get_text())
        author = meta_content(soup, 'meta[name="author"]', 'meta[property="article:author"]')
        date_raw = meta_content(soup, 'meta[property="article:published_time"]',
                                'meta[name="pubdate"]', 'meta[name="date"]', "time[datetime]")
        if not date_raw:
            tag = soup.find("time", datetime=True)
            date_raw = tag.get("datetime", "") if tag else ""
        date, clock = parse_date(date_raw)
        return {
            "title": title, "author": author, "publication_date": date,
            "article_url": normalize_url(driver.current_url) or url,
            "time": clock, "source": source_name,
        }
    except Exception as exc:
        logging.debug("Dynamic extraction failed %s: %s", url, exc)
        return None


def merge_row(row, candidate):
    row = row or {}
    for field in FIELDS:
        if not row.get(field) and candidate.get(field):
            row[field] = candidate[field]
    row["article_url"] = normalize_url(row.get("article_url") or candidate.get("article_url"))
    row["title"] = clean_text(row.get("title"))
    row["author"] = clean_text(row.get("author"))
    row["publication_date"] = clean_text(row.get("publication_date"))
    row["time"] = clean_text(row.get("time"))
    return row


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    # PHASE 1: DISCOVER ARTICLE URLS.
    logging.info("PHASE 1/4: Discovering feeds, sitemaps and article links from %d sources.", len(SOURCE_HOMEPAGES))
    candidates = []
    seen = set()

    def add_rows(rows):
        for row in rows:
            url = normalize_url(row.get("article_url", ""))
            if url and url not in seen:
                row["article_url"] = url
                candidates.append(row)
                seen.add(url)

    for index, (name, homepage) in enumerate(SOURCE_HOMEPAGES, 1):
        domain = source_domain(homepage)
        logging.info("Discovery %d/%d: %s", index, len(SOURCE_HOMEPAGES), name)
        home_response = get_page(homepage)
        home_soup = BeautifulSoup(home_response.text, "html.parser") if home_response else BeautifulSoup("", "html.parser")

        for feed_url in discover_feed_urls(homepage, home_soup):
            add_rows(parse_feed(feed_url, name, domain))
            time.sleep(REQUEST_DELAY_SECONDS)

        for sitemap_url in discover_sitemap_urls(homepage, home_soup):
            sitemap_rows = fetch_sitemap(sitemap_url, domain, name)
            add_rows(sitemap_rows)
            if len(candidates) >= TARGET_ARTICLES * 2:
                break
            time.sleep(REQUEST_DELAY_SECONDS)

        if home_response:
            add_rows(discover_homepage_links(homepage, home_soup, name, domain))

        logging.info("Candidates collected so far: %d", len(candidates))
        if len(candidates) >= TARGET_ARTICLES * 2:
            logging.info("Candidate target buffer reached; stopping discovery.")
            break

    logging.info("PHASE 1 COMPLETE: %d unique candidate URLs discovered.", len(candidates))
    candidates.sort(key=lambda r: (r.get("publication_date", ""), r.get("time", "")), reverse=True)
    candidates = candidates[:max(TARGET_ARTICLES * 2, TARGET_ARTICLES)]

    # PHASE 2: STATIC SCRAPING.
    logging.info("PHASE 2/4: Static scraping with Requests + BeautifulSoup.")
    static_rows = []
    dynamic_queue = []
    for index, candidate in enumerate(candidates, 1):
        url = candidate["article_url"]
        row = extract_static(url, candidate.get("source", "Unknown"))
        row = merge_row(row, candidate)
        if row.get("title") and row.get("article_url"):
            static_rows.append(row)
        else:
            dynamic_queue.append(candidate)
        if index % 100 == 0:
            logging.info("Static phase: inspected %d/%d; valid=%d; dynamic fallback queue=%d",
                         index, len(candidates), len(static_rows), len(dynamic_queue))
        if len(static_rows) >= TARGET_ARTICLES:
            break
        if index % 10 == 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        if len(static_rows) and len(static_rows) % SAVE_CHECKPOINT_EVERY == 0:
            save_csv(CHECKPOINT_FILE, static_rows)

    logging.info("PHASE 2 COMPLETE: %d static-valid rows; %d URLs queued for dynamic fallback.",
                 len(static_rows), len(dynamic_queue))

    # PHASE 3: DYNAMIC BROWSER FALLBACK FOR STATIC FAILURES.
    final_rows = list(static_rows)
    final_urls = {r["article_url"] for r in final_rows if r.get("article_url")}
    if USE_SELENIUM_FALLBACK and len(final_rows) < TARGET_ARTICLES and dynamic_queue:
        logging.info("PHASE 3/4: Selenium fallback for pages that lacked a static title.")
        for index, candidate in enumerate(dynamic_queue, 1):
            if len(final_rows) >= TARGET_ARTICLES:
                break
            url = candidate["article_url"]
            logging.info("Dynamic fallback %d/%d: %s", index, len(dynamic_queue), url)
            row = extract_dynamic(url, candidate.get("source", "Unknown"))
            row = merge_row(row, candidate)
            if row.get("title") and row.get("article_url") and row["article_url"] not in final_urls:
                final_rows.append(row)
                final_urls.add(row["article_url"])
                if len(final_rows) % 100 == 0:
                    logging.info("Dynamic phase: valid total=%d/%d", len(final_rows), TARGET_ARTICLES)
                    save_csv(CHECKPOINT_FILE, final_rows)
            time.sleep(REQUEST_DELAY_SECONDS)
    else:
        logging.info("PHASE 3: Skipped (target reached or no fallback URLs).")

    # PHASE 4: FINAL CLEANUP AND SAVE.
    logging.info("PHASE 4/4: Deduplicating, sorting and saving.")
    unique_rows = []
    final_urls = set()
    for row in final_rows:
        url = normalize_url(row.get("article_url", ""))
        if not row.get("title") or not url or url in final_urls:
            continue
        row["article_url"] = url
        final_urls.add(url)
        unique_rows.append({field: row.get(field, "") for field in FIELDS})

    unique_rows.sort(key=lambda r: (r.get("publication_date", ""), r.get("time", "")), reverse=True)
    save_csv(OUTPUT_FILE, unique_rows)
    save_csv(CHECKPOINT_FILE, unique_rows)

    if selenium_driver is not None:
        try:
            selenium_driver.quit()
        except Exception:
            pass

    logging.info("=" * 55)
    logging.info("DONE. Saved %d valid unique articles.", len(unique_rows))
    logging.info("Output: %s", OUTPUT_FILE)
    if len(unique_rows) < TARGET_ARTICLES:
        logging.warning("Target was %d but only %d valid articles were accessible in this run.",
                        TARGET_ARTICLES, len(unique_rows))
        logging.warning("Check logs for blocked sites, missing feeds/sitemaps or Selenium startup failures.")
    logging.info("=" * 55)


if __name__ == "__main__":
    main()
