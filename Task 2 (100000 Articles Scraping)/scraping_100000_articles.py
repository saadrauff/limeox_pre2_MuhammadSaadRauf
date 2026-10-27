import csv
import html
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# =============== SETTINGS ===============
TARGET = 100_000
OUTPUT = "articles_100000.csv"
CHECKPOINT = "articles_100000_checkpoint.csv"
DISCOVERY = "articles_100000_discovered.csv"

# High-concurrency mode. Respects robots.txt and limits parallel requests per host.
WORKERS = 160
DISCOVERY_WORKERS = 24
PER_HOST_WORKERS = 5
TIMEOUT = (2.5, 6)
CHECKPOINT_EVERY = 500
MAX_URLS_PER_SOURCE = 20000
MAX_SITEMAPS_PER_SOURCE = 60
MAX_CONTENT_CHARS = 20000
IN_FLIGHT_MULTIPLIER = 3

FIELDS = [
    "title", "author", "publication_date", "year",
    "time", "article_url", "source", "content"
]

SOURCES = [
    ("OpenAI", "https://openai.com/"),
    ("Google Research", "https://research.google/"),
    ("DeepMind", "https://deepmind.google/"),
    ("Hugging Face", "https://huggingface.co/blog"),
    ("TechCrunch", "https://techcrunch.com/"),
    ("The Verge", "https://www.theverge.com/"),
    ("Ars Technica", "https://arstechnica.com/"),
    ("Engadget", "https://www.engadget.com/"),
    ("ZDNET", "https://www.zdnet.com/"),
    ("The Register", "https://www.theregister.com/"),
    ("TechRadar", "https://www.techradar.com/"),
    ("Android Authority", "https://www.androidauthority.com/"),
    ("9to5Google", "https://9to5google.com/"),
    ("MacRumors", "https://www.macrumors.com/"),
    ("NVIDIA", "https://blogs.nvidia.com/"),
    ("AWS ML", "https://aws.amazon.com/blogs/machine-learning/"),
    ("Google Cloud", "https://cloud.google.com/blog"),
    ("Cloudflare", "https://blog.cloudflare.com/"),
    ("GitHub", "https://github.blog/"),
    ("GitLab", "https://about.gitlab.com/blog/"),
    ("InfoQ", "https://www.infoq.com/"),
    ("KDnuggets", "https://www.kdnuggets.com/"),
    ("MIT News", "https://news.mit.edu/"),
    ("ScienceDaily", "https://www.sciencedaily.com/"),
    ("Phys.org", "https://phys.org/"),
    ("BBC Technology", "https://www.bbc.com/news/technology"),
    ("The Guardian", "https://www.theguardian.com/technology"),
    ("Wired", "https://www.wired.com/"),
    ("VentureBeat", "https://venturebeat.com/"),
    ("IEEE Spectrum", "https://spectrum.ieee.org/"),
    ("Scientific American", "https://www.scientificamerican.com/"),
    ("New Scientist", "https://www.newscientist.com/"),
    ("NASA", "https://www.nasa.gov/news/"),
    ("Microsoft Research", "https://www.microsoft.com/en-us/research/blog/"),
    ("IBM Research", "https://research.ibm.com/blog"),
    ("PyTorch", "https://pytorch.org/blog/"),
    ("TensorFlow", "https://blog.tensorflow.org/"),
    ("Mozilla", "https://hacks.mozilla.org/"),
    ("Web.dev", "https://web.dev/"),
    ("Chrome Developers", "https://developer.chrome.com/"),
    ("Stack Overflow", "https://stackoverflow.blog/"),
    ("Netflix TechBlog", "https://netflixtechblog.com/"),
    ("Uber Engineering", "https://www.uber.com/blog/engineering/"),
    ("Docker", "https://www.docker.com/blog/"),
    ("Kubernetes", "https://kubernetes.io/blog/"),
    ("BleepingComputer", "https://www.bleepingcomputer.com/"),
    ("SecurityWeek", "https://www.securityweek.com/"),
    ("Digital Trends", "https://www.digitaltrends.com/"),
    ("CNET", "https://www.cnet.com/"),
    ("Mashable", "https://mashable.com/"),
    ("Live Science", "https://www.livescience.com/"),
    # Additional static technology, AI, science and engineering publishers
    ("The Next Web", "https://thenextweb.com/"),
    ("TechSpot", "https://www.techspot.com/"),
    ("Tom's Hardware", "https://www.tomshardware.com/"),
    ("PCMag", "https://www.pcmag.com/"),
    ("PCWorld", "https://www.pcworld.com/"),
    ("Computerworld", "https://www.computerworld.com/"),
    ("InfoWorld", "https://www.infoworld.com/"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/"),
    ("Analytics Vidhya", "https://www.analyticsvidhya.com/blog/"),
    ("Towards Data Science", "https://towardsdatascience.com/"),
    ("The Gradient", "https://thegradient.pub/"),
    ("BAIR Blog", "https://bair.berkeley.edu/blog/"),
    ("Berkeley AI Research", "https://bair.berkeley.edu/"),
    ("Stanford HAI", "https://hai.stanford.edu/news"),
    ("Stanford Engineering", "https://engineering.stanford.edu/news"),
    ("MIT CSAIL", "https://www.csail.mit.edu/news"),
    ("CMU School of Computer Science", "https://www.cs.cmu.edu/news"),
    ("Allen Institute for AI", "https://allenai.org/blog"),
    ("EleutherAI", "https://www.eleuther.ai/"),
    ("HackerNoon", "https://hackernoon.com/"),
    ("DZone", "https://dzone.com/"),
    ("The New Stack", "https://thenewstack.io/"),
    ("SD Times", "https://www.sdtimes.com/"),
    ("InfoSecurity Magazine", "https://www.infosecurity-magazine.com/"),
    ("Dark Reading", "https://www.darkreading.com/"),
    ("Krebs on Security", "https://krebsonsecurity.com/"),
    ("The Hacker News", "https://thehackernews.com/"),
    ("Security Boulevard", "https://securityboulevard.com/"),
    ("SANS Internet Storm Center", "https://isc.sans.edu/"),
    ("Packet Pushers", "https://packetpushers.net/"),
    ("LWN.net", "https://lwn.net/"),
    ("Phoronix", "https://www.phoronix.com/"),
    ("GamingOnLinux", "https://www.gamingonlinux.com/"),
    ("Linux.com", "https://www.linux.com/"),
    ("OMG Ubuntu", "https://www.omgubuntu.co.uk/"),
    ("It's FOSS", "https://itsfoss.com/"),
    ("How-To Geek", "https://www.howtogeek.com/"),
    ("MakeUseOf", "https://www.makeuseof.com/"),
    ("DigitalOcean Community", "https://www.digitalocean.com/community/tutorials"),
    ("HashiCorp Blog", "https://www.hashicorp.com/blog"),
    ("Stripe Blog", "https://stripe.com/blog"),
    ("Slack Engineering", "https://slack.engineering/"),
    ("LinkedIn Engineering", "https://engineering.linkedin.com/blog"),
    ("Meta Engineering", "https://engineering.fb.com/"),
    ("Pinterest Engineering", "https://medium.com/pinterest-engineering"),
    ("Dropbox Tech Blog", "https://dropbox.tech/"),
    ("Cloudflare Learning", "https://www.cloudflare.com/learning/"),
    ("Shopify Engineering", "https://shopify.engineering/"),
    ("Spotify Engineering", "https://engineering.atspotify.com/"),
    ("Slack Blog", "https://slack.com/blog"),
    ("Atlassian Engineering", "https://www.atlassian.com/blog/engineering"),
    ("Figma Blog", "https://www.figma.com/blog/"),
    ("Canva Engineering", "https://www.canva.dev/blog/engineering/"),
    ("Vercel Blog", "https://vercel.com/blog"),
    ("Netlify Blog", "https://www.netlify.com/blog/"),
    ("Sentry Blog", "https://blog.sentry.io/"),
    ("JetBrains Blog", "https://blog.jetbrains.com/"),
    ("Kotlin Blog", "https://blog.jetbrains.com/kotlin/"),
    ("Android Developers Blog", "https://android-developers.googleblog.com/"),
    ("Apple Machine Learning Research", "https://machinelearning.apple.com/"),
    ("Apple Newsroom", "https://www.apple.com/newsroom/"),
    ("Microsoft Azure Blog", "https://azure.microsoft.com/en-us/blog/"),
    ("Microsoft DevBlogs", "https://devblogs.microsoft.com/"),
    ("Oracle Blogs", "https://blogs.oracle.com/"),
    ("Red Hat Blog", "https://www.redhat.com/en/blog"),
    ("Canonical Blog", "https://canonical.com/blog"),
    ("AMD Newsroom", "https://www.amd.com/en/newsroom.html"),
    ("Intel Newsroom", "https://newsroom.intel.com/"),
    ("Qualcomm News", "https://www.qualcomm.com/news"),
    ("Samsung Newsroom", "https://news.samsung.com/global/technology"),
    ("ESA News", "https://www.esa.int/Newsroom"),
    ("European Commission Digital Strategy", "https://digital-strategy.ec.europa.eu/en/news"),
    ("Nature News", "https://www.nature.com/news"),
    ("Nature Machine Intelligence", "https://www.nature.com/natmachintell/"),
    ("Science News", "https://www.sciencenews.org/"),
    ("EurekAlert", "https://www.eurekalert.org/"),
    ("Medical Xpress", "https://medicalxpress.com/"),
    ("Technology Networks", "https://www.technologynetworks.com/"),
    ("The Conversation Technology", "https://theconversation.com/us/technology"),
    ("Quanta Magazine", "https://www.quantamagazine.org/"),
    ("Scientific Computing", "https://www.scientificcomputing.com/"),
    ("IEEE News", "https://spectrum.ieee.org/technology"),
    ("ACM TechNews", "https://technews.acm.org/"),
    ("ACM Communications", "https://cacm.acm.org/"),
    ("The Mozilla Blog", "https://blog.mozilla.org/"),
    ("Python Insider", "https://blog.python.org/"),
    ("Real Python", "https://realpython.com/"),
    ("Python Software Foundation", "https://pyfound.blogspot.com/"),
    ("RStudio Blog", "https://posit.co/blog/"),
    ("Julia Blog", "https://julialang.org/blog/"),
    ("Rust Blog", "https://blog.rust-lang.org/"),
    ("Go Blog", "https://go.dev/blog/"),
    ("Node.js Blog", "https://nodejs.org/en/blog"),
    ("PHP.Watch", "https://php.watch/"),
    ("Laravel News", "https://laravel-news.com/"),
    ("WordPress News", "https://wordpress.org/news/"),
    ("Drupal News", "https://www.drupal.org/news"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ArticleResearchBot/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

local = threading.local()
host_guard = threading.Lock()
host_semaphores = {}
robots_cache = {}
robots_guard = threading.Lock()


def session():
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.headers.update(HEADERS)
    return local.session


def clean(value):
    if value is None:
        return ""
    return re.sub(
        r"\s+", " ",
        BeautifulSoup(html.unescape(str(value)), "html.parser")
        .get_text(" ", strip=True)
    ).strip()


def normalize(url, base=""):
    if not url:
        return ""
    url = html.unescape(str(url).strip()).replace("\\/", "/")
    url = urljoin(base, url) if base else url
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        return ""
    query = [
        (k, v) for k, v in parse_qsl(p.query)
        if not k.lower().startswith("utm_")
        and k.lower() not in ("fbclid", "gclid")
    ]
    return urlunparse((
        p.scheme.lower(), p.netloc.lower(), p.path or "/",
        "", urlencode(query), ""
    ))


def domain(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def same_domain(url, host):
    d = domain(url)
    return d == host or d.endswith("." + host)


def fetch(url):
    """Requests with timeouts and a per-host concurrency limit."""
    url = normalize(url)
    if not url:
        return None

    host = urlparse(url).netloc.lower()
    with host_guard:
        semaphore = host_semaphores.setdefault(
            host, threading.Semaphore(PER_HOST_WORKERS)
        )

    with semaphore:
        try:
            response = session().get(
                url, timeout=TIMEOUT, allow_redirects=True
            )
            if response.status_code in (403, 404, 410, 429):
                response.close()
                return None
            response.raise_for_status()
            if len(response.content) > 10_000_000:
                response.close()
                return None
            return response
        except requests.RequestException:
            return None


def allowed(url):
    """Cache robots.txt results per host and respect the site's crawl rules."""
    p = urlparse(url)
    root = f"{p.scheme}://{p.netloc}"

    with robots_guard:
        parser = robots_cache.get(root)
        if parser is None:
            # A temporary placeholder prevents every worker fetching robots.txt.
            parser = RobotFileParser()
            parser.set_url(root + "/robots.txt")
            robots_cache[root] = parser
            should_load = True
        else:
            should_load = False

    if should_load:
        response = fetch(root + "/robots.txt")
        try:
            parser.parse(response.text.splitlines() if response else [])
        finally:
            if response:
                response.close()

    try:
        return parser.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return False


def parse_date(value):
    if not value:
        return "", ""
    value = str(value).strip()
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(value)
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
        except Exception:
            return "", ""


def is_article(url):
    p = urlparse(url)
    path = p.path.lower()
    if not path or path == "/":
        return False

    blocked = (
        "/tag/", "/category/", "/author/", "/search", "/page/",
        "/privacy", "/terms", "/contact", "/login", "/subscribe",
        "/sitemap", "/feed", "/wp-json", ".jpg", ".jpeg", ".png",
        ".gif", ".webp", ".svg", ".pdf", ".xml", ".zip", ".mp4",
        ".mp3", ".css", ".js"
    )
    return not any(x in path for x in blocked)


def blank_row(url, source):
    return {
        "title": "", "author": "", "publication_date": "",
        "year": "", "time": "", "article_url": url,
        "source": source, "content": ""
    }


def discover_feed(feed_url, source, host):
    response = fetch(feed_url)
    if not response:
        return []

    results = []
    try:
        root = ET.fromstring(response.content)
        for item in root.iter():
            if item.tag.rsplit("}", 1)[-1].lower() not in ("item", "entry"):
                continue

            fields = {}
            for child in item:
                key = child.tag.rsplit("}", 1)[-1].lower()
                value = child.attrib.get("href", "") if key == "link" else ""
                value = value or "".join(child.itertext()).strip()
                fields.setdefault(key, value)

            url = normalize(fields.get("link") or fields.get("guid"), feed_url)
            title = clean(fields.get("title"))
            if not url or not title or not same_domain(url, host):
                continue
            if not is_article(url):
                continue

            date, clock = parse_date(
                fields.get("pubdate") or fields.get("published")
                or fields.get("updated") or fields.get("date")
            )
            row = blank_row(url, source)
            row.update({
                "title": title,
                "author": clean(fields.get("creator") or fields.get("author")),
                "publication_date": date,
                "year": date[:4],
                "time": clock,
            })
            results.append(row)
    except Exception:
        pass
    finally:
        response.close()
    return results


def discover_sitemap(url, source, host, visited, depth=0):
    if depth > 3 or len(visited) >= MAX_SITEMAPS_PER_SOURCE:
        return []
    url = normalize(url)
    if not url or url in visited or not same_domain(url, host):
        return []

    visited.add(url)
    response = fetch(url)
    if not response:
        return []

    results, children = [], []
    try:
        root = ET.fromstring(response.content)

        for node in root:
            tag = node.tag.rsplit("}", 1)[-1].lower()
            loc = next(
                (x.text.strip() for x in node
                 if x.tag.rsplit("}", 1)[-1].lower() == "loc" and x.text),
                ""
            )
            if not loc:
                continue
            loc = normalize(loc, url)

            if tag == "sitemap":
                children.append(loc)
            elif tag == "url" and is_article(loc) and same_domain(loc, host):
                row = blank_row(loc, source)
                lastmod = next(
                    (x.text for x in node
                     if x.tag.rsplit("}", 1)[-1].lower() == "lastmod"),
                    ""
                )
                date, clock = parse_date(lastmod)
                row.update({
                    "publication_date": date,
                    "year": date[:4],
                    "time": clock,
                })
                results.append(row)

            if len(results) >= MAX_URLS_PER_SOURCE:
                break

        for child in children[:20]:
            if len(results) >= MAX_URLS_PER_SOURCE:
                break
            results.extend(
                discover_sitemap(child, source, host, visited, depth + 1)
            )
    except Exception:
        pass
    finally:
        response.close()

    return results[:MAX_URLS_PER_SOURCE]


def discover_source(source, homepage):
    host = domain(homepage)
    results = []
    response = fetch(homepage)
    soup = BeautifulSoup(response.text, "html.parser") if response else BeautifulSoup("", "html.parser")

    feeds = []
    for tag in soup.find_all("link", href=True):
        rel = " ".join(tag.get("rel", [])).lower()
        typ = (tag.get("type") or "").lower()
        if "alternate" in rel and any(x in typ for x in ("rss", "atom", "xml")):
            feeds.append(normalize(tag["href"], homepage))

    if not feeds:
        feeds = [
            urljoin(homepage, p)
            for p in ("/feed/", "/feed", "/rss", "/rss.xml", "/atom.xml", "/feed.xml")
        ]

    for feed in dict.fromkeys(feeds):
        results.extend(discover_feed(feed, source, host))

    root = f"{urlparse(homepage).scheme}://{urlparse(homepage).netloc}"
    robots = fetch(root + "/robots.txt")
    maps = []
    if robots:
        for line in robots.text.splitlines():
            if line.lower().startswith("sitemap:"):
                u = normalize(line.split(":", 1)[1].strip())
                if u and same_domain(u, host):
                    maps.append(u)
        robots.close()

    if not maps:
        maps = [root + "/sitemap.xml", root + "/sitemap_index.xml"]

    visited = set()
    for sm in dict.fromkeys(maps):
        if len(results) >= MAX_URLS_PER_SOURCE:
            break
        results.extend(discover_sitemap(sm, source, host, visited))

    if response:
        for a in soup.find_all("a", href=True):
            u = normalize(a["href"], homepage)
            title = clean(a.get_text(" ", strip=True))
            if u and title and len(title) >= 8 and same_domain(u, host) and is_article(u):
                row = blank_row(u, source)
                row["title"] = title
                results.append(row)
        response.close()

    unique = {}
    for row in results:
        u = row["article_url"]
        if u and u not in unique:
            unique[u] = row
        if len(unique) >= MAX_URLS_PER_SOURCE:
            break

    logging.info("%s: %s URLs", source, len(unique))
    return list(unique.values())


def get_meta(soup, *selectors):
    for selector in selectors:
        tag = soup.select_one(selector)
        if tag:
            value = tag.get("content") or tag.get("datetime") or tag.get_text(" ", strip=True)
            if value:
                return clean(value)
    return ""


def extract_content(soup):
    for tag in soup.find_all([
        "script", "style", "noscript", "svg", "nav", "header",
        "footer", "aside", "form", "button", "iframe"
    ]):
        tag.decompose()

    article = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = []
    for tag in article.find_all(["p", "h2", "h3", "blockquote", "li"]):
        text = clean(tag.get_text(" ", strip=True))
        if len(text) >= 35 and text not in paragraphs:
            paragraphs.append(text)

    content = "\n".join(paragraphs)
    if len(content) < 200:
        content = clean(article.get_text(" ", strip=True))
    return content[:MAX_CONTENT_CHARS]


def scrape(row):
    url = normalize(row.get("article_url"))
    if not url or not allowed(url):
        return None

    response = fetch(url)
    if not response:
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        final_url = normalize(response.url) or url

        metadata = {}
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or script.get_text())
                items = data if isinstance(data, list) else [data]
                index = 0
                while index < len(items):
                    item = items[index]
                    index += 1
                    if not isinstance(item, dict):
                        continue
                    if isinstance(item.get("@graph"), list):
                        items.extend(item["@graph"])
                    typ = item.get("@type", "")
                    types = typ if isinstance(typ, list) else [typ]
                    if any("article" in str(t).lower() for t in types):
                        metadata = item
                        break
                if metadata:
                    break
            except Exception:
                continue

        title = clean(metadata.get("headline") or metadata.get("name"))
        title = title or get_meta(
            soup, 'meta[property="og:title"]',
            'meta[name="twitter:title"]', "h1"
        )
        title = title or clean(soup.title.get_text() if soup.title else "")
        title = title or row.get("title", "")

        author = metadata.get("author", "")
        if isinstance(author, dict):
            author = author.get("name", "")
        elif isinstance(author, list):
            author = ", ".join(
                str(x.get("name", "")) if isinstance(x, dict) else str(x)
                for x in author
            )
        author = clean(author) or get_meta(
            soup, 'meta[name="author"]',
            'meta[property="article:author"]',
            'meta[name="parsely-author"]'
        )

        date_raw = (
            metadata.get("datePublished")
            or get_meta(
                soup, 'meta[property="article:published_time"]',
                'meta[name="pubdate"]', 'meta[name="date"]',
                "time[datetime]"
            )
            or row.get("publication_date", "")
        )
        date, clock = parse_date(date_raw)
        if not date:
            time_tag = soup.find("time", datetime=True)
            if time_tag:
                date, clock = parse_date(time_tag.get("datetime"))

        content = extract_content(soup)
        if not title or len(content) < 80:
            return None

        return {
            "title": title, "author": author,
            "publication_date": date, "year": date[:4] if date else "",
            "time": clock, "article_url": final_url,
            "source": row.get("source", ""), "content": content,
        }
    except Exception:
        return None
    finally:
        response.close()


def load_csv_rows(path, require_title=True):
    rows = {}
    if not os.path.exists(path):
        return rows
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                url = normalize(row.get("article_url"))
                if url and (row.get("title") or not require_title):
                    row["article_url"] = url
                    rows[url] = {k: row.get(k, "") for k in FIELDS}
    except Exception as exc:
        logging.warning("Could not read %s: %s", path, exc)
    return rows


def load_candidates():
    """Keep old discovered URLs and merge fresh URLs from expanded static sources."""
    candidates = load_csv_rows(DISCOVERY, require_title=False)
    if candidates:
        logging.info("Loaded %s existing discovered URLs from %s", len(candidates), DISCOVERY)

    logging.info("Phase 1: Discovering/merging URLs from %s static sources", len(SOURCES))
    with ThreadPoolExecutor(max_workers=DISCOVERY_WORKERS) as pool:
        futures = {
            pool.submit(discover_source, name, home): name
            for name, home in SOURCES
        }
        for future in as_completed(futures):
            try:
                for row in future.result():
                    url = normalize(row.get("article_url"))
                    if url and url not in candidates:
                        candidates[url] = row
            except Exception as exc:
                logging.warning("%s discovery error: %s",
                                futures[future], type(exc).__name__)

    save_csv(DISCOVERY, list(candidates.values()))
    logging.info("Discovered %s unique candidate URLs", len(candidates))
    return candidates


def save_csv(path, rows):
    temp = path + ".writing"
    with open(temp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def save_progress(saved):
    rows = list(saved.values())
    save_csv(CHECKPOINT, rows)
    save_csv(OUTPUT, rows)
    logging.info("Checkpoint saved: %s articles", len(rows))


def main():
    # Merge OUTPUT and CHECKPOINT so a more recent checkpoint is not ignored.
    saved = load_csv_rows(OUTPUT)
    checkpoint_rows = load_csv_rows(CHECKPOINT)
    for url, row in checkpoint_rows.items():
        if url not in saved or len(row.get("content", "")) > len(saved[url].get("content", "")):
            saved[url] = row

    logging.info("Existing saved articles loaded: %s", len(saved))
    if len(saved) >= TARGET:
        logging.info("Target already reached.")
        return

    candidates = load_candidates()
    queue = [row for url, row in candidates.items() if url not in saved]
    logging.info(
        "Phase 2: %s URLs to check; %s workers; target %s",
        len(queue), WORKERS, TARGET
    )

    completed = 0
    successful_since_save = 0
    last_log = time.monotonic()
    iterator = iter(queue)
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    pending = set()
    max_pending = WORKERS * IN_FLIGHT_MULTIPLIER

    def fill_pending():
        while len(pending) < max_pending:
            try:
                row = next(iterator)
            except StopIteration:
                break
            pending.add(pool.submit(scrape, row))

    try:
        fill_pending()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed += 1
                try:
                    result = future.result()
                except Exception:
                    result = None

                if result:
                    url = normalize(result.get("article_url"))
                    if url and url not in saved:
                        saved[url] = result
                        successful_since_save += 1

                if len(saved) >= TARGET:
                    break

                if successful_since_save >= CHECKPOINT_EVERY:
                    save_progress(saved)
                    successful_since_save = 0

                if completed % 500 == 0 or time.monotonic() - last_log >= 30:
                    logging.info(
                        "Checked %s/%s | Valid saved: %s/%s | In progress: %s",
                        completed, len(queue), len(saved), TARGET, len(pending)
                    )
                    last_log = time.monotonic()

            if len(saved) >= TARGET:
                for future in pending:
                    future.cancel()
                break
            fill_pending()

    except KeyboardInterrupt:
        logging.warning("Stop requested. Saving current progress before exiting...")
        save_progress(saved)
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        # On normal completion, wait for workers. On Ctrl+C the exception path
        # above already requested a non-waiting shutdown.
        if len(saved) >= TARGET or not pending:
            pool.shutdown(wait=True)

    save_progress(saved)
    logging.info("Finished: %s articles saved in %s", len(saved), OUTPUT)
    if len(saved) < TARGET:
        logging.warning(
            "Target not reached yet. Restart this script to continue; saved CSVs "
            "and discovered URLs will be reused."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Stopped. Checkpoint/output files were saved.")
