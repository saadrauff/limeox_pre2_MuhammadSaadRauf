
import csv
import re
import html
import os
import time
import hashlib
from html.parser import HTMLParser
from datetime import datetime
from urllib.parse import (
    urlsplit, urlunsplit, parse_qsl, urlencode
)

INPUT_FILE = "articles_100000.csv"
OUTPUT_FILE = "final_preprocessed_articles.csv"
REPORT_FILE = "preprocessing_report.txt"

COLUMNS = [
    "title", "author", "publication_date", "year",
    "time", "article_url", "source", "content"
]

TRACKING_PARAMS = {
    "fbclid", "gclid", "mc_cid", "mc_eid"
}

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
]


class TextExtractor(HTMLParser):
    """HTML markup ko readable text mein convert kare."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.hidden_depth += 1
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.hidden_depth = max(0, self.hidden_depth - 1)
        elif tag in ("p", "div", "li", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self.hidden_depth == 0:
            self.parts.append(data)


def clean_text(value):
    if value is None:
        return ""

    value = html.unescape(value)
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)

    return value.strip()


def strip_html(value):
    """HTML tags aur script/style content remove kare."""

    if not value:
        return ""

    parser = TextExtractor()

    try:
        parser.feed(value)
        parser.close()
        value = "".join(parser.parts)
    except (ValueError, AssertionError):
        # Malformed HTML ki surat mein original text preserve karein
        pass

    return clean_text(value)


def normalize_url(url):
    url = clean_text(url)

    if not url:
        return ""

    try:
        parts = urlsplit(url)

        if parts.scheme.lower() not in ("http", "https"):
            return ""

        if not parts.hostname:
            return ""

        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()

        query_items = [
            (key, value)
            for key, value in parse_qsl(
                parts.query, keep_blank_values=True
            )
            if key.lower() not in TRACKING_PARAMS
            and not key.lower().startswith("utm_")
        ]

        query = urlencode(query_items, doseq=True)

        return urlunsplit((
            scheme, netloc, parts.path or "/", query, ""
        ))

    except (ValueError, TypeError):
        return ""


def valid_date(value):
    """Known date formats check kare; original value preserve rahe."""

    value = clean_text(value)

    if not value:
        return False

    # ISO date/time, including timezone and fractional seconds
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        pass

    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue

    return False


def text_hash(value):
    """Case, punctuation aur extra whitespace normalize karke hash."""

    value = value.casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return hashlib.sha256(value.encode("utf-8")).digest()


def main():
    start = time.time()

    if not os.path.isfile(INPUT_FILE):
        print(f"ERROR: Input file nahi mili: {INPUT_FILE}")
        print("Input CSV ko isi folder mein rakhein.")
        return

    counts = {
        "input": 0,
        "saved": 0,
        "duplicate_url": 0,
        "duplicate_text": 0,
        "invalid_url": 0,
        "missing_title": 0,
        "missing_content": 0,
        "missing_author": 0,
        "missing_date": 0,
        "invalid_date": 0,
        "invalid_year": 0,
        "html_rows": 0,
        "extra_fields": 0,
    }

    seen_urls = set()
    seen_records = set()

    try:
        with open(
            INPUT_FILE, "r", encoding="utf-8-sig", newline=""
        ) as source:

            reader = csv.DictReader(source)

            if not reader.fieldnames:
                print("ERROR: CSV header nahi mila.")
                return

            reader.fieldnames = [
                name.strip() if name else ""
                for name in reader.fieldnames
            ]

            missing_columns = [
                col for col in COLUMNS
                if col not in reader.fieldnames
            ]

            if missing_columns:
                print("Missing columns:", missing_columns)
                return

            with open(
                OUTPUT_FILE, "w", encoding="utf-8-sig", newline=""
            ) as destination:

                writer = csv.DictWriter(
                    destination,
                    fieldnames=COLUMNS,
                    extrasaction="ignore"
                )
                writer.writeheader()

                for row in reader:
                    counts["input"] += 1

                    if None in row:
                        counts["extra_fields"] += 1

                    cleaned = {
                        col: clean_text(row.get(col, ""))
                        for col in COLUMNS
                    }

                    original_content = cleaned["content"]
                    cleaned_content = strip_html(original_content)

                    if cleaned_content != original_content:
                        counts["html_rows"] += 1

                    cleaned["title"] = strip_html(cleaned["title"])
                    cleaned["author"] = strip_html(cleaned["author"])
                    cleaned["source"] = strip_html(cleaned["source"])
                    cleaned["content"] = cleaned_content

                    url = normalize_url(cleaned["article_url"])

                    if not url:
                        counts["invalid_url"] += 1
                        continue

                    cleaned["article_url"] = url

                    if not cleaned["title"]:
                        counts["missing_title"] += 1

                    if not cleaned["content"]:
                        counts["missing_content"] += 1

                    if not cleaned["author"]:
                        counts["missing_author"] += 1

                    date_value = cleaned["publication_date"]

                    if not date_value:
                        counts["missing_date"] += 1
                    elif not valid_date(date_value):
                        counts["invalid_date"] += 1

                    year_value = cleaned["year"]

                    if year_value:
                        try:
                            year_number = int(year_value)
                            if not 1000 <= year_number <= 2100:
                                counts["invalid_year"] += 1
                        except ValueError:
                            counts["invalid_year"] += 1

                    # Same normalized URL
                    if url in seen_urls:
                        counts["duplicate_url"] += 1
                        continue

                    # Same title and content after text normalization
                    title_hash = text_hash(cleaned["title"])
                    content_hash = text_hash(cleaned["content"])
                    record_key = (title_hash, content_hash)

                    if record_key in seen_records:
                        counts["duplicate_text"] += 1
                        continue

                    seen_urls.add(url)
                    seen_records.add(record_key)

                    writer.writerow(cleaned)
                    counts["saved"] += 1

                    if counts["input"] % 10000 == 0:
                        print(
                            f"Processed: {counts['input']:,} | "
                            f"Saved: {counts['saved']:,}"
                        )

    except (OSError, csv.Error, UnicodeError) as error:
        print("File processing error:", error)
        return

    elapsed = time.time() - start
    duplicates = counts["duplicate_url"] + counts["duplicate_text"]

    report = f"""
FINAL DATA PREPROCESSING REPORT
================================

Input file: {INPUT_FILE}
Output file: {OUTPUT_FILE}

Input records processed: {counts['input']:,}
Output records saved: {counts['saved']:,}
Duplicate URL records skipped: {counts['duplicate_url']:,}
Duplicate text records skipped: {counts['duplicate_text']:,}
Total duplicate skips: {duplicates:,}
Invalid/missing URLs: {counts['invalid_url']:,}

Records with missing titles: {counts['missing_title']:,}
Records with missing content: {counts['missing_content']:,}
Records with missing authors: {counts['missing_author']:,}
Records with missing publication dates: {counts['missing_date']:,}

Non-empty dates that failed validation: {counts['invalid_date']:,}
Invalid year values: {counts['invalid_year']:,}
Rows where HTML content changed: {counts['html_rows']:,}
Rows with extra CSV fields: {counts['extra_fields']:,}

Elapsed time: {elapsed:.1f} seconds
Elapsed time in minutes: {elapsed / 60:.2f} minutes

NOTES
=====
Missing author/date values are preserved as blank.
Invalid dates are reported, not automatically deleted.
Year/date mismatches are not automatically corrected.
Near-duplicate articles are not detected by this script.
HTML removal is rule-based and should be spot-checked.
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:
        report_file.write(report)

    print(report)
    print("Preprocessing complete.")
    print("Cleaned CSV:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
