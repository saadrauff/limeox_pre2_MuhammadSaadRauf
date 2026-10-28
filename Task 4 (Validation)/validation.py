from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
import json
import re
import time

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# LimeoX Task 4: Data Validation & Quality Assessment
# Input remains unchanged; all reports go into validation_output.
# Policy: short title/content are INFO heuristics; detected block/error pages are ERROR.
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    BASE_DIR
    / "repair_output"
    / "repaired_articles.csv"
)
OUTPUT_DIR = BASE_DIR / "validation_output"

CONTRACT = {
    "required_columns": ["news_url_clean", "title_clean", "content_clean"],
    "optional_columns": [
        "published_at_utc", "author_clean", "domain", "scraped_at_utc"
    ],
    "min_title_chars_warning": 8,
    "min_content_chars_warning": 200,
    "known_error_patterns": [
        r"access denied", r"enable javascript", r"captcha",
        r"robot check", r"403 forbidden", r"page not found"
    ],
}

ISSUE_COLUMNS = [
    "rule_id", "severity", "scope", "row_index", "field", "message"
]


def is_missing(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def is_valid_http_url(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlparse(value.strip())
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and "." in parsed.hostname
        )
    except (ValueError, AttributeError):
        return False


def add_issue(issues, rule_id, severity, row_index, field, message, scope="row"):
    issues.append({
        "rule_id": rule_id,
        "severity": severity,
        "scope": scope,
        "row_index": row_index,
        "field": field,
        "message": message,
    })


def validate_schema(frame):
    issues = []

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        add_issue(
            issues, "SCHEMA_001", "ERROR", None, None,
            f"Duplicate column names: {duplicates}", scope="dataset"
        )

    for col in CONTRACT["required_columns"]:
        if col not in frame.columns:
            add_issue(
                issues, "SCHEMA_002", "ERROR", None, col,
                f"Missing required column: {col}", scope="dataset"
            )

    # Optional columns do not generate validation issues.
    return issues



def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV nahi mili:\n{INPUT_PATH}\n"
            "Check karein ke Task 3 (Preprocessing) folder aur CSV filename sahi hain."
        )

    # Read the previous task's final preprocessed dataset.
    original = pd.read_csv(INPUT_PATH, low_memory=False)
    print("Input:", INPUT_PATH)
    print("Original shape:", original.shape)
    print("Original columns:", original.columns.tolist())

    # Map the user's preprocessing column names to lecture notebook names.
    rename_map = {
        "article_url": "news_url_clean",
        "title": "title_clean",
        "content": "content_clean",
        "publication_date": "published_at_utc",
        "author": "author_clean",
    }
    data = original.rename(columns=rename_map).copy()

    # Derive domain from URL only when URL column exists and domain is absent.
    if "news_url_clean" in data.columns and "domain" not in data.columns:
        def get_domain(value):
            if is_missing(value):
                return None
            try:
                return urlparse(str(value).strip()).hostname
            except ValueError:
                return None
        data["domain"] = data["news_url_clean"].map(get_domain)

    print("Mapped columns:", data.columns.tolist())
    print("Mapped shape:", data.shape)

    schema_issues = validate_schema(data)
    schema_errors = [x for x in schema_issues if x["severity"] == "ERROR"]
    if schema_errors:
        pd.DataFrame(schema_issues, columns=ISSUE_COLUMNS).to_csv(
            OUTPUT_DIR / "news_validation_issues.csv",
            index=False, encoding="utf-8-sig"
        )
        raise ValueError(
            "Required schema validation fail hui. "
            "news_validation_issues.csv mein details check karein. "
            f"Missing/duplicate schema issue(s): {schema_errors}"
        )

    df = data.copy(deep=True).reset_index(drop=False)
    df = df.rename(columns={"index": "source_row_index"})
    issues = list(schema_issues)

    # 1. Required fields and types
    for _, row in df.iterrows():
        for field in CONTRACT["required_columns"]:
            if is_missing(row.get(field, pd.NA)):
                add_issue(
                    issues, "FIELD_001", "ERROR", int(row["source_row_index"]),
                    field, f"Required field '{field}' is missing or blank."
                )

        for field in ["title_clean", "content_clean", "author_clean"]:
            if field in df.columns:
                value = row[field]
                if not is_missing(value) and not isinstance(value, str):
                    add_issue(
                        issues, "TYPE_001", "WARNING",
                        int(row["source_row_index"]), field,
                        f"Expected text value, found {type(value).__name__}."
                    )

    # 2. URL and domain validation
    if "news_url_clean" in df.columns:
        for _, row in df.iterrows():
            url = row.get("news_url_clean", pd.NA)
            if not is_missing(url) and not is_valid_http_url(url):
                add_issue(
                    issues, "URL_001", "ERROR", int(row["source_row_index"]),
                    "news_url_clean",
                    "URL is not a plausible HTTP(S) URL with a hostname."
                )
            elif not is_missing(url):
                hostname = urlparse(str(url).strip()).hostname.lower()
                domain = row.get("domain", pd.NA)
                if not is_missing(domain) and str(domain).strip().lower() != hostname:
                    add_issue(
                        issues, "URL_002", "WARNING",
                        int(row["source_row_index"]), "domain",
                        f"Stored domain '{domain}' does not match URL hostname '{hostname}'."
                    )

    # 3. Publication timestamp validation. Scrape timestamps are absent in
    # the current preprocessing dataset, so we do not fabricate them.
    for col in ["published_at_utc", "scraped_at_utc"]:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        df[col + "_parsed"] = parsed
        for pos, (_, row) in enumerate(df.iterrows()):
            value = row[col]
            if is_missing(value):
                if col == "scraped_at_utc":
                    add_issue(
                        issues, "TIME_001", "INFO",
                        int(row["source_row_index"]), col,
                        "Scrape timestamp missing; run-level traceability is reduced."
                    )
            elif pd.isna(parsed.iloc[pos]):
                add_issue(
                    issues, "TIME_002", "WARNING",
                    int(row["source_row_index"]), col,
                    "Timestamp is present but cannot be parsed."
                )

    if "published_at_utc_parsed" in df.columns and "scraped_at_utc_parsed" in df.columns:
        for _, row in df.iterrows():
            pub, scrape = row["published_at_utc_parsed"], row["scraped_at_utc_parsed"]
            if pd.notna(pub) and pd.notna(scrape) and pub > scrape:
                add_issue(
                    issues, "TIME_003", "WARNING",
                    int(row["source_row_index"]), "published_at_utc",
                    "Publication time is later than scrape time; verify timezone/source metadata."
                )

    
    # 4. Content quality and extraction-failure checks
    error_regex = re.compile(
        "|".join(CONTRACT["known_error_patterns"]), flags=re.I
    )

    for _, row in df.iterrows():
        content = row.get("content_clean", pd.NA)

        # Missing or non-string content is handled by required-field/type checks.
        if not isinstance(content, str) or not content.strip():
            continue

        normalized_content = re.sub(r"\s+", " ", content).strip()
        lower_content = normalized_content.lower()

        # Strong signals of a genuine block/error page.
        strong_error_patterns = (
            r"^access denied\.?$",
            r"^403 forbidden\.?$",
            r"^404 not found\.?$",
            r"^verify you are human\.?$",
            r"^checking your browser before accessing",
            r"^please complete the security check",
            r"^this site is protected by hcaptcha",
        )

      
        is_docker_signup_page = (
            lower_content.startswith("this site is protected by hcaptcha")
            and str(row.get("title_clean", "")).strip().lower()
            in {"create your account", "sign up", "create account"}
        )

        is_short_error_page = (
            not is_docker_signup_page
            and len(normalized_content) < 500
            and any(
                re.search(pattern, lower_content, re.I)
                for pattern in strong_error_patterns
            )
        )

        if is_short_error_page:
            add_issue(
                issues,
                "TEXT_003",
                "ERROR",
                int(row["source_row_index"]),
                "content_clean",
                "Content may be a block, CAPTCHA, or access-error response.",
            )



    # 5. Duplicate candidates. Flag only; do not delete records.
    if "news_url_clean" in df.columns:
        valid_url_mask = df["news_url_clean"].map(is_valid_http_url)
        repeated_url_mask = valid_url_mask & df["news_url_clean"].duplicated(keep=False)
        for _, row in df[repeated_url_mask].iterrows():
            add_issue(
                issues, "DUP_001", "WARNING", int(row["source_row_index"]),
                "news_url_clean",
                "URL occurs more than once; inspect duplicate versus article update."
            )

    if {"title_clean", "content_clean"}.issubset(df.columns):
        title_norm = df["title_clean"].fillna("").astype(str).str.casefold().str.strip()
        content_norm = df["content_clean"].fillna("").astype(str).str.casefold().str.strip()
        signatures = title_norm + "||" + content_norm
        mask = signatures.duplicated(keep=False) & ((title_norm != "") | (content_norm != ""))
        df["duplicate_signature_candidate"] = mask

    issues_df = pd.DataFrame(issues, columns=ISSUE_COLUMNS)

    # 6. Dataset-level metrics
    if issues_df.empty:
        by_severity, by_rule, affected_rows = {}, {}, 0
    else:
        by_severity = {
            str(k): int(v)
            for k, v in issues_df.groupby("severity").size().items()
        }
        by_rule = {
            str(k): int(v)
            for k, v in issues_df.groupby("rule_id").size().items()
        }
        affected_rows = int(
            issues_df.loc[issues_df["row_index"].notna(), "row_index"].nunique()
        )

    completeness = pd.DataFrame({
        "column": df.columns,
        "non_missing": [int(df[c].notna().sum()) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "completeness_pct": [
            round(df[c].notna().mean() * 100, 2) if len(df) else np.nan
            for c in df.columns
        ],
    })

    # 7. Classify rows. Fatal row rules go to quarantine.
    fatal_rules = {"FIELD_001", "URL_001", "TEXT_003"}
    row_issue_map = {}
    for _, issue in issues_df.iterrows():
        if pd.notna(issue["row_index"]):
            row_issue_map.setdefault(int(issue["row_index"]), []).append(issue["rule_id"])

    df["validation_issue_count"] = df["source_row_index"].map(
        lambda x: len(row_issue_map.get(int(x), []))
    )
    df["validation_issue_codes"] = df["source_row_index"].map(
        lambda x: ";".join(row_issue_map.get(int(x), []))
    )

    def disposition(row_id):
        rules = row_issue_map.get(int(row_id), [])
        if any(rule in fatal_rules for rule in rules):
            return "QUARANTINE"
        if rules:
            return "WARNING"
        return "ACCEPT"

    df["validation_disposition"] = df["source_row_index"].map(disposition)

    excluded_mask = (
        df["content_clean"].fillna("").astype(str).str.strip()
        .str.match(r"(?i)^this site is protected by hcaptcha")
        & df["title_clean"].fillna("").astype(str).str.strip().str.lower()
          .isin(["create your account", "sign up", "create account"])
    )
   
    # Keep all records that pass validation rules.
    accepted_df = df[df["validation_disposition"] == "ACCEPT"].copy()
    warning_df = df[df["validation_disposition"] == "WARNING"].copy()
    quarantine_df = df[df["validation_disposition"] == "QUARANTINE"].copy()

    # No records are excluded under the current policy.
    excluded_df = df.iloc[0:0].copy()


    summary = {
        "input_file": str(INPUT_PATH),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(df)),
        "input_columns_after_mapping": int(len(df.columns)),
        "total_issues": int(len(issues_df)),
        "affected_rows": affected_rows,
        "issues_by_severity": by_severity,
        "issues_by_rule": by_rule,
        "dispositions": {
            "accept": int(len(accepted_df)),
            "warning": int(len(warning_df)),
            "quarantine": int(len(quarantine_df)),
            "excluded_non_article": int(len(excluded_df)),
        },
        "notes": [
            "The source CSV was not modified.",
            "A syntactically valid URL is not proof that the page is reachable.",
            "Duplicate candidates are flagged, not automatically deleted.",
            "No scraped timestamp was fabricated if the source did not contain one."
        ],
    }

    # 8. Unit tests for core rules
    assert is_valid_http_url("https://example.com/story/1")
    assert is_valid_http_url("http://news.example.org/")
    assert not is_valid_http_url("not-a-url")
    assert not is_valid_http_url("ftp://example.com/file")
    assert not is_valid_http_url(None)
    assert is_missing(None)
    assert is_missing(np.nan)
    assert is_missing("   ")
    assert not is_missing("Article title")
    assert pd.notna(pd.to_datetime("2026-10-27T09:30:00Z", errors="coerce", utc=True))
    assert pd.isna(pd.to_datetime("yesterday maybe", errors="coerce", utc=True))

    # 9. Export all requested artifacts
    df.to_csv(OUTPUT_DIR / "news_validation_all.csv", index=False, encoding="utf-8-sig")
    accepted_df.to_csv(OUTPUT_DIR / "news_accepted.csv", index=False, encoding="utf-8-sig")
    excluded_df.to_csv(OUTPUT_DIR / "news_excluded.csv", index=False, encoding="utf-8-sig")
    if "duplicate_signature_candidate" in df.columns:
        df[df["duplicate_signature_candidate"]].to_csv(
            OUTPUT_DIR / "news_duplicate_review.csv", index=False, encoding="utf-8-sig"
        )
    warning_df.to_csv(OUTPUT_DIR / "news_warning_review.csv", index=False, encoding="utf-8-sig")
    quarantine_df.to_csv(OUTPUT_DIR / "news_quarantine.csv", index=False, encoding="utf-8-sig")
    issues_df.to_csv(OUTPUT_DIR / "news_validation_issues.csv", index=False, encoding="utf-8-sig")
    completeness.to_csv(OUTPUT_DIR / "news_field_completeness.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "news_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nAll basic validation unit tests passed.")
    print(f"Input rows: {len(df):,}")
    print(f"Total issues: {len(issues_df):,}")
    print(f"Distinct affected rows: {affected_rows:,}")
    print("Disposition counts:", summary["dispositions"])
    print(f"Elapsed seconds: {time.time() - started:.1f}")
    print("\nOutput files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path.name} — {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
