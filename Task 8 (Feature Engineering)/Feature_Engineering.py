"""
LimeoX - Task 8: Feature Engineering for Scraped News
Uses the project's validated/categorized news data.

Expected input:
Task 5 (Categorization)/news_categorized_groq.csv

The script maps the real Task 4 field names:
news_url_clean, title_clean, content_clean, published_at_utc,
author_clean, domain, scraped_at_utc
and keeps the Task 5 category column as the ML target.
"""

from pathlib import Path
from urllib.parse import urlparse
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Change this only if your Task 5 output has a different filename.
INPUT_FILE = (
    BASE_DIR.parent
    / "Task 5 (Categorization)"
    / "news_categorized.csv"
)

OUTPUT_DIR = BASE_DIR / "feature_engineering_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD THE REAL PROJECT DATA
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Set INPUT_FILE to the actual categorized CSV produced by Task 5."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)
print("Input file:", INPUT_FILE)
print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())


# ------------------------------------------------------------
# 3. MAP TASK 4 / TASK 5 COLUMN NAMES TO STANDARD NAMES
# ------------------------------------------------------------

# These names are based on Task 4 validation.py's actual column contract.
COLUMN_ALIASES = {
    "title": ["title_clean", "title"],
    "text": ["content_clean", "article_text", "content", "text"],
    "url": ["news_url_clean", "article_url", "url"],
    "domain": ["domain", "source"],
    "author": ["author_clean", "author"],
    "published_at": [
        "published_at_utc", "publication_date", "published_at"
    ],
    "scraped_at": ["scraped_at_utc", "scraped_at"],
    "category": ["category", "predicted_category", "news_category"],
}


def find_column(frame, candidates):
    """Return the first matching column, ignoring case and spaces."""
    lookup = {
        str(col).strip().lower(): col
        for col in frame.columns
    }
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


for standard_name, aliases in COLUMN_ALIASES.items():
    actual_name = find_column(df, aliases)
    if actual_name is not None and actual_name != standard_name:
        df[standard_name] = df[actual_name]

# Task 5's category is needed only for supervised model training.
# Do not invent labels if the categorization output does not contain them.
if "category" not in df.columns:
    raise ValueError(
        "Category column was not found. Feature extraction can be done "
        "without labels, but model training/evaluation requires the "
        "category column from Task 5. Check the Task 5 output CSV."
    )

for col in ["title", "text", "url", "domain", "author",
            "published_at", "scraped_at"]:
    if col not in df.columns:
        df[col] = pd.NA

print("Standardized columns are ready.")


# ------------------------------------------------------------
# 4. TEXT NORMALIZATION
# ------------------------------------------------------------

def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value)
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


df["title_clean_fe"] = df["title"].apply(clean_text)
df["body_clean_fe"] = df["text"].apply(clean_text)
df["title_lower"] = df["title_clean_fe"].str.lower()
df["body_lower"] = df["body_clean_fe"].str.lower()


# ------------------------------------------------------------
# 5. TITLE FEATURES
# ------------------------------------------------------------

def get_word_count(value):
    return len(re.findall(r"\b\w+\b", value))


def get_average_word_length(value):
    words = re.findall(r"\b\w+\b", value)
    return float(np.mean([len(word) for word in words])) if words else 0.0


def get_unique_word_ratio(value):
    words = re.findall(r"\b\w+\b", value.lower())
    return len(set(words)) / len(words) if words else 0.0


df["title_char_count"] = df["title_clean_fe"].str.len()
df["title_word_count"] = df["title_clean_fe"].apply(get_word_count)
df["title_avg_word_length"] = df["title_clean_fe"].apply(
    get_average_word_length
)
df["title_digit_count"] = df["title_clean_fe"].apply(
    lambda x: sum(char.isdigit() for char in x)
)
df["title_uppercase_ratio"] = df["title_clean_fe"].apply(
    lambda x: (
        sum(char.isupper() for char in x) / len(x)
        if len(x) else 0.0
    )
)
df["title_exclamation_count"] = df["title_clean_fe"].str.count("!")
df["title_question_count"] = df["title_clean_fe"].str.count(r"\?")
df["title_unique_word_ratio"] = df["title_clean_fe"].apply(
    get_unique_word_ratio
)


# ------------------------------------------------------------
# 6. ARTICLE BODY FEATURES
# ------------------------------------------------------------

def get_body_statistics(value):
    value = clean_text(value)
    words = re.findall(r"\b\w+\b", value)
    sentences = [
        sentence for sentence in re.split(r"[.!?]+", value)
        if sentence.strip()
    ]

    return pd.Series({
        "body_char_count": len(value),
        "body_word_count": len(words),
        "body_sentence_count": len(sentences),
        "body_avg_sentence_length": (
            len(words) / len(sentences) if sentences else 0.0
        ),
        "body_unique_word_ratio": (
            len(set(word.lower() for word in words)) / len(words)
            if words else 0.0
        ),
        "body_digit_count": sum(char.isdigit() for char in value),
        "body_url_count": len(
            re.findall(r"https?://\S+|www\.\S+", value)
        ),
    })


body_features = df["body_clean_fe"].apply(get_body_statistics)
df[body_features.columns] = body_features


# ------------------------------------------------------------
# 7. DOMAIN AND URL FEATURES
# ------------------------------------------------------------

def extract_domain(url_value, existing_domain):
    if pd.notna(existing_domain) and str(existing_domain).strip():
        return str(existing_domain).strip().lower()
    try:
        host = urlparse(str(url_value)).hostname
        return host.lower().removeprefix("www.") if host else ""
    except ValueError:
        return ""


df["domain_feature"] = [
    extract_domain(url, domain)
    for url, domain in zip(df["url"], df["domain"])
]

def url_statistics(value):
    value = "" if pd.isna(value) else str(value)
    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    slug = path_parts[-1] if path_parts else ""

    return pd.Series({
        "url_length": len(value),
        "url_path_depth": len(path_parts),
        "url_https": int(parsed.scheme.lower() == "https"),
        "url_slug_length": len(slug),
    })


url_features = df["url"].apply(url_statistics)
df[url_features.columns] = url_features


# ------------------------------------------------------------
# 8. DATE AND TIME FEATURES
# ------------------------------------------------------------

df["published_at_parsed"] = pd.to_datetime(
    df["published_at"], errors="coerce", utc=True
)
df["scraped_at_parsed"] = pd.to_datetime(
    df["scraped_at"], errors="coerce", utc=True
)

published = df["published_at_parsed"]

df["published_year"] = published.dt.year
df["published_month"] = published.dt.month
df["published_hour"] = published.dt.hour
df["published_dayofweek"] = published.dt.dayofweek
df["published_is_weekend"] = (
    published.dt.dayofweek.isin([5, 6]).astype("Int64")
)

df["article_age_hours"] = (
    (df["scraped_at_parsed"] - published)
    .dt.total_seconds() / 3600
)

# Cyclical encoding preserves the circular nature of hour/day.
df["published_hour_sin"] = np.sin(
    2 * np.pi * df["published_hour"] / 24
)
df["published_hour_cos"] = np.cos(
    2 * np.pi * df["published_hour"] / 24
)
df["published_day_sin"] = np.sin(
    2 * np.pi * df["published_dayofweek"] / 7
)
df["published_day_cos"] = np.cos(
    2 * np.pi * df["published_dayofweek"] / 7
)


# ------------------------------------------------------------
# 9. DOMAIN-SPECIFIC KEYWORD FEATURES
# ------------------------------------------------------------

financial_keywords = [
    "earnings", "revenue", "profit", "loss", "stock", "stocks",
    "shares", "market", "forecast", "acquisition", "merger",
    "bankruptcy", "dividend", "inflation", "interest rate",
    "investment", "investor", "debt", "valuation", "ipo"
]


def count_keywords(value, keywords):
    value = str(value).lower()
    total = 0
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        total += len(re.findall(pattern, value))
    return total


df["financial_keyword_count"] = df["body_clean_fe"].apply(
    lambda value: count_keywords(value, financial_keywords)
)

df["financial_keyword_density"] = (
    df["financial_keyword_count"]
    / df["body_word_count"].replace(0, np.nan)
)

df["title_body_length_ratio"] = (
    df["title_word_count"]
    / df["body_word_count"].replace(0, np.nan)
)


# ------------------------------------------------------------
# 10. SAVE THE INTERPRETABLE FEATURE DATASET
# ------------------------------------------------------------

feature_columns = [
    "title_char_count",
    "title_word_count",
    "title_avg_word_length",
    "title_digit_count",
    "title_uppercase_ratio",
    "title_exclamation_count",
    "title_question_count",
    "title_unique_word_ratio",
    "body_char_count",
    "body_word_count",
    "body_sentence_count",
    "body_avg_sentence_length",
    "body_unique_word_ratio",
    "body_digit_count",
    "body_url_count",
    "financial_keyword_count",
    "financial_keyword_density",
    "url_length",
    "url_path_depth",
    "url_https",
    "url_slug_length",
    "published_year",
    "published_month",
    "published_hour",
    "published_dayofweek",
    "published_is_weekend",
    "article_age_hours",
    "published_hour_sin",
    "published_hour_cos",
    "published_day_sin",
    "published_day_cos",
    "title_body_length_ratio",
]

identifier_columns = [
    col for col in ["url", "domain_feature", "category"]
    if col in df.columns
]

feature_dataset = df[identifier_columns + feature_columns].copy()
feature_dataset = feature_dataset.replace(
    [np.inf, -np.inf], np.nan
)

feature_output = OUTPUT_DIR / "news_engineered_features.csv"
feature_dataset.to_csv(
    feature_output, index=False, encoding="utf-8-sig"
)

print("\nFeature engineering completed.")
print("Input shape:", df.shape)
print("Feature dataset shape:", feature_dataset.shape)
print("Saved:", feature_output)


# ------------------------------------------------------------
# 11. TRAIN A LEAKAGE-SAFE BASELINE CLASSIFIER
# ------------------------------------------------------------

# Remove rows without a category label.
model_df = df.dropna(subset=["category"]).copy()
model_df["category"] = model_df["category"].astype(str).str.strip()
model_df = model_df[model_df["category"] != ""]

if model_df["category"].nunique() < 2:
    print(
        "\nModel training skipped: at least two different categories "
        "are required."
    )
else:
    # Use a chronological split when valid publication dates are available.
    model_df = model_df.sort_values(
        "published_at_parsed", na_position="last"
    )

    X = model_df[[
        "title_lower",
        "body_lower",
        "domain_feature",
        "author",
        "title_word_count",
        "body_word_count",
        "url_length",
        "url_path_depth",
        "published_hour",
        "financial_keyword_count",
    ]].copy()

    y = model_df["category"]

    # Missing categorical values are represented explicitly.
    for col in ["domain_feature", "author"]:
        X[col] = X[col].fillna("unknown").astype(str)

    # Use a chronological 80/20 split to reduce future-to-past leakage.
    split_index = int(len(model_df) * 0.80)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2 or y_test.empty:
        print(
            "Model training skipped: chronological split does not contain "
            "enough category variation."
        )
    else:
        numeric_features = [
            "title_word_count",
            "body_word_count",
            "url_length",
            "url_path_depth",
            "published_hour",
            "financial_keyword_count",
        ]

        categorical_features = ["domain_feature", "author"]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            (
                "title_tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2
                ),
                "title_lower",
            ),
            (
                "body_tfidf",
                TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98
                ),
                "body_lower",
            ),
        ])

        model = Pipeline([
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ),
            ),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("\nModel evaluation")
        print("Training records:", len(X_train))
        print("Testing records:", len(X_test))
        print("Accuracy:", accuracy_score(y_test, predictions))
        print(
            "Precision:",
            precision_score(
                y_test, predictions,
                average="weighted", zero_division=0
            )
        )
        print(
            "Recall:",
            recall_score(
                y_test, predictions,
                average="weighted", zero_division=0
            )
        )
        print(
            "F1 Score:",
            f1_score(
                y_test, predictions,
                average="weighted", zero_division=0
            )
        )
        print("\nClassification report:")
        print(
            classification_report(
                y_test, predictions, zero_division=0
            )
        )

        labels = sorted(set(y_test) | set(predictions))
        matrix = confusion_matrix(
            y_test, predictions, labels=labels
        )
        pd.DataFrame(
            matrix, index=labels, columns=labels
        ).to_csv(
            OUTPUT_DIR / "confusion_matrix.csv",
            encoding="utf-8-sig"
        )

        model_path = OUTPUT_DIR / "news_classification_pipeline.pkl"
        joblib.dump(model, model_path)
        print("Saved model:", model_path)


print("\nOutput folder:", OUTPUT_DIR.resolve())
for output_file in sorted(OUTPUT_DIR.iterdir()):
    print(f"{output_file.name} - {output_file.stat().st_size:,} bytes")
