
"""
LimeoX - Task 8: Feature Engineering

Processes:
1. Augmented training dataset
2. Original testing dataset

Input:
Task 9/model_results/news_train_augmented.csv
Task 9/model_results/news_test_original.csv

Output:
Task 9/model_results/news_train_featured.csv
Task 9/model_results/news_test_featured.csv

32 Numerical Features
1 Categorical Feature: domain_feature
Target: category
"""

from pathlib import Path
from urllib.parse import urlparse
import re
import gc

import numpy as np
import pandas as pd


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_DIR = (
    PROJECT_DIR
    / "Task 9 (Machine Learning)"
    / "model_results"
)

TRAIN_FILE = MODEL_DIR / "news_train_augmented.csv"
TEST_FILE = MODEL_DIR / "news_test_original.csv"

TRAIN_OUTPUT = MODEL_DIR / "news_train_featured.csv"
TEST_OUTPUT = MODEL_DIR / "news_test_featured.csv"


# =========================================================
# 2. FEATURE CONFIGURATION
# =========================================================

FINANCIAL_KEYWORDS = [
    "earnings", "revenue", "profit", "loss", "stock",
    "stocks", "shares", "market", "forecast",
    "acquisition", "merger", "bankruptcy", "dividend",
    "inflation", "interest rate", "investment",
    "investor", "debt", "valuation", "ipo"
]

# Compile once instead of compiling a regex for every keyword
FINANCIAL_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(keyword)
        for keyword in FINANCIAL_KEYWORDS
    )
    + r")\b",
    flags=re.IGNORECASE
)

WORD_PATTERN = re.compile(r"\b\w+\b")
SENTENCE_PATTERN = re.compile(r"[.!?]+")
BODY_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


FEATURE_COLUMNS = [

    # Title features (8)
    "title_char_count",
    "title_word_count",
    "title_avg_word_length",
    "title_digit_count",
    "title_uppercase_ratio",
    "title_exclamation_count",
    "title_question_count",
    "title_unique_word_ratio",

    # Body features (7)
    "body_char_count",
    "body_word_count",
    "body_sentence_count",
    "body_avg_sentence_length",
    "body_unique_word_ratio",
    "body_digit_count",
    "body_url_count",

    # Financial features (2)
    "financial_keyword_count",
    "financial_keyword_density",

    # URL features (4)
    "url_length",
    "url_path_depth",
    "url_https",
    "url_slug_length",

    # Date/time features (10)
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

    # Interaction feature (1)
    "title_body_length_ratio"
]


# =========================================================
# 3. COLUMN MAPPING
# =========================================================

COLUMN_ALIASES = {

    "title": ["title_clean", "title"],

    "text": [
        "content_clean",
        "article_text",
        "content",
        "text"
    ],

    "url": [
        "news_url_clean",
        "article_url",
        "url"
    ],

    "domain": ["domain", "source"],

    "author": ["author_clean", "author"],

    "published_at": [
        "published_at_utc",
        "publication_date",
        "published_at"
    ],

    "scraped_at": [
        "scraped_at_utc",
        "scraped_at"
    ],

    "category": [
        "category",
        "predicted_category",
        "news_category"
    ]
}


def resolve_column(df, aliases):

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for alias in aliases:

        actual = lookup.get(alias.strip().lower())

        if actual is not None:
            return actual

    return None


def standardize_columns(df):

    df = df.copy()

    for standard_name, aliases in COLUMN_ALIASES.items():

        actual_column = resolve_column(df, aliases)

        if actual_column is not None:

            if actual_column != standard_name:
                df[standard_name] = df[actual_column]

        elif standard_name == "category":

            raise ValueError(
                "Category column is missing!\n"
                f"Expected one of: {aliases}"
            )

        else:

            df[standard_name] = pd.NA

    return df


# =========================================================
# 4. TEXT CLEANING
# =========================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value)
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def get_words(value):

    return WORD_PATTERN.findall(value)


# =========================================================
# 5. TITLE FEATURE EXTRACTION
# =========================================================

def extract_title_features(value):

    words = get_words(value)

    return {

        "title_char_count": len(value),

        "title_word_count": len(words),

        "title_avg_word_length": (
            float(np.mean([len(w) for w in words]))
            if words else 0.0
        ),

        "title_digit_count": sum(
            char.isdigit() for char in value
        ),

        "title_uppercase_ratio": (
            sum(char.isupper() for char in value) / len(value)
            if value else 0.0
        ),

        "title_exclamation_count": value.count("!"),

        "title_question_count": value.count("?"),

        "title_unique_word_ratio": (
            len(set(w.lower() for w in words)) / len(words)
            if words else 0.0
        )
    }


# =========================================================
# 6. BODY FEATURE EXTRACTION
# =========================================================

def extract_body_features(value):

    words = get_words(value)

    sentences = [
        sentence
        for sentence in SENTENCE_PATTERN.split(value)
        if sentence.strip()
    ]

    return {

        "body_char_count": len(value),

        "body_word_count": len(words),

        "body_sentence_count": len(sentences),

        "body_avg_sentence_length": (
            len(words) / len(sentences)
            if sentences else 0.0
        ),

        "body_unique_word_ratio": (
            len(set(w.lower() for w in words)) / len(words)
            if words else 0.0
        ),

        "body_digit_count": sum(
            char.isdigit() for char in value
        ),

        "body_url_count": len(
            BODY_URL_PATTERN.findall(value)
        )
    }


# =========================================================
# 7. DOMAIN AND URL FEATURES
# =========================================================

def parse_url(value):

    value = (
        ""
        if pd.isna(value)
        else str(value).strip()
    )

    if value and "://" not in value:
        value = "https://" + value

    return urlparse(value)


def normalize_domain(domain, url):

    if pd.notna(domain) and str(domain).strip():

        domain = str(domain).strip()

        try:

            parsed = parse_url(domain)

            if parsed.hostname:

                return (
                    parsed.hostname.lower()
                    .removeprefix("www.")
                )

        except ValueError:
            pass

        return domain.lower().removeprefix("www.")

    try:

        parsed = parse_url(url)

        if parsed.hostname:

            return (
                parsed.hostname.lower()
                .removeprefix("www.")
            )

    except ValueError:
        pass

    return "unknown"


def extract_url_features(value):

    value = (
        ""
        if pd.isna(value)
        else str(value).strip()
    )

    try:

        parsed = parse_url(value)

        path_parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        slug = path_parts[-1] if path_parts else ""

        return {

            "url_length": len(value),

            "url_path_depth": len(path_parts),

            "url_https": int(
                value.lower().startswith("https://")
            ),

            "url_slug_length": len(slug)
        }

    except ValueError:

        return {

            "url_length": len(value),

            "url_path_depth": 0,

            "url_https": 0,

            "url_slug_length": 0
        }


# =========================================================
# 8. DATE AND TIME FEATURES
# =========================================================

def extract_date_features(df):

    published = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True
    )

    scraped = pd.to_datetime(
        df["scraped_at"],
        errors="coerce",
        utc=True
    )

    day_of_week = published.dt.day_of_week

    article_age = (
        (scraped - published).dt.total_seconds() / 3600
    )

    # Remove invalid negative article ages.
    article_age = article_age.where(
        article_age >= 0,
        np.nan
    )

    features = pd.DataFrame(index=df.index)

    features["published_year"] = published.dt.year

    features["published_month"] = published.dt.month

    features["published_hour"] = published.dt.hour

    features["published_dayofweek"] = day_of_week

    features["published_is_weekend"] = np.where(
        day_of_week.notna(),
        day_of_week.isin([5, 6]).astype(float),
        np.nan
    )

    features["article_age_hours"] = article_age

    features["published_hour_sin"] = np.sin(
        2 * np.pi * features["published_hour"] / 24
    )

    features["published_hour_cos"] = np.cos(
        2 * np.pi * features["published_hour"] / 24
    )

    features["published_day_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )

    features["published_day_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    return features


# =========================================================
# 9. OPTIMIZED FINANCIAL KEYWORD FEATURES
# =========================================================

def count_financial_keywords(value):

    if not isinstance(value, str):
        return 0

    return len(
        FINANCIAL_PATTERN.findall(value)
    )


# =========================================================
# 10. COMPLETE FEATURE ENGINEERING
# =========================================================

def engineer_features(df, dataset_name):

    print(
        f"\nApplying feature engineering: {dataset_name}",
        flush=True
    )

    df = standardize_columns(df)

    original_rows = len(df)

    # Normalize category labels.
    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
    )

    # Remove records without valid labels.
    df = df.dropna(subset=["category"]).copy()

    df = df[df["category"] != ""].copy()

    # Reset index after filtering.
    df = df.reset_index(drop=True)

    print(
        "Valid labelled rows:",
        len(df),
        flush=True
    )

    # -------------------------------
    # TEXT NORMALIZATION
    # -------------------------------

    print("Cleaning text...", flush=True)

    df["title_clean_fe"] = df["title"].apply(clean_text)

    df["body_clean_fe"] = df["text"].apply(clean_text)

    # -------------------------------
    # TITLE FEATURES
    # -------------------------------

    print("Extracting title features...", flush=True)

    title_features = pd.DataFrame(
        df["title_clean_fe"].apply(
            extract_title_features
        ).tolist(),
        index=df.index
    )

    # -------------------------------
    # BODY FEATURES
    # -------------------------------

    print("Extracting body features...", flush=True)

    body_features = pd.DataFrame(
        df["body_clean_fe"].apply(
            extract_body_features
        ).tolist(),
        index=df.index
    )

    # -------------------------------
    # URL FEATURES
    # -------------------------------

    print("Extracting URL features...", flush=True)

    url_features = pd.DataFrame(
        df["url"].apply(
            extract_url_features
        ).tolist(),
        index=df.index
    )

    # -------------------------------
    # DOMAIN
    # -------------------------------

    print("Processing domains...", flush=True)

    df["domain_feature"] = [

        normalize_domain(domain, url)

        for domain, url in zip(
            df["domain"],
            df["url"]
        )
    ]

    # -------------------------------
    # DATE FEATURES
    # -------------------------------

    print("Extracting date features...", flush=True)

    date_features = extract_date_features(df)

    # -------------------------------
    # FINANCIAL FEATURES
    # -------------------------------

    print("Counting financial keywords...", flush=True)

    financial_count = (
        df["body_clean_fe"].apply(
            count_financial_keywords
        )
    )

    financial_features = pd.DataFrame(index=df.index)

    financial_features["financial_keyword_count"] = (
        financial_count
    )

    financial_features["financial_keyword_density"] = (
        financial_count
        / body_features["body_word_count"].replace(
            0, np.nan
        )
    )

    # -------------------------------
    # INTERACTION FEATURE
    # -------------------------------

    interaction_features = pd.DataFrame(index=df.index)

    interaction_features["title_body_length_ratio"] = (
        title_features["title_word_count"]
        / body_features["body_word_count"].replace(
            0, np.nan
        )
    )

    # -------------------------------
    # COMBINE FEATURES
    # -------------------------------

    print("Combining features...", flush=True)

    engineered = pd.concat(
        [
            title_features,
            body_features,
            financial_features,
            url_features,
            date_features,
            interaction_features
        ],
        axis=1
    )

    if engineered.columns.duplicated().any():

        raise ValueError(
            "Duplicate feature columns detected!"
        )

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in engineered.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    engineered = engineered[FEATURE_COLUMNS]

    engineered = engineered.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -------------------------------
    # FINAL DATASET
    # -------------------------------

    result = pd.concat(
        [
            engineered,
            df[["domain_feature", "category"]]
        ],
        axis=1
    )

    # -------------------------------
    # VALIDATION
    # -------------------------------

    assert len(result) == len(df)

    assert len(FEATURE_COLUMNS) == 32

    assert result.columns.is_unique

    assert "category" not in engineered.columns

    print("\nDataset validation:", dataset_name)

    print("Original rows:", original_rows)

    print("Valid labelled rows:", len(result))

    print("Removed rows:", original_rows - len(result))

    print("Numerical features:", len(FEATURE_COLUMNS))

    print("Categorical features: 1")

    print("Output shape:", result.shape)

    print(
        "Missing numerical values:",
        engineered.isna().sum().sum()
    )

    print("Feature engineering completed!", flush=True)

    return result


# =========================================================
# 11. PROCESS AND SAVE ONE DATASET AT A TIME
# =========================================================

def process_dataset(input_file, output_file, dataset_name):

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{input_file}"
        )

    print("\n" + "=" * 60)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 60)

    print("\nLoading dataset...", flush=True)

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print("Input shape:", df.shape, flush=True)

    features = engineer_features(
        df,
        dataset_name
    )

    expected_columns = (
        FEATURE_COLUMNS
        + ["domain_feature", "category"]
    )

    if features.columns.tolist() != expected_columns:

        raise ValueError(
            "Feature columns do not match expected columns!"
        )

    print("\nSaving engineered dataset...", flush=True)

    features.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", output_file, flush=True)

    print("Output shape:", features.shape, flush=True)

    print("\nCategory distribution:")

    print(features["category"].value_counts())

    print("\nDataset processing completed!", flush=True)

    # Store only the categories for final train/test comparison.
    categories = set(
        features["category"].dropna().unique()
    )

    rows = len(features)

    # Release large objects before processing next dataset.
    del features
    del df

    gc.collect()

    return rows, categories


# =========================================================
# 12. MAIN EXECUTION
# =========================================================

print("=" * 60)
print("LIMEOX TASK 8 - FEATURE ENGINEERING")
print("=" * 60)

print("\nProject directory:", PROJECT_DIR)

print("Model results directory:", MODEL_DIR)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# -------------------------------
# TRAINING DATA
# -------------------------------

train_rows, train_classes = process_dataset(
    TRAIN_FILE,
    TRAIN_OUTPUT,
    "AUGMENTED TRAINING DATA"
)

# -------------------------------
# TESTING DATA
# -------------------------------

gc.collect()

test_rows, test_classes = process_dataset(
    TEST_FILE,
    TEST_OUTPUT,
    "ORIGINAL TESTING DATA"
)

# =========================================================
# 13. TRAIN / TEST VALIDATION
# =========================================================

print("\n" + "=" * 60)
print("TRAIN / TEST VALIDATION")
print("=" * 60)

unseen_test_classes = test_classes - train_classes

if unseen_test_classes:

    print("\nWARNING: Unseen testing categories:")

    print(sorted(unseen_test_classes))

else:

    print("All testing categories exist in training data.")


print("\nTraining records:", train_rows)

print("Testing records:", test_rows)

print("Numerical features:", len(FEATURE_COLUMNS))

print("Categorical features: 1")

print("Total feature columns:", len(FEATURE_COLUMNS) + 2)

print("\nTraining output:", TRAIN_OUTPUT)

print("Testing output:", TEST_OUTPUT)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("=" * 60)
