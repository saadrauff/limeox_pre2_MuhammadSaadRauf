
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_FILE = (
    PROJECT_DIR
    / "Task 5 (Categorization)"
    / "news_categorized.csv"
)

OUTPUT_DIR = BASE_DIR / "_shared"

RANDOM_STATE = 42

TRAIN_RATIO = 0.80
TEST_RATIO = 0.20

MAX_FEATURES = 12000
SVD_COMPONENTS = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MEMORY-EFFICIENT TEXT GENERATOR
# ============================================================

def generate_text(dataframe):

    for title, content in zip(
        dataframe["title_clean"],
        dataframe["content_clean"]
    ):

        title = (
            "" if pd.isna(title)
            else str(title).strip()
        )

        content = (
            "" if pd.isna(content)
            else str(content).strip()
        )

        yield (title + " " + content).strip()


# ============================================================
# 3. MAIN DATA PREPARATION
# ============================================================

def prepare_data():

    start_time = time.time()

    print("\n" + "=" * 70)
    print("TASK 12 - RAW DATA LIBRARIES")
    print("COMPLETE DATASET PREPARATION")
    print("=" * 70)


    # --------------------------------------------------------
    # STEP 1: LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/8] Reading original dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    required_columns = [
        "title_clean",
        "content_clean",
        "category",
        "validation_disposition"
    ]

    df = pd.read_csv(
        INPUT_FILE,
        usecols=required_columns,
        low_memory=False
    )

    original_count = len(df)

    print(f"Original records: {original_count:,}")


    # --------------------------------------------------------
    # STEP 2: FILTER ACCEPTED RECORDS
    # --------------------------------------------------------

    print("\n[2/8] Filtering accepted records...")

    df["validation_disposition"] = (
        df["validation_disposition"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df = df[
        df["validation_disposition"] == "ACCEPT"
    ].copy()

    df = df.dropna(subset=["category"])

    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
    )

    df = df[
        df["category"].notna()
        & (df["category"] != "")
    ].copy()

    df.reset_index(drop=True, inplace=True)

    accepted_count = len(df)

    print(f"Accepted records: {accepted_count:,}")

    print("\nCategory distribution:")

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    if accepted_count < 2:
        raise ValueError(
            "Not enough accepted records."
        )

    class_counts = df["category"].value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "At least one category has fewer than "
            "2 records. Stratified splitting is not possible."
        )


    # --------------------------------------------------------
    # STEP 3: COMPLETE 80/20 STRATIFIED SPLIT
    # --------------------------------------------------------

    print("\n[3/8] Creating complete stratified split...")

    all_indices = np.arange(accepted_count)

    labels = df["category"].to_numpy(dtype=object)

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    train_df = (
        df.iloc[train_indices]
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[test_indices]
        .reset_index(drop=True)
    )

    train_count = len(train_df)
    test_count = len(test_df)

    print(f"Training records: {train_count:,}")
    print(f"Testing records: {test_count:,}")

    print(
        f"Total used records: {train_count + test_count:,}"
    )

    if train_count + test_count != accepted_count:
        raise RuntimeError(
            "Some accepted records were lost."
        )

    del df, labels, all_indices
    del train_indices, test_indices


    # --------------------------------------------------------
    # STEP 4: PREPARE TEXT AND LABELS
    # --------------------------------------------------------

    print("\n[4/8] Preparing text and labels...")

    # Unicode arrays prevent object-array pickle problems.
    y_train = (
        train_df["category"]
        .astype(str)
        .to_numpy(dtype=str)
    )

    y_test = (
        test_df["category"]
        .astype(str)
        .to_numpy(dtype=str)
    )

    print("Training labels prepared.")
    print("Testing labels prepared.")

    print("Using memory-efficient text generators.")


    # --------------------------------------------------------
    # STEP 5: TF-IDF FEATURE EXTRACTION
    # --------------------------------------------------------

    print("\n[5/8] Applying TF-IDF...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        dtype=np.float32
    )

    X_train_tfidf = vectorizer.fit_transform(
        generate_text(train_df)
    )

    X_test_tfidf = vectorizer.transform(
        generate_text(test_df)
    )

    print(
        f"Training TF-IDF shape: {X_train_tfidf.shape}"
    )

    print(
        f"Testing TF-IDF shape: {X_test_tfidf.shape}"
    )

    del train_df, test_df


    # --------------------------------------------------------
    # STEP 6: TRUNCATED SVD
    # --------------------------------------------------------

    print("\n[6/8] Applying TruncatedSVD...")

    actual_components = min(
        SVD_COMPONENTS,
        X_train_tfidf.shape[1] - 1,
        X_train_tfidf.shape[0] - 1
    )

    if actual_components < 1:
        raise ValueError(
            "Insufficient features for TruncatedSVD."
        )

    svd = TruncatedSVD(
        n_components=actual_components,
        n_iter=7,
        random_state=RANDOM_STATE
    )

    X_train = svd.fit_transform(
        X_train_tfidf
    ).astype(np.float32)

    X_test = svd.transform(
        X_test_tfidf
    ).astype(np.float32)

    print(f"Final training shape: {X_train.shape}")
    print(f"Final testing shape: {X_test.shape}")

    explained_variance = float(
        svd.explained_variance_ratio_.sum()
    )

    print(
        "Explained variance ratio:",
        round(explained_variance, 4)
    )

    del X_train_tfidf, X_test_tfidf


    # --------------------------------------------------------
    # STEP 7: SAVE PREPARED DATA AND MODELS
    # --------------------------------------------------------

    print("\n[7/8] Saving prepared data...")

    data_file = OUTPUT_DIR / "prepared_data.npz"

    np.savez_compressed(
        data_file,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    vectorizer_file = (
        OUTPUT_DIR / "tfidf_vectorizer.joblib"
    )

    svd_file = OUTPUT_DIR / "svd_model.joblib"

    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(svd, svd_file)

    metadata = {
        "dataset": str(INPUT_FILE),
        "original_records": original_count,
        "accepted_records": accepted_count,
        "used_records": train_count + test_count,
        "unused_accepted_records": 0,
        "filter": "validation_disposition == ACCEPT",
        "text_columns": [
            "title_clean",
            "content_clean"
        ],
        "target_column": "category",
        "split_method": "Stratified train_test_split",
        "train_ratio": TRAIN_RATIO,
        "test_ratio": TEST_RATIO,
        "random_state": RANDOM_STATE,
        "train_size": train_count,
        "test_size": test_count,
        "tfidf_max_features": MAX_FEATURES,
        "tfidf_ngram_range": [1, 2],
        "svd_components": actual_components,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "categories": sorted(
            np.unique(
                np.concatenate([y_train, y_test])
            ).tolist()
        ),
        "category_distribution": {
            str(k): int(v)
            for k, v in pd.concat([
                pd.Series(y_train),
                pd.Series(y_test)
            ]).value_counts().items()
        },
        "explained_variance_ratio": explained_variance
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


    # --------------------------------------------------------
    # STEP 8: VERIFY SAVED DATA
    # --------------------------------------------------------

    print("\n[8/8] Verifying saved dataset...")

    with np.load(
        data_file,
        allow_pickle=False
    ) as saved:

        assert saved["X_train"].shape == (
            train_count,
            actual_components
        )

        assert saved["X_test"].shape == (
            test_count,
            actual_components
        )

        assert len(saved["y_train"]) == train_count
        assert len(saved["y_test"]) == test_count

        assert saved["y_train"].dtype.kind == "U"
        assert saved["y_test"].dtype.kind == "U"

    print("All saved arrays verified successfully!")

    elapsed = time.time() - start_time


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print(f"\nOriginal records: {original_count:,}")
    print(f"Accepted records: {accepted_count:,}")
    print(f"Training records: {train_count:,}")
    print(f"Testing records: {test_count:,}")

    print(
        f"Total records used: {train_count + test_count:,}"
    )

    print("\nGenerated files:")

    print("1. prepared_data.npz")
    print("2. tfidf_vectorizer.joblib")
    print("3. svd_model.joblib")
    print("4. metadata.json")

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print(
        f"\nExecution time: {elapsed / 60:.2f} minutes"
    )

    print("\nAll accepted records are included.")
    print("Dataset is ready for six-library comparison.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prepare_data()
