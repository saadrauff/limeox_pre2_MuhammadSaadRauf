
import re
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "model_results"

OUTPUT = DATA / "tfidf_validation_experiments"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("LIME OX - CONTROLLED TF-IDF EXPERIMENTS")
print("=" * 70)

# 1. Load datasets
train = pd.read_csv(
    DATA / "news_train_augmented.csv",
    low_memory=False
)

test = pd.read_csv(
    DATA / "news_test_original.csv",
    low_memory=False
)

# 2. Normalize text
def clean(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

for df in [train, test]:
    df["title_clean"] = df["title_clean"].apply(clean)
    df["content_clean"] = df["content_clean"].apply(clean)
    df["combined_text"] = (
        df["title_clean"] + " " +
        df["title_clean"] + " " +
        df["content_clean"]
    )

# 3. Remove exact duplicate records
train["_key"] = train["combined_text"].str.lower()
test["_key"] = test["combined_text"].str.lower()

train = train.drop_duplicates("_key").copy()
test = test.drop_duplicates("_key").copy()

# 4. Remove training records overlapping test
test_titles = set(test["title_clean"].str.lower())
test_urls = set(test["news_url_clean"].fillna("").str.lower())

train = train[
    ~train["title_clean"].str.lower().isin(test_titles)
    & ~train["news_url_clean"].fillna("").str.lower().isin(test_urls)
].copy()

# Separate original and augmented records
original = train[train["is_augmented"] == 0].copy()
augmented = train.copy()

# 5. Use same test data for all experiments
X_test = test["combined_text"].values
y_test = test["category"].values

classes = sorted(test["category"].unique())

# 6. Experiment datasets
experiments = {
    "Original_Only": original,
    "Augmented": augmented
}

results = []

for name, training in experiments.items():

    print("\n" + "=" * 60)
    print("EXPERIMENT:", name)
    print("=" * 60)

    X_train = training["combined_text"].values
    y_train = training["category"].values

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=200000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    for weighting in [None, "balanced"]:

        print("\nClass weighting:", weighting)

        model = LinearSVC(
            C=1.0,
            class_weight=weighting,
            max_iter=10000,
            random_state=42,
            dual="auto"
        )

        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )
        weighted_f1 = f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )
        macro_f1 = f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        )

        experiment_name = (
            name + "_" +
            ("Balanced" if weighting else "Unbalanced")
        )

        print("Accuracy:", round(accuracy, 4))
        print("Weighted F1:", round(weighted_f1, 4))
        print("Macro F1:", round(macro_f1, 4))

        report = classification_report(
            y_test,
            predictions,
            labels=classes,
            zero_division=0
        )

        with open(
            OUTPUT / f"{experiment_name}_report.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report)

        results.append({
            "Experiment": experiment_name,
            "Training_Records": len(X_train),
            "Testing_Records": len(X_test),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Weighted_F1": weighted_f1,
            "Macro_F1": macro_f1
        })

        # Save models and predictions
        joblib.dump(
            {
                "vectorizer": vectorizer,
                "model": model,
                "classes": classes
            },
            OUTPUT / f"{experiment_name}.pkl",
            compress=3
        )

        pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions
        }).to_csv(
            OUTPUT / f"{experiment_name}_predictions.csv",
            index=False
        )

    del X_train_tfidf, X_test_tfidf
    gc.collect()

# 7. Save comparison
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "Macro_F1",
    ascending=False
)

comparison.to_csv(
    OUTPUT / "experiment_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINAL EXPERIMENT COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))

# 8. Generate comparison graph
comparison.set_index("Experiment")[
    ["Accuracy", "Weighted_F1", "Macro_F1"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("TF-IDF Linear SVM - Controlled Experiments")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    OUTPUT / "experiment_comparison.png",
    dpi=300
)

plt.close()

print("\nExperiments completed successfully!")
print("Results saved in:", OUTPUT)
