
import re
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_results"

TRAIN_FILE = MODEL_DIR / "news_train_augmented.csv"
TEST_FILE = MODEL_DIR / "news_test_original.csv"

OUTPUT_DIR = MODEL_DIR / "text_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# 2. LOAD DATA
# ==========================================

print("=" * 65)
print("LIME OX TASK 9 - TF-IDF TEXT CLASSIFICATION")
print("=" * 65)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# ==========================================
# 3. VALIDATE DATA
# ==========================================

required_columns = [
    "title_clean",
    "content_clean",
    "category"
]

for col in required_columns:
    if col not in train_df.columns or col not in test_df.columns:
        raise ValueError(f"Required column missing: {col}")

train_df = train_df.dropna(
    subset=["category"]
).copy()

test_df = test_df.dropna(
    subset=["category"]
).copy()

train_df["category"] = train_df["category"].astype(str).str.strip()
test_df["category"] = test_df["category"].astype(str).str.strip()

train_df = train_df[train_df["category"] != ""].copy()
test_df = test_df[test_df["category"] != ""].copy()

print("\nValid training records:", len(train_df))
print("Valid testing records:", len(test_df))


# ==========================================
# 4. CLEAN TEXT
# ==========================================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = value.replace("\x00", " ")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


print("\nCleaning article text...")

for df in [train_df, test_df]:

    df["title_clean"] = df["title_clean"].apply(clean_text)
    df["content_clean"] = df["content_clean"].apply(clean_text)

    # Give article titles additional importance.
    df["combined_text"] = (
        df["title_clean"] + " " +
        df["title_clean"] + " " +
        df["content_clean"]
    )


# ==========================================
# 5. REMOVE EXACT DUPLICATES
# ==========================================

print("\nChecking exact duplicate articles...")

train_df["_text_key"] = (
    train_df["combined_text"].str.lower().str.strip()
)

test_df["_text_key"] = (
    test_df["combined_text"].str.lower().str.strip()
)

before = len(train_df)

train_df = train_df.drop_duplicates(
    subset=["_text_key"],
    keep="first"
).copy()

print("Training duplicates removed:", before - len(train_df))


# ==========================================
# 6. PREVENT TRAIN-TEST TITLE OVERLAP
# ==========================================

print("\nChecking train-test title overlap...")

train_df["_title_key"] = train_df["title_clean"].str.lower().str.strip()
test_df["_title_key"] = test_df["title_clean"].str.lower().str.strip()

test_titles = set(test_df["_title_key"])

overlap_mask = train_df["_title_key"].isin(test_titles)

overlap_count = int(overlap_mask.sum())

print("Overlapping training records:", overlap_count)

train_df = train_df.loc[~overlap_mask].copy()

print("Training records after cleaning:", len(train_df))


# ==========================================
# 7. CHECK CATEGORY COMPATIBILITY
# ==========================================

classes = sorted(train_df["category"].unique())

unknown_categories = (
    set(test_df["category"].unique()) - set(classes)
)

if unknown_categories:
    raise ValueError(
        f"Testing contains unknown categories: {unknown_categories}"
    )

print("\nCategories:", classes)

print("\nTraining distribution:")
print(train_df["category"].value_counts())

print("\nTesting distribution:")
print(test_df["category"].value_counts())


# ==========================================
# 8. PREPARE TEXT AND LABELS
# ==========================================

X_train_text = train_df["combined_text"].fillna("").astype(str)
X_test_text = test_df["combined_text"].fillna("").astype(str)

y_train = train_df["category"].values
y_test = test_df["category"].values

del train_df, test_df

gc.collect()


# ==========================================
# 9. WORD-LEVEL TF-IDF
# ==========================================

print("\n" + "=" * 65)
print("CREATING WORD-LEVEL TF-IDF FEATURES")
print("=" * 65)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=200000,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32
)

print("\nFitting TF-IDF on training text...")

X_train_word = word_vectorizer.fit_transform(X_train_text)

print("Training TF-IDF shape:", X_train_word.shape)

print("\nTransforming testing text...")

X_test_word = word_vectorizer.transform(X_test_text)

print("Testing TF-IDF shape:", X_test_word.shape)

del X_train_text, X_test_text

gc.collect()


# ==========================================
# 10. TRAIN LINEAR SVM
# ==========================================

print("\n" + "=" * 65)
print("TRAINING LINEAR SVM")
print("=" * 65)

model = LinearSVC(
    C=1.0,
    class_weight=None,
    max_iter=10000,
    random_state=RANDOM_STATE,
    dual="auto",
    tol=1e-4
)

print("\nStarting model training...")

model.fit(X_train_word, y_train)

print("\nSVM training completed!")


# ==========================================
# 11. PREDICTIONS
# ==========================================

print("\nMaking testing predictions...")

y_pred = model.predict(X_test_word)

print("Predictions completed!")


# ==========================================
# 12. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)


# ==========================================
# 13. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 65)
print("TF-IDF LINEAR SVM RESULTS")
print("=" * 65)

print(f"\nAccuracy:       {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")
print(f"Macro F1:       {macro_f1:.4f}")

report = classification_report(
    y_test,
    y_pred,
    labels=classes,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ==========================================
# 14. SAVE REPORT
# ==========================================

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ==========================================
# 15. SAVE METRICS
# ==========================================

metrics = pd.DataFrame([{
    "Model": "TF-IDF Linear SVM",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Weighted_F1": weighted_f1,
    "Macro_F1": macro_f1,
    "Training_Records": len(y_train),
    "Testing_Records": len(y_test)
}])

metrics.to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False
)


# ==========================================
# 16. SAVE PREDICTIONS
# ==========================================

predictions = pd.DataFrame({
    "Actual_Category": y_test,
    "Predicted_Category": y_pred,
    "Correct": y_test == y_pred
})

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ==========================================
# 17. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("TF-IDF Linear SVM - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 18. SAVE TRAINED MODEL
# ==========================================

print("\nSaving trained model...")

joblib.dump(
    {
        "vectorizer": word_vectorizer,
        "model": model,
        "classes": classes
    },
    OUTPUT_DIR / "tfidf_svm_model.pkl",
    compress=3
)


# ==========================================
# 19. FINAL REPORT
# ==========================================

print("\n" + "=" * 65)
print("TF-IDF TEXT CLASSIFICATION COMPLETED")
print("=" * 65)

print("\nTraining records:", len(y_train))
print("Testing records:", len(y_test))

print(f"Final accuracy: {accuracy * 100:.2f}%")

print("\nSaved files:")

for file in OUTPUT_DIR.iterdir():
    print("-", file.name)

print("\nResults directory:")
print(OUTPUT_DIR)

print("\nNext step: Compare TF-IDF results with existing ML models.")
