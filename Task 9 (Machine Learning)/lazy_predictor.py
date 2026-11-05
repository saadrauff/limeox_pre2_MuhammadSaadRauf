
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

SPLIT_FILE = BASE_DIR / "model_results" / "train_test_data.pkl"

OUTPUT_DIR = BASE_DIR / "model_results" / "lazy_predictor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading dataset...")

data = joblib.load(SPLIT_FILE)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

preprocessor = data["preprocessor"]

print("Original training data:", X_train.shape)
print("Original testing data:", X_test.shape)


# ==========================================
# 3. SELECT SAMPLE DATA
# ==========================================

print("\nSelecting training and testing samples...")

X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=20000,
    stratify=y_train,
    random_state=42
)

X_test_sample, _, y_test_sample, _ = train_test_split(
    X_test,
    y_test,
    train_size=5000,
    stratify=y_test,
    random_state=42
)

print("Sample training data:", X_train_sample.shape)
print("Sample testing data:", X_test_sample.shape)


# ==========================================
# 4. PREPROCESS DATA
# ==========================================

print("\nApplying preprocessing...")

X_train_processed = preprocessor.fit_transform(
    X_train_sample,
    y_train_sample
)

X_test_processed = preprocessor.transform(
    X_test_sample
)

# Convert sparse matrices to dense arrays if needed

if issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()

if issparse(X_test_processed):
    X_test_processed = X_test_processed.toarray()

print("Preprocessing completed!")


# ==========================================
# 5. CREATE LAZY CLASSIFIER
# ==========================================

print("\nInitializing LazyPredict...")

clf = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    predictions=True
)


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

print("\nTraining multiple machine learning models...")
print("This may take some time...")

models, predictions = clf.fit(
    X_train_processed,
    X_test_processed,
    y_train_sample,
    y_test_sample
)


# ==========================================
# 7. SORT AND DISPLAY RESULTS
# ==========================================

print("\n========== LAZYPREDICT RESULTS ==========")

# Sort models by Accuracy (Highest to Lowest)

models = models.sort_values(
    by="Accuracy",
    ascending=False
)

print(models.to_string())


# Save sorted model comparison

models.to_csv(
    OUTPUT_DIR / "lazy_predictor_results.csv"
)


# ==========================================
# 8. SAVE PREDICTIONS
# ==========================================

if isinstance(predictions, pd.DataFrame):

    predictions.to_csv(
        OUTPUT_DIR / "lazy_predictor_predictions.csv",
        index=False
    )


# ==========================================
# 9. SAVE SUMMARY
# ==========================================

with open(OUTPUT_DIR / "summary.txt", "w") as file:

    file.write("LAZYPREDICT MODEL COMPARISON\n")
    file.write("============================\n\n")

    file.write(f"Training sample size: {len(y_train_sample)}\n")
    file.write(f"Testing sample size: {len(y_test_sample)}\n\n")

    file.write("Models sorted by Accuracy (Highest to Lowest)\n\n")

    file.write(models.to_string())


# ==========================================
# 10. COMPLETION
# ==========================================

print("\n====================================")
print("LAZYPREDICT COMPLETED SUCCESSFULLY")
print("====================================")

print("\nModels sorted by highest Accuracy!")

print("Results saved in:", OUTPUT_DIR)
