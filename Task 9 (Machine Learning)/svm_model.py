
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline
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

SPLIT_FILE = BASE_DIR / "model_results" / "train_test_data.pkl"

OUTPUT_DIR = BASE_DIR / "model_results" / "svm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD TRAIN-TEST DATA
# ==========================================

print("Loading training and testing data...")

data = joblib.load(SPLIT_FILE)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

preprocessor = data["preprocessor"]
classes = data["classes"]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==========================================
# 3. CREATE SVM MODEL
# ==========================================

print("\nCreating SVM model...")

model = Pipeline([
    ("preprocessing", preprocessor),

    ("classifier", LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=42,
        dual=False
    ))
])


# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining SVM model...")

model.fit(X_train, y_train)

print("SVM model training completed!")


# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ==========================================
# 6. EVALUATE MODEL
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
# 7. PRINT RESULTS
# ==========================================

print("\n========== SVM RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as file:
    file.write(report)


# ==========================================
# 9. SAVE METRICS
# ==========================================

metrics = pd.DataFrame([{
    "Model": "SVM",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Weighted_F1": weighted_f1,
    "Macro_F1": macro_f1
}])

metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)


# ==========================================
# 10. SAVE PREDICTIONS
# ==========================================

predictions = pd.DataFrame({
    "Actual_Category": y_test.values,
    "Predicted_Category": y_pred
})

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ==========================================
# 11. CONFUSION MATRIX
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

display.plot(xticks_rotation=45, cmap="Blues")

plt.title("SVM Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 12. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    OUTPUT_DIR / "svm_model.pkl"
)


print("\n================================")
print("SVM MODEL COMPLETED SUCCESSFULLY")
print("================================")
print("All results saved in:", OUTPUT_DIR)
