
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "model_results" / "train_test_data.pkl"

OUTPUT_DIR = BASE_DIR / "model_results" / "logistic_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training and testing data...")

data = joblib.load(DATA_FILE)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
preprocessor = data["preprocessor"]

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining Logistic Regression (Softmax)...")
model.fit(X_train, y_train)

print("Making predictions...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)
recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Weighted F1: {f1:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")

report = classification_report(
    y_test, y_pred, zero_division=0
)

print("\nClassification Report:")
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as file:
    file.write(report)

metrics = pd.DataFrame([{
    "Model": "Logistic Regression (Softmax)",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Weighted_F1": f1,
    "Macro_F1": macro_f1
}])

metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

predictions = pd.DataFrame({
    "Actual_Category": y_test.values,
    "Predicted_Category": y_pred
})

predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

cm = confusion_matrix(y_test, y_pred, labels=data["classes"])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data["classes"]
)

display.plot(xticks_rotation=45, cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

joblib.dump(model, OUTPUT_DIR / "logistic_regression_model.pkl")

print("\nLogistic Regression completed successfully!")
print("Results saved in:", OUTPUT_DIR)
