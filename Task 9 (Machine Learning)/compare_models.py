
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "model_results"

model_folders = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn"
]

all_results = []

for folder in model_folders:
    metrics_file = RESULTS_DIR / folder / "metrics.csv"

    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        all_results.append(df)
    else:
        print("Missing:", metrics_file)

if not all_results:
    raise FileNotFoundError("No model metrics files found.")

comparison = pd.concat(all_results, ignore_index=True)

comparison = comparison.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    RESULTS_DIR / "final_model_comparison.csv",
    index=False
)

metrics = ["Accuracy", "Precision", "Recall", "Weighted_F1", "Macro_F1"]

ax = comparison.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

ax.set_title("Machine Learning Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "model_comparison.png",
    dpi=300
)

plt.close()

print("\nComparison completed successfully!")
print("Saved:", RESULTS_DIR / "final_model_comparison.csv")
print("Saved:", RESULTS_DIR / "model_comparison.png")
