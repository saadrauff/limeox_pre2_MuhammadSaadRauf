
import json
import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# TASK 12 - AUTOML LIBRARIES FINAL COMPARISON
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "08_Final_Comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIBRARIES = {
    "LazyPredict": [
        RESULTS_DIR / "02_LazyPredict",
    ],
    "AutoGluon": [
        RESULTS_DIR / "03_AutoGluon",
    ],
    "FLAML": [
        BASE_DIR / "04_FLAML",
        RESULTS_DIR / "04_FLAML",
    ],
    "TPOT": [
        RESULTS_DIR / "05_TPOT",
    ],
    "PyCaret": [
        RESULTS_DIR / "06_PyCaret",
    ],
    "H2O AutoML": [
        BASE_DIR / "07_H2O_AutoML",
        RESULTS_DIR / "07_H2O_AutoML",
    ],
}

# Standard comparison metrics.
METRIC_ALIASES = {
    "Accuracy": [
        "accuracy",
        "accuracy_score",
        "acc",
    ],
    "Precision_Macro": [
        "precision_macro",
        "macro_precision",
        "precision_macro_avg",
    ],
    "Recall_Macro": [
        "recall_macro",
        "macro_recall",
        "recall_macro_avg",
    ],
    "F1_Macro": [
        "f1_macro",
        "macro_f1",
        "f1_macro_avg",
        "f1_score_macro",
    ],
    "F1_Weighted": [
        "f1_weighted",
        "weighted_f1",
        "weighted_f1_score",
    ],
    "Precision": [
        "precision",
        "precision_score",
    ],
    "Recall": [
        "recall",
        "recall_score",
    ],
    "F1_Score": [
        "f1_score",
        "f1",
    ],
}


def normalize_key(key):
    """Normalize metric names for matching."""
    return (
        str(key)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def flatten_json(data, result=None):
    """Flatten nested JSON dictionaries."""
    if result is None:
        result = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                flatten_json(value, result)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result[normalize_key(key)] = float(value)

    return result


def find_metrics_file(folder):
    """Find the available saved metrics JSON file."""
    preferred_files = [
        "evaluation_metrics.json",
        "metrics.json",
        "experiment_summary.json",
        "model_metrics.json",
    ]

    for filename in preferred_files:
        path = folder / filename
        if path.exists():
            return path

    # Search nested folders if the preferred file is not found.
    for filename in preferred_files:
        matches = list(folder.rglob(filename))
        if matches:
            return matches[0]

    return None


def load_json_metrics(json_file):
    """Load metrics from a JSON file."""
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        return flatten_json(data)

    except Exception as error:
        print(f"Could not read {json_file}: {error}")
        return {}


def get_metric(flat_data, aliases):
    """Get a metric using its possible saved names."""
    for alias in aliases:
        key = normalize_key(alias)

        if key in flat_data:
            return flat_data[key]

    return None


def load_library(library_name, possible_folders):
    """Load one library's previously saved results."""

    folder = None

    for candidate in possible_folders:
        if candidate.exists():
            folder = candidate
            break

    if folder is None:
        print(f"[MISSING] {library_name}: output folder not found")
        return None

    metrics_file = find_metrics_file(folder)

    if metrics_file is None:
        print(f"[MISSING] {library_name}: metrics JSON not found")
        print(f"          Checked: {folder}")
        return None

    flat_data = load_json_metrics(metrics_file)

    if not flat_data:
        print(f"[ERROR] {library_name}: no numeric metrics found")
        return None

    row = {
        "Library": library_name,
        "Accuracy": get_metric(
            flat_data, METRIC_ALIASES["Accuracy"]
        ),
        "Precision_Macro": get_metric(
            flat_data, METRIC_ALIASES["Precision_Macro"]
        ),
        "Recall_Macro": get_metric(
            flat_data, METRIC_ALIASES["Recall_Macro"]
        ),
        "F1_Macro": get_metric(
            flat_data, METRIC_ALIASES["F1_Macro"]
        ),
        "F1_Weighted": get_metric(
            flat_data, METRIC_ALIASES["F1_Weighted"]
        ),
        "Precision": get_metric(
            flat_data, METRIC_ALIASES["Precision"]
        ),
        "Recall": get_metric(
            flat_data, METRIC_ALIASES["Recall"]
        ),
        "F1_Score": get_metric(
            flat_data, METRIC_ALIASES["F1_Score"]
        ),
        "Metrics_File": str(metrics_file.relative_to(BASE_DIR)),
    }

    print(f"[LOADED] {library_name}")
    print(f"         File: {metrics_file}")
    print(f"         Accuracy: {row['Accuracy']}")

    return row


# ============================================================
# LOAD ALL SIX LIBRARIES
# ============================================================

print("\n" + "=" * 65)
print("TASK 12 - FINAL AUTOML COMPARISON")
print("=" * 65)

all_results = []

for library_name, folders in LIBRARIES.items():
    result = load_library(library_name, folders)

    if result is not None:
        all_results.append(result)


if not all_results:
    raise RuntimeError(
        "No library results were loaded. "
        "Please check the output folders and JSON files."
    )


# ============================================================
# CREATE COMPARISON DATAFRAME
# ============================================================

df = pd.DataFrame(all_results)

metric_columns = [
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "F1_Weighted",
    "Precision",
    "Recall",
    "F1_Score",
]

# Keep all six libraries, including those with missing metrics.
for column in metric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ============================================================
# RANKING
# ============================================================

# Accuracy is used because it is a common metric across
# the six libraries. Missing values are not replaced with zero.

df["Accuracy_Rank"] = (
    df["Accuracy"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)

df = df.sort_values(
    by="Accuracy",
    ascending=False,
    na_position="last"
).reset_index(drop=True)


# ============================================================
# SAVE CSV FILES
# ============================================================

comparison_file = OUTPUT_DIR / "all_libraries_comparison.csv"
ranking_file = OUTPUT_DIR / "automl_ranking.csv"

df.to_csv(comparison_file, index=False)

ranking_df = df[
    ["Accuracy_Rank", "Library", "Accuracy"]
].copy()

ranking_df.to_csv(ranking_file, index=False)

print("\nComparison CSV saved.")
print(comparison_file)


# ============================================================
# EXCEL EXPORT (OPTIONAL)
# ============================================================

excel_file = OUTPUT_DIR / "automl_comparison.xlsx"

try:
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name="All Metrics",
            index=False
        )

        ranking_df.to_excel(
            writer,
            sheet_name="Accuracy Ranking",
            index=False
        )

    print(f"Excel file saved: {excel_file}")

except ImportError:
    print("\n[WARNING] openpyxl is not installed.")
    print("Excel export skipped. CSV files are already saved.")
    print("Install it using: python -m pip install openpyxl")


# ============================================================
# ACCURACY GRAPH
# ============================================================

accuracy_data = df.dropna(subset=["Accuracy"]).copy()

if not accuracy_data.empty:
    plt.figure(figsize=(11, 6))

    plt.bar(
        accuracy_data["Library"],
        accuracy_data["Accuracy"]
    )

    plt.title("Accuracy Comparison - Six AutoML Libraries")
    plt.xlabel("AutoML Libraries")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for index, value in enumerate(accuracy_data["Accuracy"]):
        plt.text(
            index,
            value + 0.01,
            f"{value:.4f}",
            ha="center"
        )

    plt.tight_layout()

    accuracy_graph = OUTPUT_DIR / "accuracy_comparison.png"
    plt.savefig(accuracy_graph, dpi=300)
    plt.close()

    print(f"Accuracy graph saved: {accuracy_graph}")


# ============================================================
# F1 SCORE GRAPH
# ============================================================

f1_columns = [
    column for column in ["F1_Macro", "F1_Weighted"]
    if df[column].notna().any()
]

if f1_columns:
    f1_data = df[["Library"] + f1_columns].copy()
    f1_data = f1_data.set_index("Library")

    ax = f1_data.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.75
    )

    ax.set_title("F1 Score Comparison - Six AutoML Libraries")
    ax.set_xlabel("AutoML Libraries")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Metric")

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    f1_graph = OUTPUT_DIR / "f1_score_comparison.png"
    plt.savefig(f1_graph, dpi=300)
    plt.close()

    print(f"F1 graph saved: {f1_graph}")


# ============================================================
# GENERATE TEXT SUMMARY
# ============================================================

summary_file = OUTPUT_DIR / "comparison_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("=" * 65 + "\n")
    file.write("TASK 12 - FINAL AUTOML COMPARISON SUMMARY\n")
    file.write("=" * 65 + "\n\n")

    file.write(f"Libraries loaded: {len(df)} / 6\n\n")

    file.write("ACCURACY RANKING\n")
    file.write("-" * 65 + "\n")

    for _, row in ranking_df.iterrows():
        accuracy = row["Accuracy"]

        if pd.notna(accuracy):
            file.write(
                f"{int(row['Accuracy_Rank'])}. "
                f"{row['Library']}: {accuracy:.6f}\n"
            )
        else:
            file.write(f"- {row['Library']}: Not available\n")

    file.write("\n\nCOMPLETE METRICS\n")
    file.write("-" * 65 + "\n")

    for _, row in df.iterrows():
        file.write(f"\nLibrary: {row['Library']}\n")

        for column in metric_columns:
            value = row[column]

            if pd.notna(value):
                file.write(f"{column}: {value:.6f}\n")
            else:
                file.write(f"{column}: Not available\n")

    file.write("\n\nNOTES\n")
    file.write("-" * 65 + "\n")
    file.write(
        "1. Accuracy ranking uses the common Accuracy metric.\n"
    )
    file.write(
        "2. Missing metrics are left blank, not replaced with zero.\n"
    )
    file.write(
        "3. Generic Precision, Recall and F1 scores are kept "
        "separate from Macro and Weighted metrics.\n"
    )
    file.write(
        "4. This script reads saved results only; it does not "
        "train any model again.\n"
    )

print(f"Summary saved: {summary_file}")


# ============================================================
# FINAL DISPLAY
# ============================================================

print("\n" + "=" * 65)
print("FINAL COMPARISON TABLE")
print("=" * 65)

print(
    df[
        [
            "Library",
            "Accuracy",
            "Precision_Macro",
            "Recall_Macro",
            "F1_Macro",
            "F1_Weighted",
            "Accuracy_Rank",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 65)
print(f"LIBRARIES LOADED: {len(df)} / 6")
print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}")
print("COMPARISON COMPLETED")
print("=" * 65)
