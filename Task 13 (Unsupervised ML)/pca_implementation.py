
import gc
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "Task 8 (Feature Engineering)"
    / "news_engineered_features.csv"
)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "pca_results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

EXPLAINED_VARIANCE_THRESHOLD = 0.95

VISUALIZATION_SAMPLE_SIZE = 10000

METADATA_COLUMNS = ["url", "category"]


# ============================================================
# 2. CHECK INPUT DATASET
# ============================================================

print("\n" + "=" * 65)
print("LOADING FEATURE ENGINEERED DATASET")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

total_articles = len(df)

print(f"Dataset shape: {df.shape}")
print(f"Total articles: {total_articles}")


# ============================================================
# 3. SELECT METADATA
# ============================================================

metadata_columns = [
    col for col in METADATA_COLUMNS
    if col in df.columns
]

metadata = df[metadata_columns].copy()

print("\nMetadata columns:", metadata_columns)


# ============================================================
# 4. SELECT NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 65)
print("SELECTING NUMERICAL FEATURES")
print("=" * 65)

numerical_df = df.select_dtypes(
    include=[np.number]
).copy()

# Remove columns containing only missing values
numerical_df = numerical_df.dropna(
    axis=1,
    how="all"
)

# Remove constant features
constant_columns = [
    col for col in numerical_df.columns
    if numerical_df[col].nunique(dropna=True) <= 1
]

numerical_df.drop(
    columns=constant_columns,
    inplace=True
)

feature_columns = numerical_df.columns.tolist()

if len(feature_columns) < 2:
    raise ValueError(
        "At least two valid numerical features are required for PCA."
    )

X = numerical_df.to_numpy(dtype=np.float64)

print(f"Selected features: {len(feature_columns)}")

print("Feature names:")

for feature in feature_columns:
    print(f"- {feature}")

print(f"\nFeature matrix shape: {X.shape}")


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 65)
print("HANDLING MISSING VALUES")
print("=" * 65)

missing_values_before = int(
    np.isnan(X).sum()
)

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X)

missing_values_after = int(
    np.isnan(X_imputed).sum()
)

print("Missing values handled.")
print(f"Missing values before: {missing_values_before}")
print(f"Missing values after: {missing_values_after}")
print(f"Imputed shape: {X_imputed.shape}")


# ============================================================
# 6. APPLY STANDARD SCALER
# ============================================================

print("\n" + "=" * 65)
print("APPLYING STANDARD SCALER")
print("=" * 65)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("Scaling completed.")
print(f"Scaled shape: {X_scaled.shape}")


# ============================================================
# 7. APPLY PCA ON COMPLETE DATASET
# ============================================================

print("\n" + "=" * 65)
print("APPLYING PCA ON COMPLETE DATASET")
print("=" * 65)

print(f"Original number of features: {X_scaled.shape[1]}")
print(f"Total articles: {X_scaled.shape[0]}")

pca = PCA(
    n_components=EXPLAINED_VARIANCE_THRESHOLD,
    svd_solver="full"
)

X_pca = pca.fit_transform(X_scaled)

reduced_features = pca.n_components_

print("\nPCA completed successfully.")
print(f"Original shape: {X_scaled.shape}")
print(f"Reduced shape: {X_pca.shape}")
print(f"Original features: {len(feature_columns)}")
print(f"PCA components: {reduced_features}")


# ============================================================
# 8. EXPLAINED VARIANCE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("EXPLAINED VARIANCE ANALYSIS")
print("=" * 65)

explained_variance = pca.explained_variance_ratio_

explained_variance_percentage = (
    explained_variance * 100
)

cumulative_variance = np.cumsum(
    explained_variance
)

cumulative_variance_percentage = (
    cumulative_variance * 100
)

total_explained_variance = (
    np.sum(explained_variance) * 100
)

variance_lost = (
    100 - total_explained_variance
)

dimensionality_reduction = (
    (len(feature_columns) - reduced_features)
    / len(feature_columns)
) * 100

print("\nVariance explained by each component:\n")

for i, variance in enumerate(
    explained_variance_percentage,
    start=1
):
    print(f"PC{i}: {variance:.4f}%")

print(
    f"\nTotal Explained Variance: "
    f"{total_explained_variance:.4f} %"
)

print(
    f"Variance Lost: {variance_lost:.4f} %"
)


# ============================================================
# 9. CREATE VARIANCE REPORT
# ============================================================

variance_report = pd.DataFrame({
    "Principal_Component": [
        f"PC{i}" for i in range(1, reduced_features + 1)
    ],
    "Explained_Variance": explained_variance,
    "Explained_Variance_Percentage": explained_variance_percentage,
    "Cumulative_Variance": cumulative_variance,
    "Cumulative_Variance_Percentage": cumulative_variance_percentage
})

variance_report_path = (
    RESULTS_DIR / "pca_explained_variance.csv"
)

variance_report.to_csv(
    variance_report_path,
    index=False
)

print("\nVariance report saved.")


# ============================================================
# 10. SAVE PCA TRANSFORMED DATASET
# ============================================================

print("\n" + "=" * 65)
print("SAVING PCA TRANSFORMED DATASET")
print("=" * 65)

pca_columns = [
    f"PC{i}"
    for i in range(1, reduced_features + 1)
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)

# Preserve metadata alongside PCA components
pca_df = pd.concat(
    [
        metadata.reset_index(drop=True),
        pca_df
    ],
    axis=1
)

transformed_dataset_path = (
    RESULTS_DIR / "pca_transformed_dataset.csv"
)

pca_df.to_csv(
    transformed_dataset_path,
    index=False
)

print(f"PCA dataset saved: {transformed_dataset_path}")
print(f"Saved dataset shape: {pca_df.shape}")


# ============================================================
# 11. GENERATE PCA VISUALIZATION
# ============================================================

print("\n" + "=" * 65)
print("GENERATING PCA VISUALIZATION")
print("=" * 65)

if reduced_features >= 2:

    sample_size = min(
        VISUALIZATION_SAMPLE_SIZE,
        len(X_pca)
    )

    rng = np.random.default_rng(RANDOM_STATE)

    sample_indices = rng.choice(
        len(X_pca),
        size=sample_size,
        replace=False
    )

    X_visualization = X_pca[sample_indices]

    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        X_visualization[:, 0],
        X_visualization[:, 1],
        alpha=0.5,
        s=10
    )

    plt.xlabel(
        f"PC1 ({explained_variance_percentage[0]:.2f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance_percentage[1]:.2f}% variance)"
    )

    plt.title(
        "PCA Visualization - First Two Principal Components"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    visualization_path = (
        RESULTS_DIR / "pca_visualization.png"
    )

    plt.savefig(
        visualization_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("PCA visualization saved.")

else:
    visualization_path = None
    print("Visualization skipped: fewer than two PCA components.")


# ============================================================
# 12. SAVE PCA MODEL AND PREPROCESSING OBJECTS
# ============================================================

print("\n" + "=" * 65)
print("SAVING PCA MODEL")
print("=" * 65)

model_data = {
    "imputer": imputer,
    "scaler": scaler,
    "pca": pca,
    "feature_columns": feature_columns,
    "metadata_columns": metadata_columns,
    "explained_variance_threshold": EXPLAINED_VARIANCE_THRESHOLD,
    "random_state": RANDOM_STATE
}

model_path = (
    RESULTS_DIR / "pca_model.pkl"
)

joblib.dump(
    model_data,
    model_path
)

print(f"PCA model saved: {model_path}")


# ============================================================
# 13. GENERATE COMPLETE PCA SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("GENERATING COMPLETE PCA SUMMARY REPORT")
print("=" * 65)

execution_date = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

component_details = []

for i in range(reduced_features):

    component_details.append({
        "component": f"PC{i + 1}",
        "explained_variance_percentage": round(
            float(explained_variance_percentage[i]), 4
        ),
        "cumulative_variance_percentage": round(
            float(cumulative_variance_percentage[i]), 4
        )
    })


summary = {

    "execution_information": {
        "execution_date": execution_date,
        "task": "Task 13 - Unsupervised Machine Learning",
        "technique": "Principal Component Analysis (PCA)"
    },

    "dataset_information": {
        "input_dataset": str(DATA_FILE),
        "total_articles": int(total_articles),
        "original_dataset_rows": int(df.shape[0]),
        "original_dataset_columns": int(df.shape[1]),
        "selected_numerical_features": int(len(feature_columns)),
        "feature_names": feature_columns,
        "metadata_columns": metadata_columns
    },

    "preprocessing": {
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "missing_value_strategy": "Median Imputation",
        "scaling_method": "StandardScaler",
        "imputed_shape": list(X_imputed.shape),
        "scaled_shape": list(X_scaled.shape)
    },

    "pca_configuration": {
        "algorithm": "Principal Component Analysis",
        "variance_threshold": EXPLAINED_VARIANCE_THRESHOLD,
        "solver": pca.svd_solver,
        "random_state": RANDOM_STATE,
        "original_dimensions": int(len(feature_columns)),
        "reduced_dimensions": int(reduced_features)
    },

    "pca_results": {
        "total_articles_processed": int(len(X_pca)),
        "original_features": int(len(feature_columns)),
        "final_components": int(reduced_features),
        "total_explained_variance_percentage": round(
            float(total_explained_variance), 4
        ),
        "variance_lost_percentage": round(
            float(variance_lost), 4
        ),
        "dimensionality_reduction_percentage": round(
            float(dimensionality_reduction), 4
        ),
        "transformed_dataset_shape": list(pca_df.shape),
        "component_wise_variance": component_details
    },

    "output_files": {
        "transformed_dataset": transformed_dataset_path.name,
        "explained_variance_report": variance_report_path.name,
        "trained_pca_model": model_path.name,
        "visualization": (
            visualization_path.name
            if visualization_path is not None
            else None
        ),
        "summary_text": "pca_summary.txt",
        "summary_json": "pca_summary.json"
    }
}


# ------------------------------------------------------------
# SAVE JSON SUMMARY
# ------------------------------------------------------------

json_summary_path = (
    RESULTS_DIR / "pca_summary.json"
)

with open(
    json_summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# SAVE HUMAN-READABLE TEXT SUMMARY
# ------------------------------------------------------------

txt_summary_path = (
    RESULTS_DIR / "pca_summary.txt"
)

with open(
    txt_summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 70 + "\n")
    file.write("PCA IMPLEMENTATION - COMPLETE SUMMARY REPORT\n")
    file.write("=" * 70 + "\n\n")

    file.write("1. EXECUTION INFORMATION\n")
    file.write("-" * 70 + "\n")
    file.write(f"Execution Date: {execution_date}\n")
    file.write("Task: Task 13 - Unsupervised Machine Learning\n")
    file.write("Technique: Principal Component Analysis (PCA)\n\n")

    file.write("2. DATASET INFORMATION\n")
    file.write("-" * 70 + "\n")
    file.write(f"Input Dataset: {DATA_FILE}\n")
    file.write(f"Total Articles: {total_articles}\n")
    file.write(f"Original Rows: {df.shape[0]}\n")
    file.write(f"Original Columns: {df.shape[1]}\n")
    file.write(f"Numerical Features Selected: {len(feature_columns)}\n\n")

    file.write("3. ORIGINAL NUMERICAL FEATURES\n")
    file.write("-" * 70 + "\n")

    for index, feature in enumerate(feature_columns, start=1):
        file.write(f"{index}. {feature}\n")

    file.write("\n4. PREPROCESSING DETAILS\n")
    file.write("-" * 70 + "\n")
    file.write("Missing Value Handling: Median Imputation\n")
    file.write(f"Missing Values Before: {missing_values_before}\n")
    file.write(f"Missing Values After: {missing_values_after}\n")
    file.write("Feature Scaling: StandardScaler\n")
    file.write(f"Imputed Matrix Shape: {X_imputed.shape}\n")
    file.write(f"Scaled Matrix Shape: {X_scaled.shape}\n\n")

    file.write("5. PCA CONFIGURATION\n")
    file.write("-" * 70 + "\n")
    file.write("Algorithm: Principal Component Analysis\n")
    file.write(
        f"Explained Variance Threshold: "
        f"{EXPLAINED_VARIANCE_THRESHOLD * 100:.2f}%\n"
    )
    file.write(f"PCA Solver: {pca.svd_solver}\n")
    file.write(f"Original Dimensions: {len(feature_columns)}\n")
    file.write(f"Reduced Dimensions: {reduced_features}\n\n")

    file.write("6. COMPONENT-WISE EXPLAINED VARIANCE\n")
    file.write("-" * 70 + "\n")

    for detail in component_details:

        file.write(
            f"{detail['component']}: "
            f"{detail['explained_variance_percentage']:.4f}%"
            f" | Cumulative: "
            f"{detail['cumulative_variance_percentage']:.4f}%\n"
        )

    file.write("\n7. FINAL PCA RESULTS\n")
    file.write("-" * 70 + "\n")
    file.write(f"Total Articles Processed: {len(X_pca)}\n")
    file.write(f"Original Features: {len(feature_columns)}\n")
    file.write(f"Final PCA Components: {reduced_features}\n")
    file.write(
        f"Total Explained Variance: "
        f"{total_explained_variance:.4f}%\n"
    )
    file.write(f"Variance Lost: {variance_lost:.4f}%\n")
    file.write(
        f"Dimensionality Reduction: "
        f"{dimensionality_reduction:.4f}%\n"
    )
    file.write(
        f"Transformed Dataset Shape: {pca_df.shape}\n"
    )

    file.write("\n8. GENERATED OUTPUT FILES\n")
    file.write("-" * 70 + "\n")

    for name, filename in summary["output_files"].items():
        if filename is not None:
            file.write(f"{name}: {filename}\n")

    file.write("\n" + "=" * 70 + "\n")
    file.write("END OF PCA SUMMARY REPORT\n")
    file.write("=" * 70 + "\n")


print("TXT summary saved:", txt_summary_path)
print("JSON summary saved:", json_summary_path)


# ============================================================
# 14. FINAL EXECUTION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PCA IMPLEMENTATION COMPLETED")
print("=" * 65)

print(f"Total Articles: {total_articles}")
print(f"Original Numerical Features: {len(feature_columns)}")
print(f"Reduced PCA Features: {reduced_features}")

print(
    f"Explained Variance: "
    f"{total_explained_variance:.4f}%"
)

print(
    f"Variance Lost: "
    f"{variance_lost:.4f}%"
)

print(
    f"Dimensionality Reduction: "
    f"{dimensionality_reduction:.4f}%"
)

print(f"\nResults Directory: {RESULTS_DIR}")

print("\nGenerated Files:")

for path in sorted(RESULTS_DIR.iterdir()):
    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)

        print(
            f"{path.name} - {size_mb:.2f} MB"
        )

print("\nTASK 13 PCA EXECUTION FINISHED!")
print("=" * 65)

gc.collect()
