
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PCA_DIR = BASE_DIR / "pca_results"
KMEANS_DIR = BASE_DIR / "pca_kmeans_results"
DBSCAN_DIR = BASE_DIR / "pca_dbscan_results"
HIERARCHICAL_DIR = BASE_DIR / "pca_hierarchical_results"

OUTPUT_DIR = BASE_DIR / "pca_clustering_comparison_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def read_text(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path.read_text(encoding="utf-8")


def extract_number(text, pattern, field):
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Could not extract {field} from summary file."
        )

    return float(match.group(1))


def extract_distribution(text):
    distribution = {}

    for line in text.splitlines():

        match = re.search(
            r"^\s*(Noise|Cluster\s+\d+)\s*:\s*([\d,]+)",
            line,
            re.IGNORECASE
        )

        if match:
            name = match.group(1).strip()
            count = int(match.group(2).replace(",", ""))

            distribution[name] = count

    return distribution


def clean_json(value):

    if isinstance(value, dict):
        return {
            str(k): clean_json(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [clean_json(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ============================================================
# 3. LOAD SUMMARY FILES
# ============================================================

print("\n" + "=" * 65)
print("PCA CLUSTERING COMPARISON")
print("=" * 65)

kmeans_text = read_text(
    KMEANS_DIR / "kmeans_summary.txt"
)

dbscan_text = read_text(
    DBSCAN_DIR / "dbscan_summary.txt"
)

hierarchical_text = read_text(
    HIERARCHICAL_DIR / "hierarchical_summary.txt"
)

print("\nAll three summary files loaded successfully.")


# ============================================================
# 4. EXTRACT PCA INFORMATION
# ============================================================

pca_dataset = PCA_DIR / "pca_transformed_dataset.csv"

if not pca_dataset.exists():
    raise FileNotFoundError(pca_dataset)

pca_header = pd.read_csv(pca_dataset, nrows=0)

pca_columns = [
    column for column in pca_header.columns
    if re.match(r"^PC\d+$", str(column), re.IGNORECASE)
]

pca_components = len(pca_columns)

# Count complete dataset records
with open(pca_dataset, "r", encoding="utf-8-sig") as file:
    total_records = sum(1 for _ in file) - 1

variance_file = PCA_DIR / "pca_explained_variance.csv"

variance_df = pd.read_csv(variance_file)

# Identify cumulative variance or explained variance ratio
cumulative_columns = [
    c for c in variance_df.columns
    if "cumulative" in c.lower()
]

ratio_columns = [
    c for c in variance_df.columns
    if "ratio" in c.lower()
    and "cumulative" not in c.lower()
]

if cumulative_columns:

    values = pd.to_numeric(
        variance_df[cumulative_columns[0]],
        errors="coerce"
    ).dropna()

    explained_variance = float(values.iloc[-1])

elif ratio_columns:

    values = pd.to_numeric(
        variance_df[ratio_columns[0]],
        errors="coerce"
    ).dropna()

    explained_variance = float(values.sum())

else:
    raise ValueError(
        "Explained variance column not found in PCA report."
    )

# Support both decimal and percentage formats
if explained_variance <= 1.5:
    explained_variance *= 100

variance_lost = 100 - explained_variance

original_features = 25

print("\nPCA INFORMATION")
print("-" * 65)
print("Total records:", total_records)
print("Original features:", original_features)
print("PCA components:", pca_components)
print("Explained variance:", round(explained_variance, 4), "%")
print("Variance lost:", round(variance_lost, 4), "%")


# ============================================================
# 5. EXTRACT K-MEANS METRICS
# ============================================================

kmeans_silhouette = extract_number(
    kmeans_text,
    r"Silhouette Score:\s*([-+]?\d*\.?\d+)",
    "K-Means Silhouette"
)

kmeans_db = extract_number(
    kmeans_text,
    r"Davies-Bouldin Index:\s*([-+]?\d*\.?\d+)",
    "K-Means Davies-Bouldin"
)

kmeans_ch = extract_number(
    kmeans_text,
    r"Calinski-Harabasz Score:\s*([-+]?\d*\.?\d+)",
    "K-Means Calinski-Harabasz"
)

kmeans_k = int(extract_number(
    kmeans_text,
    r"Selected K:\s*(\d+)",
    "K-Means selected K"
))

kmeans_distribution = extract_distribution(kmeans_text)

print("\nK-MEANS METRICS EXTRACTED")
print("Selected K:", kmeans_k)
print("Silhouette:", kmeans_silhouette)
print("Davies-Bouldin:", kmeans_db)
print("Calinski-Harabasz:", kmeans_ch)


# ============================================================
# 6. EXTRACT DBSCAN METRICS
# ============================================================

dbscan_silhouette = extract_number(
    dbscan_text,
    r"Silhouette Score:\s*([-+]?\d*\.?\d+)",
    "DBSCAN Silhouette"
)

dbscan_db = extract_number(
    dbscan_text,
    r"Davies-Bouldin Index:\s*([-+]?\d*\.?\d+)",
    "DBSCAN Davies-Bouldin"
)

dbscan_ch = extract_number(
    dbscan_text,
    r"Calinski-Harabasz Score:\s*([-+]?\d*\.?\d+)",
    "DBSCAN Calinski-Harabasz"
)

dbscan_clusters = int(extract_number(
    dbscan_text,
    r"Number of Clusters:\s*(\d+)",
    "DBSCAN clusters"
))

dbscan_noise = int(extract_number(
    dbscan_text,
    r"Noise Points:\s*([\d,]+)",
    "DBSCAN noise"
))

dbscan_eps = extract_number(
    dbscan_text,
    r"EPS:\s*([-+]?\d*\.?\d+)",
    "DBSCAN EPS"
)

dbscan_min_samples = int(extract_number(
    dbscan_text,
    r"Min Samples:\s*(\d+)",
    "DBSCAN Min Samples"
))

dbscan_distribution = extract_distribution(dbscan_text)

print("\nDBSCAN METRICS EXTRACTED")
print("Clusters:", dbscan_clusters)
print("Noise:", dbscan_noise)
print("Silhouette:", dbscan_silhouette)
print("Davies-Bouldin:", dbscan_db)
print("Calinski-Harabasz:", dbscan_ch)


# ============================================================
# 7. EXTRACT HIERARCHICAL METRICS
# ============================================================

hierarchical_silhouette = extract_number(
    hierarchical_text,
    r"Silhouette Score:\s*([-+]?\d*\.?\d+)",
    "Hierarchical Silhouette"
)

hierarchical_db = extract_number(
    hierarchical_text,
    r"Davies-Bouldin Index:\s*([-+]?\d*\.?\d+)",
    "Hierarchical Davies-Bouldin"
)

hierarchical_ch = extract_number(
    hierarchical_text,
    r"Calinski-Harabasz Score:\s*([-+]?\d*\.?\d+)",
    "Hierarchical Calinski-Harabasz"
)

hierarchical_clusters = int(extract_number(
    hierarchical_text,
    r"Number of Clusters:\s*(\d+)",
    "Hierarchical clusters"
))

hierarchical_records = int(extract_number(
    hierarchical_text,
    r"Sample Size:\s*([\d,]+)",
    "Hierarchical sample size"
))

hierarchical_distribution = extract_distribution(
    hierarchical_text
)

print("\nHIERARCHICAL METRICS EXTRACTED")
print("Records:", hierarchical_records)
print("Clusters:", hierarchical_clusters)
print("Silhouette:", hierarchical_silhouette)
print("Davies-Bouldin:", hierarchical_db)
print("Calinski-Harabasz:", hierarchical_ch)


# ============================================================
# 8. CREATE MASTER COMPARISON TABLE
# ============================================================

comparison_data = [

    {
        "Algorithm": "K-Means",
        "Total_Records": total_records,
        "Records_Processed": total_records,
        "Coverage_Percent": 100.0,
        "PCA_Features": pca_components,
        "Clusters": kmeans_k,
        "Noise_Points": 0,
        "Noise_Percent": 0.0,
        "Silhouette": kmeans_silhouette,
        "Davies_Bouldin": kmeans_db,
        "Calinski_Harabasz": kmeans_ch,
        "Evaluation": "Random sample of 3000"
    },

    {
        "Algorithm": "DBSCAN",
        "Total_Records": total_records,
        "Records_Processed": total_records,
        "Coverage_Percent": 100.0,
        "PCA_Features": pca_components,
        "Clusters": dbscan_clusters,
        "Noise_Points": dbscan_noise,
        "Noise_Percent": dbscan_noise / total_records * 100,
        "Silhouette": dbscan_silhouette,
        "Davies_Bouldin": dbscan_db,
        "Calinski_Harabasz": dbscan_ch,
        "Evaluation": "Random sample of non-noise records"
    },

    {
        "Algorithm": "Hierarchical",
        "Total_Records": total_records,
        "Records_Processed": hierarchical_records,
        "Coverage_Percent": (
            hierarchical_records / total_records * 100
        ),
        "PCA_Features": pca_components,
        "Clusters": hierarchical_clusters,
        "Noise_Points": 0,
        "Noise_Percent": 0.0,
        "Silhouette": hierarchical_silhouette,
        "Davies_Bouldin": hierarchical_db,
        "Calinski_Harabasz": hierarchical_ch,
        "Evaluation": "Random sample of 3000"
    }
]

comparison_df = pd.DataFrame(comparison_data)

comparison_df.to_csv(
    OUTPUT_DIR / "clustering_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("MASTER COMPARISON TABLE")
print("=" * 65)

print(comparison_df.to_string(index=False))


# ============================================================
# 9. METRIC RANKINGS
# ============================================================

ranking_df = comparison_df[
    [
        "Algorithm",
        "Silhouette",
        "Davies_Bouldin",
        "Calinski_Harabasz"
    ]
].copy()

ranking_df["Silhouette_Rank"] = (
    ranking_df["Silhouette"]
    .rank(ascending=False, method="min")
)

ranking_df["Davies_Bouldin_Rank"] = (
    ranking_df["Davies_Bouldin"]
    .rank(ascending=True, method="min")
)

ranking_df["Calinski_Harabasz_Rank"] = (
    ranking_df["Calinski_Harabasz"]
    .rank(ascending=False, method="min")
)

ranking_df.to_csv(
    OUTPUT_DIR / "metric_rankings.csv",
    index=False
)

print("\nMETRIC RANKINGS")
print(ranking_df.to_string(index=False))


# ============================================================
# 10. CLUSTER DISTRIBUTION
# ============================================================

distribution_records = []

all_distributions = {
    "K-Means": kmeans_distribution,
    "DBSCAN": dbscan_distribution,
    "Hierarchical": hierarchical_distribution
}

for algorithm, distribution in all_distributions.items():

    for cluster, count in distribution.items():

        distribution_records.append({
            "Algorithm": algorithm,
            "Cluster": cluster,
            "Article_Count": count
        })

distribution_df = pd.DataFrame(distribution_records)

distribution_df.to_csv(
    OUTPUT_DIR / "cluster_distribution_comparison.csv",
    index=False
)


# ============================================================
# 11. METRIC VISUALIZATIONS
# ============================================================

def create_metric_chart(
    column,
    title,
    ylabel,
    filename
):

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Algorithm"],
        comparison_df[column]
    )

    plt.title(title)
    plt.xlabel("Clustering Algorithm")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


create_metric_chart(
    "Silhouette",
    "Silhouette Score Comparison",
    "Silhouette Score",
    "silhouette_comparison.png"
)

create_metric_chart(
    "Davies_Bouldin",
    "Davies-Bouldin Index Comparison",
    "Davies-Bouldin Index",
    "davies_bouldin_comparison.png"
)

create_metric_chart(
    "Calinski_Harabasz",
    "Calinski-Harabasz Score Comparison",
    "Calinski-Harabasz Score",
    "calinski_harabasz_comparison.png"
)

create_metric_chart(
    "Coverage_Percent",
    "Dataset Coverage Comparison",
    "Dataset Coverage (%)",
    "dataset_coverage_comparison.png"
)


# ============================================================
# 12. CLUSTER DISTRIBUTION GRAPHS
# ============================================================

for algorithm in all_distributions:

    current = distribution_df[
        distribution_df["Algorithm"] == algorithm
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        current["Cluster"],
        current["Article_Count"]
    )

    plt.title(f"{algorithm} Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Articles")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / (
            algorithm.lower() + "_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. AUTOMATIC RESEARCH OBSERVATIONS
# ============================================================

best_silhouette = comparison_df.loc[
    comparison_df["Silhouette"].idxmax(),
    "Algorithm"
]

best_db = comparison_df.loc[
    comparison_df["Davies_Bouldin"].idxmin(),
    "Algorithm"
]

best_ch = comparison_df.loc[
    comparison_df["Calinski_Harabasz"].idxmax(),
    "Algorithm"
]

observations = [

    f"PCA reduced {original_features} features to "
    f"{pca_components} components.",

    f"PCA retained {explained_variance:.4f}% "
    "of the original variance.",

    f"{best_silhouette} achieved the highest reported "
    "Silhouette Score.",

    f"{best_db} achieved the lowest reported "
    "Davies-Bouldin Index.",

    f"{best_ch} achieved the highest reported "
    "Calinski-Harabasz Score.",

    f"DBSCAN identified {dbscan_noise} noise records.",

    f"Hierarchical clustering processed "
    f"{hierarchical_records} out of {total_records} records.",

    "K-Means and DBSCAN processed the complete dataset.",

    "The evaluation procedures differ, so the metric "
    "rankings are preliminary rather than a fully "
    "controlled comparison."
]


# ============================================================
# 14. MASTER JSON REPORT
# ============================================================

master_report = {

    "generated_at": datetime.now().isoformat(),

    "project": "PCA-Based Unsupervised Learning Comparison",

    "pca_information": {
        "total_records": total_records,
        "original_features": original_features,
        "pca_components": pca_components,
        "explained_variance_percent": explained_variance,
        "variance_lost_percent": variance_lost
    },

    "clustering_results": comparison_data,

    "metric_rankings": ranking_df.to_dict(
        orient="records"
    ),

    "cluster_distributions": {
        algorithm: distribution
        for algorithm, distribution in all_distributions.items()
    },

    "observations": observations,

    "limitations": [
        "K-Means metrics use a random sample.",
        "DBSCAN metrics exclude noise records.",
        "Hierarchical clustering processes only 5000 records.",
        "Evaluation sample sizes and procedures differ.",
        "Cluster labels are algorithm-specific.",
        "A common evaluation sample is needed for a fair comparison."
    ]
}

with open(
    OUTPUT_DIR / "master_comparison_report.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        clean_json(master_report),
        file,
        indent=4,
        allow_nan=False
    )


# ============================================================
# 15. MASTER TEXT SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR / "master_comparison_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 70 + "\n")
    file.write("PCA CLUSTERING COMPARISON REPORT\n")
    file.write("=" * 70 + "\n\n")

    file.write("PCA INFORMATION\n")
    file.write("-" * 70 + "\n")

    for key, value in master_report["pca_information"].items():
        file.write(f"{key}: {value}\n")

    file.write("\nCLUSTERING COMPARISON\n")
    file.write("-" * 70 + "\n")
    file.write(comparison_df.to_string(index=False))

    file.write("\n\nMETRIC RANKINGS\n")
    file.write("-" * 70 + "\n")
    file.write(ranking_df.to_string(index=False))

    file.write("\n\nCLUSTER DISTRIBUTION\n")
    file.write("-" * 70 + "\n")
    file.write(distribution_df.to_string(index=False))

    file.write("\n\nRESEARCH OBSERVATIONS\n")
    file.write("-" * 70 + "\n")

    for index, observation in enumerate(
        observations, start=1
    ):
        file.write(f"{index}. {observation}\n")

    file.write("\nRESEARCH LIMITATIONS\n")
    file.write("-" * 70 + "\n")

    for limitation in master_report["limitations"]:
        file.write(f"- {limitation}\n")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PCA CLUSTERING COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nPCA:")
print(f"Original Features: {original_features}")
print(f"PCA Components: {pca_components}")
print(f"Explained Variance: {explained_variance:.4f}%")

print("\nBEST REPORTED METRICS:")
print("Silhouette:", best_silhouette)
print("Davies-Bouldin:", best_db)
print("Calinski-Harabasz:", best_ch)

print("\nRESULTS DIRECTORY:")
print(OUTPUT_DIR)

print("\nGENERATED FILES:")

for output_file in sorted(OUTPUT_DIR.iterdir()):

    if output_file.is_file():

        size_mb = (
            output_file.stat().st_size / (1024 * 1024)
        )

        print(
            f"{output_file.name} - {size_mb:.2f} MB"
        )

print("\nTASK 13 COMPARISON FINISHED!")
print("=" * 65)
