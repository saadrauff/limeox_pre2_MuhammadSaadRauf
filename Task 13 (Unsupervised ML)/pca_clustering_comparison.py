
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

# NEW FULL DATASET HIERARCHICAL RESULTS
HIERARCHICAL_DIR = (
    BASE_DIR / "pca_hierarchical_full_results"
)

# Separate output folder to preserve previous comparison
OUTPUT_DIR = (
    BASE_DIR / "pca_clustering_comparison_full_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def read_text(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return path.read_text(encoding="utf-8")


def read_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_number(text, pattern, field):

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            f"Could not extract {field} from summary."
        )

    return float(
        match.group(1).replace(",", "")
    )


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

            count = int(
                match.group(2).replace(",", "")
            )

            distribution[name] = count

    return distribution


def clean_json(value):

    if isinstance(value, dict):

        return {
            str(k): clean_json(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            clean_json(v)
            for v in value
        ]

    if isinstance(value, (np.integer,)):

        return int(value)

    if isinstance(value, (np.floating, float)):

        return (
            float(value)
            if np.isfinite(value)
            else None
        )

    if isinstance(value, np.bool_):

        return bool(value)

    return value


def safe_round(value, digits=6):

    if value is None:
        return None

    return round(float(value), digits)


# ============================================================
# 3. LOAD SUMMARY FILES
# ============================================================

print("\n" + "=" * 70)
print("PCA CLUSTERING COMPARISON - FULL DATASET")
print("=" * 70)

kmeans_text = read_text(
    KMEANS_DIR / "kmeans_summary.txt"
)

dbscan_text = read_text(
    DBSCAN_DIR / "dbscan_summary.txt"
)

# Read NEW Hierarchical JSON
hierarchical_json = read_json(
    HIERARCHICAL_DIR / "hierarchical_summary.json"
)

# Also read its text summary for cluster distribution
hierarchical_text = read_text(
    HIERARCHICAL_DIR / "hierarchical_summary.txt"
)

print("\nAll summary files loaded successfully.")


# ============================================================
# 4. EXTRACT PCA INFORMATION
# ============================================================

pca_dataset = (
    PCA_DIR / "pca_transformed_dataset.csv"
)

if not pca_dataset.exists():
    raise FileNotFoundError(pca_dataset)

pca_header = pd.read_csv(
    pca_dataset,
    nrows=0
)

pca_columns = [
    column
    for column in pca_header.columns
    if re.match(
        r"^PC\d+$",
        str(column),
        re.IGNORECASE
    )
]

pca_components = len(pca_columns)

with open(
    pca_dataset,
    "r",
    encoding="utf-8-sig"
) as file:

    total_records = sum(
        1 for _ in file
    ) - 1


variance_file = (
    PCA_DIR / "pca_explained_variance.csv"
)

variance_df = pd.read_csv(
    variance_file
)

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

    explained_variance = float(
        values.iloc[-1]
    )

elif ratio_columns:

    values = pd.to_numeric(
        variance_df[ratio_columns[0]],
        errors="coerce"
    ).dropna()

    explained_variance = float(
        values.sum()
    )

else:

    raise ValueError(
        "Explained variance column not found."
    )


if explained_variance <= 1.5:
    explained_variance *= 100

variance_lost = (
    100 - explained_variance
)

original_features = 25

print("\nPCA INFORMATION")
print("-" * 70)

print("Total records:", total_records)

print("Original features:", original_features)

print("PCA components:", pca_components)

print(
    "Explained variance:",
    round(explained_variance, 4), "%"
)

print(
    "Variance lost:",
    round(variance_lost, 4), "%"
)


# ============================================================
# 5. EXTRACT K-MEANS RESULTS
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

kmeans_distribution = extract_distribution(
    kmeans_text
)

kmeans_records = total_records

print("\nK-MEANS RESULTS")
print("-" * 70)

print("Records:", kmeans_records)

print("Clusters:", kmeans_k)

print("Silhouette:", kmeans_silhouette)

print("Davies-Bouldin:", kmeans_db)

print("Calinski-Harabasz:", kmeans_ch)


# ============================================================
# 6. EXTRACT DBSCAN RESULTS
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

dbscan_distribution = extract_distribution(
    dbscan_text
)

dbscan_records = total_records

print("\nDBSCAN RESULTS")
print("-" * 70)

print("Records:", dbscan_records)

print("Clusters:", dbscan_clusters)

print("Noise points:", dbscan_noise)

print("Silhouette:", dbscan_silhouette)

print("Davies-Bouldin:", dbscan_db)

print("Calinski-Harabasz:", dbscan_ch)


# ============================================================
# 7. EXTRACT FULL HIERARCHICAL RESULTS
# ============================================================

hierarchical_processing = (
    hierarchical_json["processing"]
)

hierarchical_metrics = (
    hierarchical_json["evaluation_metrics"]
)

hierarchical_parameters = (
    hierarchical_json["parameters"]
)

hierarchical_records = int(
    hierarchical_json["records_processed"]
)

hierarchical_total = int(
    hierarchical_json["original_total_records"]
)

hierarchical_coverage = float(
    hierarchical_json["coverage_percentage"]
)

hierarchical_clusters = int(
    hierarchical_parameters["n_clusters"]
)

hierarchical_silhouette = (
    hierarchical_metrics["silhouette_score"]
)

hierarchical_db = (
    hierarchical_metrics["davies_bouldin_index"]
)

hierarchical_ch = (
    hierarchical_metrics["calinski_harabasz_score"]
)

hierarchical_distribution = extract_distribution(
    hierarchical_text
)

# Validate full dataset processing
if hierarchical_records != total_records:

    raise ValueError(
        "Hierarchical records do not match "
        "the PCA dataset."
    )

if hierarchical_total != total_records:

    raise ValueError(
        "Hierarchical original record count mismatch."
    )

if hierarchical_coverage < 99.99:

    raise ValueError(
        "Hierarchical clustering did not achieve "
        "100% coverage."
    )

print("\nHIERARCHICAL RESULTS")
print("-" * 70)

print("Records:", hierarchical_records)

print("Coverage:", hierarchical_coverage, "%")

print("Clusters:", hierarchical_clusters)

print("Linkage:", hierarchical_parameters["linkage"])

print(
    "Connectivity:",
    hierarchical_parameters["connectivity"]
)

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

        "Records_Processed": kmeans_records,

        "Coverage_Percent": 100.0,

        "PCA_Features": pca_components,

        "Clusters": kmeans_k,

        "Noise_Points": 0,

        "Noise_Percent": 0.0,

        "Silhouette": kmeans_silhouette,

        "Davies_Bouldin": kmeans_db,

        "Calinski_Harabasz": kmeans_ch,

        "Evaluation": "Random sample of 3000",

        "Method": "Standard K-Means"

    },

    {
        "Algorithm": "DBSCAN",

        "Total_Records": total_records,

        "Records_Processed": dbscan_records,

        "Coverage_Percent": 100.0,

        "PCA_Features": pca_components,

        "Clusters": dbscan_clusters,

        "Noise_Points": dbscan_noise,

        "Noise_Percent": (
            dbscan_noise / total_records * 100
        ),

        "Silhouette": dbscan_silhouette,

        "Davies_Bouldin": dbscan_db,

        "Calinski_Harabasz": dbscan_ch,

        "Evaluation": "Random sample of non-noise records",

        "Method": "DBSCAN"

    },

    {
        "Algorithm": "Hierarchical",

        "Total_Records": total_records,

        "Records_Processed": hierarchical_records,

        "Coverage_Percent": hierarchical_coverage,

        "PCA_Features": pca_components,

        "Clusters": hierarchical_clusters,

        "Noise_Points": 0,

        "Noise_Percent": 0.0,

        "Silhouette": hierarchical_silhouette,

        "Davies_Bouldin": hierarchical_db,

        "Calinski_Harabasz": hierarchical_ch,

        "Evaluation": "Random sample of 3000",

        "Method": "KNN Connectivity + Ward Linkage"

    }

]

comparison_df = pd.DataFrame(
    comparison_data
)

comparison_df.to_csv(
    OUTPUT_DIR / "clustering_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("MASTER CLUSTERING COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(index=False)
)


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
    .rank(
        ascending=False,
        method="min"
    )
)

ranking_df["Davies_Bouldin_Rank"] = (
    ranking_df["Davies_Bouldin"]
    .rank(
        ascending=True,
        method="min"
    )
)

ranking_df["Calinski_Harabasz_Rank"] = (
    ranking_df["Calinski_Harabasz"]
    .rank(
        ascending=False,
        method="min"
    )
)

ranking_df.to_csv(
    OUTPUT_DIR / "metric_rankings.csv",
    index=False
)

print("\nMETRIC RANKINGS")
print("-" * 70)

print(
    ranking_df.to_string(index=False)
)


# ============================================================
# 10. CLUSTER DISTRIBUTION COMPARISON
# ============================================================

all_distributions = {

    "K-Means": kmeans_distribution,

    "DBSCAN": dbscan_distribution,

    "Hierarchical": hierarchical_distribution

}

distribution_records = []

for algorithm, distribution in all_distributions.items():

    for cluster, count in distribution.items():

        distribution_records.append({

            "Algorithm": algorithm,

            "Cluster": cluster,

            "Article_Count": count

        })

distribution_df = pd.DataFrame(
    distribution_records
)

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

    plt.grid(
        axis="y",
        alpha=0.3
    )

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
    "Coverage (%)",
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

    plt.title(
        f"{algorithm} Cluster Distribution"
    )

    plt.xlabel("Cluster")

    plt.ylabel("Number of Articles")

    plt.grid(
        axis="y",
        alpha=0.3
    )

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

    (
        f"PCA reduced {original_features} original features "
        f"to {pca_components} components."
    ),

    (
        f"PCA retained {explained_variance:.4f}% "
        "of the original variance."
    ),

    (
        f"{best_silhouette} achieved the highest "
        "reported Silhouette Score."
    ),

    (
        f"{best_db} achieved the lowest "
        "reported Davies-Bouldin Index."
    ),

    (
        f"{best_ch} achieved the highest "
        "reported Calinski-Harabasz Score."
    ),

    (
        f"DBSCAN identified {dbscan_noise} noise records."
    ),

    (
        f"All three algorithms processed "
        f"{total_records} records."
    ),

    (
        "Hierarchical clustering used KNN connectivity "
        "with Ward linkage."
    ),

    (
        "Hierarchical clustering generated labels "
        "for the complete PCA dataset."
    ),

    (
        "Metric comparisons should be interpreted "
        "with consideration of their different "
        "evaluation procedures."
    )

]


# ============================================================
# 14. MASTER JSON REPORT
# ============================================================

master_report = {

    "generated_at": datetime.now().isoformat(),

    "project": (
        "PCA-Based Unsupervised Learning Comparison"
    ),

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

    "cluster_distributions": all_distributions,

    "hierarchical_method": {

        "algorithm": "Agglomerative Clustering",

        "linkage": "Ward",

        "metric": "Euclidean",

        "connectivity": "K-Nearest Neighbors",

        "n_neighbors": 10

    },

    "observations": observations,

    "limitations": [

        (
            "K-Means metrics use a random sample "
            "of 3000 records."
        ),

        (
            "DBSCAN metrics exclude noise records "
            "from evaluation."
        ),

        (
            "Hierarchical metrics use a random "
            "sample of 3000 records."
        ),

        (
            "Evaluation procedures differ, so "
            "metric rankings are not a fully "
            "controlled comparison."
        ),

        (
            "Hierarchical KNN connectivity changes "
            "the allowed cluster-merging structure "
            "compared with unconstrained Ward linkage."
        ),

        (
            "Cluster labels are algorithm-specific "
            "and are not directly comparable."
        )

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

    file.write(
        "PCA CLUSTERING COMPARISON REPORT\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write("PCA INFORMATION\n")

    file.write("-" * 70 + "\n")

    for key, value in master_report[
        "pca_information"
    ].items():

        file.write(
            f"{key}: {value}\n"
        )

    file.write("\nMASTER COMPARISON\n")

    file.write("-" * 70 + "\n")

    file.write(
        comparison_df.to_string(index=False)
    )

    file.write("\n\nMETRIC RANKINGS\n")

    file.write("-" * 70 + "\n")

    file.write(
        ranking_df.to_string(index=False)
    )

    file.write("\n\nCLUSTER DISTRIBUTIONS\n")

    file.write("-" * 70 + "\n")

    file.write(
        distribution_df.to_string(index=False)
    )

    file.write("\n\nRESEARCH OBSERVATIONS\n")

    file.write("-" * 70 + "\n")

    for index, observation in enumerate(
        observations,
        start=1
    ):

        file.write(
            f"{index}. {observation}\n"
        )

    file.write("\nRESEARCH LIMITATIONS\n")

    file.write("-" * 70 + "\n")

    for limitation in master_report["limitations"]:

        file.write(
            f"- {limitation}\n"
        )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "PCA CLUSTERING COMPARISON COMPLETED SUCCESSFULLY"
)

print("=" * 70)

print("\nPCA INFORMATION")

print("Original Features:", original_features)

print("PCA Components:", pca_components)

print(
    "Explained Variance:",
    round(explained_variance, 4), "%"
)

print("\nDATASET COVERAGE")

for row in comparison_data:

    print(
        f"{row['Algorithm']}: "
        f"{row['Records_Processed']} / "
        f"{row['Total_Records']} "
        f"({row['Coverage_Percent']:.4f}%)"
    )

print("\nBEST REPORTED METRICS")

print("Silhouette:", best_silhouette)

print("Davies-Bouldin:", best_db)

print("Calinski-Harabasz:", best_ch)

print("\nRESULTS DIRECTORY:")

print(OUTPUT_DIR)

print("\nGENERATED FILES:")

for output_file in sorted(
    OUTPUT_DIR.iterdir()
):

    if output_file.is_file():

        size_mb = (
            output_file.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{output_file.name} - "
            f"{size_mb:.2f} MB"
        )

print("\nTASK 13 COMPARISON FINISHED!")

print("=" * 70)
