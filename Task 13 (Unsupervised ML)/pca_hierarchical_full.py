
import gc
import json
import joblib
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("default")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "pca_results"
    / "pca_transformed_dataset.csv"
)

# Separate folder: existing results remain untouched
OUTPUT_DIR = BASE_DIR / "pca_hierarchical_full_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

N_CLUSTERS = 5

LINKAGE = "ward"

N_NEIGHBORS = 10

METRIC_SAMPLE_SIZE = 3000

VISUALIZATION_SAMPLE_SIZE = 20000


# ============================================================
# 2. LOAD PCA DATA
# ============================================================

print("\n" + "=" * 70)
print("PCA HIERARCHICAL CLUSTERING - FULL DATASET")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"PCA dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

pc_columns = [
    col for col in df.columns
    if col.startswith("PC")
]

if not pc_columns:
    raise ValueError("No PCA component columns found.")

total_records = len(df)

print("\nDataset Information")
print("-" * 50)
print("Total records:", total_records)
print("PCA features:", len(pc_columns))
print("Dataset shape:", df.shape)


# ============================================================
# 3. PREPARE FULL PCA DATASET
# ============================================================

print("\n" + "=" * 70)
print("PREPARING FULL PCA DATASET")
print("=" * 70)

X = df[pc_columns].to_numpy(dtype=np.float64)

if not np.isfinite(X).all():
    raise ValueError(
        "PCA dataset contains NaN or infinite values."
    )

print("Records prepared:", X.shape[0])
print("Features prepared:", X.shape[1])
print("Feature matrix:", X.shape)


# ============================================================
# 4. BUILD K-NEAREST NEIGHBORS CONNECTIVITY
# ============================================================

print("\n" + "=" * 70)
print("BUILDING KNN CONNECTIVITY GRAPH")
print("=" * 70)

start_time = time.time()

connectivity = kneighbors_graph(
    X,
    n_neighbors=N_NEIGHBORS,
    mode="connectivity",
    include_self=False,
    n_jobs=-1
)

# Make connectivity symmetric
connectivity = connectivity.maximum(
    connectivity.T
).tocsr()

graph_time = time.time() - start_time

print("Connectivity graph created successfully.")
print("Graph shape:", connectivity.shape)
print("Graph non-zero connections:", connectivity.nnz)
print("Graph construction time:", round(graph_time, 2), "seconds")


# ============================================================
# 5. APPLY FULL DATASET HIERARCHICAL CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("APPLYING HIERARCHICAL CLUSTERING")
print("=" * 70)

print("Algorithm: Agglomerative Clustering")
print("Linkage:", LINKAGE)
print("Distance metric: Euclidean")
print("Clusters:", N_CLUSTERS)
print("Records:", total_records)
print("Connectivity neighbors:", N_NEIGHBORS)

hierarchical_model = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    metric="euclidean",
    linkage=LINKAGE,
    connectivity=connectivity,
    compute_full_tree=True
)

clustering_start = time.time()

labels = hierarchical_model.fit_predict(X)

clustering_time = time.time() - clustering_start

print("\nHierarchical clustering completed.")

print(
    "Clustering execution time:",
    round(clustering_time, 2),
    "seconds"
)

# Release connectivity graph after clustering
del connectivity
gc.collect()


# ============================================================
# 6. VERIFY FULL DATASET CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING CLUSTERING RESULTS")
print("=" * 70)

if len(labels) != total_records:
    raise RuntimeError(
        "Cluster label count does not match dataset records."
    )

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

if len(unique_labels) < 2:
    raise RuntimeError(
        "Clustering generated fewer than two clusters."
    )

cluster_distribution = {
    int(label): int(count)
    for label, count in zip(unique_labels, counts)
}

print("Total records:", total_records)
print("Records clustered:", len(labels))
print("Number of clusters:", len(unique_labels))

print("\nCluster Distribution:")

for cluster, count in cluster_distribution.items():

    percentage = count / total_records * 100

    print(
        f"Cluster {cluster}: "
        f"{count} records ({percentage:.2f}%)"
    )


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL CLUSTERING EVALUATION")
print("=" * 70)

rng = np.random.default_rng(RANDOM_STATE)

metric_size = min(
    METRIC_SAMPLE_SIZE,
    total_records
)

metric_indices = rng.choice(
    total_records,
    size=metric_size,
    replace=False
)

X_metric = X[metric_indices]
labels_metric = labels[metric_indices]

metrics = {
    "silhouette_score": None,
    "davies_bouldin_index": None,
    "calinski_harabasz_score": None,
    "evaluation_sample_size": int(metric_size)
}

if (
    len(np.unique(labels_metric)) >= 2
    and len(np.unique(labels_metric)) < metric_size
):

    metrics["silhouette_score"] = float(
        silhouette_score(
            X_metric,
            labels_metric
        )
    )

    metrics["davies_bouldin_index"] = float(
        davies_bouldin_score(
            X_metric,
            labels_metric
        )
    )

    metrics["calinski_harabasz_score"] = float(
        calinski_harabasz_score(
            X_metric,
            labels_metric
        )
    )

else:
    print(
        "Warning: Evaluation sample does not contain "
        "a valid number of clusters."
    )

print("\nEvaluation Results")

print("Silhouette Score:",
      metrics["silhouette_score"])

print("Davies-Bouldin Index:",
      metrics["davies_bouldin_index"])

print("Calinski-Harabasz Score:",
      metrics["calinski_harabasz_score"])


# ============================================================
# 8. SAVE FULL CLUSTERED DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING FULL CLUSTERED DATASET")
print("=" * 70)

result_df = df.copy()

result_df["hierarchical_cluster"] = labels

result_df.to_csv(
    OUTPUT_DIR / "hierarchical_full_clustered_dataset.csv",
    index=False
)

print("Full clustered dataset saved.")


# ============================================================
# 9. SAVE CLUSTER DISTRIBUTION
# ============================================================

distribution_df = pd.DataFrame(
    [
        {
            "cluster": cluster,
            "article_count": count,
            "percentage": round(
                count / total_records * 100,
                4
            )
        }
        for cluster, count in cluster_distribution.items()
    ]
)

distribution_df.to_csv(
    OUTPUT_DIR / "hierarchical_cluster_distribution.csv",
    index=False
)

print("Cluster distribution saved.")


# ============================================================
# 10. PCA CLUSTER VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING CLUSTER VISUALIZATION")
print("=" * 70)

visual_size = min(
    VISUALIZATION_SAMPLE_SIZE,
    total_records
)

visual_indices = rng.choice(
    total_records,
    size=visual_size,
    replace=False
)

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X[visual_indices, 0],
    X[visual_indices, 1],
    c=labels[visual_indices],
    cmap="tab10",
    s=12,
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Hierarchical Clustering on Full PCA Dataset"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "hierarchical_cluster_visualization.png",
    dpi=300
)

plt.close()

print("Visualization saved.")


# ============================================================
# 11. SAVE MODEL
# ============================================================

print("\nSaving Hierarchical Model...")

joblib.dump(
    hierarchical_model,
    OUTPUT_DIR / "hierarchical_full_model.pkl"
)

print("Model saved.")


# ============================================================
# 12. GENERATE SUMMARY JSON
# ============================================================

processing_percentage = (
    len(labels) / total_records * 100
)

summary = {

    "execution_date": datetime.now().isoformat(),

    "algorithm": "Connectivity-constrained Agglomerative Hierarchical Clustering",

    "input_dataset": str(INPUT_FILE),

    "original_total_records": int(total_records),

    "records_processed": int(len(labels)),

    "coverage_percentage": round(
        processing_percentage,
        4
    ),

    "total_pca_features": len(pc_columns),

    "pca_features": pc_columns,

    "method": "Agglomerative Clustering",

    "parameters": {

        "n_clusters": N_CLUSTERS,

        "linkage": LINKAGE,

        "metric": "euclidean",

        "connectivity": "K-Nearest Neighbors",

        "n_neighbors": N_NEIGHBORS

    },

    "processing": {

        "processing_method": "Full dataset",

        "sample_size": int(len(labels)),

        "full_dataset_cluster_labels_generated": True

    },

    "timing": {

        "connectivity_graph_seconds": round(
            graph_time, 2
        ),

        "clustering_seconds": round(
            clustering_time, 2
        )

    },

    "cluster_distribution": cluster_distribution,

    "evaluation_metrics": metrics,

    "model_fitted_on": "Full PCA dataset"

}

with open(
    OUTPUT_DIR / "hierarchical_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 13. GENERATE SUMMARY TXT
# ============================================================

with open(
    OUTPUT_DIR / "hierarchical_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 70 + "\n")

    file.write(
        "PCA HIERARCHICAL CLUSTERING - FULL DATASET SUMMARY\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Execution Date: {summary['execution_date']}\n"
    )

    file.write(
        f"Algorithm: {summary['algorithm']}\n"
    )

    file.write(
        f"Original Records: {total_records}\n"
    )

    file.write(
        f"Records Processed: {len(labels)}\n"
    )

    file.write(
        f"Coverage: {processing_percentage:.4f}%\n"
    )

    file.write(
        f"PCA Features: {len(pc_columns)}\n"
    )

    file.write(
        f"Clusters: {N_CLUSTERS}\n"
    )

    file.write(
        f"Linkage: {LINKAGE}\n"
    )

    file.write(
        f"Connectivity Neighbors: {N_NEIGHBORS}\n"
    )

    file.write(
        "\nCLUSTER DISTRIBUTION:\n"
    )

    for cluster, count in cluster_distribution.items():

        file.write(
            f"Cluster {cluster}: {count}\n"
        )

    file.write(
        "\nEVALUATION METRICS:\n"
    )

    file.write(
        f"Silhouette Score: {metrics['silhouette_score']}\n"
    )

    file.write(
        f"Davies-Bouldin Index: {metrics['davies_bouldin_index']}\n"
    )

    file.write(
        f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']}\n"
    )

    file.write(
        f"Evaluation Sample Size: {metric_size}\n"
    )

    file.write(
        "\nPROCESSING INFORMATION:\n"
    )

    file.write(
        "Hierarchical clustering was fitted on the full PCA dataset.\n"
    )

    file.write(
        "K-Nearest Neighbors connectivity was used.\n"
    )

    file.write(
        "All records received cluster labels.\n"
    )


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("FULL DATASET HIERARCHICAL CLUSTERING COMPLETED")
print("=" * 70)

print("Original records:", total_records)

print("Records clustered:", len(labels))

print("Coverage:", round(processing_percentage, 4), "%")

print("Clusters:", len(unique_labels))

print("Silhouette:", metrics["silhouette_score"])

print("Davies-Bouldin:",
      metrics["davies_bouldin_index"])

print("Calinski-Harabasz:",
      metrics["calinski_harabasz_score"])

print("\nResults directory:", OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):

    if path.is_file():

        print(
            f"{path.name} - "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )

gc.collect()
