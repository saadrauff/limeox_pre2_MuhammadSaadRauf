
import gc
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "pca_results" / "pca_transformed_dataset.csv"

OUTPUT_DIR = BASE_DIR / "pca_hierarchical_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

SAMPLE_SIZE = 5000

N_CLUSTERS = 5

LINKAGE = "ward"

METRIC_SAMPLE_SIZE = 3000


# ============================================================
# 2. LOAD PCA DATA
# ============================================================

print("\n" + "=" * 65)
print("PCA HIERARCHICAL CLUSTERING")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

pc_columns = [
    col for col in df.columns
    if col.startswith("PC")
]

if not pc_columns:
    raise ValueError("No PCA component columns found.")

total_records = len(df)

print("Dataset shape:", df.shape)
print("Total records:", total_records)
print("PCA components:", len(pc_columns))


# ============================================================
# 3. SELECT RANDOM SAMPLE
# ============================================================

print("\n" + "=" * 65)
print("SELECTING DATASET SAMPLE")
print("=" * 65)

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(
    SAMPLE_SIZE,
    total_records
)

sample_indices = rng.choice(
    total_records,
    size=sample_size,
    replace=False
)

sample_df = df.iloc[sample_indices].copy()

X_sample = sample_df[pc_columns].to_numpy(
    dtype=np.float64
)

print("Original dataset records:", total_records)
print("Hierarchical sample records:", sample_size)
print("PCA features:", len(pc_columns))
print("Sample matrix shape:", X_sample.shape)


# ============================================================
# 4. APPLY AGGLOMERATIVE CLUSTERING
# ============================================================

print("\n" + "=" * 65)
print("APPLYING HIERARCHICAL CLUSTERING")
print("=" * 65)

print("Number of clusters:", N_CLUSTERS)
print("Linkage method:", LINKAGE)

hierarchical_model = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    metric="euclidean",
    linkage=LINKAGE
)

labels = hierarchical_model.fit_predict(
    X_sample
)

print("\nHierarchical clustering completed.")


# ============================================================
# 5. CLUSTER DISTRIBUTION
# ============================================================

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

cluster_distribution = {
    int(label): int(count)
    for label, count in zip(unique_labels, counts)
}

print("\nCluster Distribution:")

for cluster, count in cluster_distribution.items():
    print(f"Cluster {cluster}: {count} articles")


# ============================================================
# 6. EVALUATION METRICS
# ============================================================

print("\n" + "=" * 65)
print("HIERARCHICAL CLUSTERING EVALUATION")
print("=" * 65)

metric_size = min(
    METRIC_SAMPLE_SIZE,
    sample_size
)

metric_indices = rng.choice(
    sample_size,
    size=metric_size,
    replace=False
)

X_metric = X_sample[metric_indices]
labels_metric = labels[metric_indices]

metrics = {
    "silhouette_score": None,
    "davies_bouldin_index": None,
    "calinski_harabasz_score": None,
    "evaluation_sample_size": int(metric_size)
}

if len(np.unique(labels_metric)) >= 2:

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
        "Warning: Evaluation sample contains fewer than two clusters."
    )

print("Silhouette Score:", metrics["silhouette_score"])
print("Davies-Bouldin Index:", metrics["davies_bouldin_index"])
print("Calinski-Harabasz Score:", metrics["calinski_harabasz_score"])


# ============================================================
# 7. SAVE CLUSTERED SAMPLE DATASET
# ============================================================

result_df = sample_df.copy()

result_df["hierarchical_cluster"] = labels

result_df.to_csv(
    OUTPUT_DIR / "hierarchical_clustered_sample.csv",
    index=False
)


# ============================================================
# 8. SAVE CLUSTER DISTRIBUTION
# ============================================================

distribution_df = pd.DataFrame(
    [
        {
            "cluster": cluster,
            "article_count": count,
            "percentage": round(
                count / sample_size * 100,
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


# ============================================================
# 9. PCA CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_sample[:, 0],
    X_sample[:, 1],
    c=labels,
    cmap="tab10",
    s=12,
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical Clusters on PCA Dataset")

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


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    hierarchical_model,
    OUTPUT_DIR / "hierarchical_model.pkl"
)


# ============================================================
# 11. GENERATE SUMMARY
# ============================================================

summary = {
    "execution_date": datetime.now().isoformat(),
    "algorithm": "Agglomerative Hierarchical Clustering",
    "input_dataset": str(INPUT_FILE),
    "original_total_records": int(total_records),
    "total_pca_features": len(pc_columns),
    "pca_features": pc_columns,
    "method": "Agglomerative Clustering",
    "parameters": {
        "n_clusters": N_CLUSTERS,
        "linkage": LINKAGE,
        "metric": "euclidean"
    },
    "processing": {
        "processing_method": "Random sample",
        "sample_size": int(sample_size),
        "percentage_of_dataset": round(
            sample_size / total_records * 100,
            4
        )
    },
    "cluster_distribution": cluster_distribution,
    "evaluation_metrics": metrics,
    "model_fitted_on": "Random sample only",
    "full_dataset_cluster_labels_generated": False
}

with open(
    OUTPUT_DIR / "hierarchical_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

with open(
    OUTPUT_DIR / "hierarchical_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 65 + "\n")
    file.write("PCA HIERARCHICAL CLUSTERING SUMMARY\n")
    file.write("=" * 65 + "\n\n")

    file.write(f"Execution Date: {summary['execution_date']}\n")
    file.write(f"Algorithm: {summary['algorithm']}\n")
    file.write(f"Original Dataset Records: {total_records}\n")
    file.write(f"PCA Features: {len(pc_columns)}\n")
    file.write(f"Sample Size: {sample_size}\n")
    file.write(
        f"Dataset Percentage Processed: "
        f"{summary['processing']['percentage_of_dataset']}%\n"
    )
    file.write(f"Number of Clusters: {N_CLUSTERS}\n")
    file.write(f"Linkage: {LINKAGE}\n")
    file.write("Distance Metric: Euclidean\n\n")

    file.write("CLUSTER DISTRIBUTION:\n")

    for cluster, count in cluster_distribution.items():
        file.write(f"Cluster {cluster}: {count}\n")

    file.write("\nEVALUATION METRICS:\n")
    file.write(f"Silhouette Score: {metrics['silhouette_score']}\n")
    file.write(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']}\n")
    file.write(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']}\n")
    file.write(
        f"Evaluation Sample Size: "
        f"{metrics['evaluation_sample_size']}\n"
    )

    file.write("\nPROCESSING LIMITATION:\n")
    file.write(
        "Hierarchical clustering was fitted on a random sample, "
        "not the complete dataset.\n"
    )
    file.write(
        "No cluster labels were generated for the remaining records.\n"
    )


# ============================================================
# 12. FINISHED
# ============================================================

print("\n" + "=" * 65)
print("PCA HIERARCHICAL CLUSTERING COMPLETED")
print("=" * 65)

print("Original records:", total_records)
print("Records clustered:", sample_size)
print("Number of clusters:", N_CLUSTERS)

print("Silhouette:", metrics["silhouette_score"])
print("Davies-Bouldin:", metrics["davies_bouldin_index"])
print("Calinski-Harabasz:", metrics["calinski_harabasz_score"])

print("Results:", OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(
            f"{path.name} - "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )

gc.collect()
