
import gc
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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

OUTPUT_DIR = BASE_DIR / "pca_dbscan_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

EPS = 2.5
MIN_SAMPLES = 10

K_DISTANCE_SAMPLE_SIZE = 10000
METRIC_SAMPLE_SIZE = 3000
VISUALIZATION_SAMPLE_SIZE = 10000


# ============================================================
# 2. LOAD PCA DATA
# ============================================================

print("\n" + "=" * 65)
print("PCA DBSCAN CLUSTERING")
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

X = df[pc_columns].to_numpy(dtype=np.float64)

total_records = len(X)

print("Dataset shape:", df.shape)
print("Total records:", total_records)
print("PCA components:", len(pc_columns))


# ============================================================
# 3. K-DISTANCE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("GENERATING K-DISTANCE GRAPH")
print("=" * 65)

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(
    K_DISTANCE_SAMPLE_SIZE,
    total_records
)

sample_indices = rng.choice(
    total_records,
    size=sample_size,
    replace=False
)

X_sample = X[sample_indices]

neighbors = NearestNeighbors(
    n_neighbors=MIN_SAMPLES,
    algorithm="auto",
    n_jobs=-1
)

neighbors.fit(X_sample)

distances, _ = neighbors.kneighbors(X_sample)

k_distances = np.sort(
    distances[:, -1]
)

plt.figure(figsize=(12, 6))

plt.plot(k_distances)

plt.xlabel("Points sorted by distance")
plt.ylabel(f"{MIN_SAMPLES}-Nearest Neighbor Distance")
plt.title("DBSCAN K-Distance Graph")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dbscan_k_distance.png",
    dpi=300
)

plt.close()

print("K-distance graph saved.")
print("Suggested EPS can be selected by inspecting the graph.")


# ============================================================
# 4. APPLY DBSCAN ON COMPLETE DATASET
# ============================================================

print("\n" + "=" * 65)
print("APPLYING DBSCAN ON COMPLETE DATASET")
print("=" * 65)

print("EPS:", EPS)
print("Min Samples:", MIN_SAMPLES)
print("Total records:", total_records)

dbscan_model = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    algorithm="auto",
    n_jobs=-1
)

labels = dbscan_model.fit_predict(X)

print("\nDBSCAN completed.")


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

noise_count = int(np.sum(labels == -1))

number_of_clusters = len(
    set(labels) - {-1}
)

clustered_count = total_records - noise_count

noise_percentage = (
    noise_count / total_records
) * 100

print("\nNumber of clusters:", number_of_clusters)
print("Noise points:", noise_count)
print("Clustered records:", clustered_count)
print("Noise percentage:", round(noise_percentage, 4), "%")

print("\nCluster Distribution:")

for cluster, count in cluster_distribution.items():
    label_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
    print(f"{label_name}: {count}")


# ============================================================
# 6. EVALUATION METRICS
# ============================================================

print("\n" + "=" * 65)
print("DBSCAN EVALUATION")
print("=" * 65)

# Exclude noise points from clustering quality metrics
non_noise_mask = labels != -1

X_clustered = X[non_noise_mask]
labels_clustered = labels[non_noise_mask]

unique_clusters = np.unique(labels_clustered)

metrics = {
    "silhouette_score": None,
    "davies_bouldin_index": None,
    "calinski_harabasz_score": None,
    "evaluation_note": ""
}

if len(unique_clusters) >= 2 and len(X_clustered) > len(unique_clusters):

    metric_size = min(
        METRIC_SAMPLE_SIZE,
        len(X_clustered)
    )

    metric_indices = rng.choice(
        len(X_clustered),
        size=metric_size,
        replace=False
    )

    X_metric = X_clustered[metric_indices]
    labels_metric = labels_clustered[metric_indices]

    # Sample may contain only one cluster, so verify again
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

        metrics["evaluation_note"] = (
            "Metrics calculated on a random sample of non-noise records."
        )

    else:
        metrics["evaluation_note"] = (
            "Evaluation skipped because the selected sample contains fewer than two clusters."
        )

else:
    metrics["evaluation_note"] = (
        "Evaluation skipped because DBSCAN did not identify at least two non-noise clusters."
    )


print("Silhouette Score:", metrics["silhouette_score"])
print("Davies-Bouldin Index:", metrics["davies_bouldin_index"])
print("Calinski-Harabasz Score:", metrics["calinski_harabasz_score"])
print("Evaluation note:", metrics["evaluation_note"])


# ============================================================
# 7. SAVE CLUSTERED DATASET
# ============================================================

result_df = df.copy()

result_df["dbscan_cluster"] = labels

result_df.to_csv(
    OUTPUT_DIR / "dbscan_clustered_dataset.csv",
    index=False
)


# ============================================================
# 8. SAVE CLUSTER DISTRIBUTION
# ============================================================

distribution_rows = []

for cluster, count in cluster_distribution.items():

    distribution_rows.append({
        "cluster": cluster,
        "cluster_name": (
            "Noise" if cluster == -1
            else f"Cluster {cluster}"
        ),
        "article_count": count,
        "percentage": round(
            count / total_records * 100,
            4
        )
    })

distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    OUTPUT_DIR / "dbscan_cluster_distribution.csv",
    index=False
)


# ============================================================
# 9. DBSCAN CLUSTER VISUALIZATION
# ============================================================

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
    cmap="tab20",
    s=10,
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DBSCAN Clusters on PCA Dataset")

plt.colorbar(
    scatter,
    label="Cluster (-1 = Noise)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "dbscan_cluster_visualization.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    dbscan_model,
    OUTPUT_DIR / "dbscan_model.pkl"
)


# ============================================================
# 11. GENERATE SUMMARY
# ============================================================

summary = {
    "execution_date": datetime.now().isoformat(),
    "algorithm": "DBSCAN",
    "input_dataset": str(INPUT_FILE),
    "total_records": int(total_records),
    "pca_features": pc_columns,
    "number_of_pca_features": len(pc_columns),
    "parameters": {
        "eps": EPS,
        "min_samples": MIN_SAMPLES,
        "metric": "euclidean",
        "algorithm": "auto"
    },
    "k_distance_sample_size": sample_size,
    "number_of_clusters": int(number_of_clusters),
    "noise_points": noise_count,
    "clustered_records": int(clustered_count),
    "noise_percentage": round(float(noise_percentage), 4),
    "cluster_distribution": cluster_distribution,
    "evaluation_metrics": metrics,
    "final_model_fitted_on": "Complete dataset"
}

with open(
    OUTPUT_DIR / "dbscan_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

with open(
    OUTPUT_DIR / "dbscan_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 65 + "\n")
    file.write("PCA DBSCAN CLUSTERING SUMMARY\n")
    file.write("=" * 65 + "\n\n")

    file.write(f"Execution Date: {summary['execution_date']}\n")
    file.write(f"Total Records: {total_records}\n")
    file.write(f"PCA Features: {len(pc_columns)}\n")
    file.write(f"EPS: {EPS}\n")
    file.write(f"Min Samples: {MIN_SAMPLES}\n")
    file.write(f"Number of Clusters: {number_of_clusters}\n")
    file.write(f"Noise Points: {noise_count}\n")
    file.write(f"Clustered Records: {clustered_count}\n")
    file.write(f"Noise Percentage: {noise_percentage:.4f}%\n\n")

    file.write("CLUSTER DISTRIBUTION:\n")

    for cluster, count in cluster_distribution.items():
        name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        file.write(f"{name}: {count}\n")

    file.write("\nEVALUATION METRICS:\n")
    file.write(f"Silhouette Score: {metrics['silhouette_score']}\n")
    file.write(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']}\n")
    file.write(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']}\n")
    file.write(f"Evaluation Note: {metrics['evaluation_note']}\n")


# ============================================================
# 12. FINISHED
# ============================================================

print("\n" + "=" * 65)
print("PCA DBSCAN COMPLETED")
print("=" * 65)

print("Clusters:", number_of_clusters)
print("Noise points:", noise_count)
print("Noise percentage:", round(noise_percentage, 4), "%")
print("Results:", OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(
            f"{path.name} - "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )

gc.collect()
