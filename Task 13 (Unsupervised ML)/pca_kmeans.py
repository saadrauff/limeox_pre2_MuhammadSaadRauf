
import gc
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
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

OUTPUT_DIR = BASE_DIR / "pca_kmeans_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

K_VALUES = range(2, 11)

SILHOUETTE_SAMPLE_SIZE = 3000
K_SELECTION_SAMPLE_SIZE = 10000


# ============================================================
# 2. LOAD PCA DATA
# ============================================================

print("\n" + "=" * 65)
print("PCA K-MEANS CLUSTERING")
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
# 3. SELECT SAMPLE FOR K EVALUATION
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(
    K_SELECTION_SAMPLE_SIZE,
    total_records
)

sample_indices = rng.choice(
    total_records,
    size=sample_size,
    replace=False
)

X_sample = X[sample_indices]

silhouette_size = min(
    SILHOUETTE_SAMPLE_SIZE,
    sample_size
)

print("\nK selection sample:", sample_size)
print("Silhouette sample:", silhouette_size)


# ============================================================
# 4. ELBOW AND SILHOUETTE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("FINDING OPTIMAL K")
print("=" * 65)

evaluation_results = []

inertias = []
silhouette_scores = []

for k in K_VALUES:

    print(f"\nEvaluating K = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300
    )

    labels = model.fit_predict(X_sample)

    inertia = float(model.inertia_)

    score = silhouette_score(
        X_sample,
        labels,
        sample_size=silhouette_size,
        random_state=RANDOM_STATE
    )

    inertias.append(inertia)
    silhouette_scores.append(float(score))

    evaluation_results.append({
        "k": k,
        "inertia": inertia,
        "silhouette_score": float(score)
    })

    print("Inertia:", round(inertia, 4))
    print("Silhouette Score:", round(score, 4))

    del model, labels
    gc.collect()


# ============================================================
# 5. SELECT BEST K
# ============================================================

best_result = max(
    evaluation_results,
    key=lambda item: item["silhouette_score"]
)

BEST_K = best_result["k"]

print("\nBest K according to Silhouette Score:", BEST_K)


# ============================================================
# 6. SAVE K EVALUATION
# ============================================================

evaluation_df = pd.DataFrame(evaluation_results)

evaluation_df.to_csv(
    OUTPUT_DIR / "kmeans_k_evaluation.csv",
    index=False
)


# ============================================================
# 7. ELBOW VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_elbow.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. SILHOUETTE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Analysis")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_silhouette.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. FINAL K-MEANS ON COMPLETE DATASET
# ============================================================

print("\n" + "=" * 65)
print("FITTING FINAL K-MEANS ON COMPLETE DATASET")
print("=" * 65)

final_model = KMeans(
    n_clusters=BEST_K,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300
)

final_labels = final_model.fit_predict(X)

print("Final K-Means completed.")
print("Total records clustered:", len(final_labels))


# ============================================================
# 10. CLUSTER DISTRIBUTION
# ============================================================

unique_labels, cluster_counts = np.unique(
    final_labels,
    return_counts=True
)

cluster_distribution = {
    int(label): int(count)
    for label, count in zip(unique_labels, cluster_counts)
}

print("\nCluster Distribution:")

for cluster, count in cluster_distribution.items():
    print(f"Cluster {cluster}: {count} articles")


# ============================================================
# 11. FINAL EVALUATION
# ============================================================

print("\nCalculating final evaluation metrics...")

final_silhouette = silhouette_score(
    X,
    final_labels,
    sample_size=min(SILHOUETTE_SAMPLE_SIZE, total_records),
    random_state=RANDOM_STATE
)

final_davies_bouldin = davies_bouldin_score(
    X,
    final_labels
)

final_calinski_harabasz = calinski_harabasz_score(
    X,
    final_labels
)

print("Silhouette Score:", round(final_silhouette, 4))
print("Davies-Bouldin Index:", round(final_davies_bouldin, 4))
print("Calinski-Harabasz Score:", round(final_calinski_harabasz, 4))


# ============================================================
# 12. SAVE CLUSTERED DATASET
# ============================================================

result_df = df.copy()
result_df["kmeans_cluster"] = final_labels

result_df.to_csv(
    OUTPUT_DIR / "kmeans_clustered_dataset.csv",
    index=False
)


# ============================================================
# 13. SAVE CLUSTER DISTRIBUTION
# ============================================================

distribution_df = pd.DataFrame(
    [
        {
            "cluster": cluster,
            "article_count": count,
            "percentage": round(
                count / total_records * 100, 4
            )
        }
        for cluster, count in cluster_distribution.items()
    ]
)

distribution_df.to_csv(
    OUTPUT_DIR / "kmeans_cluster_distribution.csv",
    index=False
)


# ============================================================
# 14. PCA CLUSTER VISUALIZATION
# ============================================================

visual_sample_size = min(10000, total_records)

visual_indices = rng.choice(
    total_records,
    size=visual_sample_size,
    replace=False
)

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X[visual_indices, 0],
    X[visual_indices, 1],
    c=final_labels[visual_indices],
    cmap="tab10",
    s=10,
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters on PCA Dataset")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_cluster_visualization.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    OUTPUT_DIR / "kmeans_model.pkl"
)


# ============================================================
# 16. GENERATE SUMMARY
# ============================================================

summary = {
    "execution_date": datetime.now().isoformat(),
    "algorithm": "K-Means Clustering",
    "input_dataset": str(INPUT_FILE),
    "total_records": int(total_records),
    "pca_features": pc_columns,
    "number_of_pca_features": len(pc_columns),
    "k_selection_range": list(K_VALUES),
    "k_selection_sample_size": sample_size,
    "silhouette_sample_size": silhouette_size,
    "selected_k": int(BEST_K),
    "final_model_fitted_on": "Complete dataset",
    "cluster_distribution": cluster_distribution,
    "evaluation_metrics": {
        "silhouette_score": float(final_silhouette),
        "davies_bouldin_index": float(final_davies_bouldin),
        "calinski_harabasz_score": float(final_calinski_harabasz)
    },
    "k_evaluation": evaluation_results
}

with open(
    OUTPUT_DIR / "kmeans_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

with open(
    OUTPUT_DIR / "kmeans_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 65 + "\n")
    file.write("PCA K-MEANS CLUSTERING SUMMARY\n")
    file.write("=" * 65 + "\n\n")

    file.write(f"Execution Date: {summary['execution_date']}\n")
    file.write(f"Total Records: {total_records}\n")
    file.write(f"PCA Features: {len(pc_columns)}\n")
    file.write(f"Selected K: {BEST_K}\n")
    file.write("Final Fit: Complete Dataset\n\n")

    file.write("K EVALUATION:\n")

    for item in evaluation_results:
        file.write(
            f"K={item['k']} | "
            f"Inertia={item['inertia']:.4f} | "
            f"Silhouette={item['silhouette_score']:.4f}\n"
        )

    file.write("\nCLUSTER DISTRIBUTION:\n")

    for cluster, count in cluster_distribution.items():
        file.write(f"Cluster {cluster}: {count}\n")

    file.write("\nFINAL EVALUATION:\n")
    file.write(f"Silhouette Score: {final_silhouette:.4f}\n")
    file.write(f"Davies-Bouldin Index: {final_davies_bouldin:.4f}\n")
    file.write(f"Calinski-Harabasz Score: {final_calinski_harabasz:.4f}\n")


# ============================================================
# 17. FINISHED
# ============================================================

print("\n" + "=" * 65)
print("PCA K-MEANS COMPLETED")
print("=" * 65)

print("Selected K:", BEST_K)
print("Total records:", total_records)
print("Silhouette:", round(final_silhouette, 4))
print("Davies-Bouldin:", round(final_davies_bouldin, 4))
print("Calinski-Harabasz:", round(final_calinski_harabasz, 4))
print("Results:", OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(
            f"{path.name} - "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )

gc.collect()
