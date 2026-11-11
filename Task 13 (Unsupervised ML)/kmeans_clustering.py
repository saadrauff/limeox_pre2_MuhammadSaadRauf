
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path("Task 13 (Unsupervised ML)")
DATA_FILE = Path("Task 8 (Feature Engineering)") / "news_engineered_features.csv"
RESULTS_DIR = BASE_DIR / "model_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 10000
SILHOUETTE_SAMPLE_SIZE = 3000
K_VALUES = range(2, 11)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# Preserve information for final results
metadata = df[["url", "category"]].copy()

# ==========================================
# 3. FEATURE SELECTION
# ==========================================

print("\nSelecting numerical features...")

X = df.select_dtypes(include=np.number).copy()

# Remove columns that contain only missing values
X.dropna(axis=1, how="all", inplace=True)

# Remove constant features
X = X.loc[:, X.nunique(dropna=True) > 1]

print("Selected features:", len(X.columns))
print("Features:", X.columns.tolist())

# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

print("\nHandling missing values...")

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

# ==========================================
# 5. FEATURE SCALING
# ==========================================

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("Scaled dataset shape:", X_scaled.shape)

# ==========================================
# 6. CREATE REPRESENTATIVE SAMPLE
# ==========================================

print("\nPreparing sample for K selection...")

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(SAMPLE_SIZE, len(X_scaled))

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_sample = X_scaled[sample_indices]

silhouette_size = min(
    SILHOUETTE_SAMPLE_SIZE,
    len(X_sample)
)

silhouette_indices = rng.choice(
    len(X_sample),
    size=silhouette_size,
    replace=False
)

X_silhouette = X_sample[silhouette_indices]

# ==========================================
# 7. ELBOW METHOD AND SILHOUETTE ANALYSIS
# ==========================================

print("\nRunning Elbow Method and Silhouette Analysis...")

inertias = []
silhouette_scores = []
candidate_models = {}

for k in K_VALUES:

    print(f"\nTesting K = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=5,
        max_iter=200
    )

    model.fit(X_sample)

    labels = model.predict(X_silhouette)

    score = silhouette_score(
        X_silhouette,
        labels
    )

    inertias.append(model.inertia_)
    silhouette_scores.append(score)

    candidate_models[k] = model

    print("Inertia:", round(model.inertia_, 2))
    print("Silhouette Score:", round(score, 4))

# Save evaluation results
evaluation_df = pd.DataFrame({
    "K": list(K_VALUES),
    "Inertia": inertias,
    "Silhouette_Score": silhouette_scores
})

evaluation_df.to_csv(
    RESULTS_DIR / "k_selection_metrics.csv",
    index=False
)

# Select K with the highest sampled silhouette score
best_k = int(
    evaluation_df.loc[
        evaluation_df["Silhouette_Score"].idxmax(),
        "K"
    ]
)

print("\nBest K based on sampled Silhouette Score:", best_k)

# ==========================================
# 8. ELBOW VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(list(K_VALUES))
plt.grid(alpha=0.3)

plt.savefig(
    RESULTS_DIR / "elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 9. SILHOUETTE VISUALIZATION
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for K-Means")
plt.xticks(list(K_VALUES))
plt.grid(alpha=0.3)

plt.savefig(
    RESULTS_DIR / "silhouette_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 10. FINAL K-MEANS MODEL
# ==========================================

print("\nTraining final K-Means model on ALL articles...")

final_model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=5,
    max_iter=300
)

final_labels = final_model.fit_predict(X_scaled)

print("Final clustering completed!")

# ==========================================
# 11. FINAL EVALUATION
# ==========================================

print("\nCalculating final evaluation metrics...")

final_silhouette = silhouette_score(
    X_scaled,
    final_labels,
    sample_size=min(10000, len(X_scaled)),
    random_state=RANDOM_STATE
)

final_metrics = pd.DataFrame({
    "Metric": [
        "Total Articles",
        "Number of Clusters",
        "Number of Features",
        "Inertia",
        "Silhouette Score"
    ],
    "Value": [
        len(df),
        best_k,
        X.shape[1],
        final_model.inertia_,
        final_silhouette
    ]
})

final_metrics.to_csv(
    RESULTS_DIR / "cluster_metrics.csv",
    index=False
)

print(final_metrics.to_string(index=False))

# ==========================================
# 12. SAVE CLUSTERED ARTICLES
# ==========================================

print("\nSaving clustered articles...")

results = metadata.copy()

results["cluster"] = final_labels

results.to_csv(
    RESULTS_DIR / "clustered_articles.csv",
    index=False
)

# ==========================================
# 13. CLUSTER DISTRIBUTION
# ==========================================

cluster_distribution = (
    results["cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("cluster")
    .reset_index(name="article_count")
)

cluster_distribution.to_csv(
    RESULTS_DIR / "cluster_distribution.csv",
    index=False
)

print("\nCluster Distribution:")
print(cluster_distribution.to_string(index=False))

# ==========================================
# 14. CATEGORY VS CLUSTER ANALYSIS
# ==========================================

print("\nAnalyzing actual categories against clusters...")

category_analysis = pd.crosstab(
    results["cluster"],
    results["category"]
)

category_analysis.to_csv(
    RESULTS_DIR / "category_cluster_analysis.csv"
)

print(category_analysis.to_string())

# ==========================================
# 15. CLUSTER FEATURE PROFILES
# ==========================================

profile_df = pd.DataFrame(
    X_imputed,
    columns=X.columns
)

profile_df["cluster"] = final_labels

cluster_profiles = (
    profile_df.groupby("cluster")
    .mean()
    .round(3)
)

cluster_profiles.to_csv(
    RESULTS_DIR / "cluster_profiles.csv"
)

# ==========================================
# 16. PCA VISUALIZATION
# ==========================================

print("\nGenerating PCA cluster visualization...")

pca_sample_size = min(10000, len(X_scaled))

pca_indices = rng.choice(
    len(X_scaled),
    size=pca_sample_size,
    replace=False
)

X_pca_sample = X_scaled[pca_indices]
pca_labels = final_labels[pca_indices]

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(X_pca_sample)

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pca_labels,
    cmap="viridis",
    s=12,
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering Visualization using PCA")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.savefig(
    RESULTS_DIR / "pca_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 17. SAVE TRAINED MODEL
# ==========================================

print("\nSaving trained model...")

joblib.dump(
    {
        "model": final_model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_columns": X.columns.tolist(),
        "best_k": best_k,
        "random_state": RANDOM_STATE
    },
    RESULTS_DIR / "kmeans_model.pkl"
)

# ==========================================
# 18. COMPLETION SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Total Articles:", len(df))
print("Selected Features:", X.shape[1])
print("Optimal K:", best_k)
print("Silhouette Score:", round(final_silhouette, 4))
print("Results Folder:", RESULTS_DIR)

print("\nGenerated Files:")

for file in sorted(RESULTS_DIR.iterdir()):
    if file.is_file():
        print("-", file.name)

print("\nTASK 13 EXECUTION FINISHED!")
