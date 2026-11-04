
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------
# 1. PATHS
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "Task 8 (Feature Engineering)"
    / "news_engineered_features.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "model_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = OUTPUT_DIR / "train_test_data.pkl"


# ---------------------------------------
# 2. LOAD DATA
# ---------------------------------------

print("Loading engineered dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original Dataset Shape:", df.shape)

if "category" not in df.columns:
    raise ValueError("Category column not found!")

df = df.dropna(subset=["category"]).copy()

# Target
y = df["category"].astype(str)

# Remove target and identifier/raw URL
X = df.drop(
    columns=["category", "url", "source_row_index"],
    errors="ignore"
)

# Remove completely empty columns
X = X.dropna(axis=1, how="all")

# Remove high-cardinality text columns
high_cardinality = [
    col for col in X.select_dtypes(
        include=["object", "category"]
    ).columns
    if X[col].nunique(dropna=True) > 200
]

X = X.drop(columns=high_cardinality)

print("Features Shape:", X.shape)
print("\nCategory Distribution:")
print(y.value_counts())


# ---------------------------------------
# 3. IDENTIFY FEATURE TYPES
# ---------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))


# ---------------------------------------
# 4. TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ---------------------------------------
# 5. PREPROCESSING PIPELINE
# ---------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ---------------------------------------
# 6. SAVE COMMON DATA
# ---------------------------------------

data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "preprocessor": preprocessor,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "classes": sorted(y.unique())
}

joblib.dump(data, SPLIT_FILE)

print("\nPreprocessing completed!")
print("Training and testing data saved.")
print("Output:", SPLIT_FILE)
