
import joblib
import pandas as pd

from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR / "model_results" /
    "random_forest" / "random_forest_model.pkl"
)

DATA_FILE = (
    BASE_DIR.parent /
    "Task 8 (Feature Engineering)" /
    "news_engineered_features.csv"
)

ORIGINAL_DATA_FILE = (
    BASE_DIR.parent /
    "Task 5 (Categorization)" /
    "news_categorized.csv"
)

TRAIN_DATA_FILE = (
    BASE_DIR / "model_results" /
    "train_test_data.pkl"
)

OUTPUT_DIR = BASE_DIR / "model_results"
OUTPUT_FILE = OUTPUT_DIR / "new_article_predictions.csv"


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

print("Loading trained Random Forest model...")
model = joblib.load(MODEL_FILE)


# ==========================================
# 3. LOAD DATASETS
# ==========================================

print("Loading engineered article data...")
df = pd.read_csv(DATA_FILE)

print("Loading original article data...")
original_df = pd.read_csv(ORIGINAL_DATA_FILE)


# ==========================================
# 4. LOAD TRAINING FEATURES
# ==========================================

data = joblib.load(TRAIN_DATA_FILE)

required_columns = data["X_train"].columns.tolist()


# ==========================================
# 5. MATCH ARTICLE TITLE USING URL
# ==========================================

# Check required URL columns
if "url" not in df.columns:
    raise ValueError("URL column missing from engineered dataset.")

if "news_url_clean" not in original_df.columns:
    raise ValueError("URL column missing from original dataset.")

if "title_clean" not in original_df.columns:
    raise ValueError("Title column missing from original dataset.")

# Remove duplicate URLs from original dataset
original_df = original_df.drop_duplicates(
    subset=["news_url_clean"],
    keep="first"
)

# Select one article for demonstration
sample = df.dropna(subset=["category"]).iloc[[0]].copy()

# Match original title with engineered article using URL
matched_article = sample.merge(
    original_df[
        ["news_url_clean", "title_clean"]
    ],
    left_on="url",
    right_on="news_url_clean",
    how="left",
    validate="many_to_one"
)

article_title = matched_article.iloc[0]["title_clean"]
article_url = sample.iloc[0]["url"]

if pd.isna(article_title):
    article_title = "Title not found"


# ==========================================
# 6. SELECT REQUIRED FEATURES
# ==========================================

missing_columns = [
    col for col in required_columns
    if col not in sample.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required features: {missing_columns}"
    )

X_new = sample[required_columns]


# ==========================================
# 7. PREDICT CATEGORY
# ==========================================

print("\nPredicting article category...")

prediction = model.predict(X_new)[0]

probabilities = model.predict_proba(X_new)[0]

classes = model.named_steps["classifier"].classes_


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n========== ARTICLE PREDICTION ==========")

print("\nArticle Title:", article_title)

print("\nArticle URL:", article_url)

print("\nPredicted Category:", prediction)

print("\nCategory Probabilities:")

for category, probability in sorted(
    zip(classes, probabilities),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{category}: {probability * 100:.2f}%")

confidence = max(probabilities) * 100

print(f"\nPrediction Confidence: {confidence:.2f}%")


# ==========================================
# 9. SAVE PREDICTION
# ==========================================

result = pd.DataFrame([{
    "Article_Title": article_title,
    "Article_URL": article_url,
    "Predicted_Category": prediction,
    "Prediction_Confidence_Percent": confidence
}])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

result.to_csv(OUTPUT_FILE, index=False)

print("\nPrediction saved successfully!")
print("Output:", OUTPUT_FILE)
