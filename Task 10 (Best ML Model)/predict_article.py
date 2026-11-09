
import joblib
import re
from pathlib import Path

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "model_results"
    / "text_classification"
    / "tfidf_svm_model.pkl"
)

print("=" * 60)
print("LIME OX - NEWS CATEGORY PREDICTION SYSTEM")
print("=" * 60)

print("\nLoading trained TF-IDF Linear SVM model...")

saved_data = joblib.load(MODEL_FILE)

vectorizer = saved_data["vectorizer"]
model = saved_data["model"]
classes = saved_data["classes"]

print("Model loaded successfully!")

# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================

def predict_category(title):

    title = clean_text(title)

    if not title:
        print("\nPlease enter a valid news title!")
        return

    # Same title weighting used during training.
    combined_text = title + " " + title

    # Convert title into TF-IDF numerical features.
    title_features = vectorizer.transform(
        [combined_text]
    )

    # Predict category using trained SVM.
    prediction = model.predict(title_features)[0]

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print("\nNews Title:")
    print(title)

    print("\nPredicted Category:")
    print(f">>> {prediction.upper()} <<<")

    print("=" * 60)


# ==========================================
# 4. USER INPUT LOOP
# ==========================================

while True:

    print("\nEnter a news headline below.")

    title = input("\nNews Title (or type exit): ").strip()

    if title.lower() in ["exit", "quit"]:
        print("\nThank you for using LimeoX!")
        break

    predict_category(title)
