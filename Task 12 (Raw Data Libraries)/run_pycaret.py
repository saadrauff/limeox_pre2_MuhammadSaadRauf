
import time
import json
import traceback
import warnings

import numpy as np
import pandas as pd

from pathlib import Path

from pycaret.classification import ClassificationExperiment

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SHARED_DIR = BASE_DIR / "_shared"
DATA_FILE = SHARED_DIR / "prepared_data.npz"

OUTPUT_DIR = BASE_DIR / "results" / "06_PyCaret"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TIME_BUDGET = 600
CV_FOLDS = 3
CPU_JOBS = 2

TARGET_COLUMN = "category"

# LightGBM excluded temporarily because of the
# native access violation encountered previously.

MODELS = [
    "lr",
    "dt",
    "rf",
    "et"
]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading prepared dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    with np.load(DATA_FILE, allow_pickle=False) as data:

        X_train = data["X_train"].copy()
        X_test = data["X_test"].copy()

        y_train = data["y_train"].astype(str).copy()
        y_test = data["y_test"].astype(str).copy()

    print("\nDataset loaded successfully!")

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("Training records:", len(y_train))
    print("Testing records:", len(y_test))

    print("Number of features:", X_train.shape[1])

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. VALIDATE DATA
# ============================================================

def validate_data(X_train, X_test, y_train, y_test):

    print("\nValidating dataset...")

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Features must be 2D arrays.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature counts differ.")

    if len(X_train) != len(y_train):
        raise ValueError("Training data size mismatch.")

    if len(X_test) != len(y_test):
        raise ValueError("Testing data size mismatch.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training data contains invalid values.")

    if not np.isfinite(X_test).all():
        raise ValueError("Testing data contains invalid values.")

    if set(np.unique(y_train)) != set(np.unique(y_test)):
        raise ValueError("Training/testing classes differ.")

    print("Dataset validation successful.")
    print("Number of classes:", len(np.unique(y_train)))
    print("Number of features:", X_train.shape[1])


# ============================================================
# 4. CREATE DATAFRAMES
# ============================================================

def create_dataframes(X_train, X_test, y_train):

    print("\nCreating DataFrames...")

    feature_names = [
        f"feature_{i + 1}"
        for i in range(X_train.shape[1])
    ]

    X_train_df = pd.DataFrame(
        X_train,
        columns=feature_names
    )

    X_test_df = pd.DataFrame(
        X_test,
        columns=feature_names
    )

    train_df = X_train_df.copy()
    train_df[TARGET_COLUMN] = y_train

    print("Training DataFrame:", train_df.shape)
    print("Testing DataFrame:", X_test_df.shape)

    return train_df, X_test_df


# ============================================================
# 5. PYCARET SETUP
# ============================================================

def initialize_pycaret(train_df):

    print("\n" + "=" * 65)
    print("PYCARET MODEL SETUP")
    print("=" * 65)

    exp = ClassificationExperiment()

    start = time.time()

    exp.setup(
        data=train_df,
        target=TARGET_COLUMN,

        session_id=RANDOM_STATE,

        fold=CV_FOLDS,
        fold_strategy="stratifiedkfold",
        fold_shuffle=True,

        preprocess=False,
        normalize=False,
        fix_imbalance=False,

        n_jobs=CPU_JOBS,

        html=False,
        verbose=False,
        memory=False
    )

    setup_time = time.time() - start

    print("\nPyCaret setup completed.")
    print(f"Setup time: {setup_time:.2f} seconds")

    return exp, setup_time


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

def compare_models(exp):

    print("\n" + "=" * 65)
    print("PYCARET MODEL COMPARISON")
    print("=" * 65)

    print("Time budget:", TIME_BUDGET, "seconds")
    print("Cross-validation folds:", CV_FOLDS)
    print("CPU jobs:", CPU_JOBS)

    print("Models:", MODELS)
    print("Selection metric: PyCaret built-in F1")

    start = time.time()

    best_model = exp.compare_models(
        include=MODELS,

        sort="F1",

        n_select=1,

        fold=CV_FOLDS,

        budget_time=TIME_BUDGET,

        turbo=True,

        errors="ignore",

        verbose=False
    )

    comparison_time = time.time() - start

    if best_model is None:
        raise RuntimeError(
            "PyCaret could not train a valid model."
        )

    comparison = exp.pull()

    if comparison is None or comparison.empty:
        raise RuntimeError(
            "PyCaret returned an empty comparison table."
        )

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=True
    )

    print("\nModel comparison completed.")

    print("\nComparison results:")
    print(comparison.to_string())

    print("\nBest model:", type(best_model).__name__)

    print(
        f"Comparison time: {comparison_time:.2f} seconds"
    )

    return best_model, comparison, comparison_time


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

def generate_predictions(best_model, X_test_df):

    print("\n" + "=" * 65)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 65)

    start = time.time()

    # Direct sklearn-compatible prediction.
    # Avoids PyCaret predict_model() output processing.

    y_pred = np.asarray(
        best_model.predict(X_test_df)
    ).ravel().astype(str)

    prediction_time = time.time() - start

    print("\nPredictions generated successfully.")

    print("Prediction records:", len(y_pred))

    print(
        f"Prediction time: {prediction_time:.2f} seconds"
    )

    return y_pred, prediction_time


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

def evaluate_model(y_test, y_pred):

    print("\n" + "=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "Library": "PyCaret",
        "Accuracy": float(accuracy),
        "Macro Precision": float(precision),
        "Macro Recall": float(recall),
        "Macro F1": float(macro_f1),
        "Weighted F1": float(weighted_f1)
    }

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    return metrics


# ============================================================
# 9. SAVE EVALUATION RESULTS
# ============================================================

def save_results(y_test, y_pred, metrics):

    print("\n" + "=" * 65)
    print("SAVING EVALUATION RESULTS")
    print("=" * 65)

    # Classification report

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    # Confusion matrix

    labels = sorted(np.unique(y_test))

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{x}" for x in labels],
        columns=[f"Predicted_{x}" for x in labels]
    )

    matrix_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print("\nConfusion Matrix:")
    print(matrix_df)

    # Predictions

    pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    }).to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    # Metrics

    with open(
        OUTPUT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    print("\nAll evaluation files saved successfully.")


# ============================================================
# 10. SAVE MODEL
# ============================================================

def save_best_model(exp, best_model):

    print("\n" + "=" * 65)
    print("SAVING BEST MODEL")
    print("=" * 65)

    exp.save_model(
        best_model,
        str(OUTPUT_DIR / "best_pycaret_model")
    )

    print("Best model saved successfully.")


# ============================================================
# 11. MAIN
# ============================================================

def main():

    total_start = time.time()

    print("\n" + "=" * 65)
    print("TASK 12 - PYCARET AUTOML")
    print("=" * 65)

    # Load data

    X_train, X_test, y_train, y_test = load_data()

    # Validate

    validate_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Create DataFrames

    train_df, X_test_df = create_dataframes(
        X_train,
        X_test,
        y_train
    )

    # Setup

    exp, setup_time = initialize_pycaret(train_df)

    # Model comparison

    best_model, comparison, comparison_time = compare_models(
        exp
    )

    # Predictions

    y_pred, prediction_time = generate_predictions(
        best_model,
        X_test_df
    )

    if len(y_pred) != len(y_test):
        raise ValueError(
            "Prediction count does not match test records."
        )

    # Evaluation

    metrics = evaluate_model(
        y_test,
        y_pred
    )

    metrics["Best Estimator"] = type(best_model).__name__

    metrics["Setup Time (seconds)"] = round(
        setup_time, 2
    )

    metrics["Comparison Time (seconds)"] = round(
        comparison_time, 2
    )

    metrics["Prediction Time (seconds)"] = round(
        prediction_time, 2
    )

    # Save results

    save_results(
        y_test,
        y_pred,
        metrics
    )

    # Save model

    save_best_model(
        exp,
        best_model
    )

    total_time = time.time() - total_start

    metrics["Total Execution Time (seconds)"] = round(
        total_time, 2
    )

    with open(
        OUTPUT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    # Final summary

    print("\n" + "=" * 65)
    print("PYCARET TRAINING SUMMARY")
    print("=" * 65)

    print("Best estimator:", type(best_model).__name__)

    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Macro Precision: {metrics['Macro Precision']:.4f}")
    print(f"Macro Recall: {metrics['Macro Recall']:.4f}")
    print(f"Macro F1: {metrics['Macro F1']:.4f}")
    print(f"Weighted F1: {metrics['Weighted F1']:.4f}")

    print(
        f"Total execution time: {total_time / 60:.2f} minutes"
    )

    print("\nResults saved at:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.iterdir()):
        print("-", file.name)

    print("\nPYCARET COMPLETED SUCCESSFULLY!")

    print("=" * 65)


# ============================================================
# 12. EXECUTION
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n" + "=" * 65)
        print("PYCARET EXECUTION FAILED")
        print("=" * 65)

        print("Error:", error)

        traceback.print_exc()

        raise
