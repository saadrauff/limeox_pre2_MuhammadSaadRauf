
import time
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from flaml import AutoML

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SHARED_DIR = BASE_DIR / "_shared"
DATA_FILE = SHARED_DIR / "prepared_data.npz"

OUTPUT_DIR = BASE_DIR / "04_FLAML"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TIME_BUDGET = 600


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

def load_prepared_data():

    print("\nLoading prepared dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_FILE}\n"
            "Please run prepare_data.py first."
        )

    with np.load(DATA_FILE, allow_pickle=False) as data:

        X_train = data["X_train"].copy()
        X_test = data["X_test"].copy()

        y_train = data["y_train"].copy()
        y_test = data["y_test"].copy()

    print("\nDataset loaded successfully!")

    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")

    print(f"Training records: {len(y_train)}")
    print(f"Testing records: {len(y_test)}")

    print(f"Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. SAVE EVALUATION RESULTS
# ============================================================

def save_evaluation(y_test, y_pred, elapsed, best_estimator):

    print("\nEvaluating FLAML model...")

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

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    results = {
        "Library": "FLAML",
        "Best Estimator": str(best_estimator),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "Macro F1": float(f1),
        "Training Time (seconds)": float(elapsed)
    }

    # Save metrics
    metrics_file = OUTPUT_DIR / "evaluation_metrics.json"

    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Save classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # Save confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    pd.DataFrame(matrix).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        index=False
    )

    # Save predictions
    pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    }).to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    print("\nEvaluation completed!")

    print("\nPerformance Metrics")
    print("-" * 45)

    for key, value in results.items():

        if isinstance(value, float) and key != "Training Time (seconds)":
            print(f"{key}: {value:.4f}")

        else:
            print(f"{key}: {value}")

    print("\nClassification Report:")
    print(report)

    return results


# ============================================================
# 4. MAIN FLAML TRAINING
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("TASK 12 - FLAML AUTOML")
    print("=" * 65)

    # Load shared prepared dataset
    X_train, X_test, y_train, y_test = load_prepared_data()

    # Initialize FLAML
    automl = AutoML()

    print("\nInitializing FLAML...")

    print(f"Time budget: {TIME_BUDGET} seconds")
    print("Task: Classification")
    print("Metric: Macro F1")

    print("\nEstimators:")
    print("1. LightGBM")
    print("2. XGBoost")
    print("3. Random Forest")
    print("4. Extra Trees")

    # Start training timer
    start_time = time.time()

    print("\nTraining FLAML model...")

    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        metric="macro_f1",
        time_budget=TIME_BUDGET,
        estimator_list=[
            "lgbm",
            "xgboost",
            "rf",
            "extra_tree"
        ],
        eval_method="holdout",
        split_type="stratified",
        seed=RANDOM_STATE,
        n_jobs=4,
        log_file_name=str(
            OUTPUT_DIR / "flaml_training.log"
        )
    )

    elapsed = time.time() - start_time

    print("\nTraining completed successfully!")

    # Generate predictions
    print("\nGenerating test predictions...")

    y_pred = np.asarray(
        automl.predict(X_test)
    ).astype(str)

    # Save trained model
    model_file = OUTPUT_DIR / "best_flaml_model.joblib"

    joblib.dump(automl, model_file)

    # Save model information
    model_info = {
        "library": "FLAML",
        "best_estimator": str(automl.best_estimator),
        "best_config": automl.best_config,
        "best_validation_loss": float(automl.best_loss),
        "training_time_seconds": round(elapsed, 2),
        "training_records": len(y_train),
        "testing_records": len(y_test),
        "features": X_train.shape[1],
        "time_budget_seconds": TIME_BUDGET,
        "random_state": RANDOM_STATE
    }

    with open(
        OUTPUT_DIR / "best_model_info.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model_info,
            file,
            indent=4,
            default=str
        )

    # Evaluate model
    results = save_evaluation(
        y_test,
        y_pred,
        elapsed,
        automl.best_estimator
    )

    # Final summary
    print("\n" + "=" * 65)
    print("FLAML TRAINING SUMMARY")
    print("=" * 65)

    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best validation loss: {automl.best_loss}")

    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall: {results['Recall']:.4f}")
    print(f"Macro F1: {results['Macro F1']:.4f}")

    print(f"Training time: {elapsed:.2f} seconds")

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\nGenerated files:")
    print("1. best_flaml_model.joblib")
    print("2. best_model_info.json")
    print("3. evaluation_metrics.json")
    print("4. classification_report.txt")
    print("5. confusion_matrix.csv")
    print("6. predictions.csv")
    print("7. flaml_training.log")

    print("\nFLAML completed successfully!")

    print("=" * 65)


# ============================================================
# 5. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
