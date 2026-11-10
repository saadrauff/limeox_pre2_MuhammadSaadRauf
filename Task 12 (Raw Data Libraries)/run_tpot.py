
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from tpot import TPOTClassifier

from sklearn.preprocessing import LabelEncoder

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

OUTPUT_DIR = BASE_DIR / "results" / "05_TPOT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

SEARCH_SPACE = "linear"

GENERATIONS = None
POPULATION_SIZE = 10

MAX_TIME_MINS = 10
MAX_EVAL_TIME_MINS = 2

CV_FOLDS = 3
N_JOBS = 2

SCORING = "f1_macro"


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

def load_prepared_data():

    print("\nLoading prepared dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_FILE}"
        )

    with np.load(DATA_FILE, allow_pickle=False) as data:

        X_train = data["X_train"].copy()
        X_test = data["X_test"].copy()

        y_train = data["y_train"].copy()
        y_test = data["y_test"].copy()

    y_train = np.asarray(y_train).astype(str)
    y_test = np.asarray(y_test).astype(str)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Features must be two-dimensional.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and testing feature counts differ.")

    if len(X_train) != len(y_train):
        raise ValueError("Training feature/label count mismatch.")

    if len(X_test) != len(y_test):
        raise ValueError("Testing feature/label count mismatch.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training data contains invalid values.")

    if not np.isfinite(X_test).all():
        raise ValueError("Testing data contains invalid values.")

    if np.any(y_train == "") or np.any(y_test == ""):
        raise ValueError("Empty category labels detected.")

    train_categories = set(np.unique(y_train))
    test_categories = set(np.unique(y_test))

    if len(train_categories) < 2:
        raise ValueError("At least two training categories are required.")

    if not test_categories.issubset(train_categories):
        raise ValueError("Testing data contains unseen categories.")

    print("\nDataset loaded successfully!")

    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")

    print(f"Training records: {len(y_train):,}")
    print(f"Testing records: {len(y_test):,}")

    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of categories: {len(train_categories)}")

    print("\nTraining category distribution:")

    categories, counts = np.unique(
        y_train,
        return_counts=True
    )

    for category, count in zip(categories, counts):
        print(f"{category}: {count:,}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. SAVE EVALUATION RESULTS
# ============================================================

def save_evaluation(
    y_test,
    y_pred,
    elapsed,
    best_pipeline,
    y_train,
    n_features,
    categories
):

    print("\nEvaluating TPOT model...")

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

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results = {

        "Library": "TPOT",

        "Search Space": SEARCH_SPACE,

        "Best Pipeline": str(best_pipeline),

        "Accuracy": float(accuracy),

        "Precision": float(precision),

        "Recall": float(recall),

        "Macro F1": float(macro_f1),

        "Weighted Precision": float(weighted_precision),

        "Weighted Recall": float(weighted_recall),

        "Weighted F1": float(weighted_f1),

        "Training Time (seconds)": round(elapsed, 2),

        "Training Time (minutes)": round(elapsed / 60, 2),

        "Training Records": int(len(y_train)),

        "Testing Records": int(len(y_test)),

        "Features": int(n_features),

        "Categories": categories.tolist(),

        "Number of Categories": int(len(categories))

    }

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(results, file, indent=4)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=categories,
        target_names=categories,
        zero_division=0
    )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=categories
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=categories,
        columns=categories
    )

    matrix_df.index.name = "Actual Category"
    matrix_df.columns.name = "Predicted Category"

    matrix_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.DataFrame({

        "Actual": y_test,

        "Predicted": y_pred,

        "Correct": y_test == y_pred

    })

    predictions.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TPOT MODEL EVALUATION")
    print("=" * 65)

    for key, value in results.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nClassification Report:\n")
    print(report)

    print("\nEvaluation files saved successfully!")

    return results


# ============================================================
# 4. SAVE SEARCH RESULTS
# ============================================================

def save_search_results(model):

    print("\nSaving TPOT search results...")

    evaluated = getattr(
        model,
        "evaluated_individuals",
        None
    )

    if isinstance(evaluated, pd.DataFrame):

        evaluated.to_csv(
            OUTPUT_DIR / "tpot_search_results.csv",
            index=True
        )

        print("Search results saved.")

    elif isinstance(evaluated, dict):

        pd.DataFrame.from_dict(
            evaluated,
            orient="index"
        ).to_csv(
            OUTPUT_DIR / "tpot_search_results.csv"
        )

        print("Search results saved.")

    else:

        print("Search results unavailable.")

    pareto = getattr(
        model,
        "pareto_front",
        None
    )

    if isinstance(pareto, pd.DataFrame):

        pareto.to_csv(
            OUTPUT_DIR / "tpot_pareto_front.csv",
            index=True
        )

        print("Pareto front saved.")


# ============================================================
# 5. MAIN TPOT TRAINING
# ============================================================

def main():

    start_total = time.time()

    print("\n" + "=" * 65)
    print("TASK 12 - TPOT AUTOML CLASSIFICATION")
    print("=" * 65)

    print("\nMode: FRESH OPTIMIZATION")
    print("TPOT Version: 1.1.0")
    print("Using the same prepared dataset as FLAML.")
    print("Original category names will be used for evaluation.")
    print("Existing shared dataset will NOT be modified.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_prepared_data()

    # --------------------------------------------------------
    # ENCODE LABELS FOR TPOT
    # --------------------------------------------------------

    print("\nPreparing category labels for TPOT...")

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)

    categories = label_encoder.classes_

    print("\nLabel encoding:")

    for index, category in enumerate(categories):
        print(f"{index} = {category}")

    # Ensure test labels are known to the encoder.
    if not set(np.unique(y_test)).issubset(set(categories)):
        raise ValueError("Testing data contains unknown categories.")

    # --------------------------------------------------------
    # INITIALIZE TPOT
    # --------------------------------------------------------

    print("\nInitializing TPOT Classifier...")

    model = TPOTClassifier(

        search_space=SEARCH_SPACE,

        scorers=[SCORING],

        generations=GENERATIONS,

        population_size=POPULATION_SIZE,

        max_time_mins=MAX_TIME_MINS,

        max_eval_time_mins=MAX_EVAL_TIME_MINS,

        cv=CV_FOLDS,

        n_jobs=N_JOBS,

        random_state=RANDOM_STATE,

        verbose=2,

        validation_strategy="none",

        preprocessing=False,

        allow_inner_classifiers=True

    )

    print("\nTPOT Configuration")
    print("-" * 45)

    print(f"Search space: {SEARCH_SPACE}")
    print(f"Maximum search time: {MAX_TIME_MINS} minutes")
    print(f"Maximum evaluation time: {MAX_EVAL_TIME_MINS} minutes")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print(f"Scoring: {SCORING}")
    print(f"Parallel jobs: {N_JOBS}")

    # --------------------------------------------------------
    # START OPTIMIZATION
    # --------------------------------------------------------

    print("\nStarting TPOT optimization...")

    optimization_start = time.time()

    model.fit(
        X_train,
        y_train_encoded
    )

    optimization_elapsed = (
        time.time() - optimization_start
    )

    print("\nTPOT optimization completed!")

    # --------------------------------------------------------
    # GET BEST PIPELINE
    # --------------------------------------------------------

    best_pipeline = getattr(
        model,
        "fitted_pipeline_",
        None
    )

    if best_pipeline is None:
        raise RuntimeError(
            "TPOT did not produce a fitted pipeline."
        )

    print("\nBest pipeline selected successfully.")
    print("\nBest Pipeline:\n")
    print(best_pipeline)

    # --------------------------------------------------------
    # SAVE PIPELINE AND LABEL ENCODER
    # --------------------------------------------------------

    joblib.dump(
        best_pipeline,
        OUTPUT_DIR / "best_tpot_pipeline.joblib"
    )

    joblib.dump(
        label_encoder,
        OUTPUT_DIR / "label_encoder.joblib"
    )

    print("\nBest pipeline and label encoder saved.")

    # --------------------------------------------------------
    # GENERATE TEST PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    prediction_start = time.time()

    # TPOT was trained using encoded labels.
    raw_predictions = np.asarray(
        model.predict(X_test)
    )

    prediction_elapsed = (
        time.time() - prediction_start
    )

    print(
        f"Prediction completed in {prediction_elapsed:.2f} seconds."
    )

    # --------------------------------------------------------
    # DECODE PREDICTIONS
    # --------------------------------------------------------

    try:

        prediction_indices = raw_predictions.astype(int)

    except (ValueError, TypeError) as error:

        raise ValueError(
            "TPOT returned predictions that cannot be decoded "
            "as encoded category indices."
        ) from error

    valid_indices = np.arange(len(categories))

    if not np.isin(
        prediction_indices,
        valid_indices
    ).all():

        raise ValueError(
            "TPOT returned category indices outside the "
            "LabelEncoder range."
        )

    # Convert encoded predictions back to original labels.
    y_pred = label_encoder.inverse_transform(
        prediction_indices
    )

    # --------------------------------------------------------
    # VALIDATE PREDICTIONS
    # --------------------------------------------------------

    if len(y_pred) != len(y_test):

        raise ValueError(
            "Prediction count does not match testing records."
        )

    if not set(np.unique(y_pred)).issubset(set(categories)):

        raise ValueError(
            "Decoded predictions contain unknown categories."
        )

    print("\nOriginal category labels successfully restored.")

    print("\nPredicted categories:")
    print(np.unique(y_pred))

    # --------------------------------------------------------
    # SAVE EVALUATION
    # --------------------------------------------------------

    results = save_evaluation(

        y_test=y_test,

        y_pred=y_pred,

        elapsed=optimization_elapsed,

        best_pipeline=best_pipeline,

        y_train=y_train,

        n_features=X_train.shape[1],

        categories=categories

    )

    # --------------------------------------------------------
    # SAVE SEARCH RESULTS
    # --------------------------------------------------------

    save_search_results(model)

    # --------------------------------------------------------
    # SAVE MODEL INFORMATION
    # --------------------------------------------------------

    model_info = {

        "library": "TPOT",

        "version": "1.1.0",

        "search_space": SEARCH_SPACE,

        "best_pipeline": str(best_pipeline),

        "training_records": int(len(y_train)),

        "testing_records": int(len(y_test)),

        "features": int(X_train.shape[1]),

        "categories": categories.tolist(),

        "time_budget_minutes": MAX_TIME_MINS,

        "max_eval_time_minutes": MAX_EVAL_TIME_MINS,

        "population_size": POPULATION_SIZE,

        "cv_folds": CV_FOLDS,

        "scoring": SCORING,

        "random_state": RANDOM_STATE,

        "optimization_time_seconds": round(
            optimization_elapsed,
            2
        )

    }

    with open(
        OUTPUT_DIR / "best_model_info.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model_info,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total_elapsed = time.time() - start_total

    print("\n" + "=" * 65)
    print("TPOT TRAINING SUMMARY")
    print("=" * 65)

    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall: {results['Recall']:.4f}")
    print(f"Macro F1: {results['Macro F1']:.4f}")
    print(f"Weighted F1: {results['Weighted F1']:.4f}")

    print(
        f"Optimization time: {optimization_elapsed:.2f} seconds"
    )

    print(
        f"Total execution time: {total_elapsed:.2f} seconds"
    )

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\nGenerated files:")

    print("1. best_tpot_pipeline.joblib")
    print("2. label_encoder.joblib")
    print("3. best_model_info.json")
    print("4. evaluation_metrics.json")
    print("5. classification_report.txt")
    print("6. confusion_matrix.csv")
    print("7. predictions.csv")
    print("8. tpot_search_results.csv (if available)")
    print("9. tpot_pareto_front.csv (if available)")

    print("\nTPOT EXECUTION COMPLETED SUCCESSFULLY!")

    print("=" * 65)


# ============================================================
# 6. EXECUTION
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nTPOT EXECUTION FAILED!")
        print(f"Error: {error}")

        raise
