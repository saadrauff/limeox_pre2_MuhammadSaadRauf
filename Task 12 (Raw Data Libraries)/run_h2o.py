
import json
import time
import traceback
from pathlib import Path

import h2o
import numpy as np
import pandas as pd

from h2o.automl import H2OAutoML
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# TASK 12 - H2O AUTOML CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "_shared" / "prepared_data.npz"
OUTPUT_DIR = BASE_DIR / "07_H2O_AutoML"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RUNTIME_SECONDS = 600
MAX_MODELS = 12
RANDOM_STATE = 42

MAX_MEMORY = "4G"
N_THREADS = -1

TARGET_COLUMN = "target"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_json(data, file_path):
    """Save dictionary data as JSON."""

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def load_data():
    """Load the prepared train/test dataset."""

    print("\nLoading prepared dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = np.load(DATA_FILE, allow_pickle=True)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("\nDataset loaded successfully!")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training records:", len(y_train))
    print("Testing records:", len(y_test))
    print("Number of features:", X_train.shape[1])

    return X_train, X_test, y_train, y_test


def prepare_dataframes(X_train, X_test, y_train, y_test):
    """Convert NumPy arrays into Pandas DataFrames."""

    print("\nPreparing DataFrames...")

    feature_columns = [
        f"feature_{i}"
        for i in range(X_train.shape[1])
    ]

    train_df = pd.DataFrame(
        X_train,
        columns=feature_columns
    )

    test_df = pd.DataFrame(
        X_test,
        columns=feature_columns
    )

    # Keep the same class levels in train and test.
    classes = np.unique(
        np.concatenate([y_train, y_test])
    )

    train_df[TARGET_COLUMN] = pd.Categorical(
        y_train,
        categories=classes
    )

    test_df[TARGET_COLUMN] = pd.Categorical(
        y_test,
        categories=classes
    )

    print("DataFrames created successfully!")
    print("Number of classes:", len(classes))

    return train_df, test_df, classes


def validate_dataset(X_train, X_test, y_train, y_test):
    """Validate dataset before training."""

    print("\n" + "=" * 65)
    print("DATASET VALIDATION")
    print("=" * 65)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Features must be two-dimensional.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training and testing feature counts do not match."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training features and labels have different lengths."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Testing features and labels have different lengths."
        )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training data must contain at least two classes."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Testing data contains NaN or infinite values."
        )

    print("Dataset validation successful!")
    print("Number of classes:", len(np.unique(
        np.concatenate([y_train, y_test])
    )))
    print("Number of features:", X_train.shape[1])


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 65)
    print("TASK 12 - H2O AUTOML")
    print("=" * 65)

    h2o_started = False

    try:

        # ----------------------------------------------------
        # STEP 1: LOAD DATA
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = load_data()

        # ----------------------------------------------------
        # STEP 2: VALIDATE DATA
        # ----------------------------------------------------

        validate_dataset(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ----------------------------------------------------
        # STEP 3: PREPARE DATAFRAMES
        # ----------------------------------------------------

        train_df, test_df, classes = prepare_dataframes(
            X_train,
            X_test,
            y_train,
            y_test
        )

        # ----------------------------------------------------
        # STEP 4: INITIALIZE H2O
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("INITIALIZING H2O")
        print("=" * 65)

        # IMPORTANT:
        # silent=True is intentionally omitted because
        # this H2O version does not accept it in h2o.init().

        h2o.init(
            max_mem_size=MAX_MEMORY,
            nthreads=N_THREADS
        )

        h2o_started = True

        print("\nH2O initialized successfully!")

        # ----------------------------------------------------
        # STEP 5: CONVERT DATA TO H2O FRAMES
        # ----------------------------------------------------

        print("\nConverting data into H2O Frames...")

        train_h2o = h2o.H2OFrame(train_df)
        test_h2o = h2o.H2OFrame(test_df)

        train_h2o[TARGET_COLUMN] = (
            train_h2o[TARGET_COLUMN].asfactor()
        )

        test_h2o[TARGET_COLUMN] = (
            test_h2o[TARGET_COLUMN].asfactor()
        )

        print("H2O Frames created successfully!")

        # ----------------------------------------------------
        # STEP 6: TRAIN H2O AUTOML
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("TRAINING H2O AUTOML")
        print("=" * 65)

        print("Maximum runtime:", MAX_RUNTIME_SECONDS, "seconds")
        print("Maximum models:", MAX_MODELS)

        automl = H2OAutoML(
            max_runtime_secs=MAX_RUNTIME_SECONDS,
            max_models=MAX_MODELS,
            seed=RANDOM_STATE,
            sort_metric="mean_per_class_error",
            balance_classes=False,
            nfolds=0,
            verbosity="info"
        )

        training_start = time.time()

        automl.train(
            x=[
                column
                for column in train_h2o.columns
                if column != TARGET_COLUMN
            ],
            y=TARGET_COLUMN,
            training_frame=train_h2o
        )

        training_time = time.time() - training_start

        print("\nH2O AutoML training completed!")
        print(
            "Training time:",
            round(training_time, 2),
            "seconds"
        )

        # ----------------------------------------------------
        # STEP 7: LEADERBOARD
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("H2O AUTOML LEADERBOARD")
        print("=" * 65)

        leaderboard = automl.leaderboard

        leaderboard_df = leaderboard.as_data_frame()

        print(leaderboard_df.head(15).to_string(index=False))

        leaderboard_df.to_csv(
            OUTPUT_DIR / "leaderboard.csv",
            index=False
        )

        best_model = automl.leader

        if best_model is None:
            raise RuntimeError(
                "H2O AutoML did not produce a best model."
            )

        print("\nBest model:", best_model.model_id)

        # ----------------------------------------------------
        # STEP 8: PREDICTIONS
        # ----------------------------------------------------

        print("\nGenerating predictions...")

        predictions_h2o = best_model.predict(test_h2o)

        predictions_df = predictions_h2o.as_data_frame()

        y_pred = predictions_df["predict"].to_numpy()

        # Convert predictions to the original label type.
        y_pred = np.asarray(y_pred, dtype=str)

        y_true = np.asarray(y_test, dtype=str)

        print("Predictions generated successfully!")

        # ----------------------------------------------------
        # STEP 9: EVALUATION
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("MODEL EVALUATION")
        print("=" * 65)

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=classes.astype(str)
        )

        print("\nAccuracy:", round(accuracy, 6))
        print("Precision:", round(precision, 6))
        print("Recall:", round(recall, 6))
        print("F1 Score:", round(f1, 6))

        print("\nClassification Report:")
        print(report)

        print("\nConfusion Matrix:")
        print(cm)

        # ----------------------------------------------------
        # STEP 10: SAVE EVALUATION RESULTS
        # ----------------------------------------------------

        print("\nSaving evaluation results...")

        metrics = {
            "library": "H2O AutoML",
            "best_model": best_model.model_id,
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "training_time_seconds": round(
                training_time, 2
            ),
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "max_models": MAX_MODELS,
            "training_records": int(len(y_train)),
            "testing_records": int(len(y_test)),
            "number_of_features": int(X_train.shape[1]),
            "number_of_classes": int(len(classes))
        }

        # Read the actual mean_per_class_error from leaderboard.
        if (
            "mean_per_class_error" in leaderboard_df.columns
            and len(leaderboard_df) > 0
        ):
            score = leaderboard_df.iloc[0][
                "mean_per_class_error"
            ]

            if pd.notna(score):
                metrics["best_validation_mean_per_class_error"] = float(
                    score
                )

        save_json(
            metrics,
            OUTPUT_DIR / "metrics.json"
        )

        with open(
            OUTPUT_DIR / "classification_report.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(report)

        pd.DataFrame(
            cm,
            index=classes.astype(str),
            columns=classes.astype(str)
        ).to_csv(
            OUTPUT_DIR / "confusion_matrix.csv"
        )

        # Save predictions with actual labels.
        prediction_output = pd.DataFrame({
            "actual": y_true,
            "predicted": y_pred
        })

        prediction_output.to_csv(
            OUTPUT_DIR / "predictions.csv",
            index=False
        )

        # ----------------------------------------------------
        # STEP 11: SAVE BEST MODEL
        # ----------------------------------------------------

        print("\nSaving best model...")

        model_path = h2o.save_model(
            model=best_model,
            path=str(OUTPUT_DIR),
            force=True
        )

        print("Model saved at:", model_path)

        # ----------------------------------------------------
        # STEP 12: SAVE MODEL INFORMATION
        # ----------------------------------------------------

        model_info = {
            "library": "H2O AutoML",
            "best_model_id": best_model.model_id,
            "algorithm": best_model.algo,
            "model_path": str(model_path),
            "training_time_seconds": round(
                training_time, 2
            ),
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "max_models": MAX_MODELS,
            "seed": RANDOM_STATE
        }

        save_json(
            model_info,
            OUTPUT_DIR / "model_info.json"
        )

        # ----------------------------------------------------
        # STEP 13: SAVE LOG
        # ----------------------------------------------------

        total_time = time.time() - start_time

        log_text = f"""
TASK 12 - H2O AUTOML EXECUTION LOG

Library: H2O AutoML

Best Model: {best_model.model_id}
Algorithm: {best_model.algo}

Training Records: {len(y_train)}
Testing Records: {len(y_test)}
Features: {X_train.shape[1]}
Classes: {len(classes)}

Training Time: {training_time:.2f} seconds
Total Execution Time: {total_time:.2f} seconds

Accuracy: {accuracy:.6f}
Precision: {precision:.6f}
Recall: {recall:.6f}
F1 Score: {f1:.6f}

Maximum Runtime: {MAX_RUNTIME_SECONDS} seconds
Maximum Models: {MAX_MODELS}

Execution Status: SUCCESS
"""

        with open(
            OUTPUT_DIR / "execution_log.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(log_text)

        # ----------------------------------------------------
        # STEP 14: FINAL SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("H2O AUTOML FINAL SUMMARY")
        print("=" * 65)

        print("Status: SUCCESS")
        print("Best Model:", best_model.model_id)
        print("Algorithm:", best_model.algo)
        print("Accuracy:", round(accuracy, 6))
        print("Precision:", round(precision, 6))
        print("Recall:", round(recall, 6))
        print("F1 Score:", round(f1, 6))
        print("Training Time:", round(training_time, 2), "seconds")
        print("Total Time:", round(total_time, 2), "seconds")

        print("\nOutput Directory:")
        print(OUTPUT_DIR)

        print("\nGenerated Files:")
        for file_path in sorted(OUTPUT_DIR.iterdir()):
            if file_path.is_file():
                print(" -", file_path.name)

        print("\nH2O AutoML task completed successfully!")

    except Exception as error:

        print("\n" + "=" * 65)
        print("H2O AUTOML EXECUTION FAILED")
        print("=" * 65)

        print("Error:", str(error))

        error_log = {
            "status": "FAILED",
            "error": str(error),
            "traceback": traceback.format_exc()
        }

        save_json(
            error_log,
            OUTPUT_DIR / "error_log.json"
        )

        traceback.print_exc()

        raise

    finally:

        if h2o_started:
            print("\nShutting down H2O...")
            h2o.cluster().shutdown(prompt=False)
            print("H2O shutdown completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
