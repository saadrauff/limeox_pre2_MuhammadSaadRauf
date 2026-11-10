
from pathlib import Path
import time
import json
import shutil

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "_shared").is_dir():
    TASK12_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "_shared").is_dir():
    TASK12_DIR = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Task 12 _shared folder could not be located."
    )

DATA_FILE = TASK12_DIR / "_shared" / "prepared_data.npz"

OUTPUT_DIR = (
    TASK12_DIR / "results" / "03_AutoGluon"
)

PREDICTOR_PATH = OUTPUT_DIR / "AutogluonModels"

TARGET_COLUMN = "category"

EXPECTED_TRAIN_SHAPE = (79920, 100)
EXPECTED_TEST_SHAPE = (19980, 100)

TIME_LIMIT = 600
NUM_CPUS = 4


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

def load_prepared_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_FILE}\n"
            "Run prepare_data.py first."
        )

    print("\nLoading prepared dataset...")

    with np.load(DATA_FILE, allow_pickle=False) as data:

        required_keys = {
            "X_train",
            "X_test",
            "y_train",
            "y_test"
        }

        missing_keys = required_keys - set(data.files)

        if missing_keys:
            raise KeyError(
                f"Missing dataset arrays: {missing_keys}"
            )

        X_train = data["X_train"]
        X_test = data["X_test"]

        y_train = data["y_train"].astype(str)
        y_test = data["y_test"].astype(str)

    print("\nDataset loaded successfully.")

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("Training records:", len(y_train))
    print("Testing records:", len(y_test))

    # Validate dataset dimensions.

    if X_train.shape != EXPECTED_TRAIN_SHAPE:
        raise ValueError(
            f"Unexpected training shape: {X_train.shape}"
        )

    if X_test.shape != EXPECTED_TEST_SHAPE:
        raise ValueError(
            f"Unexpected testing shape: {X_test.shape}"
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
            "Training dataset must contain at least two classes."
        )

    if not set(np.unique(y_test)).issubset(set(np.unique(y_train))):
        raise ValueError(
            "Test dataset contains classes missing from training data."
        )

    # Validate numerical features.

    if not np.issubdtype(X_train.dtype, np.number):
        raise TypeError("Training features must be numeric.")

    if not np.issubdtype(X_test.dtype, np.number):
        raise TypeError("Testing features must be numeric.")

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Testing data contains NaN or infinite values."
        )

    print("\nDataset validation successful.")

    print("Number of classes:", len(np.unique(y_train)))

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. CREATE DATAFRAMES
# ============================================================

def create_dataframes(X_train, X_test, y_train):

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

    train_df[TARGET_COLUMN] = y_train

    print("\nDataFrames created successfully.")

    print("Training DataFrame:", train_df.shape)
    print("Testing DataFrame:", test_df.shape)

    return train_df, test_df


# ============================================================
# 4. TRAIN AUTOGUON MODEL
# ============================================================

def train_model(train_df):

    print("\n" + "=" * 65)
    print("AUTOGLUON MODEL TRAINING")
    print("=" * 65)

    print("\nConfiguration:")
    print("Preset: medium_quality")
    print("Evaluation metric: f1_macro")
    print("Problem type: multiclass")
    print("Time limit:", TIME_LIMIT, "seconds")
    print("CPU cores:", NUM_CPUS)
    print("GPU: Disabled")

    # Remove only previous AutoGluon experiment output.

    if PREDICTOR_PATH.exists():

        print("\nRemoving previous AutoGluon model directory...")

        shutil.rmtree(PREDICTOR_PATH)

    # Initialize predictor.

    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        path=str(PREDICTOR_PATH),
        problem_type="multiclass",
        eval_metric="f1_macro",
        verbosity=2
    )

    print("\nStarting AutoGluon training...")

    start_time = time.time()

    predictor.fit(
        train_data=train_df,
        presets="medium_quality",
        time_limit=TIME_LIMIT,
        num_cpus=NUM_CPUS,
        num_gpus=0,
        dynamic_stacking=False
    )

    elapsed_time = time.time() - start_time

    print("\nTraining completed successfully.")

    print(
        "Training execution time:",
        round(elapsed_time / 60, 2),
        "minutes"
    )

    return predictor, elapsed_time


# ============================================================
# 5. GENERATE TEST PREDICTIONS
# ============================================================

def generate_predictions(predictor, test_df):

    print("\nGenerating test predictions...")

    predictions = predictor.predict(test_df)

    y_pred = predictions.astype(str).to_numpy()

    best_model = predictor.model_best

    print("\nBest model:", best_model)

    print("Predictions generated:", len(y_pred))

    return y_pred, best_model


# ============================================================
# 6. GENERATE LEADERBOARD
# ============================================================

def generate_leaderboard(predictor):

    print("\nGenerating AutoGluon leaderboard...")

    # Uses AutoGluon's internal validation data.
    # Test data is deliberately excluded.

    leaderboard = predictor.leaderboard(
        silent=True
    )

    leaderboard.to_csv(
        OUTPUT_DIR / "leaderboard.csv",
        index=False
    )

    print("\nLeaderboard saved successfully.")

    print("\nTop models:")

    print(
        leaderboard.head(10).to_string(index=False)
    )

    return leaderboard


# ============================================================
# 7. EVALUATE MODEL ON TEST DATA
# ============================================================

def evaluate_model(y_test, y_pred, best_model, elapsed_time):

    print("\nCalculating evaluation metrics...")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
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

        "Library": "AutoGluon",

        "Best_Model": str(best_model),

        "Accuracy": float(accuracy),

        "Macro_Precision": float(macro_precision),

        "Macro_Recall": float(macro_recall),

        "Macro_F1": float(macro_f1),

        "Weighted_F1": float(weighted_f1),

        "Training_Records": int(len(y_train_global)),

        "Testing_Records": int(len(y_test)),

        "Total_Records": int(
            len(y_train_global) + len(y_test)
        ),

        "Execution_Time_Seconds": float(elapsed_time)
    }

    return metrics


# ============================================================
# 8. SAVE ALL RESULTS
# ============================================================

def save_results(metrics, y_test, y_pred):

    print("\nSaving evaluation results...")

    # JSON metrics.

    with open(
        OUTPUT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # CSV metrics.

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv",
        index=False
    )

    # Classification report.

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    # Confusion matrix.

    labels = sorted(
        set(y_test) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    ).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    # Predictions.

    pd.DataFrame({

        "Actual_Category": y_test,

        "Predicted_Category": y_pred

    }).to_csv(
        OUTPUT_DIR / "best_model_predictions.csv",
        index=False
    )

    print("\nAll evaluation files saved successfully.")


# ============================================================
# 9. DISPLAY FINAL SUMMARY
# ============================================================

def display_summary(metrics):

    print("\n" + "=" * 65)
    print("AUTOGLUON COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nLibrary:", metrics["Library"])

    print("Best Model:", metrics["Best_Model"])

    print(
        "Accuracy:",
        round(metrics["Accuracy"], 4)
    )

    print(
        "Macro Precision:",
        round(metrics["Macro_Precision"], 4)
    )

    print(
        "Macro Recall:",
        round(metrics["Macro_Recall"], 4)
    )

    print(
        "Macro F1:",
        round(metrics["Macro_F1"], 4)
    )

    print(
        "Weighted F1:",
        round(metrics["Weighted_F1"], 4)
    )

    print(
        "\nExecution Time:",
        round(
            metrics["Execution_Time_Seconds"] / 60,
            2
        ),
        "minutes"
    )

    print("\nResults saved at:")

    print(OUTPUT_DIR)

    print("\nGenerated result files:")

    for file in sorted(OUTPUT_DIR.iterdir()):

        if file.is_file():
            print("-", file.name)

    print("\n" + "=" * 65)


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():

    global y_train_global

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load dataset.

    X_train, X_test, y_train, y_test = (
        load_prepared_data()
    )

    y_train_global = y_train

    # Create DataFrames.

    train_df, test_df = create_dataframes(
        X_train,
        X_test,
        y_train
    )

    # Train model.

    predictor, elapsed_time = train_model(
        train_df
    )

    # Predictions.

    y_pred, best_model = generate_predictions(
        predictor,
        test_df
    )

    # Leaderboard.

    generate_leaderboard(
        predictor
    )

    # Test evaluation.

    metrics = evaluate_model(
        y_test,
        y_pred,
        best_model,
        elapsed_time
    )

    # Save results.

    save_results(
        metrics,
        y_test,
        y_pred
    )

    # Final summary.

    display_summary(metrics)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n" + "=" * 65)
        print("AUTOGLUON EXPERIMENT FAILED")
        print("=" * 65)

        print("\nError:", str(error))

        raise
