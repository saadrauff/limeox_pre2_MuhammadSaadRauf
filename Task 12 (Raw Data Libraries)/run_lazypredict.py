
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from lazypredict.Supervised import LazyClassifier


# ==================================================
# 1. PROJECT PATHS
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Supports both:
# Task 12/run_lazypredict.py
# Task 12/02_LazyPredict/run_lazypredict.py

if (SCRIPT_DIR / "_shared").exists():
    TASK12_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "_shared").exists():
    TASK12_DIR = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Task 12 _shared folder could not be located."
    )

DATA_FILE = TASK12_DIR / "_shared" / "prepared_data.npz"

OUTPUT_DIR = TASK12_DIR / "results" / "02_LazyPredict"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD PREPARED DATA
# ==================================================

def load_prepared_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_FILE}\n"
            "Run prepare_data.py first."
        )

    print("\nLoading prepared dataset...")

    with np.load(DATA_FILE, allow_pickle=False) as data:

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

    print("\nDataset loaded successfully.")

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("Training records:", len(y_train))
    print("Testing records:", len(y_test))

    if X_train.shape != (79920, 100):
        raise ValueError(
            f"Unexpected training shape: {X_train.shape}"
        )

    if X_test.shape != (19980, 100):
        raise ValueError(
            f"Unexpected testing shape: {X_test.shape}"
        )

    return X_train, X_test, y_train, y_test


# ==================================================
# 3. MAIN FUNCTION
# ==================================================

def main():

    X_train, X_test, y_train, y_test = load_prepared_data()

    print("\n" + "=" * 60)
    print("LAZYPREDICT MODEL COMPARISON")
    print("=" * 60)

    start_time = time.time()

    print("\nInitializing LazyPredict...")

    try:

        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            custom_metric=None,
            predictions=True,
            timeout=30
        )

    except TypeError:

        print("Timeout parameter not supported. Using compatibility mode.")

        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            custom_metric=None,
            predictions=True
        )

    print("\nTraining and comparing models...")
    print("This process may take some time.\n")

    models, predictions = clf.fit(
        X_train,
        X_test,
        y_train,
        y_test
    )

    elapsed_time = time.time() - start_time

    # ==================================================
    # 4. SAVE MODEL COMPARISON
    # ==================================================

    models.to_csv(
        OUTPUT_DIR / "all_models_comparison.csv"
    )

    if predictions is None or predictions.empty:
        raise RuntimeError(
            "LazyPredict did not return model predictions."
        )

    # ==================================================
    # 5. CALCULATE MACRO F1 FOR EVERY MODEL
    # ==================================================

    print("\nCalculating Macro F1 scores...")

    macro_scores = {}

    for model_name in predictions.columns:

        y_pred_candidate = predictions[model_name].to_numpy()

        macro_scores[str(model_name)] = f1_score(
            y_test,
            y_pred_candidate,
            average="macro",
            zero_division=0
        )

    ranking = pd.DataFrame([
        {
            "Model": name,
            "Macro_F1": score
        }
        for name, score in macro_scores.items()
    ])

    ranking = ranking.sort_values(
        "Macro_F1",
        ascending=False
    )

    ranking.to_csv(
        OUTPUT_DIR / "lazypredict_macro_f1_ranking.csv",
        index=False
    )

    # ==================================================
    # 6. SELECT BEST MODEL
    # ==================================================

    best_name = ranking.iloc[0]["Model"]

    y_pred = predictions[best_name].to_numpy()

    print("\nBest model:", best_name)

    # ==================================================
    # 7. EVALUATION METRICS
    # ==================================================

    metrics = {
        "Library": "LazyPredict",
        "Best_Model": str(best_name),
        "Accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "Macro_Precision": float(
            precision_score(
                y_test, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "Macro_Recall": float(
            recall_score(
                y_test, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "Macro_F1": float(
            f1_score(
                y_test, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "Weighted_F1": float(
            f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "Training_Records": int(len(y_train)),
        "Testing_Records": int(len(y_test)),
        "Total_Records": int(len(y_train) + len(y_test)),
        "Execution_Time_Seconds": float(elapsed_time)
    }

    # ==================================================
    # 8. SAVE RESULTS
    # ==================================================

    with open(
        OUTPUT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv",
        index=False
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    labels = sorted(set(y_test))

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

    pd.DataFrame({
        "Actual_Category": y_test,
        "Predicted_Category": y_pred
    }).to_csv(
        OUTPUT_DIR / "best_model_predictions.csv",
        index=False
    )

    # ==================================================
    # 9. FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 60)
    print("LAZYPREDICT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nBest Model:", best_name)
    print("Accuracy:", round(metrics["Accuracy"], 4))
    print("Macro Precision:", round(metrics["Macro_Precision"], 4))
    print("Macro Recall:", round(metrics["Macro_Recall"], 4))
    print("Macro F1:", round(metrics["Macro_F1"], 4))
    print("Weighted F1:", round(metrics["Weighted_F1"], 4))

    print(
        "\nExecution Time:",
        round(elapsed_time / 60, 2),
        "minutes"
    )

    print("\nResults saved at:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.iterdir()):
        if file.is_file():
            print("-", file.name)


if __name__ == "__main__":
    main()
