
"""
TASK 15 - DEEP LEARNING
TF-IDF + PyTorch Deep Neural Network (MLP)

Separate experiment from bilstm_news_classification.py.

Input:
    Task 5 (Categorization)/news_categorized.csv

Pipeline:
    Data Cleaning
    -> Stratified 80/10/10 Split
    -> TF-IDF Unigrams + Bigrams
    -> Sparse PyTorch Input
    -> Deep Neural Network
    -> Evaluation and Results

All results are saved in:
    model_results/mlp_model_results/
"""

import os
import json
import time
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MAX_FEATURES = 30000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.98

BATCH_SIZE = 256
EPOCHS = 12

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

PATIENCE = 3
LOG_EVERY_BATCHES = 25

HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128

DROPOUT = 0.35

CPU_THREADS = min(6, os.cpu_count() or 1)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "Task 5 (Categorization)"
    / "news_categorized.csv"
)

# Separate output folder.
OUTPUT_DIR = (
    BASE_DIR
    / "model_results"
    / "mlp_model_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = OUTPUT_DIR / "tfidf_pytorch_mlp.pt"

VECTORIZER_PATH = (
    OUTPUT_DIR / "tfidf_vectorizer.joblib"
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

if DEVICE.type == "cpu":
    torch.set_num_threads(CPU_THREADS)


# ============================================================
# 4. UTILITIES
# ============================================================

def format_time(seconds):

    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)

    minutes, seconds = divmod(remainder, 60)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def scipy_to_torch_sparse(matrix):

    coo = matrix.tocoo()

    indices = np.vstack(
        (coo.row, coo.col)
    ).astype(np.int64)

    torch_indices = torch.from_numpy(indices)

    torch_values = torch.from_numpy(
        coo.data.astype(
            np.float32,
            copy=False
        )
    )

    tensor = torch.sparse_coo_tensor(
        torch_indices,
        torch_values,
        size=coo.shape,
        dtype=torch.float32
    ).coalesce()

    return tensor.to(DEVICE)


# ============================================================
# 5. DEEP NEURAL NETWORK
# ============================================================

class SparseTFIDFMLP(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.input_weight = nn.Parameter(
            torch.empty(
                input_dim,
                HIDDEN_DIM_1
            )
        )

        self.input_bias = nn.Parameter(
            torch.zeros(HIDDEN_DIM_1)
        )

        nn.init.xavier_uniform_(
            self.input_weight
        )

        self.hidden = nn.Sequential(

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM_1,
                HIDDEN_DIM_2
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                HIDDEN_DIM_2,
                num_classes
            )
        )

    def forward(self, sparse_x):

        x = torch.sparse.mm(
            sparse_x,
            self.input_weight
        )

        x = x + self.input_bias

        return self.hidden(x)


# ============================================================
# 6. TRAINING / EVALUATION
# ============================================================

def run_epoch(
    model,
    X,
    y,
    criterion,
    optimizer=None,
    epoch_number=None,
    total_epochs=None,
    training=False
):

    if training:
        model.train()
    else:
        model.eval()

    number_of_rows = X.shape[0]

    total_batches = (
        number_of_rows + BATCH_SIZE - 1
    ) // BATCH_SIZE

    indices = np.arange(number_of_rows)

    if training:
        np.random.shuffle(indices)

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    start_time = time.time()

    running_loss = 0.0
    processed = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for batch_number, start in enumerate(
            range(0, number_of_rows, BATCH_SIZE),
            start=1
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            batch_x = scipy_to_torch_sparse(
                X[batch_indices]
            )

            batch_y = torch.as_tensor(
                y[batch_indices],
                dtype=torch.long,
                device=DEVICE
            )

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=2.0
                )

                optimizer.step()

            actual_batch_size = len(batch_indices)

            batch_loss = (
                loss.item() * actual_batch_size
            )

            total_loss += batch_loss
            running_loss += batch_loss

            processed += actual_batch_size

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.detach().cpu().numpy()
            )

            all_targets.extend(
                batch_y.detach().cpu().numpy()
            )

            if training and (
                batch_number % LOG_EVERY_BATCHES == 0
                or batch_number == total_batches
            ):

                elapsed = time.time() - start_time

                average_batch_time = (
                    elapsed / batch_number
                )

                eta = (
                    average_batch_time
                    * (total_batches - batch_number)
                )

                progress = (
                    100.0 * batch_number / total_batches
                )

                current_loss = (
                    running_loss / processed
                )

                print(
                    f"Epoch {epoch_number}/{total_epochs} | "
                    f"Batch {batch_number:,}/{total_batches:,} | "
                    f"{progress:5.1f}% | "
                    f"Loss: {current_loss:.4f} | "
                    f"Elapsed: {format_time(elapsed)} | "
                    f"ETA: {format_time(eta)}",
                    flush=True
                )

    average_loss = (
        total_loss / number_of_rows
    )

    all_targets = np.asarray(all_targets)

    all_predictions = np.asarray(
        all_predictions
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_targets,
        all_predictions
    )


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    overall_start = time.time()

    print("=" * 70)
    print("TASK 15 - TF-IDF + PYTORCH DEEP NEURAL NETWORK")
    print("=" * 70)

    print("Device:", DEVICE)

    print(
        "CPU threads:",
        CPU_THREADS if DEVICE.type == "cpu" else "N/A"
    )

    print("Input:", INPUT_FILE)
    print("Output:", OUTPUT_DIR)

    print("\nConfiguration:")

    print("Maximum TF-IDF features:", MAX_FEATURES)
    print("N-grams:", NGRAM_RANGE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    print(
        "Hidden layers:",
        HIDDEN_DIM_1,
        "->",
        HIDDEN_DIM_2
    )

    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading dataset...", flush=True)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    required_columns = [
        "title_clean",
        "content_clean",
        "category",
        "validation_disposition"
    ]

    df = pd.read_csv(
        INPUT_FILE,
        usecols=required_columns
    )

    print(
        "Original dataset shape:",
        df.shape
    )

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    print("\nCleaning dataset...", flush=True)

    for column in [
        "title_clean",
        "content_clean",
        "category"
    ]:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["validation_disposition"] = (
        df["validation_disposition"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["validation_disposition"] == "ACCEPT"
    ].copy()

    df = df[
        (df["title_clean"] != "")
        & (df["content_clean"] != "")
        & (df["category"] != "")
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    if df.empty:

        raise ValueError(
            "No valid ACCEPT records found."
        )

    print(
        "Clean dataset shape:",
        df.shape
    )

    print("\nCategory distribution:")

    print(
        df["category"].value_counts()
    )

    # --------------------------------------------------------
    # COMBINE TEXT
    # --------------------------------------------------------

    print("\nCombining title and content...")

    texts = (
        df["title_clean"]
        + " "
        + df["content_clean"]
    ).tolist()

    original_titles = (
        df["title_clean"].tolist()
    )

    # --------------------------------------------------------
    # ENCODE LABELS
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(
        df["category"]
    )

    class_names = (
        label_encoder.classes_.tolist()
    )

    num_classes = len(class_names)

    print("\nClasses:", class_names)
    print("Number of classes:", num_classes)

    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

    print("\nSplitting dataset...", flush=True)

    all_indices = np.arange(
        len(texts)
    )

    train_indices, temporary_indices = (
        train_test_split(
            all_indices,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    validation_indices, test_indices = (
        train_test_split(
            temporary_indices,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=labels[temporary_indices]
        )
    )

    train_texts = [
        texts[i] for i in train_indices
    ]

    validation_texts = [
        texts[i] for i in validation_indices
    ]

    test_texts = [
        texts[i] for i in test_indices
    ]

    y_train = labels[train_indices]
    y_validation = labels[validation_indices]
    y_test = labels[test_indices]

    print(
        "Training records:",
        len(train_indices)
    )

    print(
        "Validation records:",
        len(validation_indices)
    )

    print(
        "Testing records:",
        len(test_indices)
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    distribution_rows = []

    for split_name, split_labels in [
        ("Training", y_train),
        ("Validation", y_validation),
        ("Testing", y_test)
    ]:

        counts = np.bincount(
            split_labels,
            minlength=num_classes
        )

        for class_name, count in zip(
            class_names,
            counts
        ):

            distribution_rows.append({

                "split": split_name,

                "category": class_name,

                "count": int(count)
            })

    pd.DataFrame(
        distribution_rows
    ).to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print(
        "\nFitting TF-IDF on training data only...",
        flush=True
    )

    vectorizer = TfidfVectorizer(

        max_features=MAX_FEATURES,

        ngram_range=NGRAM_RANGE,

        min_df=MIN_DF,

        max_df=MAX_DF,

        sublinear_tf=True,

        strip_accents="unicode",

        dtype=np.float32
    )

    tfidf_start = time.time()

    X_train = (
        vectorizer
        .fit_transform(train_texts)
        .tocsr()
    )

    print(
        "Training TF-IDF shape:",
        X_train.shape
    )

    print(
        "TF-IDF fitting time:",
        format_time(time.time() - tfidf_start)
    )

    X_validation = (
        vectorizer
        .transform(validation_texts)
        .tocsr()
    )

    X_test = (
        vectorizer
        .transform(test_texts)
        .tocsr()
    )

    print(
        "Validation TF-IDF shape:",
        X_validation.shape
    )

    print(
        "Testing TF-IDF shape:",
        X_test.shape
    )

    print(
        "Vocabulary size:",
        len(vectorizer.vocabulary_)
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
        compress=3
    )

    print(
        "TF-IDF vectorizer saved:",
        VECTORIZER_PATH
    )

    # Release unnecessary text lists.
    del train_texts
    del validation_texts
    del test_texts
    del texts

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=num_classes
    )

    # Moderate inverse-frequency weighting.
    class_weights = np.sqrt(
        len(y_train)
        / (num_classes * class_counts)
    )

    class_weights = (
        class_weights / class_weights.mean()
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print("\nTraining class counts:")

    for name, count, weight in zip(
        class_names,
        class_counts,
        class_weights
    ):

        print(
            f"{name:16s} "
            f"Count: {count:6d} | "
            f"Weight: {weight:.3f}"
        )

    # --------------------------------------------------------
    # INITIALIZE MODEL
    # --------------------------------------------------------

    print("\nInitializing PyTorch MLP...")

    model = SparseTFIDFMLP(
        input_dim=X_train.shape[1],
        num_classes=num_classes
    ).to(DEVICE)

    print(model)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1
        )
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\nSTARTING TRAINING")
    print("=" * 70)

    history = []

    best_macro_f1 = -1.0
    best_epoch = 0

    patience_counter = 0

    best_model_state = None

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        print("\n" + "-" * 65)

        print(
            f"STARTING EPOCH {epoch}/{EPOCHS}"
        )

        print("-" * 65)

        train_loss, _, _, _, _ = run_epoch(

            model=model,

            X=X_train,

            y=y_train,

            criterion=criterion,

            optimizer=optimizer,

            epoch_number=epoch,

            total_epochs=EPOCHS,

            training=True
        )

        print("\nRunning validation...")

        (
            validation_loss,
            validation_accuracy,
            validation_macro_f1,
            _,
            _
        ) = run_epoch(

            model=model,

            X=X_validation,

            y=y_validation,

            criterion=criterion,

            training=False
        )

        scheduler.step(
            validation_macro_f1
        )

        epoch_time = (
            time.time() - epoch_start
        )

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "validation_loss": validation_loss,

            "validation_accuracy": validation_accuracy,

            "validation_macro_f1": validation_macro_f1,

            "learning_rate": optimizer.param_groups[0]["lr"],

            "epoch_seconds": epoch_time
        })

        print("\n" + "=" * 65)

        print(
            f"EPOCH {epoch}/{EPOCHS} COMPLETED"
        )

        print("=" * 65)

        print(
            f"Training Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

        print(
            f"Validation Macro F1: {validation_macro_f1:.4f}"
        )

        print(
            f"Epoch Time: {format_time(epoch_time)}"
        )

        print(
            "Learning Rate:",
            optimizer.param_groups[0]["lr"]
        )

        # Save best model.
        if validation_macro_f1 > best_macro_f1:

            best_macro_f1 = validation_macro_f1

            best_epoch = epoch

            patience_counter = 0

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save({

                "model_state_dict": best_model_state,

                "input_dim": X_train.shape[1],

                "class_names": class_names,

                "best_epoch": best_epoch,

                "config": {

                    "max_features": MAX_FEATURES,

                    "ngram_range": NGRAM_RANGE,

                    "hidden_dim_1": HIDDEN_DIM_1,

                    "hidden_dim_2": HIDDEN_DIM_2,

                    "dropout": DROPOUT
                }

            }, MODEL_PATH)

            print(
                "\nBest model checkpoint saved."
            )

        else:

            patience_counter += 1

            print(
                f"\nNo improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:

            print("\nEarly stopping activated.")

            break

    if best_model_state is None:

        raise RuntimeError(
            "Training did not produce a valid model."
        )

    model.load_state_dict(
        best_model_state
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    print("\nBest epoch:", best_epoch)

    print(
        "Best validation Macro F1:",
        round(best_macro_f1, 4)
    )

    print(
        "Total training time:",
        format_time(time.time() - overall_start)
    )

    # --------------------------------------------------------
    # TRAINING PLOTS
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    axes[0].plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Training Loss"
    )

    axes[0].plot(
        history_df["epoch"],
        history_df["validation_loss"],
        marker="o",
        label="Validation Loss"
    )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history_df["epoch"],
        history_df["validation_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    axes[1].plot(
        history_df["epoch"],
        history_df["validation_macro_f1"],
        marker="o",
        label="Validation Macro F1"
    )

    axes[1].set_title(
        "Validation Performance"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")

    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_history.png",
        dpi=250
    )

    plt.close()

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    print(
        "\nEvaluating best model on test data...",
        flush=True
    )

    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        y_true,
        y_pred
    ) = run_epoch(

        model=model,

        X=X_test,

        y=y_test,

        criterion=criterion,

        training=False
    )

    test_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    test_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    test_weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)

    print(
        "FINAL TF-IDF + PYTORCH MLP RESULTS"
    )

    print("=" * 70)

    print(f"Test Loss:       {test_loss:.4f}")

    print(f"Accuracy:        {test_accuracy:.4f}")

    print(f"Weighted Precision: {test_precision:.4f}")

    print(f"Weighted Recall:    {test_recall:.4f}")

    print(f"Weighted F1:     {test_weighted_f1:.4f}")

    print(f"Macro F1:        {test_macro_f1:.4f}")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(

        y_true,

        y_pred,

        labels=np.arange(num_classes),

        target_names=class_names,

        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    (
        OUTPUT_DIR / "classification_report.txt"
    ).write_text(
        report,
        encoding="utf-8"
    )

    report_dict = classification_report(

        y_true,

        y_pred,

        labels=np.arange(num_classes),

        target_names=class_names,

        output_dict=True,

        zero_division=0
    )

    with open(
        OUTPUT_DIR / "classification_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report_dict,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {

        "model": "TF-IDF + PyTorch Deep Neural Network (MLP)",

        "best_epoch": best_epoch,

        "test_loss": test_loss,

        "accuracy": test_accuracy,

        "weighted_precision": test_precision,

        "weighted_recall": test_recall,

        "weighted_f1": test_weighted_f1,

        "macro_f1": test_macro_f1,

        "training_records": len(train_indices),

        "validation_records": len(validation_indices),

        "testing_records": len(test_indices),

        "tfidf_features": int(X_train.shape[1]),

        "ngram_range": str(NGRAM_RANGE)
    }

    pd.DataFrame(
        [metrics]
    ).to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({

        "title": [
            original_titles[i]
            for i in test_indices
        ],

        "actual_category": (
            label_encoder.inverse_transform(y_true)
        ),

        "predicted_category": (
            label_encoder.inverse_transform(y_pred)
        )
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=np.arange(num_classes)
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    display = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=class_names
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    display.plot(

        ax=ax,

        cmap="Blues",

        xticks_rotation=45,

        values_format="d"
    )

    plt.title(
        "TF-IDF + PyTorch MLP - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=250
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE CLASS NAMES
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "class_names.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "TASK 15 - TF-IDF + PYTORCH MLP COMPLETED"
    )

    print("=" * 70)

    print("\nSaved model:")
    print(MODEL_PATH)

    print("\nSaved vectorizer:")
    print(VECTORIZER_PATH)

    print("\nAll results saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for file_path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if file_path.is_file():

            size_mb = (
                file_path.stat().st_size
                / (1024 * 1024)
            )

            print(
                f"{file_path.name} "
                f"({size_mb:.2f} MB)"
            )

    print("\nTask 15 completed successfully!")


if __name__ == "__main__":

    main()
