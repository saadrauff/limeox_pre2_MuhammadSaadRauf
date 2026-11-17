
"""
TASK 15 - DEEP LEARNING
News Classification using Bidirectional LSTM (BiLSTM)

Input:
    Task 5 (Categorization)/news_categorized.csv

Output:
    Task 15 (Deep Learning)/model_results/bilstm_results/

Architecture:
    Tokenization -> Padding -> Embedding -> BiLSTM -> Dense -> Classification
"""

import re
import json
import copy
import time
import random
import os

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
# 1. CONFIGURATION - CPU OPTIMIZED
# ============================================================

RANDOM_STATE = 42

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100

EMBEDDING_DIM = 32
HIDDEN_DIM = 32
DROPOUT = 0.3

BATCH_SIZE = 128

EPOCHS = 5
LEARNING_RATE = 0.001
PATIENCE = 2

PROGRESS_INTERVAL = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# CPU thread optimization
if DEVICE.type == "cpu":
    torch.set_num_threads(
        min(6, os.cpu_count() or 1)
    )

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "Task 5 (Categorization)"
    / "news_categorized.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "model_results"
    / "bilstm_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    OUTPUT_DIR
    / "bilstm_news_classifier.pt"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# 3. TIME FORMATTER
# ============================================================

def format_time(seconds):

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# 4. TEXT TOKENIZATION
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        str(text).lower()
    )


# ============================================================
# 5. BUILD VOCABULARY
# ============================================================

def build_vocabulary(texts):

    word_counter = Counter()

    print("\nBuilding vocabulary...")

    start_time = time.perf_counter()

    for index, text in enumerate(texts):

        word_counter.update(tokenize(text))

        if (index + 1) % 10000 == 0:

            print(
                f"Vocabulary progress: "
                f"{index + 1:,}/{len(texts):,}",
                flush=True
            )

    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, _ in word_counter.most_common(
        MAX_VOCAB_SIZE - 2
    ):

        vocabulary[word] = len(vocabulary)

    elapsed = time.perf_counter() - start_time

    print("Vocabulary size:", len(vocabulary))

    print(
        "Vocabulary time:",
        format_time(elapsed)
    )

    return vocabulary


# ============================================================
# 6. ENCODE TEXT
# ============================================================

def encode_texts(texts, vocabulary, split_name):

    encoded_sequences = []

    print(f"\nEncoding {split_name} data...")

    start_time = time.perf_counter()

    unknown_id = vocabulary["<UNK>"]

    for index, text in enumerate(texts):

        tokens = tokenize(text)

        tokens = tokens[:MAX_SEQUENCE_LENGTH]

        token_ids = [
            vocabulary.get(token, unknown_id)
            for token in tokens
        ]

        if len(token_ids) == 0:
            token_ids = [unknown_id]

        encoded_sequences.append(
            np.asarray(
                token_ids,
                dtype=np.int64
            )
        )

        if (index + 1) % 10000 == 0:

            print(
                f"{split_name} encoding: "
                f"{index + 1:,}/{len(texts):,}",
                flush=True
            )

    elapsed = time.perf_counter() - start_time

    print(
        f"{split_name} encoding completed in "
        f"{format_time(elapsed)}"
    )

    return encoded_sequences


# ============================================================
# 7. PYTORCH DATASET
# ============================================================

class NewsDataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = sequences

        self.labels = np.asarray(
            labels,
            dtype=np.int64
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        sequence = torch.from_numpy(
            self.sequences[index]
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return sequence, label


# ============================================================
# 8. BATCH PADDING
# ============================================================

def collate_batch(batch):

    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels)

    return (
        padded_sequences,
        lengths,
        labels
    )


# ============================================================
# 9. BiLSTM MODEL
# ============================================================

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        dropout
    ):

        super().__init__()

        # Word Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification Layer
        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, text, lengths):

        embedded = self.embedding(text)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        combined = self.dropout(combined)

        output = self.fc(combined)

        return output


# ============================================================
# 10. TRAIN ONE EPOCH WITH LIVE PROGRESS
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch,
    total_epochs
):

    model.train()

    total_loss = 0.0

    total_batches = len(loader)

    epoch_start = time.perf_counter()

    print("\n" + "-" * 65)

    print(
        f"STARTING EPOCH {epoch}/{total_epochs}"
    )

    print(
        f"Total batches: {total_batches:,}"
    )

    print("-" * 65, flush=True)

    for batch_index, (
        texts,
        lengths,
        labels
    ) in enumerate(loader, start=1):

        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            texts,
            lengths
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        # Live batch progress
        if (
            batch_index % PROGRESS_INTERVAL == 0
            or batch_index == total_batches
        ):

            elapsed = (
                time.perf_counter()
                - epoch_start
            )

            average_time = (
                elapsed / batch_index
            )

            remaining_batches = (
                total_batches - batch_index
            )

            eta = (
                average_time * remaining_batches
            )

            progress = (
                batch_index / total_batches
            ) * 100

            current_loss = (
                total_loss
                / (
                    batch_index
                    * loader.batch_size
                )
            )

            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"Batch {batch_index:,}/{total_batches:,} | "
                f"Progress: {progress:.1f}% | "
                f"Loss: {current_loss:.4f} | "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(eta)}",
                flush=True
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    epoch_time = (
        time.perf_counter() - epoch_start
    )

    return average_loss, epoch_time


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for texts, lengths, labels in loader:

            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                texts,
                lengths
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        np.asarray(all_labels),
        np.asarray(all_predictions)
    )


# ============================================================
# 12. SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history):

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    return history_df


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 65)
    print("DEEP LEARNING - BiLSTM NEWS CLASSIFICATION")
    print("=" * 65)

    print("\nDevice:", DEVICE)

    print(
        "CPU threads:",
        torch.get_num_threads()
    )

    print("Input:", INPUT_FILE)
    print("Output:", OUTPUT_DIR)

    print("\nConfiguration:")

    print("Maximum vocabulary:", MAX_VOCAB_SIZE)
    print("Maximum sequence length:", MAX_SEQUENCE_LENGTH)
    print("Embedding dimension:", EMBEDDING_DIM)
    print("Hidden dimension:", HIDDEN_DIM)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "title_clean",
            "content_clean",
            "category",
            "validation_disposition"
        ]
    )

    print(
        "Original dataset shape:",
        df.shape
    )

    # --------------------------------------------------------
    # DATA CLEANING
    # --------------------------------------------------------

    print("\nCleaning dataset...")

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

    # Keep ACCEPT records only
    df = df[
        df["validation_disposition"] == "ACCEPT"
    ].copy()

    # Remove empty records
    df = df[
        (df["title_clean"] != "")
        & (df["content_clean"] != "")
        & (df["category"] != "")
    ].copy()

    df = df.reset_index(drop=True)

    if len(df) == 0:

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
    # COMBINE TITLE AND CONTENT
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
    # LABEL ENCODING
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

    print(
        "Number of classes:",
        num_classes
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    indices = np.arange(len(texts))

    train_indices, temporary_indices = (
        train_test_split(
            indices,
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

    # Save split distribution
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
    # BUILD VOCABULARY
    # --------------------------------------------------------

    vocabulary = build_vocabulary(
        train_texts
    )

    with open(
        OUTPUT_DIR / "vocabulary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vocabulary,
            file,
            ensure_ascii=False
        )

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
    # ENCODE TEXT
    # --------------------------------------------------------

    X_train = encode_texts(
        train_texts,
        vocabulary,
        "Training"
    )

    X_validation = encode_texts(
        validation_texts,
        vocabulary,
        "Validation"
    )

    X_test = encode_texts(
        test_texts,
        vocabulary,
        "Testing"
    )

    # --------------------------------------------------------
    # PYTORCH DATASETS
    # --------------------------------------------------------

    train_dataset = NewsDataset(
        X_train,
        y_train
    )

    validation_dataset = NewsDataset(
        X_validation,
        y_validation
    )

    test_dataset = NewsDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    print("\nTraining batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Testing batches:", len(test_loader))

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_counts = np.bincount(
        y_train,
        minlength=num_classes
    )

    class_weights = (
        len(y_train)
        / (num_classes * class_counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(DEVICE)

    print(
        "\nClass weights:",
        class_weights.cpu().numpy()
    )

    # --------------------------------------------------------
    # INITIALIZE MODEL
    # --------------------------------------------------------

    print("\nInitializing BiLSTM model...")

    model = BiLSTMClassifier(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(DEVICE)

    print(model)

    # --------------------------------------------------------
    # LOSS AND OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\nSTARTING BiLSTM TRAINING")

    print("=" * 65)

    history = []

    best_macro_f1 = -1
    best_epoch = 0

    patience_counter = 0

    best_model_state = None

    checkpoint_config = {

        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "num_classes": num_classes

    }

    total_training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        # Train
        train_loss, epoch_time = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            EPOCHS
        )

        # Validate
        print(
            f"\nRunning validation for epoch {epoch}...",
            flush=True
        )

        (
            validation_loss,
            validation_accuracy,
            validation_macro_f1,
            _,
            _
        ) = evaluate_model(
            model,
            validation_loader,
            criterion
        )

        # Save history
        history.append({

            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_macro_f1,
            "epoch_time_seconds": epoch_time

        })

        history_df = save_training_history(
            history
        )

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

        # Save best model
        if validation_macro_f1 > best_macro_f1:

            best_macro_f1 = validation_macro_f1
            best_epoch = epoch

            patience_counter = 0

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "vocabulary": vocabulary,
                    "classes": class_names,
                    "config": checkpoint_config
                },
                MODEL_PATH
            )

            print(
                "\nBest model saved successfully!"
            )

        else:

            patience_counter += 1

            print(
                "\nNo improvement in validation Macro F1."
            )

        print(
            f"Early stopping counter: "
            f"{patience_counter}/{PATIENCE}"
        )

        print("=" * 65)

        # Early stopping
        if patience_counter >= PATIENCE:

            print(
                "\nEARLY STOPPING ACTIVATED"
            )

            break

    total_training_time = (
        time.perf_counter()
        - total_training_start
    )

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_model_state is None:

        raise RuntimeError(
            "Training did not produce a valid model."
        )

    model.load_state_dict(
        best_model_state
    )

    print("\nBest epoch:", best_epoch)

    print(
        "Best validation Macro F1:",
        round(best_macro_f1, 4)
    )

    print(
        "Total training time:",
        format_time(total_training_time)
    )

    # --------------------------------------------------------
    # PLOT TRAINING HISTORY
    # --------------------------------------------------------

    history_df = save_training_history(
        history
    )

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
        "BiLSTM Training and Validation Loss"
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
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # FINAL TESTING
    # --------------------------------------------------------

    print("\nEvaluating final model on test data...")

    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        y_true,
        y_pred
    ) = evaluate_model(
        model,
        test_loader,
        criterion
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

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("FINAL BiLSTM TEST RESULTS")

    print("=" * 65)

    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Accuracy:        {test_accuracy:.4f}")
    print(f"Precision:       {test_precision:.4f}")
    print(f"Recall:          {test_recall:.4f}")
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

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

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

        "model": "BiLSTM",

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

        "vocabulary_size": len(vocabulary),

        "max_sequence_length": MAX_SEQUENCE_LENGTH,

        "embedding_dimension": EMBEDDING_DIM,

        "hidden_dimension": HIDDEN_DIM,

        "batch_size": BATCH_SIZE,

        "total_training_time_seconds": total_training_time

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

        "actual_category": label_encoder.inverse_transform(
            y_true
        ),

        "predicted_category": label_encoder.inverse_transform(
            y_pred
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
        "BiLSTM News Classification - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # Save confusion matrix data
    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("BiLSTM TRAINING COMPLETED SUCCESSFULLY")

    print("=" * 65)

    print("\nSaved Model:")
    print(MODEL_PATH)

    print("\nAll Results Saved In:")
    print(OUTPUT_DIR)

    print("\nGenerated Files:")

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

    print("\nTask 15 BiLSTM completed successfully!")


# ============================================================
# 14. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
