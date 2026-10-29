
import re
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = (
    BASE_DIR / "Task 4 (Validation)" /
    "validation_output" / "news_accepted.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "news_categorized.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "categorization_checkpoint.json"
REPORT_FILE = OUTPUT_DIR / "categorization_report.txt"

CHUNK_SIZE = 500

RULES = {
    "Market/Stocks": [
        "stock", "stocks", "share price", "stock market",
        "shares", "nasdaq", "dow jones", "investor",
        "investment", "earnings report", "wall street",
        "market cap", "ipo", "quarterly earnings",
        "financial markets", "bond yields", "trading"
    ],
    "Technology": [
        "technology", "tech", "artificial intelligence",
        "machine learning", "software", "smartphone",
        "iphone", "android", "google", "apple",
        "samsung", "microsoft", "cybersecurity",
        "robotics", "computer", "app", "semiconductor",
        "chip", "laptop", "tablet", "internet",
        "cloud computing", "operating system", "gadget",
        "processor", "browser", "social media"
    ],
    "Health": [
        "health", "healthcare", "medical", "medicine",
        "hospital", "doctor", "patient", "disease",
        "cancer", "clinical trial", "vaccine", "mental health",
        "diagnosis", "treatment", "drug approval",
        "public health", "nutrition", "epidemic"
    ],
    "Energy": [
        "energy", "oil prices", "crude oil", "natural gas",
        "electricity", "solar power", "wind power",
        "renewable energy", "nuclear power", "power grid",
        "petroleum", "fuel prices", "energy storage",
        "oil production", "gas prices", "electric vehicle"
    ],
    "Politics": [
        "politics", "political", "election", "president",
        "prime minister", "parliament", "government",
        "senate", "congress", "political party",
        "voting", "campaign", "legislation", "minister",
        "white house", "geopolitics", "diplomatic"
    ],
}


def count_matches(text, keywords):
    text = str(text).lower()
    total = 0

    for keyword in keywords:
        pattern = (
            r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
        )
        total += len(re.findall(pattern, text))

    return total


def categorize_article(row):
    title = str(row.get("title_clean", "") or "")
    content = str(row.get("content_clean", "") or "")

    scores = {}

    for category, keywords in RULES.items():
        title_score = count_matches(title, keywords)
        content_score = count_matches(content, keywords)
        scores[category] = title_score * 3 + content_score

    best_category = max(scores, key=scores.get)

    if scores[best_category] == 0:
        return "Other"

    return best_category


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return 0

    try:
        with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("input_file") != str(INPUT_FILE.resolve()):
            raise ValueError("Checkpoint input path does not match.")

        if not OUTPUT_FILE.exists():
            raise ValueError("Output file is missing; cannot resume safely.")

        return int(data["rows_processed"])

    except Exception as exc:
        raise RuntimeError(
            "Checkpoint read nahi ho saka. Files ko manually delete "
            "ya reset karne se pehle error check karein."
        ) from exc


def save_checkpoint(rows_processed):
    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

    data = {
        "input_file": str(INPUT_FILE.resolve()),
        "rows_processed": rows_processed,
        "chunk_size": CHUNK_SIZE,
    }

    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    temp_file.replace(CHECKPOINT_FILE)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_row = load_checkpoint()

    if start_row == 0 and OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} pehle se maujood hai, lekin checkpoint "
            "nahi mila. Data overwrite nahi kiya gaya. File rename "
            "kar ke phir script run karein."
        )

    if start_row > 0:
        print(f"Resuming from row {start_row:,}...")
    else:
        print("Starting full categorization...")

    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    rows_seen = 0
    rows_written = start_row
    category_counts = {}

    try:
        for chunk in reader:
            chunk_start = rows_seen
            rows_seen += len(chunk)

            if rows_seen <= start_row:
                continue

            if chunk_start < start_row:
                chunk = chunk.iloc[start_row - chunk_start:].copy()

            chunk["category"] = chunk.apply(categorize_article, axis=1)

            write_header = not OUTPUT_FILE.exists()

            chunk.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=write_header,
                index=False,
                encoding="utf-8-sig"
            )

            rows_written += len(chunk)
            save_checkpoint(rows_written)

            counts = chunk["category"].value_counts().to_dict()
            for category, count in counts.items():
                category_counts[category] = (
                    category_counts.get(category, 0) + int(count)
                )

            print(
                f"Processed and saved: {rows_written:,} articles"
            )

    except KeyboardInterrupt:
        print("\nProcessing stopped by user.")
        print("Saved output and checkpoint retained.")
        print("Script dobara run karne par resume karne ki koshish hogi.")
        return

    # Final report is calculated from the completed output.
    if OUTPUT_FILE.exists():
        final_df = pd.read_csv(
            OUTPUT_FILE,
            usecols=["category"],
            low_memory=False
        )
        final_counts = final_df["category"].value_counts(dropna=False)

        with REPORT_FILE.open("w", encoding="utf-8") as f:
            f.write("NEWS CATEGORIZATION REPORT\n")
            f.write("==========================\n")
            f.write(f"Total categorized articles: {len(final_df):,}\n\n")
            f.write("Category counts:\n")

            for category, count in final_counts.items():
                f.write(f"{category}: {count:,}\n")

        print("\nCategorization completed!")
        print(f"Total articles: {len(final_df):,}")
        print("\nCategory counts:")
        print(final_counts.to_string())
        print(f"\nOutput: {OUTPUT_FILE}")
        print(f"Report: {REPORT_FILE}")
        print(f"Checkpoint: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
