# Task 12 (Raw Data Libraries)

This task compares six AutoML libraries using the existing categorized dataset:

`Task 5 (Categorization)/news_categorized.csv`

## Fixed experiment design

- Use only rows where `validation_disposition == ACCEPT`.
- Use the existing `category` column as the target.
- Text inputs: `title_clean` and `content_clean`.
- No new categorization is performed.
- No mapping back to the raw CSV is performed.
- Stratified 80/20 split, `random_state=42`.
- TF-IDF is fitted on training text only; TruncatedSVD converts text vectors into numeric features.
- The same saved training/test arrays are used by all six libraries.
- Metrics: Accuracy, weighted Precision, weighted Recall, weighted F1, Macro F1, Micro F1, and training time.

## Laptop-friendly common sample

The full ACCEPT dataset is split first. To keep all six experiments practical on a typical laptop, the default common benchmark uses a stratified sample of 25,000 training rows and 10,000 test rows. Every library receives exactly these same rows.

To use the complete 80/20 split, edit `automl_common.py`:
- `TRAIN_SAMPLE_LIMIT = None`
- `TEST_SAMPLE_LIMIT = None`

This requires substantially more RAM and training time. Do not change these values between libraries if you want a fair comparison.

## Important Python version note

Use a fresh Python 3.11 (64-bit) environment. The user's existing Python 3.15 environment may not be supported by all pinned AutoML packages. H2O also requires a compatible Java runtime; install a 64-bit JDK (Java 17 is a practical choice) and confirm `java -version` works before running H2O.

## Setup in PowerShell

Run these commands from the project root (`WebScraping`):

```powershell
py -0p
py -3.11 -m venv .venv_task12
.\.venv_task12\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r "Task 12 (Raw Data Libraries)\requirements.txt"
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv_task12\Scripts\Activate.ps1
```

## Run order

Run the shared feature conversion once:

```powershell
python "Task 12 (Raw Data Libraries)\prepare_data.py"
```

Then run each library separately:

```powershell
python "Task 12 (Raw Data Libraries)\02_LazyPredict\run_lazypredict.py"
python "Task 12 (Raw Data Libraries)\03_AutoGluon\run_autogluon.py"
python "Task 12 (Raw Data Libraries)\04_FLAML\run_flaml.py"
python "Task 12 (Raw Data Libraries)\05_TPOT\run_tpot.py"
python "Task 12 (Raw Data Libraries)\06_PyCaret\run_pycaret.py"
python "Task 12 (Raw Data Libraries)\07_H2O_AutoML\run_h2o.py"
```

After all six finish:

```powershell
python "Task 12 (Raw Data Libraries)\08_Final_Comparison\compare_results.py"
```

## Output files

Each library folder saves:
- `metrics.csv`
- `predictions.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- Library-specific model / leaderboard artifacts

Final combined file:
- `08_Final_Comparison/six_library_comparison.csv`

## Notes

- Task 11 is not read, changed, or overwritten.
- `Macro F1` is computed externally using the same held-out labels for every library.
- The internal AutoML leaderboard metric can differ by library; final comparison must use the common external metrics.
- A model can fail because of local RAM, CPU, or package compatibility. Save the complete terminal error before changing package versions.
