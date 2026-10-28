# Task 4: Data Validation

## Overview
This task validates the repaired article dataset and generates files for accepted records, validation issues, duplicate review, and other validation results.

## Input Dataset
- `repair_output/repaired_articles.csv`

## Main Script
- `validation.py` — executes the validation process.

## Output Files
The validation script generates results in `validation_output/`, including:

- `news_accepted.csv` — accepted records.
- `news_validation_all.csv` — complete validation dataset.
- `news_validation_issues.csv` — detected validation issues.
- `news_validation_summary.json` — summary of validation results.
- `news_field_completeness.csv` — field completeness report.
- `news_duplicate_review.csv` — duplicate review records.
- `news_excluded.csv` — excluded records.
- `news_quarantine.csv` — quarantined records.
- `news_warning_review.csv` — records requiring warning review.

## How to Run
From the project root, run:

```powershell
python "Task 4 (Validation)/validation.py"
```

Alternatively, run this command inside the Task 4 folder:

```powershell
python validation.py
```

Ensure that the input dataset exists at the path expected by the script.

## Latest Recorded Results
The latest successful run reported:

- Input records: 99,900
- Accepted records: 99,900
- Total validation issues: 0
- Affected records: 0
- Warning records: 0
- Quarantined records: 0
- Excluded non-article records: 0

These results reflect the latest recorded run and may change if the dataset or validation rules are modified.

## Notes
Keep the generated output files in `validation_output/`. Check the script for the exact file paths if you have modified its configuration.
