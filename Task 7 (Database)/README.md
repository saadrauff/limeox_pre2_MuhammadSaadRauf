# Task 7 – PostgreSQL Database Integration

## Project Overview
This task integrates the LimeoX web scraping and data preprocessing pipeline with PostgreSQL. The validated news article dataset is imported into a relational database for structured storage and SQL-based analysis.

## Dataset
- **Source:** Task 4 – Validation
- **Input file:** `news_accepted.csv`
- **Total records imported:** 99,900
- **Total columns in CSV:** 15

## Database Configuration
- **Database:** `webscraping`
- **Table:** `articles`
- **Database system:** PostgreSQL 18
- **Python database driver:** Psycopg 3

## Project Files
- `import_to_postgres.py` – Python script used to import CSV records into PostgreSQL.

## Import Method
The project uses PostgreSQL's `COPY` command through Psycopg to efficiently transfer CSV records into the database.

The import process:
1. Checks whether the input CSV exists.
2. Establishes a PostgreSQL connection.
3. Checks whether the destination table already contains records.
4. Streams CSV data into PostgreSQL.
5. Handles empty CSV fields as SQL NULL values.
6. Verifies the total number of imported records.

## Validation and Verification
The database import was verified using SQL queries:

Sample records were also inspected:

**Final verified record count: 99,900**

## Requirements
- Python 3
- PostgreSQL 18
- Psycopg 3 with binary support

## Notes
- The original validated CSV file remains unchanged.
- Database credentials are not stored in the Python script.
- The import script prevents importing data when the destination table is already populated.
