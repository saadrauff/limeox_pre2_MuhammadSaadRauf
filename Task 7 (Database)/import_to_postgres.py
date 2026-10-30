import psycopg
from getpass import getpass
from pathlib import Path

# CSV file location
CSV_PATH = Path(
    r"C:\Users\muham\OneDrive\Desktop\WebScraping\Task 4 (Validation)\validation_output\news_accepted.csv"
)

# PostgreSQL connection details
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "webscraping",
    "user": "postgres",
}

COLUMNS = """
source_row_index,
title_clean,
author_clean,
published_at_utc,
year,
time,
news_url_clean,
source,
content_clean,
domain,
published_at_utc_parsed,
duplicate_signature_candidate,
validation_issue_count,
validation_issue_codes,
validation_disposition
"""

def main():
    if not CSV_PATH.exists():
        print("ERROR: CSV file not found!")
        return

    password = getpass("Enter PostgreSQL password: ")

    try:
        with psycopg.connect(**DB_CONFIG, password=password) as conn:

            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM articles;")
                existing_rows = cur.fetchone()[0]

                if existing_rows > 0:
                    print(f"Import cancelled: Table already contains {existing_rows} records.")
                    return

                print("Starting CSV import...")

                copy_sql = f"""
                    COPY articles ({COLUMNS})
                    FROM STDIN
                    WITH (
                        FORMAT CSV,
                        HEADER TRUE,
                        NULL ''
                    )
                """

                with cur.copy(copy_sql) as copy:
                    with CSV_PATH.open("rb") as csv_file:
                        while True:
                            chunk = csv_file.read(1024 * 1024)

                            if not chunk:
                                break

                            copy.write(chunk)

                cur.execute("SELECT COUNT(*) FROM articles;")
                total_rows = cur.fetchone()[0]

            print("\nImport completed successfully!")
            print(f"Total records imported: {total_rows}")

    except Exception as error:
        print("\nIMPORT FAILED!")
        print(error)


if __name__ == "__main__":
    main()