# Task 2: 100,000 Articles Web Scraping

## Project Overview
This project uses Python to collect technology, artificial intelligence, science, and research-related article data from multiple online sources.

The scraper uses Requests and BeautifulSoup for static webpage extraction and Selenium as a fallback for selected pages where static extraction does not retrieve a title.

## Project Files

| File | Description |
|---|---|
| `scrap_100000_articles.py` | Python script used for article discovery and scraping |
| `100000_articles.csv` | CSV dataset containing the scraped article records |
| `100000_articles.xlsx` | Excel version of the dataset for convenient viewing and analysis |
| `README.md` | Project documentation |

## Data Fields
The scraper is configured to collect these fields:

- Article title
- Author
- Publication date
- Article URL
- Time
- Source website

The actual availability of individual fields depends on the information published by each source.

## Technologies Used
- Python
- Requests
- BeautifulSoup
- Selenium
- CSV
- Microsoft Excel

## How to Run

1. Install Python.
2. Install the required libraries:

   ```bash
   pip install requests beautifulsoup4 selenium
   ```

3. Open a terminal in the project folder.
4. Run the scraper:

   ```bash
   python scrap_100000_articles.py
   ```

The script is configured with a target of 100,000 articles. The final number collected depends on accessible pages, available source data, and website restrictions.

## Notes
- The CSV and Excel files are separate representations of the dataset and may contain the same records.
- Checkpoint and discovery files are used for local scraping progress and are not included in this project folder's published file list.
- Respect website terms of service, robots.txt rules, copyright requirements, and rate limits.
