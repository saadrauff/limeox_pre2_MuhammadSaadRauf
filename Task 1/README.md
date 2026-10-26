# Web Scraping Project — Task 1

## Project Overview

This project focuses on collecting article data from multiple online sources using Python-based web scraping techniques. The project was developed in three stages, progressively increasing the target number of articles from 1,000 to 10,000 and then to 100,000.

The project uses tools and libraries such as Requests, Beautiful Soup, and Selenium to support web page access, HTML parsing, and article data extraction.

## Project Development Stages

### 1. Scraping 1,000 Articles

**Python File:** `scraper.py`

The first version of the scraper was developed with a target of collecting 1,000 articles. This stage established the basic workflow for retrieving article information from websites and saving the collected data.

**Output File:** `articles excel file.csv`

**Data Collected:** 1,000 articles

### 2. Scraping 10,000 Articles

**Python File:** `scraper_10000.py`

The second version expanded the scraping target to 10,000 articles. This stage extended the project to handle a larger volume of article data.

**Output File:** `articles.csv`

**Data Collected:** 10,000 articles

### 3. Scraping 100,000 Articles

**Python File:** `scrap_100000_articles.py`

The third version increased the target to 100,000 articles. This stage aimed to scale up the data collection process and gather article information from multiple sources.

**Output File:** `100000_articles.csv`

**Data Collected So Far:** Approximately 17,000 articles

The target for this version is 100,000 articles; however, approximately 17,000 articles have been collected and saved in the current output file so far.

## Project Files

| File | Description |
|---|---|
| `scraper.py` | Python scraper targeting 1,000 articles |
| `scraper_10000.py` | Python scraper targeting 10,000 articles |
| `scrap_100000_articles.py` | Python scraper targeting 100,000 articles |
| `articles excel file.csv` | Contains data for 1,000 articles |
| `articles.csv` | Contains data for 10,000 articles |
| `100000_articles.csv` | Contains approximately 17,000 collected articles |
| `articles_discovery_checkpoint.csv` | Stores article discovery checkpoint data |
| `sources.txt` | Contains website URLs used as scraping sources |
| `requirements.txt` | Lists the Python dependencies required by the project |
| `README.md` | Documents the project, development stages, and files |

## Technologies Used

- **Python** — Core programming language for the scraping system.
- **Requests** — Retrieves web pages through HTTP requests.
- **Beautiful Soup** — Parses HTML and extracts relevant information.
- **Selenium** — Automates browser interactions and supports pages that require browser-based rendering.
- **CSV** — Stores collected article data in a structured tabular format.

## Data Collection

The project collects article information from multiple online sources. Depending on the website and the available information, article records may include fields such as:

- Article title
- Author
- Publication date
- Article URL
- Publication time
- Source website

The availability of individual fields depends on the structure and metadata provided by each website.

## Project Objective

The main objectives of this project are to:

1. Develop a Python-based web scraping solution.
2. Progressively scale article collection from 1,000 to 10,000 and then 100,000 articles.
3. Extract article information from multiple online sources.
4. Store collected information in CSV files for further analysis and processing.
5. Explore different scraping techniques using Requests, Beautiful Soup, and Selenium.


### 4. Run a Scraper

To run the scraper targeting 1,000 articles:

```bash
python scraper.py
```

To run the scraper targeting 10,000 articles:

```bash
python scraper_10000.py
```

To run the scraper targeting 100,000 articles:

```bash
python scrap_100000_articles.py
```

**Note:** Before running the scripts, check their configuration and ensure that `sources.txt` contains the required website URLs. Actual results depend on website availability, source configuration, and scraping limitations.

## Important Notes

- The article counts listed above describe the current reported output counts, not guaranteed results for every execution.
- The 100,000-article scraper has a target of 100,000 articles, with approximately 17,000 currently saved in `100000_articles.csv`.
- Checkpoint files may be used to retain intermediate discovery or scraping progress.
- Website structures, rate limits, and access restrictions can affect scraping results.
- Scraping should be performed responsibly and in accordance with applicable website terms, robots.txt guidance, and relevant laws.

## Conclusion

This project demonstrates the progressive development of a Python web scraping system, starting with 1,000 articles, expanding to 10,000 articles, and then targeting 100,000 articles. It provides a practical foundation for collecting, storing, and working with article data from multiple online sources.