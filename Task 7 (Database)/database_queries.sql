-- 1. Total number of articles
SELECT COUNT(*) AS total_articles
FROM articles;


-- 2. Display first 10 articles
SELECT id, title_clean, author_clean, source, domain
FROM articles
LIMIT 10;


-- 3. Count articles by source
SELECT source, COUNT(*) AS total_articles
FROM articles
GROUP BY source
ORDER BY total_articles DESC;


-- 4. Count articles by year
SELECT year, COUNT(*) AS total_articles
FROM articles
GROUP BY year
ORDER BY year DESC;


-- 5. Find articles by keyword in title
SELECT id, title_clean, news_url_clean
FROM articles
WHERE title_clean ILIKE '%artificial intelligence%'
LIMIT 20;


-- 6. Find articles from a specific domain
SELECT id, title_clean, domain
FROM articles
WHERE domain = 'www.nasa.gov'
LIMIT 20;


-- 7. Check missing values in important columns
SELECT
    COUNT(*) FILTER (WHERE title_clean IS NULL) AS missing_titles,
    COUNT(*) FILTER (WHERE news_url_clean IS NULL) AS missing_urls,
    COUNT(*) FILTER (WHERE content_clean IS NULL) AS missing_content
FROM articles;


-- 8. Find latest published articles
SELECT title_clean, published_at_utc, source
FROM articles
ORDER BY published_at_utc_parsed DESC NULLS LAST
LIMIT 10;