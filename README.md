# Module 1 — Data Pipeline

## Overview
This module extracts book data from `books.toscrape.com`, cleans and normalizes the dataset, and loads it into a SQLite database. It also demonstrates executing complex SQL queries and reproducing a JOIN operation using pandas.

## Setup and Execution

### Requirements
* Python 3.8+
* Install dependencies from the root `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Pipeline
Run the pipeline script from the root directory:
```bash
python data_pipeline/pipeline.py
```
*(Note: If you encounter Unicode display issues on Windows, set the encoding first: `$env:PYTHONIOENCODING="UTF-8"`)*

The script will:
1. Scrape the data.
2. Generate the SQLite database `data_pipeline.db` in the root directory.
3. Print the results of the 5 SQL queries and the pandas validation.

## Parsing and Cleaning Decisions
- **Categories**: Since we need multiple categories and at least 60 books, the script targets specific categories from the sidebar and scrapes their pages until a minimum of 80 books is reached.
- **Price (`price_gbp`)**: The currency symbol (e.g., `£`) is stripped using regex (`r'[^\d.]'`), and the string is cast to a `float`.
- **Price INR (`price_inr`)**: Calculated using the project-defined fixed rate of exactly `1 GBP = 105.50 INR`.
- **Rating (`rating`)**: Extracted from the HTML class name (e.g., `star-rating Three`). The text representation ('One', 'Two', etc.) is mapped to an integer (1-5).
- **Availability (`in_stock`)**: Parsed by checking if the string "In stock" is present in the availability text, yielding a boolean.
- **Normalization**: The data is split into two tables: `categories` (category_id, category_name) and `books` (book_id, title, price_gbp, price_inr, rating, in_stock, category_id) with a foreign key relationship to avoid data duplication.

## SQL Query Outputs

### Query 1: SELECT with LIMIT
```sql
SELECT title, price_gbp FROM books LIMIT 5;
```
**Output:**
```text
                                               title  price_gbp
0                            It's Only the Himalayas      45.17
1  Full Moon over Noahâ€™s Ark: An Odyssey to Mou...      49.43
2  See America: A Celebration of Our National Par...      48.87
3  Vagabonding: An Uncommon Guide to the Art of L...      36.94
4                               Under the Tuscan Sun      37.33
```

### Query 2: WHERE and ORDER BY
```sql
SELECT title, rating, price_gbp FROM books WHERE rating >= 4 ORDER BY price_gbp ASC LIMIT 5;
```
**Output:**
```text
                                               title  rating  price_gbp
0           Fruits Basket, Vol. 2 (Fruits Basket #2)       5      11.64
1  Superman Vol. 1: Before Truth (Superman by Gen...       5      11.89
2                                  The Girl You Lost       5      12.29
3  Princess Jellyfish 2-in-1 Omnibus, Vol. 01 (Pr...       5      13.61
4                                        Roller Girl       5      14.10
```

### Query 3: DISTINCT
```sql
SELECT DISTINCT category_id FROM books ORDER BY category_id;
```
**Output:**
```text
   category_id
0            1
1            2
2            3
3            4
```

### Query 4: IN / BETWEEN
```sql
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 15.0 AND 30.0 LIMIT 5;
```
**Output:**
```text
                                               title  price_gbp
0  The Road to Little Dribbling: Adventures of an...      23.21
1                 1,000 Places to See Before You Die      26.08
2                               In a Dark, Dark Wood      19.63
3                                   A Murder in Time      16.64
4            A Study in Scarlet (Sherlock Holmes #1)      16.73
```

### Query 5: JOIN (SQL vs Pandas)

**SQL Query:**
```sql
SELECT b.title, c.category_name, b.price_gbp
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE c.category_name IN ('Mystery', 'Historical Fiction')
ORDER BY b.price_gbp DESC
LIMIT 10;
```

**SQL `pd.read_sql` Output:**
```text
                                               title       category_name  price_gbp
0                      Boar Island (Anna Pigeon #19)             Mystery      59.48
1  The No. 1 Ladies' Detective Agency (No. 1 Ladi...             Mystery      57.70
2                                The Past Never Ends             Mystery      56.50
3                   The Last Painting of Sara de Vos  Historical Fiction      55.55
4            A Flight of Arrows (The Pathfinders #2)  Historical Fiction      55.53
5  Murder at the 42nd Street Library (Raymond Amb...             Mystery      54.36
6                     The Last Mile (Amos Decker #2)             Mystery      54.21
7                1st to Die (Women's Murder Club #1)             Mystery      53.98
8                                 Tipping the Velvet  Historical Fiction      53.74
9  The Bachelor Girl's Guide to Murder (Herringfo...             Mystery      52.30
```

**Pandas `pd.merge` Output:**
```text
                                               title       category_name  price_gbp
0                      Boar Island (Anna Pigeon #19)             Mystery      59.48
1  The No. 1 Ladies' Detective Agency (No. 1 Ladi...             Mystery      57.70
2                                The Past Never Ends             Mystery      56.50
3                   The Last Painting of Sara de Vos  Historical Fiction      55.55
4            A Flight of Arrows (The Pathfinders #2)  Historical Fiction      55.53
5  Murder at the 42nd Street Library (Raymond Amb...             Mystery      54.36
6                     The Last Mile (Amos Decker #2)             Mystery      54.21
7                1st to Die (Women's Murder Club #1)             Mystery      53.98
8                                 Tipping the Velvet  Historical Fiction      53.74
9  The Bachelor Girl's Guide to Murder (Herringfo...             Mystery      52.30
```

**Result:** Both outputs match exactly (`pandas_result.equals(sql_result) == True`).
