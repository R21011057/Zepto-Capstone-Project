import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import os

GBP_TO_INR = 105.50

def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def parse_rating(rating_str):
    mapping = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    return mapping.get(rating_str, 0)

def scrape_books():
    base_url = "http://books.toscrape.com"
    soup = get_soup(f"{base_url}/index.html")
    category_list = soup.select('.side_categories ul li ul li a')
    categories_to_scrape = []
    
    for cat in category_list:
        cat_name = cat.text.strip()
        cat_url = base_url + "/" + cat['href']
        categories_to_scrape.append((cat_name, cat_url))
        
    books_data = []
    
    # Scrape specific categories until we have at least 80 books
    # (The requirement is >= 60 books across >= 3 categories)
    for cat_name, cat_url in categories_to_scrape:
        if len(books_data) >= 80:
            break
            
        current_url = cat_url
        while current_url:
            try:
                cat_soup = get_soup(current_url)
            except Exception as e:
                print(f"Failed to fetch {current_url}: {e}")
                break
                
            books = cat_soup.select('.product_pod')
            
            for book in books:
                try:
                    title = book.h3.a['title']
                    price_str = book.select_one('.price_color').text
                    # Clean currency symbol and convert to float
                    price_gbp = float(re.sub(r'[^\d.]', '', price_str))
                    
                    # Extract string representation of rating and map to int
                    rating_class = book.select_one('.star-rating')['class']
                    rating_str = [c for c in rating_class if c != 'star-rating'][0]
                    rating = parse_rating(rating_str)
                    
                    # Convert availability to boolean
                    availability_str = book.select_one('.availability').text.strip()
                    in_stock = 'In stock' in availability_str
                    
                    # Convert GBP to INR
                    price_inr = round(price_gbp * GBP_TO_INR, 2)
                    
                    books_data.append({
                        'title': title,
                        'price_gbp': price_gbp,
                        'price_inr': price_inr,
                        'rating': rating,
                        'in_stock': in_stock,
                        'category': cat_name
                    })
                except Exception as e:
                    print(f"Failed to parse a book in {cat_name}: {e}")
            
            # Pagination
            next_btn = cat_soup.select_one('.next a')
            if next_btn:
                base_page_url = current_url.rsplit('/', 1)[0]
                current_url = f"{base_page_url}/{next_btn['href']}"
            else:
                current_url = None
                
    return pd.DataFrame(books_data)

def main():
    print("Scraping books...")
    df = scrape_books()
    print(f"Successfully scraped {len(df)} books from {df['category'].nunique()} categories.")
    
    # ---------------------------------------------------------
    # Database Normalization and Storage
    # ---------------------------------------------------------
    categories_df = pd.DataFrame({'category_name': df['category'].unique()})
    categories_df['category_id'] = range(1, len(categories_df) + 1)
    
    df = df.merge(categories_df, left_on='category', right_on='category_name')
    books_df = df[['title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_id']].copy()
    books_df['book_id'] = range(1, len(books_df) + 1)
    
    db_path = 'data_pipeline.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("\nConnecting to SQLite and saving tables...")
    conn = sqlite3.connect(db_path)
    
    # Enable foreign key constraints in sqlite (for correctness)
    conn.execute("PRAGMA foreign_keys = ON;")
    
    # Explicitly create tables with primary and foreign keys
    conn.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY,
        category_name TEXT UNIQUE NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER,
        FOREIGN KEY(category_id) REFERENCES categories(category_id)
    );
    """)
    
    categories_df[['category_id', 'category_name']].to_sql('categories', conn, index=False, if_exists='append')
    books_df[['book_id', 'title', 'price_gbp', 'price_inr', 'rating', 'in_stock', 'category_id']].to_sql('books', conn, index=False, if_exists='append')
    
    print("Data saved successfully.")
    
    # ---------------------------------------------------------
    # SQL Queries Execution
    # ---------------------------------------------------------
    print("\nExecuting 5 SQL Queries covering required clauses...\n")
    
    queries = {
        "Q1_SELECT_LIMIT": "SELECT title, price_gbp FROM books LIMIT 5;",
        "Q2_WHERE_ORDER": "SELECT title, rating, price_gbp FROM books WHERE rating >= 4 ORDER BY price_gbp ASC LIMIT 5;",
        "Q3_DISTINCT": "SELECT DISTINCT category_id FROM books ORDER BY category_id;",
        "Q4_IN_BETWEEN": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 15.0 AND 30.0 LIMIT 5;",
        "Q5_JOIN": """
            SELECT b.title, c.category_name, b.price_gbp
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE c.category_name IN ('Mystery', 'Historical Fiction')
            ORDER BY b.price_gbp DESC
            LIMIT 10;
        """
    }
    
    for q_name, query in queries.items():
        print(f"--- {q_name} ---")
        print(f"Query: {query}")
        result = pd.read_sql(query, conn)
        print("Result:")
        print(result)
        print("-" * 40)
        
    # ---------------------------------------------------------
    # Pandas Verification
    # ---------------------------------------------------------
    print("\n--- Pandas Merge Verification ---")
    print("Reading full tables back into Pandas...")
    df_books_from_db = pd.read_sql("SELECT * FROM books", conn)
    df_categories_from_db = pd.read_sql("SELECT * FROM categories", conn)
    
    print("Reproducing Q5 (JOIN) using pd.merge()...")
    # Join
    merged_df = pd.merge(df_books_from_db, df_categories_from_db, on='category_id')
    # Filter (IN clause)
    filtered_df = merged_df[merged_df['category_name'].isin(['Mystery', 'Historical Fiction'])]
    # Order By and Limit
    sorted_df = filtered_df.sort_values(by='price_gbp', ascending=False).head(10)
    # Select specific columns
    pandas_result = sorted_df[['title', 'category_name', 'price_gbp']].reset_index(drop=True)
    
    print("Pandas Output:")
    print(pandas_result)
    
    # Retrieve SQL result to compare
    sql_result = pd.read_sql(queries["Q5_JOIN"], conn)
    
    print("\nAre the SQL result and Pandas result identical?")
    # We use round to avoid floating point mismatch, though they should be identical
    is_equal = pandas_result.equals(sql_result)
    print(f"Identical: {is_equal}")
    
    conn.close()

if __name__ == "__main__":
    main()
