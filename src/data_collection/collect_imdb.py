import os
import requests
import json
import time
import argparse
from pathlib import Path

# === CONFIGURATION ===
BASE_URL = "https://caching.graphql.imdb.com/"
HEADERS = {
    'accept': 'application/graphql+json, application/json',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'origin': 'https://www.imdb.com',
    'priority': 'u=1, i',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# === TRENDING MOVIES LOGIC ===
def get_trending_movies(count=150):
    """Fetch trending movies from IMDb"""
    print(f"Fetching top {count} trending movies...")
    
    payload = {
        'query': """query Trending($first: Int!, $input: TopTrendingInput!) {
          topTrendingTitles(first: $first, input: $input) {
            edges {
              node {
                item {
                  id
                  titleText {
                    text
                  }
                  releaseYear {
                    year
                  }
                }
                rank
              }
            }
          }
        }""",
        'operationName': "Trending",
        'variables': {
            "first": count,
            "input": {
                "dataWindow": "HOURS", 
                "trafficSource": "XWW"
            }
        }
    }
    
    # Use the exact headers from the working script
    headers = HEADERS.copy()
    headers['user-agent'] = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36'
    
    response = requests.post(BASE_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error fetching trending movies: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()
        
    data = response.json()
    
    movies = []
    edges = data.get("data", {}).get("topTrendingTitles", {}).get("edges", [])
    
    if not edges:
        print("No trending movies found in response.")
        if "errors" in data:
            print(f"GraphQL Errors: {data['errors']}")
    
    for edge in edges:
        node = edge.get("node", {})
        item = node.get("item", {})
        movies.append({
            "id": item.get("id"),
            "title": item.get("titleText", {}).get("text"),
            "year": item.get("releaseYear", {}).get("year"),
            "rank": node.get("rank")
        })
    
    return movies
import requests
import json
import time
import argparse
from pathlib import Path
import csv

# === CONFIGURATION ===
BASE_URL = "https://caching.graphql.imdb.com/"
HEADERS = {
    'accept': 'application/graphql+json, application/json',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'origin': 'https://www.imdb.com',
    'priority': 'u=1, i',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# === TRENDING MOVIES LOGIC ===
def get_trending_movies(count=150):
    """Fetch trending movies from IMDb"""
    print(f"Fetching top {count} trending movies...")
    
    payload = {
        'query': """query Trending($first: Int!, $input: TopTrendingInput!) {
          topTrendingTitles(first: $first, input: $input) {
            edges {
              node {
                item {
                  id
                  titleText {
                    text
                  }
                  releaseYear {
                    year
                  }
                }
                rank
              }
            }
          }
        }""",
        'operationName': "Trending",
        'variables': {
            "first": count,
            "input": {
                "dataWindow": "HOURS", 
                "trafficSource": "XWW"
            }
        }
    }
    
    # Use the exact headers from the working script
    headers = HEADERS.copy()
    headers['user-agent'] = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36'
    
    response = requests.post(BASE_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error fetching trending movies: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()
        
    data = response.json()
    
    movies = []
    edges = data.get("data", {}).get("topTrendingTitles", {}).get("edges", [])
    
    if not edges:
        print("No trending movies found in response.")
        if "errors" in data:
            print(f"GraphQL Errors: {data['errors']}")
    
    for edge in edges:
        node = edge.get("node", {})
        item = node.get("item", {})
        movies.append({
            "id": item.get("id"),
            "title": item.get("titleText", {}).get("text"),
            "year": item.get("releaseYear", {}).get("year"),
            "rank": node.get("rank")
        })
    
    return movies

# === REVIEW EXTRACTION LOGIC ===
def fetch_review_page(movie_id, after_cursor=None):
    variables = {
        "const": movie_id,
        "filter": {},
        "first": 25, 
        "sort": { "by": "HELPFULNESS_SCORE", "order": "DESC" }
    }
    if after_cursor:
        variables["after"] = after_cursor

    payload = {
        "query": """query TitleReviewsRefine($const: ID!, $filter: ReviewsFilter, $first: Int!, $sort: ReviewsSort, $after: ID) {
          title(id: $const) {
            reviews(filter: $filter, first: $first, sort: $sort, after: $after) {
              edges {
                node {
                  authorRating
                  summary { originalText }
                  text { originalText { plainText } }
                  helpfulness { upVotes }
                }
              }
              pageInfo {
                hasNextPage
                endCursor
              }
            }
          }
        }""",
        "operationName": "TitleReviewsRefine",
        "variables": variables
    }

    response = requests.post(BASE_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

def get_movie_reviews(movie_id, movie_title, max_reviews=200):
    """Fetch detailed reviews for a movie"""
    print(f"Fetching reviews for: {movie_title} ({movie_id})...")
    reviews = []
    after_cursor = None
    
    while True:
        try:
            data = fetch_review_page(movie_id, after_cursor)
            
            title_data = data.get("data", {}).get("title")
            if not title_data:
                print(f"No title data found for {movie_id}. Full response: {json.dumps(data, indent=2)}")
                break
                
            reviews_data = title_data.get("reviews")
            if not reviews_data:
                print(f"No reviews data found for {movie_id}. Full response: {json.dumps(data, indent=2)}")
                break
                
            edges = reviews_data.get("edges", [])
            if not edges:
                break
                
            for edge in edges:
                node = edge.get("node", {})
                summary = node.get("summary", {})
                text_data = node.get("text", {})
                original_text = text_data.get("originalText", {}) if text_data else {}
                helpfulness = node.get("helpfulness", {})
                
                review = {
                    "movie_id": movie_id,
                    "movie_title": movie_title,
                    "rating": node.get("authorRating"),
                    "review_title": summary.get("originalText"),
                    "review_body": original_text.get("plainText"),
                    "upvotes": helpfulness.get("upVotes", 0)
                }
                
                # Only add if we have at least a title or body
                if review["review_title"] or review["review_body"]:
                    reviews.append(review)
            
            if len(reviews) >= max_reviews:
                reviews = reviews[:max_reviews]
                break
                
            page_info = reviews_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
                
            after_cursor = page_info.get("endCursor")
            time.sleep(0.2) 
            
        except Exception as e:
            print(f"Error fetching reviews for {movie_title}: {e}")
            break
            
    print(f"Found {len(reviews)} reviews.")
    return reviews

def main():
    parser = argparse.ArgumentParser(description="Extract reviews for top trending movies")
    parser.add_argument("-n", "--num-movies", type=int, default=150, help="Number of movies to scrape (default: 150)")
    parser.add_argument("-r", "--reviews-per-movie", type=int, default=100, help="Max reviews per movie (default: 100)")
    parser.add_argument("-o", "--output", default="data/imdb_reviews.csv", help="Output CSV file (default: data/imdb_reviews.csv)")
    
    args = parser.parse_args()
    
    try:
        movies = get_trending_movies(args.num_movies)
        print(f"Found {len(movies)} movies.")
    except Exception as e:
        print(f"Failed to get trending movies: {e}")
        return

    # 2. Prepare CSV File
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    csv_columns = ["movie_id", "movie_title", "rating", "review_title", "review_body", "upvotes"]
    
    # Open file in write mode to add header, then we will append
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # 3. Loop through movies and append to CSV
    total_reviews = 0
    for i, movie in enumerate(movies, 1):
        print(f"\n[{i}/{len(movies)}] Processing {movie['title']}...")
        
        reviews = get_movie_reviews(movie['id'], movie['title'], args.reviews_per_movie)
        
        if reviews:
            with open(args.output, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writerows(reviews)
            total_reviews += len(reviews)
            
    print(f"\nDone! Saved {total_reviews} reviews to {args.output}")

if __name__ == "__main__":
    main()
