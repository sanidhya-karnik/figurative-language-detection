import newspaper
from newspaper import Article
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime

# === CONFIGURATION ===
# List of opinion/commentary section URLs or main URLs
# Expanded to improve yield toward ~3k samples
SOURCES = [
    {'url': 'https://www.theguardian.com/commentisfree/all', 'name': 'The Guardian'},
    {'url': 'https://www.aljazeera.com/opinion/', 'name': 'Al Jazeera'},
    {'url': 'https://www.huffpost.com/opinion', 'name': 'HuffPost'},
    {'url': 'https://www.nbcnews.com/think', 'name': 'NBC News Think'},
    {'url': 'https://www.theatlantic.com/category/opinion/', 'name': 'The Atlantic'},
    {'url': 'https://www.npr.org/sections/opinion/', 'name': 'NPR'},
    {'url': 'https://www.nytimes.com/section/opinion', 'name': 'NYTimes Opinion'},
    {'url': 'https://www.washingtonpost.com/opinions/', 'name': 'Washington Post'},
    {'url': 'https://www.latimes.com/opinion', 'name': 'LA Times'},
    {'url': 'https://www.wsj.com/news/opinion', 'name': 'WSJ Opinion'},
    {'url': 'https://www.theglobeandmail.com/opinion/', 'name': 'Globe and Mail'},
    {'url': 'https://www.chicagotribune.com/opinion/', 'name': 'Chicago Tribune'},
    {'url': 'https://www.nydailynews.com/opinion/', 'name': 'NY Daily News'},
    {'url': 'https://www.usatoday.com/opinion/', 'name': 'USA Today'},
    {'url': 'https://www.politico.com/tag/opinion', 'name': 'Politico Opinion'},
    {'url': 'https://time.com/tag/opinion/', 'name': 'Time Opinion'},
    {'url': 'https://www.reuters.com/world/', 'name': 'Reuters World Analysis'},
    {'url': 'https://www.ft.com/stream/sectionsId/MTA0ZDUxMmUtNDkzNi00NTAxLWFlM2QtNDI1YmY4OWY5YjBk-QWxs', 'name': 'FT Opinion'},
]

OUTPUT_FILE = "data/raw/news_articles.csv"
# Target ~1200 English-language opinion pieces total
MAX_TOTAL_SAMPLES = 1200
SAMPLES_PER_SOURCE = 80  # Overshoot slightly per source to reach the global target

def scrape_source(source_info):
    url = source_info['url']
    source_name = source_info['name']
    
    print(f"\nProcessing {source_name} ({url})...")
    
    # Build newspaper paper object
    try:
        # memoize_articles=False ensures we re-scan even if run before
        paper = newspaper.build(url, memoize_articles=False, language='en')
    except Exception as e:
        print(f"Error building source {source_name}: {e}")
        return []

    print(f"Found {len(paper.articles)} potential articles.")
    
    collected_articles = []
    
    # Sort or shuffle? For now, we take the top ones which are usually newest
    # Newspaper3k articles are generators, so we iterate
    
    count = 0
    for article in paper.articles:
        if count >= SAMPLES_PER_SOURCE:
            break
            
        try:
            # Download and parse
            article.download()
            article.parse()
            
            # Skip non-English content when language metadata is present
            if article.meta_lang and article.meta_lang.lower() != 'en':
                continue
            
            # Basic validation
            text = article.text
            if not text or len(text) < 500: # Skip very short or empty texts
                continue
                
            # Check keywords to ensure it's likely opinion if we are on a main page
            # (Less critical if we target opinion section URLs directly)
            
            collected_articles.append({
                'source': source_name,
                'url': article.url,
                'title': article.title,
                'text': text,
                'publish_date': str(article.publish_date) if article.publish_date else datetime.now().strftime("%Y-%m-%d")
            })
            
            count += 1
            print(f"  - Scraped: {article.title[:50]}...")
            
            # Be polite
            time.sleep(1)
            
        except Exception as e:
            # print(f"  - Failed: {e}")
            continue
            
    return collected_articles

def main():
    print(f"Starting Multi-Source News Scraping (Target: ~{MAX_TOTAL_SAMPLES} samples)...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    all_articles = []
    
    for source in SOURCES:
        if len(all_articles) >= MAX_TOTAL_SAMPLES:
            break
        
        articles = scrape_source(source)
        all_articles.extend(articles)
        
    # Check if we have existing data to append to or overwrite?
    # User said "just like imdb", usually we overwrite raw or append. 
    # Let's overwrite for a fresh batch as requested.
    
    if all_articles:
        df = pd.DataFrame(all_articles)
        
        # Shuffle to mix sources
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Limit to global max
        if len(df) > MAX_TOTAL_SAMPLES:
            df = df.head(MAX_TOTAL_SAMPLES)
            
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Scraped {len(df)} articles from {len(SOURCES)} sources.")
        print(f"Saved to: {OUTPUT_FILE}")
        
        # Show breakdown
        print("\nSource Breakdown:")
        print(df['source'].value_counts())
    else:
        print("Failed to scrape any articles.")

if __name__ == "__main__":
    main()
