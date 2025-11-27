import newspaper
from newspaper import Article
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit
import re

# === CONFIGURATION ===
# List of opinion/commentary section URLs or main URLs
# Focus on opinion/analysis outlets with higher volume; recency filter will keep last ~30 days
SOURCES = [
    {'url': 'https://www.theguardian.com/commentisfree/all', 'name': 'The Guardian'},
    {'url': 'https://www.washingtonpost.com/opinions/', 'name': 'Washington Post'},
    {'url': 'https://www.nytimes.com/section/opinion', 'name': 'NYTimes Opinion'},
    {'url': 'https://www.nbcnews.com/think', 'name': 'NBC News Think'},
    {'url': 'https://www.reuters.com/world/', 'name': 'Reuters Analysis'},
    {'url': 'https://www.politico.com/news/opinion', 'name': 'Politico Opinion'},
    {'url': 'https://time.com/section/opinion', 'name': 'TIME Opinion'},
    {'url': 'https://www.cnn.com/opinions', 'name': 'CNN Opinion'},
    {'url': 'https://www.foxnews.com/opinion', 'name': 'Fox News Opinion'},
    {'url': 'https://apnews.com/hub/analysis', 'name': 'AP Analysis'},
    {'url': 'https://www.aljazeera.com/opinions', 'name': 'Al Jazeera Opinion'},
]

OUTPUT_FILE = "data/raw/news_articles.csv"
# No hard cap; rely on source volume + recency filtering
MAX_TOTAL_SAMPLES = None
SAMPLES_PER_SOURCE = None  # Unbounded per source

def truncate_head_tail(text, max_tokens=450, head_ratio=0.75):
    """
    Preserve both the lead and the ending to keep context for downstream models.
    Token budget is approximate (whitespace split).
    """
    if not text:
        return text
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    
    head_tokens = max(1, int(max_tokens * head_ratio))
    tail_tokens = max_tokens - head_tokens
    
    # Sentence-aware: split sentences, then rebuild while respecting head/tail budgets
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    
    head_parts = []
    head_count = 0
    for sent in sentences:
        sent_tokens = sent.split()
        if head_count + len(sent_tokens) > head_tokens:
            break
        head_parts.append(sent)
        head_count += len(sent_tokens)
    
    tail_parts = []
    tail_count = 0
    for sent in reversed(sentences):
        sent_tokens = sent.split()
        if tail_count + len(sent_tokens) > tail_tokens:
            break
        tail_parts.append(sent)
        tail_count += len(sent_tokens)
    tail_parts = list(reversed(tail_parts))
    
    if not head_parts and not tail_parts:
        # fallback to simple token truncation
        return " ".join(tokens[:head_tokens] + ["..."] + tokens[-tail_tokens:])
    
    return " ".join(head_parts + ["..."] + tail_parts)

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
    seen_urls = set()
    seen_titles = set()
    
    # Iterate all available articles; filter to last ~30 days and English
    for article in paper.articles:
        try:
            # Normalize URL early to avoid duplicate downloads
            raw_url = article.url
            if raw_url:
                parts = urlsplit(raw_url)
                normalized_url = urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
                if normalized_url in seen_urls:
                    continue
            else:
                normalized_url = None

            # Download and parse
            article.download()
            article.parse()
            
            # Deduplicate by URL or title (some feeds repeat items)
            if normalized_url:
                if normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
            parsed_title = (article.title or "").strip().lower()
            if parsed_title and parsed_title in seen_titles:
                continue
            if parsed_title:
                seen_titles.add(parsed_title)

            # Skip non-English content when language metadata is present
            if article.meta_lang and article.meta_lang.lower() != 'en':
                continue

            # Enforce recency (publish_date within ~30 days)
            publish_date = article.publish_date
            if publish_date:
                try:
                    if isinstance(publish_date, str):
                        publish_date = datetime.fromisoformat(publish_date.split('T')[0])
                except Exception:
                    pass
            if publish_date:
                age_days = (datetime.now() - publish_date).days
                if age_days > 31:
                    continue
            else:
                # If no reliable date, skip to keep data recent
                continue

            # Skip non-English content when language metadata is present
            # (duplicate check above ensures we don't process same URL twice)

            # Basic validation
            text = article.text
            if not text or len(text) < 500: # Skip very short or empty texts
                continue
            
            # Truncate to keep lead and ending for downstream models
            text = truncate_head_tail(text, max_tokens=450, head_ratio=0.75)
            
            # Check keywords to ensure it's likely opinion if we are on a main page
            # (Less critical if we target opinion section URLs directly)
            
            collected_articles.append({
                'source': source_name,
                'url': article.url,
                'title': article.title,
                'text': text,
                'publish_date': publish_date.strftime("%Y-%m-%d") if publish_date else ""
            })
            
            print(f"  - Scraped: {article.title[:50]}...")
            
            # Be polite
            time.sleep(1)
            
        except Exception as e:
            # print(f"  - Failed: {e}")
            continue
            
    return collected_articles

def main():
    print(f"Starting Multi-Source News Scraping (recency-limited, no hard cap)...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    all_articles = []
    
    for source in SOURCES:
        articles = scrape_source(source)
        all_articles.extend(articles)
        
    # Check if we have existing data to append to or overwrite?
    # User said "just like imdb", usually we overwrite raw or append. 
    # Let's overwrite for a fresh batch as requested.
    
    if all_articles:
        df = pd.DataFrame(all_articles)
        
        # Shuffle to mix sources
        df = df.sample(frac=1).reset_index(drop=True)
        
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
