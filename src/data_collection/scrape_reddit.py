"""
Reddit data scraper with integrated text length handling

This script scrapes Reddit posts and comments, and automatically handles
long texts that exceed transformer model token limits (512 tokens).
"""

import praw
import pandas as pd
from datetime import datetime
import time
import os
import sys
from pathlib import Path
from transformers import RobertaTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config

class TextLengthHandler:
    """Handle text length for transformer models"""
    
    def __init__(self, model_name='roberta-base', max_length=512):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.max_content_length = max_length - 2  # Account for [CLS] and [SEP]
    
    def get_token_count(self, text: str) -> int:
        """Get number of tokens for a text"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def smart_truncate(self, text: str) -> str:
        """
        Intelligent truncation that tries to keep complete sentences
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_content_length:
            return text
        
        # Try to find a good breaking point (sentence boundary)
        sentences = text.split('. ')
        
        current_text = ""
        for sentence in sentences:
            test_text = current_text + sentence + ". "
            test_tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
            
            if len(test_tokens) > self.max_content_length:
                break
            
            current_text = test_text
        
        # If we got at least some content, return it
        if current_text:
            return current_text.strip()
        
        # Otherwise, fall back to simple truncation
        truncated_tokens = tokens[:self.max_content_length]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)


class RedditScraper:
    """
    Reddit scraper with text length handling
    """
    
    def __init__(self, client_id=None, client_secret=None, user_agent=None, 
                 handle_long_texts=True, max_tokens=512):
        """
        Initialize Reddit API connection
        
        Args:
            client_id: Reddit client ID (defaults to config.REDDIT_CLIENT_ID)
            client_secret: Reddit client secret (defaults to config.REDDIT_CLIENT_SECRET)
            user_agent: User agent string (defaults to config.REDDIT_USER_AGENT)
            handle_long_texts: Whether to truncate long texts
            max_tokens: Maximum token length (default 512 for RoBERTa/DeBERTa)
        """
        # Use provided credentials or fall back to config
        self.client_id = client_id or config.REDDIT_CLIENT_ID
        self.client_secret = client_secret or config.REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or config.REDDIT_USER_AGENT
        
        # Text handling
        self.handle_long_texts = handle_long_texts
        if handle_long_texts:
            print("Initializing text length handler...")
            self.text_handler = TextLengthHandler(max_length=max_tokens)
            print(f"✓ Text will be truncated to {max_tokens} tokens")
        else:
            self.text_handler = None
            print("⚠️  Warning: Long text handling disabled")
        
        # Validate credentials
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Reddit API credentials not found. "
                "Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file"
            )
        
        # Initialize Reddit API
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            # Test connection
            self.reddit.user.me()
            print("✓ Reddit API connection successful")
        except Exception as e:
            print(f"✗ Reddit API connection failed: {e}")
            print("\nPlease check your credentials in .env file:")
            print("1. Go to https://www.reddit.com/prefs/apps")
            print("2. Create an app (script type)")
            print("3. Copy client_id and client_secret to .env")
            raise
    
    def process_text(self, text: str) -> dict:
        """
        Process text and return metadata
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with processed text and metadata
        """
        if not text or len(text.strip()) == 0:
            return None
        
        original_text = text
        original_length = len(text)
        
        # Get token count
        if self.text_handler:
            original_tokens = self.text_handler.get_token_count(text)
            
            # Truncate if needed
            if original_tokens > self.text_handler.max_content_length:
                text = self.text_handler.smart_truncate(text)
                final_tokens = self.text_handler.get_token_count(text)
                was_truncated = True
            else:
                final_tokens = original_tokens
                was_truncated = False
        else:
            original_tokens = None
            final_tokens = None
            was_truncated = False
        
        return {
            'text': text,
            'original_length': original_length,
            'final_length': len(text),
            'original_tokens': original_tokens,
            'final_tokens': final_tokens,
            'was_truncated': was_truncated
        }
    
    def scrape_subreddit(self, subreddit_name, limit=1000, search_query=None):
        """
        Scrape posts and comments from a subreddit
        
        Args:
            subreddit_name: Name of subreddit
            limit: Number of items to scrape
            search_query: Optional search query
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        data = []
        
        print(f"\nScraping r/{subreddit_name}...")
        
        try:
            # Scrape posts
            if search_query:
                submissions = subreddit.search(search_query, limit=limit//2)
            else:
                submissions = subreddit.hot(limit=limit//2)
            
            posts_scraped = 0
            posts_truncated = 0
            
            for submission in submissions:
                # Combine title and body
                full_text = submission.title
                if submission.selftext:
                    full_text += " " + submission.selftext
                
                if len(full_text.strip()) < 50:  # Skip very short posts
                    continue
                
                # Process text
                processed = self.process_text(full_text)
                if not processed:
                    continue
                
                data.append({
                    'text': processed['text'],
                    'source': f'reddit_{subreddit_name}',
                    'type': 'post',
                    'score': submission.score,
                    'timestamp': datetime.fromtimestamp(submission.created_utc),
                    'url': f"https://reddit.com{submission.permalink}",
                    'original_tokens': processed['original_tokens'],
                    'final_tokens': processed['final_tokens'],
                    'was_truncated': processed['was_truncated']
                })
                
                posts_scraped += 1
                if processed['was_truncated']:
                    posts_truncated += 1
            
            print(f"  Posts: {posts_scraped} scraped, {posts_truncated} truncated")
            
            # Scrape comments
            comments_scraped = 0
            comments_truncated = 0
            
            for comment in subreddit.comments(limit=limit//2):
                if len(comment.body) < 50:  # Skip very short comments
                    continue
                
                # Process text
                processed = self.process_text(comment.body)
                if not processed:
                    continue
                
                data.append({
                    'text': processed['text'],
                    'source': f'reddit_{subreddit_name}',
                    'type': 'comment',
                    'score': comment.score,
                    'timestamp': datetime.fromtimestamp(comment.created_utc),
                    'url': f"https://reddit.com{comment.permalink}",
                    'original_tokens': processed['original_tokens'],
                    'final_tokens': processed['final_tokens'],
                    'was_truncated': processed['was_truncated']
                })
                
                comments_scraped += 1
                if processed['was_truncated']:
                    comments_truncated += 1
                
                time.sleep(0.1)  # Rate limiting
            
            print(f"  Comments: {comments_scraped} scraped, {comments_truncated} truncated")
            
        except Exception as e:
            print(f"Error scraping r/{subreddit_name}: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    def scrape_multiple_subreddits(self, subreddits, limit_per_sub=500):
        """Scrape multiple subreddits"""
        all_data = []
        total_truncated = 0
        total_samples = 0
        
        for sub in subreddits:
            try:
                df = self.scrape_subreddit(sub, limit=limit_per_sub)
                if not df.empty:
                    all_data.append(df)
                    total_samples += len(df)
                    total_truncated += df['was_truncated'].sum()
                    print(f"✓ Scraped {len(df)} items from r/{sub}")
                time.sleep(2)
            except Exception as e:
                print(f"✗ Error scraping r/{sub}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print("SCRAPING SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples: {total_samples}")
        print(f"Truncated samples: {total_truncated} ({total_truncated/total_samples*100:.1f}%)")
        
        if self.text_handler:
            print(f"\nToken statistics:")
            print(f"  Mean original tokens: {combined_df['original_tokens'].mean():.1f}")
            print(f"  Mean final tokens: {combined_df['final_tokens'].mean():.1f}")
            print(f"  Max original tokens: {combined_df['original_tokens'].max()}")
            print(f"  Max final tokens: {combined_df['final_tokens'].max()}")
            
            # Distribution
            print(f"\nToken distribution (original):")
            bins = [0, 128, 256, 384, 512, 1024, 2048, float('inf')]
            labels = ['0-128', '128-256', '256-384', '384-512', '512-1K', '1K-2K', '2K+']
            combined_df['token_bin'] = pd.cut(combined_df['original_tokens'], bins=bins, labels=labels)
            print(combined_df['token_bin'].value_counts().sort_index())
        
        return combined_df


def main():
    """Main function to scrape Reddit data"""
    
    print("="*70)
    print("REDDIT DATA SCRAPER")
    print("="*70)
    
    # Credentials are automatically loaded from .env via config.py
    try:
        scraper = RedditScraper(
            handle_long_texts=True,  # Enable text truncation
            max_tokens=512           # RoBERTa/DeBERTa limit
        )
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Use subreddits from config
    subreddits = config.SCRAPING_CONFIG['reddit_subreddits']
    
    print(f"\nScraping from {len(subreddits)} subreddits:")
    for sub in subreddits:
        print(f"  - r/{sub}")
    
    print(f"\nConfiguration:")
    print(f"  - Text truncation: ENABLED")
    print(f"  - Max tokens: 512")
    print(f"  - Truncation strategy: Smart (preserve sentences)")
    
    response = input("\nProceed with scraping? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Scrape data
    df = scraper.scrape_multiple_subreddits(subreddits, limit_per_sub=300)
    
    if df.empty:
        print("\n✗ No data collected")
        return
    
    # Save
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/reddit_data.csv'
    
    # Save basic version (just text and label-relevant columns)
    df_basic = df[['text', 'source', 'type', 'score', 'timestamp', 'url']].copy()
    df_basic.to_csv(output_path, index=False)
    
    # Save detailed version with metadata
    detailed_path = 'data/raw/reddit_data_detailed.csv'
    df.to_csv(detailed_path, index=False)
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    print(f"✓ Saved {len(df)} Reddit posts/comments")
    print(f"✓ Basic data: {output_path}")
    print(f"✓ Detailed data (with metadata): {detailed_path}")
    
    print(f"\nBreakdown by subreddit:")
    print(df['source'].value_counts())
    
    print(f"\nBreakdown by type:")
    print(df['type'].value_counts())
    
    # Save statistics
    stats_path = 'data/raw/reddit_data_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("REDDIT SCRAPING STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Truncated samples: {df['was_truncated'].sum()}\n")
        f.write(f"Truncation rate: {df['was_truncated'].mean()*100:.1f}%\n\n")
        f.write(f"Token Statistics:\n")
        f.write(f"  Mean original tokens: {df['original_tokens'].mean():.1f}\n")
        f.write(f"  Mean final tokens: {df['final_tokens'].mean():.1f}\n")
        f.write(f"  Max original tokens: {df['original_tokens'].max()}\n")
        f.write(f"  Max final tokens: {df['final_tokens'].max()}\n\n")
        f.write("Subreddit breakdown:\n")
        f.write(df['source'].value_counts().to_string())
    
    print(f"✓ Saved statistics: {stats_path}")
    
    print("\n" + "="*70)
    print("SCRAPING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the data in data/raw/reddit_data.csv")
    print("  2. Check statistics in data/raw/reddit_data_stats.txt")
    print("  3. Scrape more sources (IMDb, news): python -m src.data_collection.scrape_imdb")
    print("  4. Annotate data: jupyter notebook notebooks/annotation_guide.ipynb")


if __name__ == "__main__":
    main()