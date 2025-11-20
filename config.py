"""
Configuration file for API keys and project settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Reddit API Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Validate Reddit credentials
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    print("Warning: Reddit API credentials not found in .env file")
    print("Please copy .env.example to .env and add your credentials")

# Data collection settings
SCRAPING_CONFIG = {
    "reddit_subreddits": [
        "politics",
        "worldnews",
        "unpopularopinion",
        "AmItheAsshole",
        "changemyview",
        "movies",
        "television"
    ]
}