"""
Database module for the Market Data Bot
Handles only news caching to prevent duplicate posts.
"""
import sqlite3
from utils.logger import logger
from config.settings import DB_NAME, NEWS_CACHE_HOURS

class MarketBot:
    """Minimal database for tracking posted news."""

    def __init__(self):
        self.db_name = DB_NAME
        self.bot = None
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for news cache only."""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()

            # News cache - track posted news to avoid duplicates
            c.execute('''CREATE TABLE IF NOT EXISTS news_cache
                         (news_id TEXT PRIMARY KEY, title TEXT, source TEXT, posted_at TIMESTAMP, url TEXT)''')

            conn.commit()
            conn.close()

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def is_news_cached(self, news_id: str) -> bool:
        """Check if news has already been posted (avoid duplicates)."""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("""SELECT 1 FROM news_cache
                        WHERE news_id = ? AND
                        datetime(posted_at) > datetime('now', '-' || ? || ' hours')""",
                     (news_id, NEWS_CACHE_HOURS))
            result = c.fetchone()
            conn.close()
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check news cache: {e}")
            return False

    def cache_news(self, news_id: str, title: str, source: str, url: str):
        """Cache posted news to avoid duplicates."""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO news_cache
                        (news_id, title, source, posted_at, url)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)""",
                     (news_id, title, source, url))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to cache news: {e}")
