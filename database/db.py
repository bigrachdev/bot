"""
Database module for the Market Data Bot
"""
import sqlite3
from typing import List, Tuple, Optional
from utils.logger import logger
from config.settings import DB_NAME, NEWS_CACHE_HOURS, TARGET_CHANNELS

class MarketBot:
    """Handle all database operations"""
    
    def __init__(self, admin_id: int):
        self.db_name = DB_NAME
        self.admin_id = admin_id
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for settings"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            
            # Settings table for customizable messages
            c.execute('''CREATE TABLE IF NOT EXISTS settings
                         (key TEXT PRIMARY KEY, value TEXT)''')
            
            # Default welcome message
            c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                      ('welcome_message', 'Welcome to the group, {username}! Stay tuned for market updates. 📈'))
            
            # Subscriptions table - track which chats receive news
            c.execute('''CREATE TABLE IF NOT EXISTS subscriptions
                         (chat_id INTEGER PRIMARY KEY, chat_type TEXT, chat_name TEXT, subscribed_at TIMESTAMP)''')
            
            # News cache - track posted news to avoid duplicates
            c.execute('''CREATE TABLE IF NOT EXISTS news_cache
                         (news_id TEXT PRIMARY KEY, title TEXT, source TEXT, posted_at TIMESTAMP, url TEXT)''')
            
            # Advertisements table - store admin-posted ads
            c.execute('''CREATE TABLE IF NOT EXISTS advertisements
                         (ad_id INTEGER PRIMARY KEY AUTOINCREMENT,
                          admin_id INTEGER,
                          chat_id INTEGER,
                          content TEXT,
                          content_type TEXT,
                          buttons TEXT,
                          image_url TEXT,
                          scheduled_time TIMESTAMP,
                          posted_at TIMESTAMP,
                          status TEXT,
                          created_at TIMESTAMP)''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def get_setting(self, key, default=None):
        """Get setting from database"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("SELECT value FROM settings WHERE key = ?", (key,))
            result = c.fetchone()
            conn.close()
            return result[0] if result else default
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return default

    def set_setting(self, key, value):
        """Set setting in database"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
            conn.commit()
            conn.close()
            logger.info(f"Updated setting {key}")
        except Exception as e:
            logger.error(f"Failed to set setting {key}: {e}")

    def subscribe_chat(self, chat_id: int, chat_type: str, chat_name: str = None):
        """Subscribe a chat to news updates"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO subscriptions 
                        (chat_id, chat_type, chat_name, subscribed_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                     (chat_id, chat_type, chat_name or f"Chat {chat_id}"))
            conn.commit()
            conn.close()
            logger.info(f"Chat {chat_id} subscribed to news")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe chat {chat_id}: {e}")
            return False

    def unsubscribe_chat(self, chat_id: int):
        """Unsubscribe a chat from news updates"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("DELETE FROM subscriptions WHERE chat_id = ?", (chat_id,))
            conn.commit()
            conn.close()
            logger.info(f"Chat {chat_id} unsubscribed from news")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe chat {chat_id}: {e}")
            return False

    def get_subscribed_chats(self) -> List[Tuple[int, str]]:
        """Get all target channels from config"""
        chats = []
        for chat_id_str in TARGET_CHANNELS:
            try:
                chats.append((int(chat_id_str), 'channel'))
            except ValueError:
                logger.warning(f"Invalid channel ID in config: {chat_id_str}")
        if not chats:
            logger.warning("No TARGET_CHANNELS configured - broadcasts will have nowhere to post")
        return chats

    def is_news_cached(self, news_id: str) -> bool:
        """Check if news has already been posted (avoid duplicates)"""
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
        """Cache posted news to avoid duplicates"""
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

    def add_ad(self, admin_id: int, chat_id: int, content: str, content_type: str = 'text', 
               buttons: str = None, image_url: str = None, scheduled_time: str = None) -> int:
        """Add advertisement to database"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            
            status = 'scheduled' if scheduled_time else 'pending'
            
            c.execute("""INSERT INTO advertisements 
                        (admin_id, chat_id, content, content_type, buttons, image_url, 
                         scheduled_time, status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                     (admin_id, chat_id, content, content_type, buttons, image_url, 
                      scheduled_time, status))
            conn.commit()
            ad_id = c.lastrowid
            conn.close()
            
            logger.info(f"Ad #{ad_id} created by admin {admin_id}")
            return ad_id
        except Exception as e:
            logger.error(f"Failed to add ad: {e}")
            return None

    def get_all_ads(self, limit: int = 10) -> List[dict]:
        """Get recent ads"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("""SELECT ad_id, admin_id, chat_id, content, content_type, status, posted_at, created_at
                        FROM advertisements 
                        ORDER BY created_at DESC 
                        LIMIT ?""", (limit,))
            ads = c.fetchall()
            conn.close()
            
            return [{'ad_id': row[0], 'admin_id': row[1], 'chat_id': row[2], 
                    'content': row[3], 'content_type': row[4], 'status': row[5],
                    'posted_at': row[6], 'created_at': row[7]} for row in ads]
        except Exception as e:
            logger.error(f"Failed to get ads: {e}")
            return []

    def get_ads_by_status(self, status: str) -> List[dict]:
        """Get ads by status (pending, posted, scheduled)"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("""SELECT ad_id, admin_id, chat_id, content, content_type, status, scheduled_time
                        FROM advertisements 
                        WHERE status = ?
                        ORDER BY created_at DESC""", (status,))
            ads = c.fetchall()
            conn.close()
            
            return [{'ad_id': row[0], 'admin_id': row[1], 'chat_id': row[2], 
                    'content': row[3], 'content_type': row[4], 'status': row[5],
                    'scheduled_time': row[6]} for row in ads]
        except Exception as e:
            logger.error(f"Failed to get ads by status: {e}")
            return []

    def update_ad_status(self, ad_id: int, status: str):
        """Update ad status (pending -> posted)"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("""UPDATE advertisements 
                        SET status = ?, posted_at = CURRENT_TIMESTAMP
                        WHERE ad_id = ?""", (status, ad_id))
            conn.commit()
            conn.close()
            logger.info(f"Ad #{ad_id} marked as {status}")
        except Exception as e:
            logger.error(f"Failed to update ad status: {e}")
