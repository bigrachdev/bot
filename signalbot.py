import sqlite3
import requests
import os
import signal
import sys
from datetime import datetime, timedelta
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, CallbackQuery
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
import pytz
import logging
import asyncio
import html
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
import nest_asyncio
from tradingview_ta import TA_Handler, Interval
from typing import List, Tuple, Optional

# Apply nest_asyncio for Windows compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure logging for production
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('market_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables for cleanup
application = None

# Production configuration
BOT_TOKEN = os.getenv('BOT_TOKEN')
YOUR_ADMIN_ID = int(os.getenv('YOUR_ADMIN_ID', '0'))

# Validate critical configuration
if not BOT_TOKEN:
    logger.error("Missing critical configuration: BOT_TOKEN")
    sys.exit(1)

if YOUR_ADMIN_ID == 0:
    logger.warning("ADMIN_ID not set - admin commands will be disabled")

DB_NAME = 'market_bot.db'
NEWS_CACHE_HOURS = 24  # Don't post same news for 24 hours

# News API Keys (add to .env file)
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
ALPHAVANTAGE_KEY = os.getenv('ALPHAVANTAGE_KEY', '')
FINNHUB_KEY = os.getenv('FINNHUB_KEY', '')
CNBC_RSS_URL = os.getenv('CNBC_RSS_URL', 'https://www.cnbc.com/id/100003114/device/rss/rss.html')

# Hardcoded top lists
TOP_STOCKS = [
    {'name': 'NVIDIA', 'symbol': 'NVDA', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Microsoft', 'symbol': 'MSFT', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Apple', 'symbol': 'AAPL', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Alphabet', 'symbol': 'GOOGL', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Amazon', 'symbol': 'AMZN', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Meta Platforms', 'symbol': 'META', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Saudi Aramco', 'symbol': '2222', 'exchange': 'TADAWUL', 'screener': 'america'},  # Adjust screener if needed
    {'name': 'Berkshire Hathaway', 'symbol': 'BRK-B', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Tesla', 'symbol': 'TSLA', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Broadcom', 'symbol': 'AVGO', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Eli Lilly', 'symbol': 'LLY', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'JPMorgan Chase', 'symbol': 'JPM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Visa', 'symbol': 'V', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Walmart', 'symbol': 'WMT', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'UnitedHealth Group', 'symbol': 'UNH', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Mastercard', 'symbol': 'MA', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Procter & Gamble', 'symbol': 'PG', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Johnson & Johnson', 'symbol': 'JNJ', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Home Depot', 'symbol': 'HD', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Costco', 'symbol': 'COST', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Oracle', 'symbol': 'ORCL', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Merck', 'symbol': 'MRK', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Chevron', 'symbol': 'CVX', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Coca-Cola', 'symbol': 'KO', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'AbbVie', 'symbol': 'ABBV', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'PepsiCo', 'symbol': 'PEP', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Adobe', 'symbol': 'ADBE', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Salesforce', 'symbol': 'CRM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Bank of America', 'symbol': 'BAC', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'McDonald\'s', 'symbol': 'MCD', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Accenture', 'symbol': 'ACN', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Cisco', 'symbol': 'CSCO', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'T-Mobile', 'symbol': 'TMUS', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'IBM', 'symbol': 'IBM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'General Electric', 'symbol': 'GE', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Abbott', 'symbol': 'ABT', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'ServiceNow', 'symbol': 'NOW', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Caterpillar', 'symbol': 'CAT', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Philip Morris', 'symbol': 'PM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Qualcomm', 'symbol': 'QCOM', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Danaher', 'symbol': 'DHR', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'American Express', 'symbol': 'AXP', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'NextEra Energy', 'symbol': 'NEE', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Verizon', 'symbol': 'VZ', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Texas Instruments', 'symbol': 'TXN', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Intuit', 'symbol': 'INTU', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Disney', 'symbol': 'DIS', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Pfizer', 'symbol': 'PFE', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'S&P Global', 'symbol': 'SPGI', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'RTX', 'symbol': 'RTX', 'exchange': 'NYSE', 'screener': 'america'},
]

TOP_FOREX = [
    {'name': 'EUR/USD', 'symbol': 'EURUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/JPY', 'symbol': 'USDJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'GBP/USD', 'symbol': 'GBPUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'AUD/USD', 'symbol': 'AUDUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CAD', 'symbol': 'USDCAD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CHF', 'symbol': 'USDCHF', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'NZD/USD', 'symbol': 'NZDUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'EUR/JPY', 'symbol': 'EURJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'GBP/JPY', 'symbol': 'GBPJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'EUR/GBP', 'symbol': 'EURGBP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'AUD/JPY', 'symbol': 'AUDJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/SEK', 'symbol': 'USDSEK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/NOK', 'symbol': 'USDNOK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/MXN', 'symbol': 'USDMXN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/ZAR', 'symbol': 'USDZAR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/TRY', 'symbol': 'USDTRY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/RUB', 'symbol': 'USDRUB', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/BRL', 'symbol': 'USDBRL', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/INR', 'symbol': 'USDINR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CNY', 'symbol': 'USDCNY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/HKD', 'symbol': 'USDHKD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/SGD', 'symbol': 'USDSGD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/DKK', 'symbol': 'USDDKK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PLN', 'symbol': 'USDPLN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CZK', 'symbol': 'USDCZK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/THB', 'symbol': 'USDTHB', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/ILS', 'symbol': 'USDILS', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PHP', 'symbol': 'USDPHP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/TWD', 'symbol': 'USDTWD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/IDR', 'symbol': 'USDIDR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/MYR', 'symbol': 'USDMYR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/KRW', 'symbol': 'USDKRW', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/SAR', 'symbol': 'USDSAR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PKR', 'symbol': 'USDPKR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/BDT', 'symbol': 'USDBDT', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/EGP', 'symbol': 'USDEGP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/NGN', 'symbol': 'USDNGN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/ARS', 'symbol': 'USDARS', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/COP', 'symbol': 'USDCOP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PEN', 'symbol': 'USDPEN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CLP', 'symbol': 'USDCLP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/UAH', 'symbol': 'USDUAH', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/RON', 'symbol': 'USDRON', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/UZS', 'symbol': 'USDUZS', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/KZT', 'symbol': 'USDKZT', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/KWD', 'symbol': 'USDKWD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/BHD', 'symbol': 'USDBHD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/OMR', 'symbol': 'USDOMR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/QAR', 'symbol': 'USDQAR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/JOD', 'symbol': 'USDJOD', 'exchange': 'OANDA', 'screener': 'forex'},
]

class MarketBot:
    def __init__(self):
        self.db_name = DB_NAME
        self.admin_id = YOUR_ADMIN_ID
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
        """Get all subscribed chats"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            c.execute("SELECT chat_id, chat_type FROM subscriptions")
            chats = c.fetchall()
            conn.close()
            return chats
        except Exception as e:
            logger.error(f"Failed to get subscribed chats: {e}")
            return []

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

    async def fetch_news_newsapi(self) -> List[dict]:
        """Fetch general market news from NewsAPI"""
        if not NEWSAPI_KEY:
            logger.warning("NEWSAPI_KEY not configured")
            return []
        
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': 'stock market OR forex OR crypto OR trading',
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 20,
                'apiKey': NEWSAPI_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            
            news = []
            for article in articles:
                news_id = f"newsapi_{article['url']}"
                if not self.is_news_cached(news_id):
                    news.append({
                        'id': news_id,
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'source': article['source']['name'],
                        'published': article['publishedAt'],
                        'image': article.get('urlToImage')
                    })
            
            logger.info(f"Fetched {len(news)} new articles from NewsAPI")
            return news
            
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []

    async def fetch_news_alphavantage(self) -> List[dict]:
        """Fetch market news from Alpha Vantage"""
        if not ALPHAVANTAGE_KEY:
            logger.warning("ALPHAVANTAGE_KEY not configured")
            return []

        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'sort': 'LATEST',
                'limit': 20,
                'apikey': ALPHAVANTAGE_KEY
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            news = []
            feed = data.get('feed', [])
            for article in feed:
                news_id = f"av_{article['url']}"
                if not self.is_news_cached(news_id):
                    news.append({
                        'id': news_id,
                        'title': article['title'],
                        'description': article.get('summary', ''),
                        'url': article['url'],
                        'source': 'Alpha Vantage',
                        'published': article.get('time_published', ''),
                        'image': article.get('banner_image', None)
                    })

            logger.info(f"Fetched {len(news)} new articles from Alpha Vantage")
            return news

        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed: {e}")
            return []

    @staticmethod
    def _parse_publish_time(published: str) -> datetime:
        """Parse mixed timestamp formats into a comparable UTC datetime."""
        if not published:
            return datetime.min.replace(tzinfo=pytz.UTC)

        try:
            iso = published.replace('Z', '+00:00')
            dt = datetime.fromisoformat(iso)
            return dt if dt.tzinfo else dt.replace(tzinfo=pytz.UTC)
        except Exception:
            pass

        try:
            dt = parsedate_to_datetime(published)
            return dt if dt.tzinfo else dt.replace(tzinfo=pytz.UTC)
        except Exception:
            return datetime.min.replace(tzinfo=pytz.UTC)

    async def fetch_news_cnbc(self) -> List[dict]:
        """Fetch live market news from CNBC RSS."""
        try:
            response = requests.get(CNBC_RSS_URL, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            items = root.findall('.//item')

            news = []
            for item in items[:20]:
                title = (item.findtext('title') or '').strip()
                url = (item.findtext('link') or '').strip()
                description = (item.findtext('description') or '').strip()
                published = (item.findtext('pubDate') or '').strip()

                if not title or not url:
                    continue

                news_id = f"cnbc_{url}"
                if self.is_news_cached(news_id):
                    continue

                # RSS descriptions often include HTML; keep plain readable text.
                description = re.sub(r'<[^>]+>', '', description)
                description = html.unescape(description).strip()

                news.append({
                    'id': news_id,
                    'title': html.unescape(title),
                    'description': description,
                    'url': url,
                    'source': 'CNBC',
                    'published': published,
                    'image': None
                })

            logger.info(f"Fetched {len(news)} new articles from CNBC RSS")
            return news

        except Exception as e:
            logger.error(f"CNBC RSS fetch failed: {e}")
            return []

    async def fetch_all_news(self) -> List[dict]:
        """Fetch and combine news from all sources"""
        all_news = []

        all_news.extend(await self.fetch_news_cnbc())

        if NEWSAPI_KEY:
            all_news.extend(await self.fetch_news_newsapi())

        if ALPHAVANTAGE_KEY:
            all_news.extend(await self.fetch_news_alphavantage())

        # Sort by published date (newest first)
        all_news.sort(key=lambda x: self._parse_publish_time(x.get('published', '')), reverse=True)

        logger.info(f"Aggregated {len(all_news)} total news items")
        return all_news

    def format_news_message(self, news_items: List[dict]) -> str:
        """Format news items for Telegram message"""
        if not news_items:
            return "📰 No recent market news available at this time."
        
        message = "📰 *Market News Update*\n\n"
        
        for i, news in enumerate(news_items[:5], 1):  # Top 5 news
            message += f"*{i}. {news['title']}*\n"
            message += f"Source: {news['source']}\n"
            
            if news['description']:
                desc = news['description'][:150]
                if len(news['description']) > 150:
                    desc += "..."
                message += f"_{desc}_\n"
            
            message += f"🔗 [Read more]({news['url']})\n\n"
        
        return message

    async def fetch_analysis(self, symbol: str, exchange: str, screener: str) -> Optional[dict]:
        """Fetch technical analysis for a symbol"""
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY,
                timeout=10
            )
            analysis = handler.get_analysis()
            
            return {
                'symbol': symbol,
                'recommendation': analysis.summary['RECOMMENDATION'],
                'buy': analysis.summary['BUY'],
                'sell': analysis.summary['SELL'],
                'neutral': analysis.summary['NEUTRAL'],
                'oscillators': analysis.oscillators,
                'moving_averages': analysis.moving_averages
            }
        except Exception as e:
            logger.error(f"Failed to fetch analysis for {symbol}: {e}")
            return None

    async def fetch_top_stocks_analysis(self) -> List[dict]:
        """Fetch analysis for top 10 stocks, filter for strong signals only"""
        top_10_stocks = TOP_STOCKS[:10]  # Top 10 by market cap
        strong_signal_analyses = []
        
        for stock in top_10_stocks:
            analysis = await self.fetch_analysis(
                symbol=stock['symbol'],
                exchange=stock['exchange'],
                screener=stock['screener']
            )
            
            if analysis:
                # Only include STRONG BUY and STRONG SELL signals
                if analysis['recommendation'] in ['STRONG_BUY', 'STRONG_SELL']:
                    strong_signal_analyses.append({
                        'name': stock['name'],
                        **analysis
                    })
            
            await asyncio.sleep(0.5)  # Rate limiting to avoid API throttling
        
        logger.info(f"Found {len(strong_signal_analyses)} stocks with strong signals")
        return strong_signal_analyses

    def format_analysis_message(self, analysis_items: List[dict]) -> Optional[str]:
        """Format analysis items for Telegram message (medium detail)"""
        if not analysis_items:
            return None
        
        message = "📈 *Strong Trading Signals - Top Stocks*\n"
        message += f"_{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
        
        for analysis in analysis_items:
            # Emoji based on signal
            signal_emoji = "🔥" if analysis['recommendation'] == 'STRONG_BUY' else "🔴"
            
            message += f"{signal_emoji} *{analysis['name']} ({analysis['symbol']})*\n"
            message += f"🎯 Signal: `{analysis['recommendation']}`\n"
            message += f"📊 Vote: Buy {analysis['buy']} | Sell {analysis['sell']} | Neutral {analysis['neutral']}\n"
            
            # Show top oscillator
            if analysis['oscillators']:
                oscillator_items = list(analysis['oscillators'].items())[:2]
                message += "📡 Oscillators: "
                message += ", ".join([f"{k.replace('_', ' ')}: {v}" for k, v in oscillator_items])
                message += "\n"
            
            # Show top moving average
            if analysis['moving_averages']:
                ma_items = list(analysis['moving_averages'].items())[:2]
                message += "📉 MAs: "
                message += ", ".join([f"{k.replace('_', ' ')}: {v}" for k, v in ma_items])
                message += "\n"
            
            message += "\n"
        
        message += "_Analysis based on daily timeframe_"
        return message


    async def fetch_top_crypto(self):
        """Fetch top 50 cryptos from CoinGecko"""
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 50,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return [{'name': coin['name'], 'symbol': coin['symbol'].upper() + 'USDT', 'exchange': 'BINANCE', 'screener': 'crypto'} for coin in data]
        except Exception as e:
            logger.error(f"Crypto fetch failed: {e}")
            return []

# Initialize bot instance
bot_instance = MarketBot()

# Start command - show category menu
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Subscribe the chat to news updates
    chat = update.effective_chat
    bot_instance.subscribe_chat(chat.id, chat.type, chat.title or chat.first_name)
    
    keyboard = [
        [InlineKeyboardButton("🚀 Crypto", callback_data='category:crypto')],
        [InlineKeyboardButton("💹 Stocks", callback_data='category:stocks')],
        [InlineKeyboardButton("📈 Forex Pairs", callback_data='category:forex')],
        [InlineKeyboardButton("📰 Get News Now", callback_data='action:news_now')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    text = "📊 *Market Data Bot*\nSelect a category to view top 50 markets:"
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode='Markdown')

# Callback handler for categories, pages, selections
async def market_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    parts = data.split(':')
    action = parts[0]

    if action == 'category':
        cat = parts[1]
        await show_page(query, cat, 1)

    elif action == 'page':
        cat = parts[1]
        page = int(parts[2])
        await show_page(query, cat, page)

    elif action == 'select':
        symbol = parts[1]
        exchange = parts[2]
        screener = parts[3]
        cat = parts[4]
        page = int(parts[5])
        await show_analysis(query, symbol, exchange, screener, cat, page)

    elif action == 'action':
        action_type = parts[1]
        if action_type == 'news_now':
            await send_news_to_chat(query.message.chat.id, context)

    elif action == 'home':
        await start_command(update, context)

async def show_page(query: CallbackQuery, cat: str, page: int):
    if cat == 'crypto':
        items = await bot_instance.fetch_top_crypto()
    elif cat == 'stocks':
        items = TOP_STOCKS
    elif cat == 'forex':
        items = TOP_FOREX
    else:
        return
    
    if not items:
        await query.edit_message_text("⚠️ Failed to fetch data")
        return
    
    per_page = 10
    total_pages = (len(items) - 1) // per_page + 1
    start = (page - 1) * per_page
    end = min(start + per_page, len(items))
    page_items = items[start:end]
    
    text = f"📊 *Top {len(items)} {cat.capitalize()}* - Page {page}/{total_pages}\n\n"
    keyboard = []
    
    for idx, item in enumerate(page_items, start + 1):
        text += f"{idx}. {item['name']}\n"
        cb_data = f"select:{item['symbol']}:{item['exchange']}:{item['screener']}:{cat}:{page}"
        keyboard.append([InlineKeyboardButton(f"{idx}. {item['name']}", callback_data=cb_data)])
    
    nav_row = []
    if page > 1:
        nav_row.append(InlineKeyboardButton("◀️ Prev", callback_data=f"page:{cat}:{page-1}"))
    if end < len(items):
        nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"page:{cat}:{page+1}"))
    nav_row.append(InlineKeyboardButton("🏠 Home", callback_data='home'))
    
    keyboard.append(nav_row)
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')

async def show_analysis(query: CallbackQuery, symbol: str, exchange: str, screener: str, cat: str, page: int):
    try:
        handler = TA_Handler(
            symbol=symbol,
            screener=screener,
            exchange=exchange,
            interval=Interval.INTERVAL_1_DAY,
            timeout=10
        )
        analysis = handler.get_analysis()
        
        rec = analysis.summary['RECOMMENDATION']
        buy = analysis.summary['BUY']
        sell = analysis.summary['SELL']
        neutral = analysis.summary['NEUTRAL']
        
        text = f"📈 *Analysis for {symbol} ({cat.capitalize()})*\n\n"
        text += f"**Signal:** {rec}\n"
        text += f"Buy: {buy} | Sell: {sell} | Neutral: {neutral}\n\n"
        
        text += "**Oscillators:**\n"
        for k, v in analysis.oscillators.items():
            text += f"{k}: {v}\n"
        
        text += "\n**Moving Averages:**\n"
        for k, v in analysis.moving_averages.items():
            text += f"{k}: {v}\n"
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to List", callback_data=f"page:{cat}:{page}")],
            [InlineKeyboardButton("🏠 Home", callback_data='home')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        await query.edit_message_text(f"⚠️ Failed to analyze: {str(e)}")

async def send_news_to_chat(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Fetch and send news to a specific chat"""
    try:
        news_items = await bot_instance.fetch_all_news()
        
        if not news_items:
            await context.bot.send_message(
                chat_id=chat_id,
                text="📰 No market news available at this time.",
                parse_mode='Markdown'
            )
            return
        
        # Send top news items
        for news in news_items[:3]:  # Send top 3 news
            message = f"📰 *{news['title']}*\n\n"
            message += f"Source: {news['source']}\n"
            
            if news['description']:
                desc = news['description'][:200]
                if len(news['description']) > 200:
                    desc += "..."
                message += f"_{desc}_\n\n"
            
            message += f"[Read full article]({news['url']})"
            
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                
                # Cache the news
                bot_instance.cache_news(news['id'], news['title'], news['source'], news['url'])
                
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to send news to chat {chat_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error sending news: {e}")

async def broadcast_news(context: ContextTypes.DEFAULT_TYPE):
    """Scheduled job to broadcast news to all subscribed chats (hourly)"""
    logger.info("Starting hourly news broadcast")
    
    try:
        news_items = await bot_instance.fetch_all_news()
        
        if not news_items:
            logger.warning("No news items to broadcast")
            return
        
        subscribed_chats = bot_instance.get_subscribed_chats()
        logger.info(f"Broadcasting to {len(subscribed_chats)} subscribed chats")
        
        for chat_id, chat_type in subscribed_chats:
            try:
                # Send top news
                for news in news_items[:2]:  # Send top 2 news per broadcast
                    message = f"📰 *{news['title']}*\n\n"
                    message += f"Source: {news['source']}\n"
                    
                    if news['description']:
                        desc = news['description'][:150]
                        if len(news['description']) > 150:
                            desc += "..."
                        message += f"_{desc}_\n\n"
                    
                    message += f"[Read more]({news['url']})"
                    
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                    
                    # Cache the news
                    bot_instance.cache_news(news['id'], news['title'], news['source'], news['url'])
                    
                    await asyncio.sleep(0.5)  # Rate limiting
            
            except Exception as e:
                logger.error(f"Failed to send news to chat {chat_id}: {e}")
    
    except Exception as e:
        logger.error(f"Broadcast news failed: {e}")

async def broadcast_analysis(context: ContextTypes.DEFAULT_TYPE):
    """Scheduled job to broadcast strong trading signals to all subscribed chats (hourly)"""
    logger.info("Starting hourly analysis broadcast")
    
    try:
        analysis_items = await bot_instance.fetch_top_stocks_analysis()
        
        if not analysis_items:
            logger.info("No strong signals found for broadcast")
            return
        
        formatted_message = bot_instance.format_analysis_message(analysis_items)
        
        if not formatted_message:
            logger.info("No formatted analysis to broadcast")
            return
        
        subscribed_chats = bot_instance.get_subscribed_chats()
        logger.info(f"Broadcasting {len(analysis_items)} strong signals to {len(subscribed_chats)} chats")
        
        for chat_id, chat_type in subscribed_chats:
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=formatted_message,
                    parse_mode='Markdown'
                )
                await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to send analysis to chat {chat_id}: {e}")
    
    except Exception as e:
        logger.error(f"Broadcast analysis failed: {e}")


# Welcome handler for new members
async def welcome_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome new members"""
    for member in update.message.new_chat_members:
        welcome_msg = bot_instance.get_setting('welcome_message', 'Welcome, {username}!')
        formatted_msg = welcome_msg.format(username=member.username or member.first_name)
        await update.message.reply_text(formatted_msg)

# Admin command for set welcome (text-based)
async def setwelcome_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    if not context.args:
        current = bot_instance.get_setting('welcome_message', 'Default welcome')
        await update.message.reply_text(f"Current welcome: {current}\nUsage: /setwelcome <new message>\nUse {{username}} for name")
        return
    
    new_message = " ".join(context.args)
    bot_instance.set_setting('welcome_message', new_message)
    await update.message.reply_text("✅ Welcome message updated")

# Admin command for sending news immediately
async def sendnews_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send news immediately to all subscribed chats (admin only)"""
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    await update.message.reply_text("📰 Broadcasting news to all subscribed chats...")
    await broadcast_news(context)
    await update.message.reply_text("✅ News broadcast completed")

# Admin command for sending analysis immediately
async def sendanalysis_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send analysis immediately to all subscribed chats (admin only)"""
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    await update.message.reply_text("📈 Analyzing top 10 stocks and broadcasting strong signals...")
    await broadcast_analysis(context)
    await update.message.reply_text("✅ Analysis broadcast completed")

# Admin command for viewing subscriptions
async def subscriptions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View current subscriptions (admin only)"""
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    chats = bot_instance.get_subscribed_chats()
    
    if not chats:
        await update.message.reply_text("No subscribed chats yet")
        return
    
    message = f"📊 Current Subscriptions ({len(chats)} chats):\n\n"
    for chat_id, chat_type in chats:
        message += f"Chat ID: {chat_id} ({chat_type})\n"
    
    await update.message.reply_text(message)

# Admin command for posting text ads
async def postmessage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Post a text ad to a specific chat (admin only)"""
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /postmessage <chat_id> <message>\n\n"
            "Example: /postmessage -1001234567890 Check out our new trading signals! 🚀\n\n"
            "To add buttons: /postmessage <chat_id> <message> --button=Label:URL --button=Label2:URL2\n"
            "Get chat ID from /subscriptions command"
        )
        return
    
    try:
        chat_id = int(context.args[0])
        
        # Separate message from buttons
        message_parts = context.args[1:]
        buttons_list = []
        message_text = []
        
        for part in message_parts:
            if part.startswith('--button='):
                button_data = part[9:]  # Remove '--button='
                if ':' in button_data:
                    label, url = button_data.split(':', 1)
                    buttons_list.append({'label': label, 'url': url})
            else:
                message_text.append(part)
        
        message = " ".join(message_text)
        
        if not message:
            await update.message.reply_text("❌ Message cannot be empty")
            return
        
        # Store ad in database
        ad_id = bot_instance.add_ad(
            admin_id=update.message.from_user.id,
            chat_id=chat_id,
            content=message,
            content_type='text',
            buttons=str(buttons_list) if buttons_list else None
        )
        
        if not ad_id:
            await update.message.reply_text("❌ Failed to create ad")
            return
        
        # Build message with buttons
        keyboard = []
        if buttons_list:
            for btn in buttons_list:
                keyboard.append([InlineKeyboardButton(btn['label'], url=btn['url'])])
        
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        
        # Post the ad
        try:
            if reply_markup:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"📢 *Advertisement*\n\n{message}",
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"📢 *Advertisement*\n\n{message}",
                    parse_mode='Markdown'
                )
            
            # Mark as posted
            bot_instance.update_ad_status(ad_id, 'posted')
            await update.message.reply_text(f"✅ Ad #{ad_id} posted successfully to chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Failed to post ad: {e}")
            await update.message.reply_text(f"❌ Failed to post: {str(e)}")
    
    except ValueError:
        await update.message.reply_text("❌ Invalid chat ID. Must be a number.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Admin command for posting image ads
async def postimage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Post an image ad with caption to a specific chat (admin only)"""
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /postimage <chat_id> <image_url> <caption>\n\n"
            "Example: /postimage -1001234567890 https://example.com/image.jpg Check out our signals!\n"
            "Get chat ID from /subscriptions command"
        )
        return
    
    try:
        chat_id = int(context.args[0])
        image_url = context.args[1]
        caption = " ".join(context.args[2:]) if len(context.args) > 2 else ""
        
        # Store ad in database
        ad_id = bot_instance.add_ad(
            admin_id=update.message.from_user.id,
            chat_id=chat_id,
            content=caption,
            content_type='image',
            image_url=image_url
        )
        
        if not ad_id:
            await update.message.reply_text("❌ Failed to create ad")
            return
        
        # Post the image
        try:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=image_url,
                caption=f"📢 *Advertisement*\n\n{caption}" if caption else "📢 *Advertisement*",
                parse_mode='Markdown'
            )
            
            # Mark as posted
            bot_instance.update_ad_status(ad_id, 'posted')
            await update.message.reply_text(f"✅ Image ad #{ad_id} posted successfully to chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Failed to post image ad: {e}")
            await update.message.reply_text(f"❌ Failed to post: {str(e)}")
    
    except ValueError:
        await update.message.reply_text("❌ Invalid chat ID. Must be a number.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Admin command for scheduling posts
async def schedulepost_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Schedule an ad post for later (admin only)"""
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    if len(context.args) < 3:
        await update.message.reply_text(
            "Usage: /schedulepost <chat_id> <datetime> <message>\n\n"
            "Example: /schedulepost -1001234567890 '2026-04-05 14:30' Great trading opportunity!\n"
            "Format: YYYY-MM-DD HH:MM (UTC)\n"
            "Get chat ID from /subscriptions command"
        )
        return
    
    try:
        chat_id = int(context.args[0])
        scheduled_time = context.args[1].strip("'\"")  # Remove optional quotes
        message = " ".join(context.args[2:])
        
        if not message:
            await update.message.reply_text("❌ Message cannot be empty")
            return
        
        # Validate datetime format
        try:
            datetime.strptime(scheduled_time, '%Y-%m-%d %H:%M')
        except ValueError:
            await update.message.reply_text("❌ Invalid datetime. Use format: YYYY-MM-DD HH:MM")
            return
        
        # Store scheduled ad in database
        ad_id = bot_instance.add_ad(
            admin_id=update.message.from_user.id,
            chat_id=chat_id,
            content=message,
            content_type='text',
            scheduled_time=scheduled_time
        )
        
        if not ad_id:
            await update.message.reply_text("❌ Failed to create scheduled ad")
            return
        
        await update.message.reply_text(f"✅ Ad #{ad_id} scheduled for {scheduled_time} UTC in chat {chat_id}")
        
    except ValueError as e:
        await update.message.reply_text("❌ Invalid chat ID. Must be a number.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Admin command for viewing ads
async def viewads_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View posted ads and history (admin only)"""
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    ads = bot_instance.get_all_ads(limit=20)
    
    if not ads:
        await update.message.reply_text("📭 No ads in history")
        return
    
    message = "📊 *Recent Ads (Last 20)*\n\n"
    
    for ad in ads:
        status_emoji = "✅" if ad['status'] == 'posted' else ("⏰" if ad['status'] == 'scheduled' else "⏳")
        created = ad['created_at'][:10] if ad['created_at'] else "Unknown"
        
        content_preview = ad['content'][:50]
        if len(ad['content']) > 50:
            content_preview += "..."
        
        message += f"{status_emoji} *Ad #{ad['ad_id']}* (Status: {ad['status']})\n"
        message += f"Content: _{content_preview}_\n"
        message += f"Chat: `{ad['chat_id']}`\n"
        message += f"Created: {created}\n\n"
    
    await update.message.reply_text(message, parse_mode='Markdown')

# Signal handler
def signal_handler(signum, frame):
    global application
    logger.info(f"Shutdown signal {signum}")
    
    if application:
        application.stop_running()
    
    sys.exit(0)

# Main application
def main():
    global application
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting Market Data Bot")
        
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Commands
        application.add_handler(CommandHandler('start', start_command))
        application.add_handler(CommandHandler('setwelcome', setwelcome_command))
        application.add_handler(CommandHandler('sendnews', sendnews_command))
        application.add_handler(CommandHandler('sendanalysis', sendanalysis_command))
        application.add_handler(CommandHandler('subscriptions', subscriptions_command))
        
        # Admin ad commands
        application.add_handler(CommandHandler('postmessage', postmessage_command))
        application.add_handler(CommandHandler('postimage', postimage_command))
        application.add_handler(CommandHandler('schedulepost', schedulepost_command))
        application.add_handler(CommandHandler('viewads', viewads_command))
        
        # Callbacks
        application.add_handler(CallbackQueryHandler(market_callback))
        
        # Welcome handler
        application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_handler))        # Setup scheduler for hourly broadcasts using PTB job queue
        job_queue = application.job_queue
        if job_queue:
            now = datetime.now(pytz.UTC)

            next_news = now.replace(minute=0, second=0, microsecond=0)
            if next_news <= now:
                next_news += timedelta(hours=1)

            next_analysis = now.replace(minute=30, second=0, microsecond=0)
            if next_analysis <= now:
                next_analysis += timedelta(hours=1)

            job_queue.run_repeating(
                broadcast_news,
                interval=3600,
                first=next_news,
                name='hourly_news'
            )
            job_queue.run_repeating(
                broadcast_analysis,
                interval=3600,
                first=next_analysis,
                name='hourly_analysis'
            )
            logger.info("Hourly news/analysis jobs scheduled via PTB job queue")
        else:
            logger.warning("Job queue unavailable - hourly broadcasts are disabled")
        print("\n✅ Market Data Bot Live!")
        print("📰 News broadcasts every hour at :00")
        print("📈 Analysis broadcasts every hour at :30")
        print("\nUser Commands:")
        print("  /start - View markets")
        print("\nAdmin Commands:")
        print("  /sendnews - Broadcast news now")
        print("  /sendanalysis - Broadcast analysis now")
        print("  /subscriptions - View subscribed chats")
        print("  /setwelcome - Set welcome message")
        print("\nAdmin Ad Commands:")
        print("  /postmessage <chat_id> <message> - Post text ad")
        print("  /postimage <chat_id> <url> <caption> - Post image ad")
        print("  /schedulepost <chat_id> <time> <message> - Schedule ad for later")
        print("  /viewads - View ad history")
        
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            timeout=10,
            bootstrap_retries=5
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if application:
            application.stop_running()

if __name__ == '__main__':
    main()


