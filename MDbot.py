import sqlite3
import requests
import os
import signal
import sys
from datetime import datetime, timedelta
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import logging
import asyncio
from dotenv import load_dotenv
import nest_asyncio
from typing import List, Tuple, Optional
from cachetools import TTLCache, cached
from tenacity import retry, stop_after_attempt, wait_fixed

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
scheduler = None

# Production configuration
BOT_TOKEN = os.getenv('BOT_TOKEN')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
YOUR_ADMIN_ID = int(os.getenv('YOUR_ADMIN_ID', '0'))
CHANNEL_ID = os.getenv('CHANNEL_ID', '')
WEBHOOK_URL = os.getenv('WEBHOOK_URL', None)  # For webhook mode
PORT = int(os.getenv('PORT', 8443))

# Validate critical configuration
if not all([BOT_TOKEN, CHANNEL_ID]):
    logger.error("Missing critical configuration: BOT_TOKEN or CHANNEL_ID")
    sys.exit(1)

if YOUR_ADMIN_ID == 0:
    logger.warning("ADMIN_ID not set - admin commands will be disabled")

# Fix channel ID format if needed
if CHANNEL_ID.startswith('100') and len(CHANNEL_ID) > 10:
    CHANNEL_ID = f"-{CHANNEL_ID}"
    logger.info(f"Auto-corrected CHANNEL_ID to: {CHANNEL_ID}")

DB_NAME = 'market_posts.db'

class MarketNewsBot:
    cache = TTLCache(maxsize=100, ttl=300)
    def __init__(self):
        self.db_name = DB_NAME
        self.channel_id = CHANNEL_ID
        self.admin_id = YOUR_ADMIN_ID
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5 min cache for API calls
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with proper indexing and settings table"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            
            # Posts table
            c.execute('''CREATE TABLE IF NOT EXISTS posts
                         (message_id INTEGER, channel_id TEXT, post_time TEXT,
                          type TEXT DEFAULT 'auto', PRIMARY KEY (message_id))''')
            
            # Settings table for customizable messages
            c.execute('''CREATE TABLE IF NOT EXISTS settings
                         (key TEXT PRIMARY KEY, value TEXT)''')
            
            # Default welcome message
            c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                      ('welcome_message', 'Welcome to the group, {username}! Stay tuned for market updates. 📈'))
            
            # Add index for faster cleanup
            c.execute('CREATE INDEX IF NOT EXISTS idx_posts_time ON posts(post_time)')
            
            # Cleanup old posts on startup (older than 48 hours to be safe)
            threshold = (datetime.utcnow() - timedelta(hours=48)).isoformat()
            c.execute("DELETE FROM posts WHERE post_time < ?", (threshold,))
            deleted = c.rowcount
            conn.commit()
            conn.close()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old posts on startup")
            else:
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

    def add_post(self, message_id, post_type='auto'):
        """Store post details for cleanup"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            post_time = datetime.utcnow().isoformat()
            c.execute("INSERT OR REPLACE INTO posts VALUES (?, ?, ?, ?)", 
                     (message_id, self.channel_id, post_time, post_type))
            conn.commit()
            conn.close()
            logger.debug(f"Post {message_id} ({post_type}) added to database")
        except Exception as e:
            logger.error(f"Failed to add post to database: {e}")

    def get_posts_to_delete(self):
        """Get posts older than 24 hours for cleanup"""
        try:
            conn = sqlite3.connect(self.db_name, check_same_thread=False)
            c = conn.cursor()
            threshold = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            c.execute("SELECT message_id FROM posts WHERE post_time < ?", (threshold,))
            to_delete = [row[0] for row in c.fetchall()]
            
            if to_delete:
                c.execute("DELETE FROM posts WHERE post_time < ?", (threshold,))
                conn.commit()
                logger.info(f"Scheduled cleanup of {len(to_delete)} old posts")
            
            conn.close()
            return to_delete
        except Exception as e:
            logger.error(f"Failed to get posts to delete: {e}")
            return []

    @cached(cache)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def fetch_crypto_news(self) -> Tuple[str, str]:
        """Fetch top 5 crypto prices and changes from CoinGecko with simplified formatting"""
        url = 'https://api.coingecko.com/api/v3/coins/markets'
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 5,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            headline = "🚀 Top Crypto"
            summary_lines = []
            
            for coin in data:
                name = coin['name']
                price = coin['current_price']
                change = coin.get('price_change_percentage_24h', 0)
                arrow = "↑" if change > 0 else "↓"
                summary_lines.append(f"*{name}*: ${price:,.2f} {arrow} {change:+.1f}%")
            
            summary = "\n\n".join(summary_lines)  # Added double spacing
            return headline, summary
            
        except Exception as e:
            logger.error(f"Crypto fetch failed: {e}")
            return "🚀 Top Crypto", "⚠️ Data unavailable"

    @cached(cache)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def fetch_stock_news(self) -> Tuple[str, str]:
        """Fetch major stock quotes with simplified layout"""
        if not ALPHA_VANTAGE_KEY or ALPHA_VANTAGE_KEY == 'demo':
            return "💹 Top Stocks", "🔑 API key required"
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        summaries = []
        successful_fetches = 0
        
        for symbol in symbols:
            if successful_fetches >= 4:
                break
                
            url = f'https://www.alphavantage.co/query'
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': ALPHA_VANTAGE_KEY
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'Global Quote' in data and data['Global Quote']:
                    quote = data['Global Quote']
                    price = float(quote['05. price'])
                    change_percent = quote['10. change percent']
                    arrow = "↑" if '+' in change_percent else "↓"
                    summaries.append(f"*{symbol}*: ${price:.2f} {arrow} {change_percent}")
                    successful_fetches += 1
                else:
                    summaries.append(f"*{symbol}*: Unavailable")
                    
            except Exception as e:
                logger.warning(f"Stock {symbol} fetch failed: {e}")
                summaries.append(f"*{symbol}*: Error")
        
        headline = "💹 Top Stocks"
        summary = "\n\n".join(summaries)  # Added double spacing
        return headline, summary

    @cached(cache)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def fetch_economic_news(self) -> Tuple[str, str]:
        """Fetch expanded economic headlines from NewsAPI - more market-influencing news"""
        if not NEWSAPI_KEY:
            return "📰 Market News", "🔑 API key required"
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': 'economy OR finance OR market OR fed OR inflation OR rates OR recession OR gdp OR unemployment OR stocks OR crypto OR central bank OR trade OR tariffs',
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 6,  # Increased for more news
            'apiKey': NEWSAPI_KEY
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            
            if not articles:
                return "📰 Market News", "No recent headlines"
            
            headline = "📰 Key Market News"
            summary_lines = []
            
            for i, article in enumerate(articles[:5], 1):  # Up to 5 for quality
                title = article.get('title', 'No title')[:80] + '...' if len(article.get('title', '')) > 80 else article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]
                url = article.get('url', '')
                
                summary_lines.append(f"{i}. *{title}* ({source}, {published}) [Read]({url})")
            
            summary = "\n\n".join(summary_lines)  # Added double spacing
            return headline, summary
            
        except Exception as e:
            logger.error(f"Economic news fetch failed: {e}")
            return "📰 Market News", "⚠️ Unavailable"

    def generate_post(self, include_footer=True) -> Tuple[str, str]:
        """Generate simplified market update post with quality info"""
        try:
            crypto_h, crypto_s = self.fetch_crypto_news()
            stock_h, stock_s = self.fetch_stock_news()
            econ_h, econ_s = self.fetch_economic_news()
            
            current_time = datetime.utcnow().strftime('%H:%M UTC')
            update_type = "📊 Update" 
            
            post_text = f"📈 *Market Update* ({current_time})\n\n{update_type}\n\n{crypto_h}\n\n{crypto_s}\n\n{stock_h}\n\n{stock_s}\n\n{econ_h}\n\n{econ_s}"  # Added extra spacing
            
            if include_footer:
                post_text += f"\n\n*Data as of {current_time} UTC*"
            
            return post_text, 'auto'
            
        except Exception as e:
            logger.error(f"Post generation failed: {e}")
            current_time = datetime.utcnow().strftime('%H:%M UTC')
            error_text = f"📈 *Market Update* ({current_time})\n\n⚠️ Error fetching data\n\n*Data as of {current_time} UTC*"
            return error_text, 'error'

    def get_next_update_time(self):
        """Calculate next scheduled update time"""
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Schedule: 0:00 (night), 8:00 (morning), 16:00 (afternoon)
        schedule_hours = [0, 8, 16]
        
        for hour in schedule_hours:
            if current_hour < hour:
                next_update = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                return next_update.strftime('%H:%M UTC')
        
        next_update = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return next_update.strftime('%H:%M UTC')

# Initialize bot instance
bot_instance = MarketNewsBot()

# Post sending function - no buttons in posts
async def send_update(bot_app, post_type='auto'):
    """Send market update to channel with cleanup - no buttons"""
    try:
        bot = bot_app.bot
        
        # Clean up old posts first
        to_delete = bot_instance.get_posts_to_delete()
        for msg_id in to_delete:
            try:
                await bot.delete_message(chat_id=bot_instance.channel_id, message_id=msg_id)
            except Exception as e:
                logger.debug(f"Could not delete message {msg_id}: {e}")
        
        # Generate post
        post_text, db_type = bot_instance.generate_post()
        
        message = await bot.send_message(
            chat_id=bot_instance.channel_id,
            text=post_text,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
        
        bot_instance.add_post(message.message_id, post_type)
        logger.info(f"Posted {post_type} update at {datetime.utcnow().strftime('%H:%M UTC')}")
        
    except Exception as e:
        logger.error(f"Failed to send update: {e}")

# Scheduler job wrapper
def scheduler_job(bot_app):
    """Sync wrapper for async scheduler job"""
    async def run():
        await send_update(bot_app, 'auto')
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())
        loop.close()
    except Exception as e:
        logger.error(f"Scheduler job failed: {e}")

# Schedule automated posts - 3x daily: 00:00, 08:00, 16:00 UTC
def schedule_posts(bot_app):
    global scheduler
    try:
        scheduler = BackgroundScheduler(timezone=pytz.UTC)
        
        scheduler.add_job(
            scheduler_job,
            CronTrigger(hour='0,8,16', minute=0, timezone=pytz.UTC),
            args=(bot_app,),
            id='market_update',
            replace_existing=True,
            misfire_grace_time=300
        )
        
        scheduler.start()
        logger.info("📅 Scheduler started - 3 daily updates at 00:00, 08:00, 16:00 UTC")
        
    except Exception as e:
        logger.error(f"Scheduler setup failed: {e}")
        raise

# Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    next_update = bot_instance.get_next_update_time()
    current_time = datetime.utcnow().strftime('%H:%M UTC')
    
    welcome_text = f"🤖 *Market Bot*\n\nStatus: Active\nChannel: `{bot_instance.channel_id}`\nUpdates: 3x daily\n\nCurrent: {current_time}\nNext: {next_update}"
    
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != bot_instance.admin_id:
        await update.message.reply_text("🔒 Admin only")
        return
    
    keyboard = [
        [InlineKeyboardButton("🔄 Refresh", callback_data='admin_refresh')],
        [InlineKeyboardButton("📊 Markets", callback_data='admin_markets')],
        [InlineKeyboardButton("📢 Broadcast", callback_data='admin_broadcast')],
        [InlineKeyboardButton("👋 Set Welcome", callback_data='admin_setwelcome')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text("🛠️ *Admin Panel*", parse_mode='Markdown', reply_markup=reply_markup)

# Callback handlers for admin menu
async def admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.from_user.id != bot_instance.admin_id:
        await query.answer("🔒 Admin only")
        return
    
    data = query.data
    
    if data == 'admin_refresh':
        await query.answer("🔄 Refreshing...")
        await send_update(context.application, 'manual')
        await query.edit_message_text("✅ Update refreshed")
    
    elif data == 'admin_markets':
        await query.answer("📊 Loading...")
        crypto_h, crypto_s = bot_instance.fetch_crypto_news()
        stock_h, stock_s = bot_instance.fetch_stock_news()
        stats_text = f"📊 *Quick Markets*\n\n{crypto_h}\n\n{crypto_s}\n\n{stock_h}\n\n{stock_s}"  # Added spacing
        await query.edit_message_text(stats_text, parse_mode='Markdown')
    
    elif data == 'admin_broadcast':
        await query.answer("📢 Send broadcast message now")
        context.user_data['admin_action'] = 'broadcast'
        await query.edit_message_text("📢 *Broadcast Mode*\nSend your message now (or /cancel)")
    
    elif data == 'admin_setwelcome':
        await query.answer("👋 Send new welcome message now")
        context.user_data['admin_action'] = 'setwelcome'
        await query.edit_message_text("👋 *Set Welcome Mode*\nSend new message now (use {username} for name, or /cancel)")

# Message handler for admin inputs
async def admin_input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != bot_instance.admin_id:
        return
    
    action = context.user_data.get('admin_action')
    if not action:
        return
    
    message_text = update.message.text.strip()
    
    if action == 'broadcast':
        try:
            await context.bot.send_message(
                chat_id=bot_instance.channel_id,
                text=message_text,
                parse_mode='Markdown'
            )
            await update.message.reply_text("✅ Broadcast sent")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed: {str(e)}")
    
    elif action == 'setwelcome':
        bot_instance.set_setting('welcome_message', message_text)
        await update.message.reply_text("✅ Welcome message updated")
    
    # Clear action
    context.user_data.pop('admin_action', None)

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != bot_instance.admin_id:
        return
    
    if 'admin_action' in context.user_data:
        context.user_data.pop('admin_action')
        await update.message.reply_text("❌ Action cancelled")

async def welcome_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome new members"""
    for member in update.message.new_chat_members:
        welcome_msg = bot_instance.get_setting('welcome_message', 'Welcome, {username}!')
        formatted_msg = welcome_msg.format(username=member.username or member.first_name)
        await update.message.reply_text(formatted_msg)

# Startup message
async def send_startup_message(bot_app):
    try:
        next_update = bot_instance.get_next_update_time()
        startup_text = f"🚀 *Market Bot Live*\n\nSystems online\nNext update: {next_update} UTC\n\nCoverage: Crypto, Stocks, Key News"
        
        await bot_app.bot.send_message(
            chat_id=bot_instance.channel_id,
            text=startup_text,
            parse_mode='Markdown'
        )
        logger.info("Startup message sent")
        
    except Exception as e:
        logger.error(f"Startup message failed: {e}")

# Signal handler
def signal_handler(signum, frame):
    global application, scheduler
    logger.info(f"Shutdown signal {signum}")
    
    if scheduler:
        scheduler.shutdown(wait=False)
    
    if application:
        application.stop_running()
    
    sys.exit(0)

# Main application
def main():
    global application, scheduler
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting Market Bot")
        
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Commands
        application.add_handler(CommandHandler('start', start_command))
        application.add_handler(CommandHandler('admin', admin_command))
        application.add_handler(CommandHandler('cancel', cancel_command))
        
        # Callbacks
        application.add_handler(CallbackQueryHandler(admin_callback, pattern='^admin_'))
        
        # Admin input handler (text messages)
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, admin_input_handler))
        
        # Welcome handler for new members
        application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_handler))
        
        schedule_posts(application)
        
        asyncio.run(send_startup_message(application))
        
        next_update = bot_instance.get_next_update_time()
        print(f"\n✅ Bot Live!\nChannel: {bot_instance.channel_id}\nNext update: {next_update} UTC\nUse /admin for menu")
        
        if WEBHOOK_URL:
            logger.info("Starting in webhook mode")
            application.run_webhook(
                listen='0.0.0.0',
                port=PORT,
                url_path=BOT_TOKEN.split(':')[1],
                webhook_url=WEBHOOK_URL
            )
        else:
            logger.info("Starting in polling mode")
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
        if scheduler:
            scheduler.shutdown(wait=False)
        if application:
            application.stop_running()

if __name__ == '__main__':
    main()