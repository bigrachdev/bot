"""
Main bot initialization and event loop
"""
import asyncio
import sys
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from utils.logger import setup_logging
from config.settings import BOT_TOKEN, YOUR_ADMIN_ID, TOP_STOCKS, TOP_FOREX, KEEP_ALIVE
from database.db import MarketBot
from handlers.user_commands import setup_user_handlers
from handlers.admin_commands import setup_admin_handlers
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler
from utils.keep_alive import start_keep_alive, ping_server

# Initialize logger
logger = setup_logging()

# Global bot instance
bot_instance = None

async def setup_bot():
    """Initialize bot, database, and handlers"""
    global bot_instance
    
    try:
        logger.info("🚀 Initializing Market Bot...")
        
        # Initialize database
        bot_instance = MarketBot(YOUR_ADMIN_ID)
        logger.info("✅ Database initialized")
        
        # Create Telegram application
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Store bot_instance in application context for access in handlers
        application.bot_data['bot_instance'] = bot_instance
        application.bot_data['bot'] = application.bot
        bot_instance.bot = application.bot
        
        # Setup handlers
        setup_user_handlers(application)
        setup_admin_handlers(application)
        logger.info("✅ Command handlers registered")
        
        # Setup scheduler for broadcasts
        scheduler = AsyncIOScheduler(timezone="UTC")
        
        # Schedule news broadcast at :00 every hour
        scheduler.add_job(
            NewsScheduler.broadcast_news,
            CronTrigger(minute=0),
            args=[bot_instance],
            id='news_broadcast',
            name='Broadcast news hourly'
        )
        
        # Schedule analysis broadcast at :30 every hour
        scheduler.add_job(
            AnalysisScheduler.broadcast_analysis,
            CronTrigger(minute=30),
            args=[bot_instance],
            id='analysis_broadcast',
            name='Broadcast analysis every 30 minutes'
        )
        
        application.job_queue.scheduler = scheduler
        
        logger.info("✅ Schedulers configured")
        logger.info("📊 News broadcast: Every hour at :00")
        logger.info("📈 Analysis broadcast: Every hour at :30")
        
        return application
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize bot: {e}")
        raise

async def main():
    """Main async entry point"""
    try:
        # On Windows, use the ProactorEventLoop to avoid issues with socket operations
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Start keep-alive server if enabled
        if KEEP_ALIVE:
            start_keep_alive()
            asyncio.create_task(ping_server())
            logger.info("✅ Keep-alive pings scheduled")

        application = await setup_bot()

        logger.info("🌐 Starting bot polling...")
        await application.run_polling(
            poll_interval=1.0,
            allowed_updates=['message', 'callback_query'],
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⏹️ Bot shutdown complete")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
