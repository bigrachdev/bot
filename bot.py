"""
Main bot initialization and event loop
"""
import asyncio
import signal
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

# Global bot instance and stop event
bot_instance = None
stop_event: asyncio.Event = None

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

        # Start the scheduler
        scheduler.start()
        logger.info("✅ Scheduler started")

        # Store scheduler in application context for cleanup
        application.bot_data['scheduler'] = scheduler

        logger.info("✅ Schedulers configured")
        logger.info("📊 News broadcast: Every hour at :00")
        logger.info("📈 Analysis broadcast: Every hour at :30")

        return application
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize bot: {e}")
        raise

async def main():
    """Main async entry point"""
    # Start keep-alive server if enabled
    if KEEP_ALIVE:
        start_keep_alive()
        asyncio.create_task(ping_server())
        logger.info("✅ Keep-alive pings scheduled")

    application = await setup_bot()

    logger.info("🌐 Starting bot polling...")
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            poll_interval=1.0,
            allowed_updates=['message', 'callback_query'],
            drop_pending_updates=True
        )

        # Keep running until signalled to stop
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("⏹️ Bot stopped")
    finally:
        # Shutdown scheduler
        scheduler = application.bot_data.get('scheduler')
        if scheduler:
            scheduler.shutdown()
            logger.info("✅ Scheduler shut down")
        
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("✅ Bot shutdown complete")

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        loop.run_until_complete(main())
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        loop.close()
