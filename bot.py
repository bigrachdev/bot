"""
Main bot initialization and event loop
"""
import asyncio
import signal
from datetime import datetime, timedelta, timezone
from telegram.ext import Application
from telegram.error import Conflict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.triggers.interval import IntervalTrigger
from utils.logger import setup_logging
from config.settings import BOT_TOKEN, YOUR_ADMIN_ID, KEEP_ALIVE
from database.db import MarketBot
from handlers.user_commands import setup_user_handlers
from handlers.admin_commands import setup_admin_handlers
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler
from schedulers.ad_scheduler import AdScheduler
from utils.keep_alive import start_keep_alive, ping_server

# Initialize logger
logger = setup_logging()

# Global bot instance and stop event
bot_instance = None
stop_event: asyncio.Event = None


def _scheduler_listener(event):
    """Log APScheduler job outcomes for observability."""
    if event.code == EVENT_JOB_EXECUTED:
        logger.info(f"Scheduled job executed successfully: {event.job_id}")
    elif event.code == EVENT_JOB_MISSED:
        logger.warning(f"Scheduled job MISSED: {event.job_id}")
    elif event.code == EVENT_JOB_ERROR:
        logger.error(f"Scheduled job failed: {event.job_id}", exc_info=event.exception)

async def startup_auto_broadcast(
    bot_instance,
    max_wait_seconds: int = 600,
    poll_interval_seconds: int = 20,
):
    """Wait for at least one target chat/channel, then auto-broadcast without admin commands."""
    elapsed = 0
    while elapsed <= max_wait_seconds:
        chat_list = bot_instance.get_subscribed_chats()
        if chat_list:
            logger.info(
                f"Detected {len(chat_list)} target chat(s)/channel(s). Running startup broadcasts."
            )
            try:
                await NewsScheduler.broadcast_news(bot_instance, chat_list=chat_list)
            except Exception as e:
                logger.error(f"Startup news broadcast failed: {e}")
            try:
                await AdScheduler.broadcast_omnex_ad(bot_instance, chat_list=chat_list)
            except Exception as e:
                logger.error(f"Startup ad broadcast failed: {e}")

            return

        logger.info("No targets detected yet. Waiting before auto-broadcast retry...")
        await asyncio.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    logger.warning(
        "Startup auto-broadcast timed out without detected targets. "
        "Scheduler jobs remain active and will continue retrying."
    )

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

        target_chats = bot_instance.get_subscribed_chats()
        logger.info(f"Initial broadcast targets loaded: {len(target_chats)} chats/channels")
        
        # Setup handlers
        setup_user_handlers(application)
        setup_admin_handlers(application)
        logger.info("✅ Command handlers registered")
        
        # Setup scheduler for broadcasts
        scheduler = AsyncIOScheduler(
            timezone="UTC",
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 3600,  # Tolerate delayed wakeups/restarts.
            },
        )
        scheduler.add_listener(
            _scheduler_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_MISSED | EVENT_JOB_ERROR,
        )

        now_utc = datetime.now(timezone.utc)

        # Schedule news broadcast every 15 minutes.
        scheduler.add_job(
            NewsScheduler.broadcast_news,
            IntervalTrigger(minutes=15, start_date=now_utc + timedelta(minutes=15)),
            args=[bot_instance],
            id='news_broadcast',
            name='Broadcast news every 15 minutes'
        )

        # Schedule analysis broadcast every hour.
        scheduler.add_job(
            AnalysisScheduler.broadcast_analysis,
            IntervalTrigger(hours=1, start_date=now_utc + timedelta(hours=1)),
            args=[bot_instance],
            id='analysis_broadcast',
            name='Broadcast analysis every hour'
        )

        # Schedule Omnex ad broadcast every 4 hours.
        scheduler.add_job(
            AdScheduler.broadcast_omnex_ad,
            IntervalTrigger(hours=4, start_date=now_utc + timedelta(hours=4)),
            args=[bot_instance],
            id='omnex_ad_broadcast',
            name='Broadcast Omnex ad every 4 hours'
        )

        # Start the scheduler
        scheduler.start()
        logger.info("✅ Scheduler started")

        # Store scheduler in application context for cleanup
        application.bot_data['scheduler'] = scheduler

        logger.info("Schedulers configured")
        logger.info("News broadcast: Every 15 minutes")
        logger.info("Analysis broadcast: Every hour")
        logger.info("Omnex ad broadcast: Every 4 hours")
        for job in scheduler.get_jobs():
            logger.info(f"Scheduled job registered: id={job.id}, next_run={job.next_run_time}")

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
        def on_polling_error(exc: Exception):
            """Stop this instance when Telegram reports concurrent polling."""
            if isinstance(exc, Conflict):
                logger.error(
                    "Polling conflict detected (another getUpdates consumer is active). "
                    "Stopping this instance."
                )
                if stop_event and not stop_event.is_set():
                    stop_event.set()

        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            poll_interval=1.0,
            allowed_updates=['message', 'callback_query'],
            drop_pending_updates=True,
            error_callback=on_polling_error,
        )

        # Auto-start broadcasting as soon as targets are detected.
        warmup_task = asyncio.create_task(startup_auto_broadcast(bot_instance))
        def _log_warmup_outcome(task: asyncio.Task):
            if task.cancelled():
                logger.warning("Startup auto-broadcast task was cancelled")
                return
            err = task.exception()
            if err:
                logger.error(f"Startup auto-broadcast task failed: {err}")
        warmup_task.add_done_callback(_log_warmup_outcome)

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
        
        if application.updater:
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


