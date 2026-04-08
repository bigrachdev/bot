"""
Main bot initialization and event loop
"""
import asyncio
import signal
from telegram.ext import Application
from telegram.error import Conflict

from utils.logger import setup_logging
from config.settings import BOT_TOKEN, YOUR_ADMIN_ID, KEEP_ALIVE
from database.db import MarketBot
from handlers.user_commands import setup_user_handlers
from handlers.admin_commands import setup_admin_handlers
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler
from schedulers.ad_scheduler import AdScheduler
from utils.keep_alive import start_keep_alive, ping_server


logger = setup_logging()

bot_instance = None
stop_event: asyncio.Event = None

NEWS_INTERVAL_SECONDS = 25 * 60  # 25 minutes - one headline from each source
ANALYSIS_INTERVAL_SECONDS = 60 * 60
AD_INTERVAL_SECONDS = 4 * 60 * 60


async def _run_periodic_job(name: str, interval_seconds: int, job_coro):
    """Run one job forever with self-healing retries."""
    logger.info(f"{name} loop started (interval={interval_seconds}s)")
    loop = asyncio.get_running_loop()
    next_run = loop.time() + interval_seconds

    while not stop_event.is_set():
        wait_seconds = max(0.0, next_run - loop.time())

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
            break
        except asyncio.TimeoutError:
            pass

        if stop_event.is_set():
            break

        try:
            logger.info(f"{name}: execution started")
            await job_coro(bot_instance)
            logger.info(f"{name}: execution completed")
            next_run = loop.time() + interval_seconds
        except Exception as e:
            logger.error(f"{name}: execution failed: {e}")
            # Retry sooner after failure rather than waiting full interval.
            next_run = loop.time() + min(300, interval_seconds)

    logger.info(f"{name} loop stopped")


async def startup_auto_broadcast(
    bot_instance,
    max_wait_seconds: int = 600,
    poll_interval_seconds: int = 20,
):
    """Wait for targets, then run immediate startup broadcasts."""
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
        "Periodic loops remain active and will continue retrying."
    )


async def setup_bot():
    """Initialize bot, database, and handlers."""
    global bot_instance

    try:
        logger.info("Initializing Market Bot...")

        bot_instance = MarketBot(YOUR_ADMIN_ID)
        logger.info("Database initialized")

        application = Application.builder().token(BOT_TOKEN).build()

        application.bot_data["bot_instance"] = bot_instance
        application.bot_data["bot"] = application.bot
        bot_instance.bot = application.bot

        target_chats = bot_instance.get_subscribed_chats()
        logger.info(f"Initial broadcast targets loaded: {len(target_chats)} chats/channels")

        setup_user_handlers(application)
        setup_admin_handlers(application)
        logger.info("Command handlers registered")

        logger.info("Periodic broadcasters configured")
        logger.info("News broadcast interval: 25 minutes (one headline per source)")
        logger.info("Analysis broadcast interval: 1 hour")
        logger.info("Omnex ad broadcast interval: 4 hours")

        return application

    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        raise


async def main():
    """Main async entry point."""
    if KEEP_ALIVE:
        start_keep_alive()
        asyncio.create_task(ping_server())
        logger.info("Keep-alive pings scheduled")

    application = await setup_bot()

    logger.info("Starting bot polling...")
    periodic_tasks = []

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
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
            error_callback=on_polling_error,
        )

        # Fire immediate startup posts when a target exists.
        warmup_task = asyncio.create_task(startup_auto_broadcast(bot_instance))

        def _log_warmup_outcome(task: asyncio.Task):
            if task.cancelled():
                logger.warning("Startup auto-broadcast task was cancelled")
                return
            err = task.exception()
            if err:
                logger.error(f"Startup auto-broadcast task failed: {err}")

        warmup_task.add_done_callback(_log_warmup_outcome)

        # Durable periodic loops (self-healing).
        periodic_tasks = [
            asyncio.create_task(
                _run_periodic_job("news_broadcast", NEWS_INTERVAL_SECONDS, NewsScheduler.broadcast_news)
            ),
            asyncio.create_task(
                _run_periodic_job("analysis_broadcast", ANALYSIS_INTERVAL_SECONDS, AnalysisScheduler.broadcast_analysis)
            ),
            asyncio.create_task(
                _run_periodic_job("omnex_ad_broadcast", AD_INTERVAL_SECONDS, AdScheduler.broadcast_omnex_ad)
            ),
        ]

        await stop_event.wait()

    except asyncio.CancelledError:
        logger.info("Bot stopped")
    finally:
        for task in periodic_tasks:
            task.cancel()
        if periodic_tasks:
            await asyncio.gather(*periodic_tasks, return_exceptions=True)

        if application.updater:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
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
        logger.error(f"Fatal error: {e}")
    finally:
        loop.close()
