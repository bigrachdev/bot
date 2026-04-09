"""
Main bot initialization and event loop
Posts finance news to a single target channel on a schedule.
"""
import asyncio
import signal
from telegram.ext import Application
from telegram.error import Conflict

from utils.logger import setup_logging
from config.settings import (
    ANALYSIS_INTERVAL_MINUTES,
    BOT_TOKEN,
    KEEP_ALIVE,
    NEWS_INTERVAL_MINUTES,
    TARGET_CHANNEL_ID,
)
from database.db import MarketBot
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler
from utils.keep_alive import start_keep_alive, ping_server


logger = setup_logging()

bot_instance = None
stop_event: asyncio.Event = None

NEWS_INTERVAL_SECONDS = NEWS_INTERVAL_MINUTES * 60
ANALYSIS_INTERVAL_SECONDS = ANALYSIS_INTERVAL_MINUTES * 60
QUICK_RETRY_SECONDS = 10 * 60  # 10 min retry when no news available


class NoContentAvailable(Exception):
    """Raised when all news sources are empty and bot has nothing to post."""
    pass


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
        except NoContentAvailable:
            # All sources empty - recheck in 10 minutes instead of full interval.
            logger.warning(f"{name}: no content available, quick retry in {QUICK_RETRY_SECONDS}s")
            next_run = loop.time() + QUICK_RETRY_SECONDS
        except Exception as e:
            logger.error(f"{name}: execution failed: {e}")
            # Retry sooner after failure rather than waiting full interval.
            next_run = loop.time() + min(300, interval_seconds)

    logger.info(f"{name} loop stopped")


async def startup_post(bot_instance, max_wait_seconds: int = 60, poll_interval_seconds: int = 10):
    """Wait for target channel to be configured, then do immediate first post."""
    elapsed = 0
    while elapsed <= max_wait_seconds:
        if TARGET_CHANNEL_ID:
            try:
                chat_id = int(TARGET_CHANNEL_ID)
                chat_list = [(chat_id, 'channel')]
                logger.info(f"Target channel {chat_id} configured. Running startup posts.")

                # Post online confirmation immediately.
                from datetime import datetime, timezone
                from services.posting import PostingService
                try:
                    health_msg = (
                        f"<b>Market Bot Online</b>\n"
                        f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
                        f"News interval: {NEWS_INTERVAL_MINUTES}m | Analysis interval: {ANALYSIS_INTERVAL_MINUTES}m\n\n"
                        f"<i>{PostingService.DISCLAIMER}</i>"
                    )
                    await bot_instance.bot.send_message(
                        chat_id=chat_id,
                        text=health_msg,
                        parse_mode="HTML",
                    )
                    logger.info("Startup health message posted.")
                except Exception as e:
                    logger.error(f"Failed to post startup health message: {e}")

                try:
                    await NewsScheduler.broadcast_news(bot_instance, chat_list=chat_list)
                except Exception as e:
                    logger.error(f"Startup news post failed: {e}")

                try:
                    await AnalysisScheduler.broadcast_analysis(bot_instance, chat_list=chat_list)
                except Exception as e:
                    logger.error(f"Startup analysis post failed: {e}")

                return
            except ValueError:
                logger.error(f"Invalid TARGET_CHANNEL_ID: {TARGET_CHANNEL_ID}")
                return

        logger.info("Waiting for TARGET_CHANNEL_ID configuration...")
        await asyncio.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    logger.warning("Startup post timed out. Periodic loops remain active.")


async def setup_bot():
    """Initialize bot and database."""
    global bot_instance

    try:
        logger.info("Initializing Market Bot...")

        bot_instance = MarketBot()
        logger.info("Database initialized")

        application = Application.builder().token(BOT_TOKEN).build()

        application.bot_data["bot_instance"] = bot_instance
        application.bot_data["bot"] = application.bot
        bot_instance.bot = application.bot

        if TARGET_CHANNEL_ID:
            logger.info(f"Target channel: {TARGET_CHANNEL_ID}")
        else:
            logger.warning("TARGET_CHANNEL_ID not set - bot will not post anywhere")

        logger.info("News broadcast interval: %d minutes", NEWS_INTERVAL_MINUTES)
        logger.info("Analysis broadcast interval: %d minutes", ANALYSIS_INTERVAL_MINUTES)

        return application

    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        raise


async def _ensure_periodic_jobs_alive():
    """Monitor and restart periodic jobs if they crash."""
    logger.info("Starting periodic job monitor...")
    
    async def monitor_and_restart(name: str, interval: int, job_coro, task_list: list, index: int):
        """Keep a periodic job alive by restarting if it crashes."""
        while not stop_event.is_set():
            try:
                logger.info(f"Starting {name} job...")
                await _run_periodic_job(name, interval, job_coro)
            except Exception as e:
                logger.error(f"❌ {name} job crashed: {e}", exc_info=True)
                if not stop_event.is_set():
                    logger.info(f"Restarting {name} job in 5 seconds...")
                    await asyncio.sleep(5)
            
            if stop_event.is_set():
                break
    
    # Keep track of monitor tasks
    monitor_tasks = [
        asyncio.create_task(
            monitor_and_restart("news_broadcast", NEWS_INTERVAL_SECONDS, NewsScheduler.broadcast_news, [], 0)
        ),
        asyncio.create_task(
            monitor_and_restart("analysis_broadcast", ANALYSIS_INTERVAL_SECONDS, AnalysisScheduler.broadcast_analysis, [], 1)
        ),
    ]
    
    return monitor_tasks


async def main():
    """Main async entry point."""
    if KEEP_ALIVE:
        start_keep_alive()
        # Ping every 4 minutes (240s) to stay well below Render's 15-min spin-down threshold
        ping_task = asyncio.create_task(ping_server(port=8080, interval_seconds=240))
        logger.info("Keep-alive HTTP pings scheduled (every 4 minutes)")

    application = await setup_bot()

    logger.info("Starting bot polling...")

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

        # Fire immediate startup posts.
        warmup_task = asyncio.create_task(startup_post(bot_instance))

        def _log_warmup_outcome(task: asyncio.Task):
            if task.cancelled():
                logger.warning("Startup post task was cancelled")
                return
            err = task.exception()
            if err:
                logger.error(f"Startup post task failed: {err}")

        warmup_task.add_done_callback(_log_warmup_outcome)

        # Start resilient periodic jobs that auto-restart if they crash
        periodic_tasks = await _ensure_periodic_jobs_alive()

        await stop_event.wait()

    except asyncio.CancelledError:
        logger.info("Bot stopped")
    finally:
        for task in periodic_tasks:
            if not task.done():
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
