"""
News broadcasting scheduler
"""
import asyncio

from config.settings import (
    POST_MAX_NEWS_PER_CYCLE,
    POST_MIN_SECONDS_BETWEEN_MESSAGES,
    TARGET_CHANNEL_ID,
)
from services.news import NewsService
from services.posting import PostingService
from utils.logger import logger


class NewsScheduler:
    """Handle scheduled news broadcasting."""

    @staticmethod
    def _select_cycle_articles(bot_instance, articles: list) -> list:
        """Pick articles for this cycle - deduplication already handled in fetch_all_news."""
        # Deduplication is already done at the fetch level (URL + story signature).
        # We no longer block posting based on cache - just cap for pacing.
        return articles[:POST_MAX_NEWS_PER_CYCLE]

    @staticmethod
    async def _send_article_with_fallback(
        bot_instance,
        chat_id: int,
        caption: str,
        image_url: str,
        video_url: str,
    ):
        """Try rich media first, then fall back to plain text to avoid dropping posts."""
        if video_url and NewsService._is_supported_video_url(video_url):
            try:
                await bot_instance.bot.send_video(
                    chat_id=chat_id,
                    video=video_url,
                    caption=caption,
                    parse_mode="HTML",
                    supports_streaming=True,
                )
                return
            except Exception as e:
                logger.warning(f"Video send failed for chat {chat_id}, falling back to text: {e}")

        if image_url:
            try:
                await bot_instance.bot.send_photo(
                    chat_id=chat_id,
                    photo=image_url,
                    caption=caption,
                    parse_mode="HTML",
                )
                return
            except Exception as e:
                logger.warning(f"Photo send failed for chat {chat_id}, falling back to text: {e}")

        await bot_instance.bot.send_message(
            chat_id=chat_id,
            text=caption,
            parse_mode="HTML",
            disable_web_page_preview=False,
        )

    @staticmethod
    async def broadcast_news(bot_instance, chat_list: list = None):
        """Broadcast one professional briefing cycle to the target channel."""
        try:
            logger.info("🔍 Starting professional news briefing cycle...")

            if chat_list is None:
                if TARGET_CHANNEL_ID:
                    try:
                        chat_id = int(TARGET_CHANNEL_ID)
                        chat_list = [(chat_id, 'channel')]
                        logger.info(f"✓ Target channel resolved: {chat_id}")
                    except ValueError:
                        logger.error(f"Invalid TARGET_CHANNEL_ID: {TARGET_CHANNEL_ID}")
                        return
                else:
                    logger.warning("TARGET_CHANNEL_ID not set")
                    return

            logger.info("📰 Fetching news articles...")
            articles = await NewsService.fetch_all_news()
            logger.info(f"📊 Fetched {len(articles) if articles else 0} articles from news service")
            
            selected_articles = NewsScheduler._select_cycle_articles(bot_instance, articles) if articles else []
            logger.info(f"📌 Selected {len(selected_articles)} articles for this cycle")
            
            chat_id, _ = chat_list[0]

            if selected_articles:
                # Post news article cards directly.
                successful = 0
                for cid, _chat_type in chat_list:
                    for rank, article in enumerate(selected_articles, start=1):
                        caption = PostingService.format_news_article_card(
                            article=article,
                            rank=rank,
                            total=len(selected_articles),
                        )
                        image_url = (article.get("image_url") or "").strip()
                        video_url = (article.get("video_url") or "").strip()

                        try:
                            await NewsScheduler._send_article_with_fallback(
                                bot_instance=bot_instance,
                                chat_id=cid,
                                caption=caption,
                                image_url=image_url,
                                video_url=video_url,
                            )
                            successful += 1
                            news_id = NewsService.make_news_id(article)
                            bot_instance.cache_news(
                                news_id,
                                article.get("title", ""),
                                article.get("source", {}).get("name", "Unknown"),
                                article.get("url", ""),
                            )
                        except Exception as e:
                            source_name = article.get("source", {}).get("name", "Unknown")
                            logger.error(f"Failed to send article from {source_name} to chat {cid}: {e}")

                        await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)

                logger.info("News broadcast complete: %d/%d articles posted", successful, len(selected_articles))
            else:
                # No articles from any source - post market snapshot as fallback.
                logger.warning("Zero articles fetched; using market snapshot fallback.")
                try:
                    from services.analysis import AnalysisService
                    stock_prices = await AnalysisService.fetch_top_stock_prices()
                    snapshot_msg = PostingService.format_market_snapshot_message(stock_prices)
                    await bot_instance.bot.send_message(
                        chat_id=chat_id,
                        text=snapshot_msg,
                        parse_mode="HTML",
                    )
                    logger.info("Market snapshot fallback posted.")
                except Exception as fallback_err:
                    # Even the fallback failed - send a basic status message.
                    logger.error(f"Fallback market snapshot failed: {fallback_err}")
                    try:
                        from datetime import datetime, timezone
                        status_msg = (
                            "<b>Market Desk | Live Update</b>\n"
                            f"Published: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                            "Market data sources are temporarily unavailable.\n"
                            "The next update will include fresh headlines.\n\n"
                            "<i>Educational market update only. Not financial advice.</i>"
                        )
                        await bot_instance.bot.send_message(
                            chat_id=chat_id,
                            text=status_msg,
                            parse_mode="HTML",
                        )
                        logger.info("Status message posted as fallback.")
                    except Exception as last_resort_err:
                        logger.error(f"Last-resort status message also failed: {last_resort_err}")
                        # Signal to retry loop that nothing could be posted.
                        from bot import NoContentAvailable
                        raise NoContentAvailable("All posting attempts exhausted")

        except Exception as e:
            logger.error(f"News broadcast failed: {e}")

    @staticmethod
    async def format_news_with_stocks(article: dict) -> str:
        """Backward-compatible wrapper retained for older call paths."""
        return PostingService.format_news_article_card(article=article, rank=1, total=1)

    @staticmethod
    async def send_news_to_chat(bot_instance, chat_id: int):
        """Send current curated briefing to a specific chat."""
        try:
            logger.info(f"Fetching news for chat {chat_id}...")

            articles = await NewsService.fetch_all_news()
            if not articles:
                await bot_instance.bot.send_message(
                    chat_id=chat_id,
                    text="No market news available at the moment. The next update will include new headlines.",
                )
                return

            selected_articles = NewsScheduler._select_cycle_articles(bot_instance, articles)
            if not selected_articles:
                await bot_instance.bot.send_message(
                    chat_id=chat_id,
                    text="No fresh stories in this cycle. The next update will include new market headlines.",
                )
                return

            for rank, article in enumerate(selected_articles, start=1):
                caption = PostingService.format_news_article_card(
                    article=article,
                    rank=rank,
                    total=len(selected_articles),
                )
                image_url = (article.get("image_url") or "").strip()
                video_url = (article.get("video_url") or "").strip()

                await NewsScheduler._send_article_with_fallback(
                    bot_instance=bot_instance,
                    chat_id=chat_id,
                    caption=caption,
                    image_url=image_url,
                    video_url=video_url,
                )

                news_id = NewsService.make_news_id(article)
                bot_instance.cache_news(
                    news_id,
                    article.get("title", ""),
                    article.get("source", {}).get("name", "Unknown"),
                    article.get("url", ""),
                )
                await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)

            logger.info(f"News sent to chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to send news to {chat_id}: {e}")
            raise
