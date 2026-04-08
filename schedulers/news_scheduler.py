"""
News broadcasting scheduler
"""
import asyncio

from config.settings import (
    POST_MAX_NEWS_PER_CYCLE,
    POST_MIN_SECONDS_BETWEEN_MESSAGES,
    SEND_BRIEFING_INTRO,
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
            logger.info("Starting professional news briefing cycle...")

            if chat_list is None:
                if TARGET_CHANNEL_ID:
                    try:
                        chat_id = int(TARGET_CHANNEL_ID)
                        chat_list = [(chat_id, 'channel')]
                    except ValueError:
                        logger.error(f"Invalid TARGET_CHANNEL_ID: {TARGET_CHANNEL_ID}")
                        return
                else:
                    logger.warning("TARGET_CHANNEL_ID not set")
                    return

            if not chat_list:
                logger.info("No target channel configured for news broadcast")
                return

            articles = await NewsService.fetch_all_news()
            selected_articles = NewsScheduler._select_cycle_articles(bot_instance, articles) if articles else []

            chat_id, _ = chat_list[0]

            # If we have fresh news, broadcast it.
            if selected_articles:
                intro = PostingService.format_news_briefing_intro(selected_articles)

                successful = 0
                failed = 0

                for cid, _chat_type in chat_list:
                    if SEND_BRIEFING_INTRO:
                        try:
                            await bot_instance.bot.send_message(
                                chat_id=cid,
                                text=intro,
                                parse_mode="HTML",
                                disable_web_page_preview=True,
                            )
                            await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)
                        except Exception as e:
                            logger.error(f"Failed to send briefing intro to chat {cid}: {e}")

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
                            failed += 1

                        await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)

                logger.info(
                    "News broadcast complete: %d successful, %d failed, %d stories/cycle",
                    successful,
                    failed,
                    len(selected_articles),
                )
            else:
                # No fresh news - post market snapshot instead so the channel always gets content.
                logger.info("No fresh news available this cycle; posting market snapshot instead.")
                from services.analysis import AnalysisService

                stock_prices = await AnalysisService.fetch_top_stock_prices()
                snapshot_msg = PostingService.format_market_snapshot_message(stock_prices)

                try:
                    await bot_instance.bot.send_message(
                        chat_id=chat_id,
                        text=snapshot_msg,
                        parse_mode="HTML",
                    )
                    logger.info("Fallback market snapshot posted successfully")
                except Exception as e:
                    logger.error(f"Failed to send fallback market snapshot: {e}")

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
                    text="No high-quality stock or commodities stories found right now. Please try again later.",
                )
                return

            selected_articles = NewsScheduler._select_cycle_articles(bot_instance, articles)
            if not selected_articles:
                await bot_instance.bot.send_message(
                    chat_id=chat_id,
                    text="No fresh stories in this cycle. The next update will include new market headlines.",
                )
                return

            intro = PostingService.format_news_briefing_intro(selected_articles)

            if SEND_BRIEFING_INTRO:
                await bot_instance.bot.send_message(
                    chat_id=chat_id,
                    text=intro,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)

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
