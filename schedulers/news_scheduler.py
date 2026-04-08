"""
News broadcasting scheduler
"""
import asyncio

from config.settings import (
    POST_MAX_NEWS_PER_CYCLE,
    POST_MIN_SECONDS_BETWEEN_MESSAGES,
    SEND_BRIEFING_INTRO,
)
from services.news import NewsService
from services.posting import PostingService
from utils.logger import logger


class NewsScheduler:
    """Handle scheduled news broadcasting."""

    @staticmethod
    def _select_cycle_articles(bot_instance, articles: list) -> list:
        """Pick fresh uncached articles and cap output size for consistent pacing."""
        fresh_articles = []
        for article in articles:
            news_id = NewsService.make_news_id(article)
            if not bot_instance.is_news_cached(news_id):
                fresh_articles.append(article)

        if not fresh_articles:
            return []

        return fresh_articles[:POST_MAX_NEWS_PER_CYCLE]

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
            disable_web_page_preview=True,
        )

    @staticmethod
    async def broadcast_news(bot_instance, chat_list: list = None):
        """Broadcast one professional briefing cycle to subscribed chats."""
        try:
            logger.info("Starting professional news briefing cycle...")

            articles = await NewsService.fetch_all_news()
            if not articles:
                logger.warning("No articles fetched for broadcast")
                return

            if chat_list is None:
                chat_list = bot_instance.get_subscribed_chats()

            if not chat_list:
                logger.info("No subscribed chats for news broadcast")
                return

            selected_articles = NewsScheduler._select_cycle_articles(bot_instance, articles)
            if not selected_articles:
                logger.info("No fresh news selected for this cycle after cache checks")
                return

            intro = PostingService.format_news_briefing_intro(selected_articles)

            successful = 0
            failed = 0

            for chat_id, _chat_type in chat_list:
                if SEND_BRIEFING_INTRO:
                    try:
                        await bot_instance.bot.send_message(
                            chat_id=chat_id,
                            text=intro,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                        await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)
                    except Exception as e:
                        logger.error(f"Failed to send briefing intro to chat {chat_id}: {e}")

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
                            chat_id=chat_id,
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
                        logger.error(f"Failed to send article from {source_name} to chat {chat_id}: {e}")
                        failed += 1

                    await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)

            logger.info(
                "News broadcast complete: %d successful, %d failed, %d stories/cycle",
                successful,
                failed,
                len(selected_articles),
            )

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
