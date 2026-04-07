"""
News broadcasting scheduler
"""
import asyncio
from utils.logger import logger
from services.news import NewsService


class NewsScheduler:
    """Handle scheduled news broadcasting"""

    @staticmethod
    async def broadcast_news(bot_instance, chat_list: list = None):
        """Broadcast latest curated news as separate detailed posts."""
        try:
            logger.info("Starting news broadcast...")

            # Fetch news
            articles = await NewsService.fetch_all_news()
            if not articles:
                logger.warning("No articles fetched for broadcast")
                return

            # Get subscribed chats if not provided
            if chat_list is None:
                chat_list = bot_instance.get_subscribed_chats()

            if not chat_list:
                logger.info("No subscribed chats for news broadcast")
                return

            fresh_articles = []
            for article in articles:
                news_id = NewsService.make_news_id(article)
                if not bot_instance.is_news_cached(news_id):
                    fresh_articles.append((news_id, article))

            if not fresh_articles:
                logger.info("All curated articles are already cached; skipping broadcast")
                return

            successful = 0
            failed = 0
            sent_per_article = {news_id: 0 for news_id, _ in fresh_articles}

            for chat_id, _chat_type in chat_list:
                intro = NewsService.format_broadcast_intro([article for _id, article in fresh_articles])
                try:
                    await bot_instance.bot.send_message(
                        chat_id=chat_id,
                        text=intro,
                        parse_mode='HTML',
                    )
                    successful += 1
                    await asyncio.sleep(0.4)
                except Exception as e:
                    logger.error(f"Failed to send briefing intro to chat {chat_id}: {e}")
                    failed += 1

                for news_id, article in fresh_articles:
                    caption = NewsService.format_article_caption(article)
                    image_url = (article.get('image_url') or '').strip()
                    video_url = (article.get('video_url') or '').strip()
                    try:
                        if video_url and NewsService._is_supported_video_url(video_url):
                            await bot_instance.bot.send_video(
                                chat_id=chat_id,
                                video=video_url,
                                caption=caption,
                                parse_mode='HTML',
                                supports_streaming=True,
                            )
                        elif image_url:
                            await bot_instance.bot.send_photo(
                                chat_id=chat_id,
                                photo=image_url,
                                caption=caption,
                                parse_mode='HTML',
                            )
                        else:
                            await bot_instance.bot.send_message(
                                chat_id=chat_id,
                                text=caption,
                                parse_mode='HTML',
                                disable_web_page_preview=False,
                            )
                        sent_per_article[news_id] += 1
                        successful += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to send article '{article.get('title', '')[:40]}' to chat {chat_id}: {e}"
                        )
                        failed += 1

                    # Brief delay to avoid rate limiting
                    await asyncio.sleep(0.6)

            # Cache article after it was posted to at least one chat.
            for news_id, article in fresh_articles:
                if sent_per_article.get(news_id, 0) > 0:
                    bot_instance.cache_news(
                        news_id,
                        article.get('title', ''),
                        article.get('source', {}).get('name', 'Unknown'),
                        article.get('url', ''),
                    )

            logger.info(f"News broadcast complete: {successful} successful, {failed} failed")

        except Exception as e:
            logger.error(f"News broadcast failed: {e}")

    @staticmethod
    async def send_news_to_chat(bot_instance, chat_id: int):
        """Send current curated news to a specific chat as separate article posts."""
        try:
            logger.info(f"Fetching news for chat {chat_id}...")

            articles = await NewsService.fetch_all_news()
            if not articles:
                await bot_instance.bot.send_message(
                    chat_id=chat_id,
                    text="No high-quality stock or commodities stories found right now. Please try again later.",
                )
                return

            await bot_instance.bot.send_message(
                chat_id=chat_id,
                text=NewsService.format_broadcast_intro(articles),
                parse_mode='HTML',
            )
            await asyncio.sleep(0.4)

            for article in articles:
                caption = NewsService.format_article_caption(article)
                image_url = (article.get('image_url') or '').strip()
                video_url = (article.get('video_url') or '').strip()
                if video_url and NewsService._is_supported_video_url(video_url):
                    await bot_instance.bot.send_video(
                        chat_id=chat_id,
                        video=video_url,
                        caption=caption,
                        parse_mode='HTML',
                        supports_streaming=True,
                    )
                elif image_url:
                    await bot_instance.bot.send_photo(
                        chat_id=chat_id,
                        photo=image_url,
                        caption=caption,
                        parse_mode='HTML',
                    )
                else:
                    await bot_instance.bot.send_message(
                        chat_id=chat_id,
                        text=caption,
                        parse_mode='HTML',
                        disable_web_page_preview=False,
                    )
                await asyncio.sleep(0.6)

            logger.info(f"News sent to chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to send news to {chat_id}: {e}")
            raise
