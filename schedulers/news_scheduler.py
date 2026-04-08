"""
News broadcasting scheduler
"""
import asyncio
import random
from utils.logger import logger
from services.news import NewsService


class NewsScheduler:
    """Handle scheduled news broadcasting"""

    @staticmethod
    async def _send_article_with_fallback(bot_instance, chat_id: int, caption: str, image_url: str, video_url: str):
        """Try rich media first, then fall back to plain text to avoid dropping posts."""
        if video_url and NewsService._is_supported_video_url(video_url):
            try:
                await bot_instance.bot.send_video(
                    chat_id=chat_id,
                    video=video_url,
                    caption=caption,
                    parse_mode='HTML',
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
                    parse_mode='HTML',
                )
                return
            except Exception as e:
                logger.warning(f"Photo send failed for chat {chat_id}, falling back to text: {e}")

        await bot_instance.bot.send_message(
            chat_id=chat_id,
            text=caption,
            parse_mode='HTML',
            disable_web_page_preview=False,
        )

    @staticmethod
    async def broadcast_news(bot_instance, chat_list: list = None):
        """Broadcast one major headline from each source with stock prices every 20-30 mins."""
        try:
            logger.info("Starting news broadcast (one headline per source)...")

            # Fetch all news
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

            # Group articles by source and pick the top one from each
            source_articles = {}
            for article in articles:
                source_name = (article.get('source', {}).get('name', 'Unknown') or 'Unknown').lower()
                if source_name not in source_articles:
                    source_articles[source_name] = article

            # Get unique sources
            unique_sources = list(source_articles.keys())
            logger.info(f"Found {len(unique_sources)} unique news sources")

            # Filter out already cached articles
            fresh_sources = {}
            for source_name, article in source_articles.items():
                news_id = NewsService.make_news_id(article)
                if not bot_instance.is_news_cached(news_id):
                    fresh_sources[source_name] = article

            if not fresh_sources:
                logger.info("All source headlines are already cached; skipping broadcast")
                return

            # Send one headline from each fresh source
            successful = 0
            failed = 0

            for chat_id, _chat_type in chat_list:
                for source_name, article in fresh_sources.items():
                    # Format message with article + stock prices
                    caption = await NewsScheduler.format_news_with_stocks(article)
                    image_url = (article.get('image_url') or '').strip()
                    video_url = (article.get('video_url') or '').strip()
                    
                    try:
                        await NewsScheduler._send_article_with_fallback(
                            bot_instance=bot_instance,
                            chat_id=chat_id,
                            caption=caption,
                            image_url=image_url,
                            video_url=video_url,
                        )
                        successful += 1
                        
                        # Cache this article
                        news_id = NewsService.make_news_id(article)
                        bot_instance.cache_news(
                            news_id,
                            article.get('title', ''),
                            article.get('source', {}).get('name', 'Unknown'),
                            article.get('url', ''),
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to send article from {source_name} to chat {chat_id}: {e}"
                        )
                        failed += 1

                    # Brief delay to avoid rate limiting
                    await asyncio.sleep(0.8)

            logger.info(f"News broadcast complete: {successful} successful, {failed} failed")

        except Exception as e:
            logger.error(f"News broadcast failed: {e}")

    @staticmethod
    async def format_news_with_stocks(article: dict) -> str:
        """Format a news article with current stock prices."""
        from services.analysis import AnalysisService
        
        # Format the news article
        news_caption = NewsService.format_article_caption(article)
        
        # Add a separator and stock market snapshot
        stock_header = (
            "\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📊 <b>Market Snapshot</b>\n"
        )
        
        # Fetch top stock prices
        try:
            stock_prices = await AnalysisService.fetch_top_stock_prices()
            
            if stock_prices:
                stock_prices_text = ""
                for stock in stock_prices:
                    stock_prices_text += f"• {stock['name']} ({stock['symbol']}): {stock['price']}\n"
                stock_header += stock_prices_text
            else:
                stock_header += "Market data temporarily unavailable\n"
        except Exception as e:
            logger.warning(f"Failed to fetch stock prices for news post: {e}")
            stock_header += "Market data loading...\n"
        
        return news_caption + stock_header

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

            for article in articles:
                caption = await NewsScheduler.format_news_with_stocks(article)
                image_url = (article.get('image_url') or '').strip()
                video_url = (article.get('video_url') or '').strip()
                await NewsScheduler._send_article_with_fallback(
                    bot_instance=bot_instance,
                    chat_id=chat_id,
                    caption=caption,
                    image_url=image_url,
                    video_url=video_url,
                )
                await asyncio.sleep(0.6)

            logger.info(f"News sent to chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to send news to {chat_id}: {e}")
            raise
