"""
News broadcasting scheduler
"""
from utils.logger import logger
from services.news import NewsService

class NewsScheduler:
    """Handle scheduled news broadcasting"""
    
    @staticmethod
    async def broadcast_news(bot_instance, chat_list: list = None):
        """Broadcast latest news to subscribed chats"""
        try:
            logger.info("🔄 Starting news broadcast...")
            
            # Fetch news
            articles = await NewsService.fetch_all_news()
            if not articles:
                logger.warning("No articles fetched for broadcast")
                return
            
            # Format message
            message = NewsService.format_news_message(articles)
            
            # Get subscribed chats if not provided
            if chat_list is None:
                chat_list = bot_instance.get_subscribed_chats()
            
            if not chat_list:
                logger.info("No subscribed chats for news broadcast")
                return
            
            # Send to each subscribed chat
            successful = 0
            failed = 0
            
            for chat_id, chat_type in chat_list:
                try:
                    # Check cache before posting
                    news_id = f"{articles[0]['title'][:50]}".lower().replace(' ', '_')
                    
                    if not bot_instance.is_news_cached(news_id):
                        await bot_instance.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML',
                            disable_web_page_preview=False
                        )
                        
                        # Cache the posted news
                        bot_instance.cache_news(
                            news_id,
                            articles[0].get('title', ''),
                            articles[0].get('source', {}).get('name', 'Unknown'),
                            articles[0].get('url', '')
                        )
                        
                        successful += 1
                    else:
                        logger.debug(f"News already cached, skipping chat {chat_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to send news to chat {chat_id}: {e}")
                    failed += 1
                    
                    # Brief delay to avoid rate limiting
                    import asyncio
                    await asyncio.sleep(0.5)
            
            logger.info(f"✅ News broadcast complete: {successful} successful, {failed} failed")
            
        except Exception as e:
            logger.error(f"News broadcast failed: {e}")

    @staticmethod
    async def send_news_to_chat(bot_instance, chat_id: int):
        """Send current news to a specific chat"""
        try:
            logger.info(f"📰 Fetching news for chat {chat_id}...")
            
            articles = await NewsService.fetch_all_news()
            message = NewsService.format_news_message(articles)
            
            await bot_instance.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='HTML',
                disable_web_page_preview=False
            )
            
            logger.info(f"✅ News sent to chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Failed to send news to {chat_id}: {e}")
            raise
