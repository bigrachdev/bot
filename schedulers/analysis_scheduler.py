"""
Analysis broadcasting scheduler
"""
import asyncio
from config.settings import POST_MIN_SECONDS_BETWEEN_MESSAGES
from services.posting import PostingService
from utils.logger import logger
from services.analysis import AnalysisService


class AnalysisScheduler:
    """Handle scheduled analysis broadcasting."""

    @staticmethod
    async def broadcast_analysis(bot_instance, chat_list: list = None):
        """Broadcast stock performance snapshots to subscribed chats."""
        try:
            logger.info("Starting analysis broadcast...")

            if chat_list is None:
                chat_list = bot_instance.get_subscribed_chats()

            signals = await AnalysisService.fetch_top_stocks_analysis()
            if not signals:
                status = AnalysisService.get_last_status()
                logger.info("No stock performance data detected")
                if status == "rate_limited" and chat_list:
                    notice = (
                        "<b>Stocks Signal Bulletin</b>\n"
                        "Data providers are temporarily rate-limiting requests. "
                        "The bot will retry automatically on the next cycle."
                    )
                    for chat_id, _chat_type in chat_list:
                        try:
                            await bot_instance.bot.send_message(
                                chat_id=chat_id,
                                text=notice,
                                parse_mode='HTML',
                            )
                            await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)
                        except Exception as e:
                            logger.error(f"Failed to send analysis status notice to chat {chat_id}: {e}")
                return

            message = PostingService.format_analysis_bulletin(signals, "Stocks")

            if not chat_list:
                logger.info("No subscribed chats for analysis broadcast")
                return

            successful = 0
            failed = 0

            for chat_id, _chat_type in chat_list:
                try:
                    await bot_instance.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML',
                    )
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to send analysis to chat {chat_id}: {e}")
                    failed += 1

                await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)

            logger.info(f"Analysis broadcast complete: {successful} successful, {failed} failed")

        except Exception as e:
            logger.error(f"Analysis broadcast failed: {e}")

    @staticmethod
    async def send_analysis_to_chat(bot_instance, chat_id: int, market: str = 'stocks'):
        """Send current analysis to a specific chat."""
        try:
            logger.info(f"Fetching {market} analysis for chat {chat_id}...")

            if market.lower() == 'stocks':
                signals = await AnalysisService.fetch_top_stocks_analysis()
                message = PostingService.format_analysis_bulletin(signals, "Stocks")
            else:
                message = "Unknown market type. Use 'stocks'."

            await bot_instance.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='HTML',
            )

            logger.info(f"{market.title()} analysis sent to chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to send {market} analysis to {chat_id}: {e}")
            raise
