"""
Analysis broadcasting scheduler
"""
import asyncio
from utils.logger import logger
from services.analysis import AnalysisService


class AnalysisScheduler:
    """Handle scheduled analysis broadcasting."""

    @staticmethod
    async def broadcast_analysis(bot_instance, chat_list: list = None):
        """Broadcast stock performance snapshots to subscribed chats."""
        try:
            logger.info("Starting analysis broadcast...")

            signals = await AnalysisService.fetch_top_stocks_analysis()
            if not signals:
                logger.info("No stock performance data detected")
                return

            message = AnalysisService.format_analysis_message(signals, 'Stocks')

            if chat_list is None:
                chat_list = bot_instance.get_subscribed_chats()

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

                await asyncio.sleep(0.5)

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
                message = AnalysisService.format_analysis_message(signals, 'Stocks')
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
