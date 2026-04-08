"""
Analysis broadcasting scheduler
"""
import asyncio
from config.settings import POST_MIN_SECONDS_BETWEEN_MESSAGES, TARGET_CHANNEL_ID
from services.posting import PostingService
from utils.logger import logger
from services.analysis import AnalysisService


class AnalysisScheduler:
    """Handle scheduled analysis broadcasting."""

    @staticmethod
    async def broadcast_analysis(bot_instance, chat_list: list = None):
        """Broadcast stock performance snapshots to the target channel."""
        try:
            logger.info("Starting analysis broadcast...")

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

            chat_id, _ = chat_list[0]

            stock_prices = await AnalysisService.fetch_top_stock_prices()
            snapshot_message = PostingService.format_market_snapshot_message(stock_prices)

            signals = await AnalysisService.fetch_top_stocks_analysis()
            if signals:
                message = PostingService.format_analysis_bulletin(signals, "Stocks")
            else:
                message = PostingService.format_analysis_bulletin([], "Stocks")

            for cid, _chat_type in chat_list:
                try:
                    await bot_instance.bot.send_message(
                        chat_id=cid,
                        text=snapshot_message,
                        parse_mode='HTML',
                    )
                    await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)
                    await bot_instance.bot.send_message(
                        chat_id=cid,
                        text=message,
                        parse_mode='HTML',
                    )
                except Exception as e:
                    logger.error(f"Failed to send analysis to chat {cid}: {e}")

            logger.info("Analysis broadcast complete")

        except Exception as e:
            logger.error(f"Analysis broadcast failed: {e}")
            # Last resort: try to post at least something.
            try:
                if chat_list:
                    cid, _ = chat_list[0]
                    from datetime import datetime, timezone
                    fallback = (
                        f"<b>Analysis Update</b>\n"
                        f"Published: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                        "Analysis data temporarily unavailable.\n"
                        "<i>Educational market update only. Not financial advice.</i>"
                    )
                    await bot_instance.bot.send_message(
                        chat_id=cid,
                        text=fallback,
                        parse_mode='HTML',
                    )
                    logger.info("Analysis fallback message posted.")
            except Exception as last_err:
                logger.error(f"Analysis fallback also failed: {last_err}")
                try:
                    from bot import NoContentAvailable
                    raise NoContentAvailable("Analysis posting failed completely")
                except ImportError:
                    pass

    @staticmethod
    async def send_analysis_to_chat(bot_instance, chat_id: int, market: str = 'stocks'):
        """Send current analysis to a specific chat."""
        try:
            logger.info(f"Fetching {market} analysis for chat {chat_id}...")

            if market.lower() == 'stocks':
                stock_prices = await AnalysisService.fetch_top_stock_prices()
                snapshot_message = PostingService.format_market_snapshot_message(stock_prices)
                signals = await AnalysisService.fetch_top_stocks_analysis()
                message = PostingService.format_analysis_bulletin(signals, "Stocks")
                await bot_instance.bot.send_message(
                    chat_id=chat_id,
                    text=snapshot_message,
                    parse_mode='HTML',
                )
                await asyncio.sleep(POST_MIN_SECONDS_BETWEEN_MESSAGES)
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
