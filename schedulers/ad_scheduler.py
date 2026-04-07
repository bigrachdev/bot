"""
Recurring promotional ad scheduler.
"""
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from utils.logger import logger


class AdScheduler:
    """Handle recurring promotional posts."""

    OMNEX_MESSAGE = (
        "<b>Omnex Financial | Smarter Investing Across 5 Asset Classes</b>\n\n"
        "Your gateway to smarter investing:\n"
        "Corporate Bonds: 8-15% fixed annual returns\n"
        "Public Stocks: NYSE, NASDAQ, LSE and TSX listed companies\n"
        "REITs: Real estate exposure with regular dividends\n"
        "Vested Stocks: Pre-IPO and private equity opportunities\n"
        "Tokenized Commodities: 28 commodities with live pricing\n\n"
        "Join 5,300+ investors across 40+ countries managing over $80M in assets.\n\n"
        "<b>Investment Options at Omnex Financial</b>\n"
        "Corporate Bonds: Steady income with flexible payouts (weekly, monthly, quarterly).\n"
        "Public Stocks: Long-term growth plus potential dividend income.\n"
        "REITs: Commercial, residential, and industrial property exposure.\n"
        "Vested Stocks: Private market access with structured vesting timelines.\n"
        "Commodities: Fractional ownership in real assets across metals, energy, agriculture, and livestock.\n\n"
        "We also share market news, stock follow-ups, signals when available, and timely insights.\n\n"
        "<b>Build a balanced portfolio. Spread risk. Maximize returns.</b>"
    )

    @staticmethod
    async def broadcast_omnex_ad(bot_instance, chat_list: list = None):
        """Broadcast Omnex promotional ad to subscribed chats."""
        try:
            logger.info("Starting Omnex ad broadcast...")

            if chat_list is None:
                chat_list = bot_instance.get_subscribed_chats()

            if not chat_list:
                logger.info("No subscribed chats for Omnex ad broadcast")
                return

            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("Explore Omnex Financial", url="https://omnexfinancial.com")]]
            )

            successful = 0
            failed = 0

            for chat_id, _chat_type in chat_list:
                try:
                    await bot_instance.bot.send_message(
                        chat_id=chat_id,
                        text=AdScheduler.OMNEX_MESSAGE,
                        parse_mode='HTML',
                        disable_web_page_preview=False,
                        reply_markup=keyboard,
                    )
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to send Omnex ad to chat {chat_id}: {e}")
                    failed += 1

                await asyncio.sleep(0.6)

            logger.info(f"Omnex ad broadcast complete: {successful} successful, {failed} failed")

        except Exception as e:
            logger.error(f"Omnex ad broadcast failed: {e}")
