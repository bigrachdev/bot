"""
Simple test to send a message directly
"""
import asyncio
import os
from dotenv import load_dotenv
from telegram.ext import Application
from database.db import MarketBot
from utils.logger import setup_logging

load_dotenv()
logger = setup_logging()

async def main():
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    
    print("Creating bot instance...")
    bot = MarketBot(6417609151)
    
    chats = bot.get_subscribed_chats()
    print(f"Target channels: {chats}")
    
    if not chats:
        print("❌ No channels configured!")
        return
    
    # Create a bot application
    app = Application.builder().token(BOT_TOKEN).build()
    await app.initialize()
    
    for chat_id, chat_type in chats:
        print(f"\nAttempting to send message to {chat_id}...")
        try:
            await app.bot.send_message(
                chat_id=chat_id,
                text="🧪 <b>Test Message</b>\n\nThis is a test broadcast from the bot.\nIf you see this, the bot is working!",
                parse_mode='HTML'
            )
            print(f"✅ Successfully sent to {chat_id}")
        except Exception as e:
            print(f"❌ Failed to send to {chat_id}: {e}")
            logger.error(f"Failed to send to {chat_id}: {e}")
    
    await app.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
