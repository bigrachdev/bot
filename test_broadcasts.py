"""
Test the actual broadcast functions
"""
import asyncio
import os
from dotenv import load_dotenv
from telegram.ext import Application
from database.db import MarketBot
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler
from utils.logger import setup_logging

load_dotenv()
logger = setup_logging()

async def test_news_broadcast():
    print("\n" + "="*60)
    print("TESTING NEWS BROADCAST")
    print("="*60)
    
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    app = Application.builder().token(BOT_TOKEN).build()
    await app.initialize()
    
    bot_instance = MarketBot(6417609151)
    bot_instance.bot = app.bot
    
    try:
        await NewsScheduler.broadcast_news(bot_instance)
        print("\n✅ News broadcast completed")
    except Exception as e:
        print(f"\n❌ News broadcast failed: {e}")
        logger.error(f"News broadcast failed: {e}")
    
    await app.shutdown()

async def test_analysis_broadcast():
    print("\n" + "="*60)
    print("TESTING ANALYSIS BROADCAST")
    print("="*60)
    
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    app = Application.builder().token(BOT_TOKEN).build()
    await app.initialize()
    
    bot_instance = MarketBot(6417609151)
    bot_instance.bot = app.bot
    
    try:
        await AnalysisScheduler.broadcast_analysis(bot_instance)
        print("\n✅ Analysis broadcast completed")
    except Exception as e:
        print(f"\n❌ Analysis broadcast failed: {e}")
        logger.error(f"Analysis broadcast failed: {e}")
    
    await app.shutdown()

async def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'analysis':
        await test_analysis_broadcast()
    else:
        await test_news_broadcast()

if __name__ == '__main__':
    asyncio.run(main())
