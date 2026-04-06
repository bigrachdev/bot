"""
Test script to manually trigger broadcasts and see what happens
"""
import asyncio
import sys
from database.db import MarketBot
from schedulers.news_scheduler import NewsScheduler
from schedulers.analysis_scheduler import AnalysisScheduler
from utils.logger import setup_logging

logger = setup_logging()

async def test_news():
    """Test news broadcast"""
    print("\n" + "="*50)
    print("TESTING NEWS BROADCAST")
    print("="*50)
    
    bot = MarketBot(6417609151)
    chats = bot.get_subscribed_chats()
    
    print(f"\nSubscribed chats: {chats}")
    
    if not chats:
        print("❌ No chats configured! Check TARGET_CHANNELS in .env")
        return
    
    articles = await NewsScheduler.send_news_to_chat.__func__(bot, chats[0][0])
    print(f"\nNews broadcast result: {articles}")

async def test_analysis():
    """Test analysis broadcast"""
    print("\n" + "="*50)
    print("TESTING ANALYSIS BROADCAST")
    print("="*50)
    
    bot = MarketBot(6417609151)
    chats = bot.get_subscribed_chats()
    
    print(f"\nSubscribed chats: {chats}")
    
    if not chats:
        print("❌ No chats configured! Check TARGET_CHANNELS in .env")
        return
    
    try:
        await AnalysisScheduler.broadcast_analysis(bot)
        print("\nAnalysis broadcast completed")
    except Exception as e:
        print(f"\nAnalysis broadcast failed: {e}")

async def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'analysis':
        await test_analysis()
    else:
        await test_news()

if __name__ == '__main__':
    asyncio.run(main())
