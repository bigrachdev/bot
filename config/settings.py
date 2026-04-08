"""
Configuration and constants for the Market Data Bot
"""
import os
from dotenv import load_dotenv
from utils.logger import logger

# Load environment variables
load_dotenv()

# Bot Token (Required)
BOT_TOKEN = os.getenv('BOT_TOKEN')
YOUR_ADMIN_ID = int(os.getenv('YOUR_ADMIN_ID', '0'))

# Validate critical configuration
if not BOT_TOKEN:
    logger.error("Missing critical configuration: BOT_TOKEN")
    raise Exception("BOT_TOKEN not found in .env file")

if YOUR_ADMIN_ID == 0:
    logger.warning("ADMIN_ID not set - admin commands will be disabled")

# Database
DB_NAME = 'market_bot.db'
NEWS_CACHE_HOURS = 24  # Don't post same news for 24 hours
MAX_NEWS_AGE_HOURS = int(os.getenv('MAX_NEWS_AGE_HOURS', '6'))

# News API Keys
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
ALPHAVANTAGE_KEY = os.getenv('ALPHAVANTAGE_KEY', '')
FINNHUB_KEY = os.getenv('FINNHUB_KEY', '')

# Target Channels/Groups (comma-separated IDs)
TARGET_CHANNELS_RAW = os.getenv('TARGET_CHANNELS', '')
TARGET_CHANNELS = [ch.strip() for ch in TARGET_CHANNELS_RAW.split(',') if ch.strip()] if TARGET_CHANNELS_RAW else []

# Keep Alive (for Render/Heroku free tier)
KEEP_ALIVE = os.getenv('KEEP_ALIVE', 'True').lower() in ('true', '1', 'yes')
KEEP_ALIVE_INTERVAL = int(os.getenv('KEEP_ALIVE_INTERVAL', '300'))

# Posting governance (professional pacing and volume controls)
POST_MAX_NEWS_PER_CYCLE = max(1, int(os.getenv('POST_MAX_NEWS_PER_CYCLE', '4')))
POST_MIN_SECONDS_BETWEEN_MESSAGES = max(0.2, float(os.getenv('POST_MIN_SECONDS_BETWEEN_MESSAGES', '0.8')))
SEND_BRIEFING_INTRO = os.getenv('SEND_BRIEFING_INTRO', 'True').lower() in ('true', '1', 'yes')

# Scheduler cadence (minutes)
NEWS_INTERVAL_MINUTES = max(5, int(os.getenv('NEWS_INTERVAL_MINUTES', '25')))
ANALYSIS_INTERVAL_MINUTES = max(15, int(os.getenv('ANALYSIS_INTERVAL_MINUTES', '60')))
AD_INTERVAL_MINUTES = max(30, int(os.getenv('AD_INTERVAL_MINUTES', '240')))

# Stock data - top 50 stocks
TOP_STOCKS = [
    {'name': 'NVIDIA', 'symbol': 'NVDA', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Microsoft', 'symbol': 'MSFT', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Apple', 'symbol': 'AAPL', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Alphabet', 'symbol': 'GOOGL', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Amazon', 'symbol': 'AMZN', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Meta Platforms', 'symbol': 'META', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Saudi Aramco', 'symbol': '2222', 'exchange': 'TADAWUL', 'screener': 'america'},
    {'name': 'Berkshire Hathaway', 'symbol': 'BRK-B', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Tesla', 'symbol': 'TSLA', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Broadcom', 'symbol': 'AVGO', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Eli Lilly', 'symbol': 'LLY', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'JPMorgan Chase', 'symbol': 'JPM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Visa', 'symbol': 'V', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Walmart', 'symbol': 'WMT', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'UnitedHealth Group', 'symbol': 'UNH', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Mastercard', 'symbol': 'MA', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Procter & Gamble', 'symbol': 'PG', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Johnson & Johnson', 'symbol': 'JNJ', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Home Depot', 'symbol': 'HD', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Costco', 'symbol': 'COST', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Oracle', 'symbol': 'ORCL', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Merck', 'symbol': 'MRK', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Chevron', 'symbol': 'CVX', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Coca-Cola', 'symbol': 'KO', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'AbbVie', 'symbol': 'ABBV', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'PepsiCo', 'symbol': 'PEP', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Adobe', 'symbol': 'ADBE', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Salesforce', 'symbol': 'CRM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Bank of America', 'symbol': 'BAC', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'McDonald\'s', 'symbol': 'MCD', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Accenture', 'symbol': 'ACN', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Cisco', 'symbol': 'CSCO', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'T-Mobile', 'symbol': 'TMUS', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'IBM', 'symbol': 'IBM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'General Electric', 'symbol': 'GE', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Abbott', 'symbol': 'ABT', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'ServiceNow', 'symbol': 'NOW', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Caterpillar', 'symbol': 'CAT', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Philip Morris', 'symbol': 'PM', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Qualcomm', 'symbol': 'QCOM', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Danaher', 'symbol': 'DHR', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'American Express', 'symbol': 'AXP', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'NextEra Energy', 'symbol': 'NEE', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Verizon', 'symbol': 'VZ', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Texas Instruments', 'symbol': 'TXN', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Intuit', 'symbol': 'INTU', 'exchange': 'NASDAQ', 'screener': 'america'},
    {'name': 'Disney', 'symbol': 'DIS', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'Pfizer', 'symbol': 'PFE', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'S&P Global', 'symbol': 'SPGI', 'exchange': 'NYSE', 'screener': 'america'},
    {'name': 'RTX', 'symbol': 'RTX', 'exchange': 'NYSE', 'screener': 'america'},
]

# Forex data - top forex pairs
TOP_FOREX = [
    {'name': 'EUR/USD', 'symbol': 'EURUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/JPY', 'symbol': 'USDJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'GBP/USD', 'symbol': 'GBPUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'AUD/USD', 'symbol': 'AUDUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CAD', 'symbol': 'USDCAD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CHF', 'symbol': 'USDCHF', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'NZD/USD', 'symbol': 'NZDUSD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'EUR/JPY', 'symbol': 'EURJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'GBP/JPY', 'symbol': 'GBPJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'EUR/GBP', 'symbol': 'EURGBP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'AUD/JPY', 'symbol': 'AUDJPY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/SEK', 'symbol': 'USDSEK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/NOK', 'symbol': 'USDNOK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/MXN', 'symbol': 'USDMXN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/ZAR', 'symbol': 'USDZAR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/TRY', 'symbol': 'USDTRY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/RUB', 'symbol': 'USDRUB', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/BRL', 'symbol': 'USDBRL', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/INR', 'symbol': 'USDINR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CNY', 'symbol': 'USDCNY', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/HKD', 'symbol': 'USDHKD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/SGD', 'symbol': 'USDSGD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/DKK', 'symbol': 'USDDKK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PLN', 'symbol': 'USDPLN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CZK', 'symbol': 'USDCZK', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/THB', 'symbol': 'USDTHB', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/ILS', 'symbol': 'USDILS', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PHP', 'symbol': 'USDPHP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/TWD', 'symbol': 'USDTWD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/IDR', 'symbol': 'USDIDR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/MYR', 'symbol': 'USDMYR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/KRW', 'symbol': 'USDKRW', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/SAR', 'symbol': 'USDSAR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PKR', 'symbol': 'USDPKR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/BDT', 'symbol': 'USDBDT', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/EGP', 'symbol': 'USDEGP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/NGN', 'symbol': 'USDNGN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/ARS', 'symbol': 'USDARS', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/COP', 'symbol': 'USDCOP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/PEN', 'symbol': 'USDPEN', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/CLP', 'symbol': 'USDCLP', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/UAH', 'symbol': 'USDUAH', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/RON', 'symbol': 'USDRON', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/UZS', 'symbol': 'USDUZS', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/KZT', 'symbol': 'USDKZT', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/KWD', 'symbol': 'USDKWD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/BHD', 'symbol': 'USDBHD', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/OMR', 'symbol': 'USDOMR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/QAR', 'symbol': 'USDQAR', 'exchange': 'OANDA', 'screener': 'forex'},
    {'name': 'USD/JOD', 'symbol': 'USDJOD', 'exchange': 'OANDA', 'screener': 'forex'},
]
