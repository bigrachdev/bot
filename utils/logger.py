"""
Logging configuration for the Market Data Bot
"""
import logging
import sys

def setup_logging():
    """Configure logging for production"""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('market_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Avoid leaking bot token via verbose HTTP request logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_logging()
