"""
Legacy compatibility entrypoint.

The active bot implementation is modular and lives in bot.py plus
services/handlers/schedulers packages. Keep this file as a thin shim so
older run commands (`python signalbot.py`) continue to work.
"""
import runpy
from utils.logger import logger


if __name__ == "__main__":
    logger.warning("signalbot.py is deprecated. Use `python bot.py` instead.")
    runpy.run_module("bot", run_name="__main__")
