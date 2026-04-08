"""
Technical analysis service for stock market signals
"""
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
from tradingview_ta import TA_Handler, Interval
from utils.logger import logger
from config.settings import TOP_STOCKS, TOP_FOREX, FINNHUB_KEY

class AnalysisService:
    """Handle technical analysis and signal detection"""
    _signals_cache = []
    _signals_cache_at = None
    _last_status = "idle"
    _cache_ttl_minutes = 90

    @staticmethod
    def get_last_status() -> str:
        """Return last analysis fetch status for scheduler decisions."""
        return AnalysisService._last_status

    @staticmethod
    def _normalize_symbol_for_tradingview(symbol: str, exchange: str) -> str:
        """Normalize symbols for TradingView-TA quirks (e.g., BRK-B -> BRK.B)."""
        clean_symbol = (symbol or "").strip().upper()
        clean_exchange = (exchange or "").strip().upper()
        if clean_exchange in {"NYSE", "NASDAQ"} and "-" in clean_symbol:
            return clean_symbol.replace("-", ".")
        return clean_symbol

    @staticmethod
    async def fetch_finnhub_indicators(symbol: str) -> dict:
        """Fetch technical indicators from Finnhub API"""
        indicators = {}
        
        if not FINNHUB_KEY:
            return indicators

        try:
            async with aiohttp.ClientSession() as session:
                # Fetch RSI
                url = f"https://finnhub.io/api/v1/stock/indicator?symbol={symbol}&indicator=rsi&resolution=D&token={FINNHUB_KEY}"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('d') and len(data['d']) > 0:
                            indicators['rsi'] = round(data['d'][-1], 2)
                    
            async with aiohttp.ClientSession() as session:
                # Fetch MACD
                url = f"https://finnhub.io/api/v1/stock/indicator?symbol={symbol}&indicator=macd&resolution=D&token={FINNHUB_KEY}"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('d') and len(data['d']) > 0:
                            indicators['macd'] = round(data['d'][-1], 2)
                    
            async with aiohttp.ClientSession() as session:
                # Fetch SMA (Simple Moving Average 20)
                url = f"https://finnhub.io/api/v1/stock/indicator?symbol={symbol}&indicator=sma&resolution=D&timeperiod=20&token={FINNHUB_KEY}"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('d') and len(data['d']) > 0:
                            indicators['sma20'] = round(data['d'][-1], 2)
                    
            async with aiohttp.ClientSession() as session:
                # Fetch SMA (Simple Moving Average 50)
                url = f"https://finnhub.io/api/v1/stock/indicator?symbol={symbol}&indicator=sma&resolution=D&timeperiod=50&token={FINNHUB_KEY}"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('d') and len(data['d']) > 0:
                            indicators['sma50'] = round(data['d'][-1], 2)
                    
            logger.info(f"Fetched Finnhub indicators for {symbol}: {list(indicators.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to fetch Finnhub indicators for {symbol}: {e}")

        return indicators

    @staticmethod
    async def fetch_analysis(
        symbol: str,
        market: str = 'STOCKS',
        exchange: str = None,
        screener: str = None,
    ) -> dict:
        """Fetch technical analysis for a single symbol"""
        try:
            if market == 'STOCKS':
                screener = (screener or "america").lower()
                exchange = (exchange or "nasdaq").lower()
            elif market == 'CRYPTO':
                screener = "crypto"
                exchange = "binance"
            else:  # FOREX
                screener = "forex"
                exchange = "fxstreet"

            symbol = AnalysisService._normalize_symbol_for_tradingview(symbol, exchange)

            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY
            )

            analysis = handler.get_analysis()

            # Enhance with Finnhub indicators
            finnhub_data = await AnalysisService.fetch_finnhub_indicators(symbol)

            return {
                'symbol': symbol,
                'recommendation': analysis.summary.get('RECOMMENDATION', 'N/A'),
                'oscillators': {
                    'rsi': finnhub_data.get('rsi', analysis.oscillators.get('RSI', 'N/A')),
                    'stoch': analysis.oscillators.get('STOCH.K', analysis.oscillators.get('STOCHK', 'N/A')),
                    'macd': finnhub_data.get('macd', 'N/A'),
                },
                'moving_averages': {
                    'sma20': finnhub_data.get('sma20', analysis.moving_averages.get('SMA20', 'N/A')),
                    'sma50': finnhub_data.get('sma50', analysis.moving_averages.get('SMA50', 'N/A')),
                    'ema12': analysis.moving_averages.get('EMA12', 'N/A'),
                }
            }
        except Exception as e:
            err_text = str(e).lower()
            if "429" in err_text or "rate limit" in err_text:
                AnalysisService._last_status = "rate_limited"
            logger.error(f"Failed to fetch analysis for {symbol}: {e}")
            return None

    @staticmethod
    async def fetch_top_stocks_analysis() -> list:
        """Fetch analysis for top 10 stocks, filter strong signals"""
        strong_signals = []
        consecutive_rate_limits = 0

        try:
            # Skip known unsupported exchanges for TradingView stock screener path.
            candidates = [
                s for s in TOP_STOCKS
                if (s.get("exchange", "").upper() in {"NASDAQ", "NYSE"})
            ][:10]

            for stock_data in candidates:
                symbol = stock_data['symbol']
                exchange = stock_data.get('exchange', 'NASDAQ')
                screener = stock_data.get('screener', 'america')
                analysis = await AnalysisService.fetch_analysis(
                    symbol=symbol,
                    market='STOCKS',
                    exchange=exchange,
                    screener=screener,
                )
                if not analysis:
                    # Detect provider throttling and stop early to avoid burning limits.
                    # fetch_analysis already logs the details.
                    # We infer 429 from recent failures by symbol density and short cadence.
                    consecutive_rate_limits += 1
                    if consecutive_rate_limits >= 3:
                        logger.warning("Analysis provider appears rate-limited; stopping this cycle early.")
                        AnalysisService._last_status = "rate_limited"
                        break
                else:
                    consecutive_rate_limits = 0

                if analysis and analysis['recommendation'] in ['STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL']:
                    strong_signals.append(analysis)

                # Mild pacing to reduce throttling pressure.
                await asyncio.sleep(0.8)

        except Exception as e:
            logger.error(f"Failed to fetch stocks analysis: {e}")

        if strong_signals:
            AnalysisService._signals_cache = strong_signals
            AnalysisService._signals_cache_at = datetime.now(timezone.utc)
            AnalysisService._last_status = "ok"
            return strong_signals

        # Fall back to recent cached signals if provider is throttling.
        if (
            AnalysisService._last_status == "rate_limited"
            and AnalysisService._signals_cache
            and AnalysisService._signals_cache_at
            and (datetime.now(timezone.utc) - AnalysisService._signals_cache_at)
            <= timedelta(minutes=AnalysisService._cache_ttl_minutes)
        ):
            logger.warning("Using cached analysis signals due to temporary provider throttling.")
            return AnalysisService._signals_cache

        AnalysisService._last_status = "empty"
        return strong_signals

    @staticmethod
    def format_analysis_message(signals: list, market_type: str = 'Stocks') -> str:
        """Format analysis results for Telegram"""
        if not signals:
            return f"📊 No strong signals detected in {market_type} today. 🔍"

        message = f"📊 <b>{market_type} Analysis - Strong Signals</b>\n\n"

        for signal in signals:
            symbol = signal.get('symbol', 'N/A')
            rec = signal.get('recommendation', 'N/A')

            emoji = '🔥' if rec == 'STRONG_BUY' else '🔴'

            message += f"{emoji} <b>{symbol}</b>: {rec}\n"

            # Oscillators
            rsi = signal.get('oscillators', {}).get('rsi', 'N/A')
            macd = signal.get('oscillators', {}).get('macd', 'N/A')
            stoch = signal.get('oscillators', {}).get('stoch', 'N/A')
            message += f"   RSI: {rsi} | MACD: {macd} | Stoch: {stoch}\n"

            # Moving Averages
            sma20 = signal.get('moving_averages', {}).get('sma20', 'N/A')
            sma50 = signal.get('moving_averages', {}).get('sma50', 'N/A')
            message += f"   SMA20: {sma20} | SMA50: {sma50}\n\n"

        return message

    @staticmethod
    async def fetch_top_stock_prices() -> list:
        """Fetch current prices for top stocks using Finnhub API (lightweight and fast)."""
        prices = []
        
        if not FINNHUB_KEY:
            logger.warning("FINNHUB_KEY not set - cannot fetch stock prices")
            return prices
        
        # Only fetch first 5 to keep it fast and lightweight
        candidates = [
            s for s in TOP_STOCKS
            if (s.get("exchange", "").upper() in {"NASDAQ", "NYSE"})
        ][:5]

        async with aiohttp.ClientSession() as session:
            for stock_data in candidates:
                symbol = stock_data['symbol']
                name = stock_data['name']
                
                try:
                    # Use Finnhub quote API for current price
                    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
                    
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            
                            # Finnhub returns: c (current price), h (high), l (low), o (open), pc (previous close), t (timestamp)
                            current_price = data.get('c')
                            
                            if current_price and isinstance(current_price, (int, float)) and current_price > 0:
                                prices.append({
                                    'name': name,
                                    'symbol': symbol,
                                    'price': f"${current_price:.2f}"
                                })
                            else:
                                logger.debug(f"Invalid price data for {symbol}: {data}")
                        else:
                            logger.warning(f"Finnhub quote API returned {resp.status} for {symbol}")
                    
                    # Brief delay to respect rate limits (60 calls/min on free tier)
                    await asyncio.sleep(1.2)
                    
                except Exception as e:
                    logger.debug(f"Could not fetch price for {symbol} via Finnhub: {e}")
                    continue

        logger.info(f"Fetched prices for {len(prices)} stocks via Finnhub")
        return prices
