"""
Technical analysis service for stock market signals
"""
import aiohttp
from tradingview_ta import TA_Handler, Interval
from utils.logger import logger
from config.settings import TOP_STOCKS, TOP_FOREX, FINNHUB_KEY

class AnalysisService:
    """Handle technical analysis and signal detection"""

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
    async def fetch_analysis(symbol: str, market: str = 'STOCKS') -> dict:
        """Fetch technical analysis for a single symbol"""
        try:
            if market == 'STOCKS':
                screener = "america"
                exchange = "nasdaq"
            elif market == 'CRYPTO':
                screener = "crypto"
                exchange = "binance"
            else:  # FOREX
                screener = "forex"
                exchange = "fxstreet"

            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1D
            )

            analysis = handler.get_analysis()
            
            # Enhance with Finnhub indicators
            finnhub_data = await AnalysisService.fetch_finnhub_indicators(symbol)

            return {
                'symbol': symbol,
                'recommendation': analysis.recommendation,
                'oscillators': {
                    'rsi': finnhub_data.get('rsi', analysis.oscillators.get('RSI', 'N/A')),
                    'stoch': analysis.oscillators.get('Stoch.K', 'N/A'),
                    'macd': finnhub_data.get('macd', 'N/A'),
                },
                'moving_averages': {
                    'sma20': finnhub_data.get('sma20', analysis.moving_averages.get('SMA20', 'N/A')),
                    'sma50': finnhub_data.get('sma50', analysis.moving_averages.get('SMA50', 'N/A')),
                    'ema12': analysis.moving_averages.get('EMA12', 'N/A'),
                }
            }
        except Exception as e:
            logger.error(f"Failed to fetch analysis for {symbol}: {e}")
            return None

    @staticmethod
    async def fetch_top_stocks_analysis() -> list:
        """Fetch analysis for top 10 stocks, filter strong signals"""
        strong_signals = []
        
        try:
            for symbol in TOP_STOCKS[:10]:
                analysis = await AnalysisService.fetch_analysis(symbol, 'STOCKS')
                
                if analysis and analysis['recommendation'] in ['STRONG_BUY', 'STRONG_SELL']:
                    strong_signals.append(analysis)
                
                # Rate limiting
                await aiohttp.ClientSession().close()
        except Exception as e:
            logger.error(f"Failed to fetch stocks analysis: {e}")
        
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
