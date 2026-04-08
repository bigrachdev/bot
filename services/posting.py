"""
Professional post formatting and post-governance helpers.
"""
from datetime import datetime, timezone
from html import escape

from utils.logger import logger
from services.news import NewsService


class PostingService:
    """Centralized post structure for reliable, professional broadcasts."""

    DISCLAIMER = "Educational market update only. Not financial advice."

    @staticmethod
    def _build_impact_line(article: dict) -> str:
        """Generate a concise impact statement to add context beyond headline text."""
        category = (article.get("category") or "market").lower()
        topic = NewsService._topic_label(article)
        if category == "commodities":
            return f"Why it matters: {topic} pricing can influence inflation expectations and sector leadership."
        if topic in {"Earnings", "Macro"}:
            return "Why it matters: This can shift risk sentiment, forward guidance assumptions, and valuation multiples."
        return "Why it matters: This may affect market positioning, volatility, and near-term price discovery."

    @staticmethod
    def _format_published_utc(article: dict) -> str:
        """Return article publish timestamp in compact UTC display format."""
        published = article.get("publishedAt") or ""
        published_dt = NewsService._parse_published_time(published)
        if published_dt == datetime.min.replace(tzinfo=timezone.utc):
            return "Time unavailable"
        return published_dt.strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def format_news_briefing_intro(articles: list) -> str:
        """Header message sent once before news story cards."""
        market_count = sum(1 for a in articles if a.get("category") == "market")
        commodity_count = sum(1 for a in articles if a.get("category") == "commodities")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return (
            "<b>Market Desk | High-Impact News Briefing</b>\n"
            f"Published: <b>{generated_at}</b>\n"
            f"Coverage: <b>{market_count}</b> market | <b>{commodity_count}</b> commodities\n\n"
            "Focus: market-moving headlines with context, timing, and source credibility.\n"
            "News feed only in this bulletin. Stock signal data is published separately.\n\n"
            f"<i>{PostingService.DISCLAIMER}</i>"
        )

    @staticmethod
    def format_news_article_card(article: dict, rank: int, total: int) -> str:
        """Consistent per-article card format."""
        title = escape(article.get("title", "No title"))[:220]
        description = escape(article.get("description", "No summary available."))[:820]
        category = escape((article.get("category") or "market").title())
        topic = escape(NewsService._topic_label(article))
        impact_line = escape(PostingService._build_impact_line(article))
        url = (article.get("url", "") or "").replace("'", "%27")

        return (
            f"<b>{category} | {topic}</b>\n"
            f"<b>{title}</b>\n\n"
            f"<b>What happened</b>\n{description}\n\n"
            f"<b>{impact_line}</b>\n\n"
            f"<a href='{url}'>Read full story and market implications</a>\n\n"
            f"<i>{PostingService.DISCLAIMER}</i>"
        )[:1000]

    @staticmethod
    def format_analysis_bulletin(signals: list, market_type: str = "Stocks") -> str:
        """Professional analysis broadcast format."""
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if not signals:
            return (
                f"<b>{escape(market_type)} Signal Bulletin</b>\n"
                f"Published: <b>{generated_at}</b>\n\n"
                "No strong directional setups detected this cycle.\n"
                "Condition: Neutral / mixed momentum.\n\n"
                f"<i>{PostingService.DISCLAIMER}</i>"
            )

        lines = [
            f"<b>{escape(market_type)} Signal Bulletin</b>",
            f"Published: <b>{generated_at}</b>",
            "",
        ]
        for signal in signals[:8]:
            symbol = escape(signal.get("symbol", "N/A"))
            rec = escape(signal.get("recommendation", "N/A"))
            rsi = escape(str(signal.get("oscillators", {}).get("rsi", "N/A")))
            macd = escape(str(signal.get("oscillators", {}).get("macd", "N/A")))
            stoch = escape(str(signal.get("oscillators", {}).get("stoch", "N/A")))
            sma20 = escape(str(signal.get("moving_averages", {}).get("sma20", "N/A")))
            sma50 = escape(str(signal.get("moving_averages", {}).get("sma50", "N/A")))
            lines.append(f"<b>{symbol}</b>: {rec}")
            lines.append(f"RSI {rsi} | MACD {macd} | STOCH {stoch}")
            lines.append(f"SMA20 {sma20} | SMA50 {sma50}")
            lines.append("")

        lines.append(f"<i>{PostingService.DISCLAIMER}</i>")
        message = "\n".join(lines).strip()
        if len(message) > 3800:
            logger.warning("Analysis bulletin exceeded preferred length; trimming output.")
            message = message[:3790] + "..."
        return message
