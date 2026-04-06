"""
News fetching and formatting service
"""
import aiohttp
import hashlib
from html import escape
from urllib.parse import quote
from utils.logger import logger
from config.settings import NEWSAPI_KEY, ALPHAVANTAGE_KEY


class NewsService:
    """Handle news fetching from multiple sources"""

    @staticmethod
    async def fetch_news_newsapi() -> list:
        """Fetch news from NewsAPI.org"""
        keywords = [
            'stock market',
            'tech stocks',
            'financial news',
            'cryptocurrency',
            'forex trading',
        ]
        articles = []

        try:
            async with aiohttp.ClientSession() as session:
                for keyword in keywords:
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q={quote(keyword)}&language=en&sortBy=publishedAt&page=1"
                    )
                    headers = {"X-Api-Key": NEWSAPI_KEY}

                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get('articles'):
                                articles.extend(data['articles'][:2])
                        else:
                            logger.warning(f"NewsAPI returned {resp.status} for keyword {keyword}")
        except Exception as e:
            logger.error(f"Failed to fetch from NewsAPI: {e}")

        return articles

    @staticmethod
    async def fetch_news_alphavantage() -> list:
        """Fetch news from Alpha Vantage"""
        articles = []

        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function=NEWS_SENTIMENT&sort=LATEST&limit=10&apikey={ALPHAVANTAGE_KEY}"
                )

                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('feed'):
                            for item in data['feed']:
                                articles.append(
                                    {
                                        'source': {'name': item.get('source', 'Alpha Vantage')},
                                        'title': item.get('title', ''),
                                        'description': item.get('summary', ''),
                                        'url': item.get('url', ''),
                                        'publishedAt': item.get('time_published', ''),
                                    }
                                )
                        else:
                            logger.warning("Alpha Vantage returned no feed data")
                    else:
                        logger.warning(f"Alpha Vantage returned {resp.status}")
        except Exception as e:
            logger.error(f"Failed to fetch from Alpha Vantage: {e}")

        return articles

    @staticmethod
    async def fetch_all_news() -> list:
        """Aggregate and deduplicate news from all sources"""
        try:
            newsapi_articles = await NewsService.fetch_news_newsapi()
            av_articles = await NewsService.fetch_news_alphavantage()

            all_articles = newsapi_articles + av_articles

            # Deduplicate by exact normalized title hash.
            unique_articles = []
            seen_titles = set()

            for article in all_articles:
                title = article.get('title', '')
                title_hash = hashlib.md5(title.lower().strip().encode()).hexdigest()

                if title_hash not in seen_titles:
                    unique_articles.append(article)
                    seen_titles.add(title_hash)

            return sorted(
                unique_articles,
                key=lambda x: x.get('publishedAt', ''),
                reverse=True,
            )[:10]

        except Exception as e:
            logger.error(f"Failed to fetch all news: {e}")
            return []

    @staticmethod
    def format_news_message(articles: list) -> str:
        """Format news articles for Telegram HTML mode"""
        if not articles:
            return "No news available at the moment. Try again later."

        message = "<b>Latest Market News</b>\n\n"

        for i, article in enumerate(articles[:5], 1):
            try:
                title = escape(article.get('title', 'No title'))
                source = escape(article.get('source', {}).get('name', 'Unknown'))
                url = (article.get('url', '') or '').replace("'", "%27")

                if url:
                    message += f"{i}. <a href='{url}'>{title[:80]}...</a>\n"
                    message += f"   Source: {source}\n\n"
                else:
                    message += f"{i}. {title[:80]}...\n"
                    message += f"   Source: {source}\n\n"
            except Exception as e:
                logger.error(f"Failed to format news article: {e}")
                continue

        return message
