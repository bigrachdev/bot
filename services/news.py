"""
News fetching and formatting service
"""
import aiohttp
import hashlib
from datetime import datetime
from email.utils import parsedate_to_datetime
from html import escape
from urllib.parse import urlparse
from urllib.parse import quote
from xml.etree import ElementTree as ET
from utils.logger import logger
from config.settings import NEWSAPI_KEY, ALPHAVANTAGE_KEY


class NewsService:
    """Handle news fetching from multiple sources"""

    TRUSTED_NEWS_SOURCES = {
        'reuters',
        'bloomberg',
        'cnbc',
        'the wall street journal',
        'wall street journal',
        'financial times',
        'marketwatch',
        'barron',
        'associated press',
        'ap news',
        'investing.com',
        'yahoo finance',
        'seeking alpha',
        'forbes',
    }

    TRUSTED_NEWS_DOMAINS = (
        'reuters.com',
        'bloomberg.com',
        'cnbc.com',
        'wsj.com',
        'ft.com',
        'marketwatch.com',
        'barrons.com',
        'apnews.com',
        'investing.com',
        'finance.yahoo.com',
        'seekingalpha.com',
        'forbes.com',
        )

    @staticmethod
    def _classify_category(title: str, description: str, fallback: str = 'market') -> str:
        """Classify article into market vs commodities."""
        text = f"{title} {description}".lower()
        commodity_terms = (
            'oil',
            'gold',
            'silver',
            'copper',
            'natural gas',
            'commodity',
            'opec',
            'wti',
            'brent',
        )
        if any(term in text for term in commodity_terms):
            return 'commodities'
        return fallback

    @staticmethod
    def _is_trusted_source(source_name: str, url: str) -> bool:
        """Return True when article source is from trusted publishers."""
        source_name = (source_name or '').lower()
        url = (url or '').lower()
        domain = urlparse(url).netloc.replace('www.', '')

        if any(src in source_name for src in NewsService.TRUSTED_NEWS_SOURCES):
            return True

        return any(domain.endswith(allowed) for allowed in NewsService.TRUSTED_NEWS_DOMAINS)

    @staticmethod
    def _normalize_article(raw: dict, category: str) -> dict:
        """Normalize article shape across providers."""
        source_name = raw.get('source', {}).get('name', 'Unknown')
        return {
            'source': {'name': source_name},
            'title': (raw.get('title') or '').strip(),
            'description': (raw.get('description') or raw.get('content') or '').strip(),
            'url': (raw.get('url') or '').strip(),
            'publishedAt': (raw.get('publishedAt') or '').strip(),
            'image_url': (raw.get('urlToImage') or '').strip(),
            'category': category,
        }

    @staticmethod
    async def fetch_news_cnbc_rss() -> list:
        """Fetch detailed updates directly from CNBC RSS feeds."""
        feed_urls = [
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top News
            "https://www.cnbc.com/id/15839069/device/rss/rss.html",   # Markets
        ]
        articles = []

        try:
            async with aiohttp.ClientSession() as session:
                for feed_url in feed_urls:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status != 200:
                            logger.warning(f"CNBC RSS returned {resp.status} for {feed_url}")
                            continue

                        raw_xml = await resp.text()
                        root = ET.fromstring(raw_xml)

                        for item in root.findall(".//item"):
                            title = (item.findtext("title") or "").strip()
                            link = (item.findtext("link") or "").strip()
                            description = (item.findtext("description") or "").strip()
                            pub_date = (item.findtext("pubDate") or "").strip()

                            # Skip incomplete records.
                            if not title or not link:
                                continue

                            published = ""
                            if pub_date:
                                try:
                                    published = parsedate_to_datetime(pub_date).isoformat()
                                except Exception:
                                    published = pub_date

                            category = NewsService._classify_category(title, description, fallback='market')
                            normalized = {
                                'source': {'name': 'CNBC'},
                                'title': title,
                                'description': description,
                                'url': link,
                                'publishedAt': published,
                                'image_url': '',
                                'category': category,
                            }

                            if NewsService._is_trusted_source(
                                normalized['source']['name'],
                                normalized['url'],
                            ):
                                articles.append(normalized)

        except Exception as e:
            logger.error(f"Failed to fetch from CNBC RSS: {e}")

        return articles

    @staticmethod
    async def fetch_news_newsapi(keywords: list, category: str, per_keyword: int = 5) -> list:
        """Fetch curated news from NewsAPI.org for specific keywords."""
        articles = []

        try:
            async with aiohttp.ClientSession() as session:
                for keyword in keywords:
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q={quote(keyword)}"
                        "&language=en"
                        "&searchIn=title,description"
                        "&sortBy=publishedAt"
                        f"&pageSize={per_keyword}"
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
                                for raw in data['articles']:
                                    normalized = NewsService._normalize_article(raw, category)
                                    if not normalized['title'] or not normalized['url']:
                                        continue
                                    if not NewsService._is_trusted_source(
                                        normalized['source']['name'],
                                        normalized['url'],
                                    ):
                                        continue
                                    articles.append(normalized)
                        else:
                            logger.warning(f"NewsAPI returned {resp.status} for keyword {keyword}")
        except Exception as e:
            logger.error(f"Failed to fetch from NewsAPI: {e}")

        return articles

    @staticmethod
    async def fetch_news_alphavantage() -> list:
        """Fetch additional market news from Alpha Vantage."""
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
                                published = (item.get('time_published') or '').strip()
                                if len(published) == 15 and published.endswith('Z'):
                                    published = f"{published[:4]}-{published[4:6]}-{published[6:8]}T{published[9:11]}:{published[11:13]}:{published[13:15]}Z"

                                normalized = {
                                    'source': {'name': item.get('source', 'Alpha Vantage')},
                                    'title': (item.get('title') or '').strip(),
                                    'description': (item.get('summary') or '').strip(),
                                    'url': (item.get('url') or '').strip(),
                                    'publishedAt': published,
                                    'image_url': (item.get('banner_image') or '').strip(),
                                    'category': 'market',
                                }
                                if not normalized['title'] or not normalized['url']:
                                    continue
                                if not NewsService._is_trusted_source(
                                    normalized['source']['name'],
                                    normalized['url'],
                                ):
                                    continue
                                articles.append(normalized)
                        else:
                            logger.warning("Alpha Vantage returned no feed data")
                    else:
                        logger.warning(f"Alpha Vantage returned {resp.status}")
        except Exception as e:
            logger.error(f"Failed to fetch from Alpha Vantage: {e}")

        return articles

    @staticmethod
    async def fetch_all_news() -> list:
        """Aggregate and deduplicate curated market + commodities news."""
        try:
            market_keywords = [
                'stock market earnings guidance',
                'federal reserve inflation rates stocks',
                'us equities outlook reuters',
                'wall street market close',
            ]
            commodity_keywords = [
                'crude oil market supply opec',
                'gold prices inflation central bank demand',
                'natural gas futures storage',
                'copper commodity demand china',
            ]

            cnbc_articles = await NewsService.fetch_news_cnbc_rss()
            market_articles = await NewsService.fetch_news_newsapi(
                market_keywords, category='market', per_keyword=4
            )
            commodity_articles = await NewsService.fetch_news_newsapi(
                commodity_keywords, category='commodities', per_keyword=4
            )
            av_articles = await NewsService.fetch_news_alphavantage()

            all_articles = cnbc_articles + market_articles + commodity_articles + av_articles

            # Deduplicate by exact normalized title hash.
            unique_articles = []
            seen_titles = set()
            seen_urls = set()

            for article in all_articles:
                title = article.get('title', '')
                title_hash = hashlib.md5(title.lower().strip().encode()).hexdigest()
                url = article.get('url', '').strip().lower()

                if title_hash not in seen_titles and url not in seen_urls:
                    unique_articles.append(article)
                    seen_titles.add(title_hash)
                    seen_urls.add(url)

            def sort_key(article):
                published = article.get('publishedAt', '')
                return published or datetime.min.isoformat()

            market_sorted = sorted(
                [a for a in unique_articles if a.get('category') == 'market'],
                key=sort_key,
                reverse=True,
            )[:4]
            commodity_sorted = sorted(
                [a for a in unique_articles if a.get('category') == 'commodities'],
                key=sort_key,
                reverse=True,
            )[:3]

            curated = market_sorted + commodity_sorted
            return curated[:7]

        except Exception as e:
            logger.error(f"Failed to fetch all news: {e}")
            return []

    @staticmethod
    def make_news_id(article: dict) -> str:
        """Generate stable cache key for an article."""
        seed = f"{article.get('url', '')}|{article.get('title', '')}".lower().strip()
        return hashlib.md5(seed.encode()).hexdigest()

    @staticmethod
    def format_news_message(articles: list) -> str:
        """Compatibility fallback summary message."""
        if not articles:
            return "No news available at the moment. Try again later."

        lines = ["<b>Latest Market and Commodities News</b>", ""]
        for i, article in enumerate(articles[:7], 1):
            title = escape(article.get('title', 'No title'))[:90]
            source = escape(article.get('source', {}).get('name', 'Unknown'))
            category = escape((article.get('category') or 'market').title())
            lines.append(f"{i}. {title} ({category})")
            lines.append(f"   Source: {source}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def format_article_caption(article: dict) -> str:
        """Format one detailed article for Telegram caption/text."""
        title = escape(article.get('title', 'No title'))[:220]
        source = escape(article.get('source', {}).get('name', 'Unknown'))
        description = escape(article.get('description', 'No summary available.'))[:500]
        url = (article.get('url', '') or '').replace("'", "%27")
        category = escape((article.get('category') or 'market').title())
        published = escape((article.get('publishedAt') or '')[:19].replace('T', ' '))

        caption = (
            f"<b>{category} News</b>\n"
            f"<b>{title}</b>\n\n"
            f"{description}\n\n"
            f"Source: <b>{source}</b>\n"
        )
        if published:
            caption += f"Published: {published} UTC\n"
        caption += f"<a href='{url}'>Read full article</a>"

        # Telegram caption hard limit safety.
        return caption[:1000]
