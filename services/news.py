"""
News fetching and formatting service
"""
import aiohttp
import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import escape
from urllib.parse import urlparse
from urllib.parse import quote
from xml.etree import ElementTree as ET
from utils.logger import logger
from config.settings import NEWSAPI_KEY, ALPHAVANTAGE_KEY, MAX_NEWS_AGE_HOURS


class NewsService:
    """Handle news fetching from multiple sources"""
    RELEVANCE_MAX_SCORE = 15

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

    SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.webm', '.mov')
    STOCK_TERMS = (
        'stock',
        'stocks',
        'equity',
        'equities',
        'earnings',
        'guidance',
        'nasdaq',
        'dow',
        's&p',
        'sp500',
        'wall street',
        'valuation',
        'analyst',
        'share',
    )
    COMMODITY_TERMS = (
        'commodity',
        'commodities',
        'crude',
        'oil',
        'brent',
        'wti',
        'gold',
        'silver',
        'copper',
        'natural gas',
        'lme',
        'opec',
        'metals',
        'futures',
    )
    IMPACT_TERMS = (
        'rate',
        'inflation',
        'fed',
        'ecb',
        'central bank',
        'tariff',
        'sanctions',
        'outlook',
        'demand',
        'supply',
        'forecast',
    )

    @staticmethod
    def _parse_published_time(value: str) -> datetime:
        """Parse provider publish timestamps into timezone-aware UTC datetimes."""
        raw = (value or '').strip()
        if not raw:
            return datetime.min.replace(tzinfo=timezone.utc)

        try:
            parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            pass

        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    @staticmethod
    def _is_recent_article(article: dict) -> bool:
        """True when article has a publish time within the configured freshness window."""
        published = article.get('publishedAt') or ''
        published_dt = NewsService._parse_published_time(published)
        if published_dt == datetime.min.replace(tzinfo=timezone.utc):
            return False

        cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_NEWS_AGE_HOURS)
        return published_dt >= cutoff

    @staticmethod
    def _classify_category(title: str, description: str, fallback: str = 'market') -> str:
        """Classify article into market vs commodities."""
        text = f"{title} {description}".lower()
        if any(term in text for term in NewsService.COMMODITY_TERMS):
            return 'commodities'
        return fallback

    @staticmethod
    def _is_stock_or_commodity_article(article: dict) -> bool:
        """Keep only items clearly related to stocks or commodities."""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        stock_hit = any(term in text for term in NewsService.STOCK_TERMS)
        commodity_hit = any(term in text for term in NewsService.COMMODITY_TERMS)
        return stock_hit or commodity_hit

    @staticmethod
    def _topic_label(article: dict) -> str:
        """Derive a concise topic label for channel presentation."""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        if any(term in text for term in ('oil', 'crude', 'brent', 'wti', 'opec')):
            return 'Oil'
        if any(term in text for term in ('gold', 'silver', 'copper', 'metals', 'lme')):
            return 'Metals'
        if any(term in text for term in ('natural gas', 'gas futures')):
            return 'Natural Gas'
        if any(term in text for term in ('earnings', 'guidance', 'analyst', 'valuation')):
            return 'Earnings'
        if any(term in text for term in ('federal reserve', 'fed', 'rate', 'inflation', 'ecb', 'central bank')):
            return 'Macro'
        return 'Market'

    @staticmethod
    def _article_score(article: dict) -> int:
        """Score for professional relevance and freshness ordering."""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        score = 0
        if article.get('category') == 'commodities':
            score += 3
        if any(term in text for term in NewsService.STOCK_TERMS):
            score += 3
        if any(term in text for term in NewsService.COMMODITY_TERMS):
            score += 3
        if any(term in text for term in NewsService.IMPACT_TERMS):
            score += 2
        if NewsService._is_trusted_source(
            article.get('source', {}).get('name', ''),
            article.get('url', ''),
        ):
            score += 2
        if NewsService._is_recent_article(article):
            score += 2
        return score

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
        video_url = (
            raw.get('urlToVideo')
            or raw.get('video_url')
            or raw.get('video')
            or ''
        )
        return {
            'source': {'name': source_name},
            'title': (raw.get('title') or '').strip(),
            'description': (raw.get('description') or raw.get('content') or '').strip(),
            'url': (raw.get('url') or '').strip(),
            'publishedAt': (raw.get('publishedAt') or '').strip(),
            'image_url': (raw.get('urlToImage') or '').strip(),
            'video_url': (video_url or '').strip(),
            'category': category,
        }

    @staticmethod
    def _extract_video_from_rss_item(item) -> str:
        """Extract direct video URL from RSS item when available."""
        # Standard RSS enclosure tag.
        enclosure = item.find("enclosure")
        if enclosure is not None:
            url = (enclosure.attrib.get("url") or "").strip()
            media_type = (enclosure.attrib.get("type") or "").lower()
            if url and ("video" in media_type or NewsService._is_supported_video_url(url)):
                return url

        # media:content tag with namespace.
        media_nodes = item.findall("{http://search.yahoo.com/mrss/}content")
        for media_node in media_nodes:
            url = (media_node.attrib.get("url") or "").strip()
            media_type = (media_node.attrib.get("type") or "").lower()
            if url and ("video" in media_type or NewsService._is_supported_video_url(url)):
                return url

        return ""

    @staticmethod
    def _is_supported_video_url(video_url: str) -> bool:
        """Check if URL looks like a directly playable short video."""
        video_url = (video_url or "").lower().split("?")[0]
        return any(video_url.endswith(ext) for ext in NewsService.SUPPORTED_VIDEO_EXTENSIONS)

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
                                'video_url': NewsService._extract_video_from_rss_item(item),
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
                    from_time = (
                        datetime.now(timezone.utc) - timedelta(hours=MAX_NEWS_AGE_HOURS)
                    ).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q={quote(keyword)}"
                        "&language=en"
                        "&searchIn=title,description"
                        "&sortBy=publishedAt"
                        f"&from={quote(from_time)}"
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
                                    'video_url': '',
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

            fresh_articles = [a for a in unique_articles if NewsService._is_recent_article(a)]
            focused_articles = [a for a in fresh_articles if NewsService._is_stock_or_commodity_article(a)]
            if not focused_articles:
                logger.warning("No stock/commodity-focused articles found after filtering")
                return []

            def sort_key(article):
                published = NewsService._parse_published_time(article.get('publishedAt', ''))
                return (
                    NewsService._article_score(article),
                    published,
                )

            market_sorted = sorted(
                [a for a in focused_articles if a.get('category') == 'market'],
                key=sort_key,
                reverse=True,
            )
            commodity_sorted = sorted(
                [a for a in focused_articles if a.get('category') == 'commodities'],
                key=sort_key,
                reverse=True,
            )

            selected = market_sorted[:4] + commodity_sorted[:3]
            if len(selected) < 6:
                combined_ranked = sorted(focused_articles, key=sort_key, reverse=True)
                seen_urls = {a.get('url', '') for a in selected}
                for candidate in combined_ranked:
                    url = candidate.get('url', '')
                    if url and url not in seen_urls:
                        selected.append(candidate)
                        seen_urls.add(url)
                    if len(selected) >= 7:
                        break
            return selected[:7]

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
        topic = escape(NewsService._topic_label(article))
        published = escape((article.get('publishedAt') or '')[:19].replace('T', ' '))
        score = NewsService._article_score(article)

        caption = (
            f"<b>Market Desk | Stocks & Commodities</b>\n"
            f"<b>{category} | {topic}</b>\n"
            f"<b>{title}</b>\n\n"
            f"{description}\n\n"
            f"Source: <b>{source}</b>\n"
            f"Relevance Score: <b>{score}/{NewsService.RELEVANCE_MAX_SCORE}</b>\n"
        )
        if published:
            caption += f"Published: {published} UTC\n"
        caption += f"<a href='{url}'>Read full article</a>"

        # Telegram caption hard limit safety.
        return caption[:1000]

    @staticmethod
    def format_broadcast_intro(articles: list) -> str:
        """Short professional header sent before article batch."""
        market_count = sum(1 for a in articles if a.get('category') == 'market')
        commodity_count = sum(1 for a in articles if a.get('category') == 'commodities')
        return (
            "<b>Stocks & Commodities Briefing</b>\n"
            f"Coverage: <b>{market_count}</b> stock-market stories, "
            f"<b>{commodity_count}</b> commodities stories.\n"
            "Selection focuses on earnings, macro drivers, and supply-demand shifts."
        )
