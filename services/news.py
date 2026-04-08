"""
News fetching and formatting service
"""
import asyncio
import aiohttp
import hashlib
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import escape
from urllib.parse import urlparse, urlunparse
from urllib.parse import quote
from xml.etree import ElementTree as ET
from utils.logger import logger
from config.settings import NEWSAPI_KEY, ALPHAVANTAGE_KEY


class NewsService:
    """Handle news fetching from multiple sources"""
    RELEVANCE_MAX_SCORE = 15
    LATEST_NEWS_WINDOW_HOURS = 3  # Only very recent news for freshness
    MIN_DISTINCT_TOKENS_FOR_SIGNATURE = 3
    REQUEST_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

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
        'cnn business',
        'cnn',
        'the new york times',
        'new york times',
        'the washington post',
        'washington post',
        'business insider',
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
        'nasdaq.com',
        'kitco.com',
        'cnn.com',
        'nytimes.com',
        'washingtonpost.com',
        'businessinsider.com',
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
    SOURCE_MAX_PER_CYCLE = 2
    NEWSAPI_COOLDOWN_HOURS = 6
    _newsapi_cooldown_until = None

    STORY_SIGNATURE_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'in',
        'into', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'to', 'up', 'with',
        'after', 'amid', 'over', 'under', 'near', 'new', 'says', 'say'
    }

    # Fallback sources - crypto, war, general news (always available)
    CRYPTO_RSS_FEEDS = [
        ("https://cointelegraph.com/rss", "CoinTelegraph"),
        ("https://cryptonews.com/newsfeed", "CryptoNews"),
        ("https://bitcoinmagazine.com/.rss/full", "Bitcoin Magazine"),
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
    ]

    WAR_CONFLICT_RSS_FEEDS = [
        ("https://feeds.reuters.com/reuters/worldNews", "Reuters World"),
        ("https://apnews.com/rss/world-news", "AP World"),
        ("https://www.bbc.com/news/world/rss.xml", "BBC World"),
        ("https://feeds.skynews.com/feeds/rss/world.xml", "Sky News World"),
        ("https://www.aljazeera.com/xml/rss/all.xml", "Al Jazeera"),
        ("https://feeds.reuters.com/reuters/worldMostRead", "Reuters Global"),
    ]

    GENERAL_NEWS_RSS_FEEDS = [
        ("https://feeds.reuters.com/reuters/topNews", "Reuters Top"),
        ("https://apnews.com/rss/apf-topnews", "AP Top"),
        ("https://feeds.bbci.co.uk/news/rss.xml", "BBC News"),
        ("https://www.theguardian.com/world/rss", "Guardian World"),
        ("https://feeds.nbcnews.com/nbcnews/public/news", "NBC News"),
        ("https://www.newsweek.com/rss", "Newsweek"),
    ]

    FALLBACK_WINDOW_HOURS = 12  # Fallback sources can use articles up to 12h old

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
        """True when article has a publish time within strict latest-news window."""
        published = article.get('publishedAt') or ''
        published_dt = NewsService._parse_published_time(published)
        if published_dt == datetime.min.replace(tzinfo=timezone.utc):
            return False

        cutoff = datetime.now(timezone.utc) - timedelta(hours=NewsService.LATEST_NEWS_WINDOW_HOURS)
        return published_dt >= cutoff

    @staticmethod
    def _latest_from_time_iso() -> str:
        """Return ISO timestamp used for upstream API recency filtering."""
        return (
            datetime.now(timezone.utc) - timedelta(hours=NewsService.LATEST_NEWS_WINDOW_HOURS)
        ).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

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
    def _canonicalize_url(url: str) -> str:
        """Normalize URL for stronger duplicate detection."""
        raw = (url or '').strip()
        if not raw:
            return ''
        try:
            parsed = urlparse(raw)
            scheme = parsed.scheme.lower() or "https"
            netloc = parsed.netloc.lower().replace("www.", "")
            path = re.sub(r"/+", "/", parsed.path).rstrip("/")
            return urlunparse((scheme, netloc, path, "", "", ""))
        except Exception:
            return raw.lower()

    @staticmethod
    def _normalize_title_for_dedupe(title: str) -> str:
        """Normalize headline text so near-identical reposts collapse into one."""
        t = (title or "").lower().strip()
        # Remove common source suffixes like " - cnbc" or " | reuters".
        t = re.sub(r"\s*[-|]\s*(cnbc|reuters|bloomberg|marketwatch|yahoo finance|investing\.com)\s*$", "", t)
        # Remove non-alphanumerics for rough semantic dedupe.
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _story_signature(article: dict) -> str:
        """Create a semantic signature so same story across outlets clusters together."""
        text = NewsService._normalize_title_for_dedupe(article.get('title', ''))
        tokens = [tok for tok in text.split() if tok and tok not in NewsService.STORY_SIGNATURE_STOPWORDS]
        if not tokens:
            return ''

        prioritized = []
        for token in tokens:
            if token not in prioritized:
                prioritized.append(token)
            if len(prioritized) >= 8:
                break

        if len(prioritized) < NewsService.MIN_DISTINCT_TOKENS_FOR_SIGNATURE:
            return ''
        return "|".join(prioritized)

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
    def _newsapi_in_cooldown() -> bool:
        """True when NewsAPI is temporarily disabled due to recent 429."""
        if NewsService._newsapi_cooldown_until is None:
            return False
        return datetime.now(timezone.utc) < NewsService._newsapi_cooldown_until

    @staticmethod
    def _activate_newsapi_cooldown() -> None:
        """Back off NewsAPI calls for a while after hitting rate limits."""
        NewsService._newsapi_cooldown_until = datetime.now(timezone.utc) + timedelta(
            hours=NewsService.NEWSAPI_COOLDOWN_HOURS
        )
        logger.warning(
            "NewsAPI rate-limited; skipping NewsAPI fetches until %s UTC",
            NewsService._newsapi_cooldown_until.replace(microsecond=0).isoformat(),
        )

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
                    async with session.get(
                        feed_url,
                        headers=NewsService.REQUEST_HEADERS,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
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
    async def fetch_news_public_rss() -> list:
        """Fetch market headlines from public RSS feeds as a fallback path."""
        feed_urls = [
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://finance.yahoo.com/news/rssindex",
            "https://www.investing.com/rss/news_25.rss",
            "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
            "https://www.nasdaq.com/feed/rssoutbound?category=Commodities",
            "https://rss.cnn.com/rss/money_latest.rss",
            "https://rss.cnn.com/rss/money_markets.rss",
            "https://www.businessinsider.com/rss",
        ]
        articles = []

        try:
            async with aiohttp.ClientSession() as session:
                for feed_url in feed_urls:
                    try:
                        async with session.get(
                            feed_url,
                            headers=NewsService.REQUEST_HEADERS,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status != 200:
                                if 500 <= resp.status < 600:
                                    logger.warning(f"Public RSS returned {resp.status} for {feed_url}")
                                else:
                                    logger.info(f"Public RSS returned {resp.status} for {feed_url}")
                                continue

                            raw_xml = await resp.text()
                            root = ET.fromstring(raw_xml)

                            for item in root.findall(".//item"):
                                title = (item.findtext("title") or "").strip()
                                link = (item.findtext("link") or "").strip()
                                description = (item.findtext("description") or "").strip()
                                pub_date = (item.findtext("pubDate") or "").strip()
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
                                    'source': {'name': urlparse(link).netloc.replace("www.", "") or 'RSS'},
                                    'title': title,
                                    'description': description,
                                    'url': link,
                                    'publishedAt': published,
                                    'image_url': '',
                                    'video_url': '',
                                    'category': category,
                                }

                                if NewsService._is_trusted_source(
                                    normalized['source']['name'],
                                    normalized['url'],
                                ):
                                    articles.append(normalized)
                    except Exception as feed_err:
                        logger.info(f"Public RSS parse/fetch failed for {feed_url}: {feed_err}")
        except Exception as e:
            logger.error(f"Failed to fetch from public RSS feeds: {e}")

        return articles

    @staticmethod
    async def _fetch_rss_feed_articles(feed_url: str, source_name: str, category: str) -> list:
        """Fetch articles from a single RSS feed, tolerant of errors."""
        articles = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    feed_url,
                    headers=NewsService.REQUEST_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as resp:
                    if resp.status != 200:
                        return articles

                    raw_xml = await resp.text()
                    root = ET.fromstring(raw_xml)

                    for item in root.findall(".//item"):
                        title = (item.findtext("title") or "").strip()
                        link = (item.findtext("link") or "").strip()
                        description = (item.findtext("description") or "").strip()
                        pub_date = (item.findtext("pubDate") or "").strip()

                        if not title or not link:
                            continue

                        published = ""
                        if pub_date:
                            try:
                                published = parsedate_to_datetime(pub_date).isoformat()
                            except Exception:
                                published = pub_date

                        articles.append({
                            'source': {'name': source_name},
                            'title': title,
                            'description': description,
                            'url': link,
                            'publishedAt': published,
                            'image_url': '',
                            'video_url': NewsService._extract_video_from_rss_item(item),
                            'category': category,
                        })
        except Exception:
            pass

        return articles

    @staticmethod
    async def fetch_crypto_news() -> list:
        """Fetch crypto/blockchain news as a fallback source."""
        articles = []
        tasks = []
        for feed_url, source_name in NewsService.CRYPTO_RSS_FEEDS:
            tasks.append(NewsService._fetch_rss_feed_articles(feed_url, source_name, 'crypto'))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                articles.extend(result)

        # Trust crypto sources directly (they may not be in TRUSTED_NEWS_SOURCES).
        logger.info(f"Fetched {len(articles)} crypto articles from RSS feeds")
        return articles

    @staticmethod
    async def fetch_war_conflict_news() -> list:
        """Fetch war/conflict/world news as a fallback source."""
        articles = []
        tasks = []
        for feed_url, source_name in NewsService.WAR_CONFLICT_RSS_FEEDS:
            tasks.append(NewsService._fetch_rss_feed_articles(feed_url, source_name, 'world'))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                articles.extend(result)

        logger.info(f"Fetched {len(articles)} world/conflict articles from RSS feeds")
        return articles

    @staticmethod
    async def fetch_general_news() -> list:
        """Fetch general/world news as a last-resort fallback source."""
        articles = []
        tasks = []
        for feed_url, source_name in NewsService.GENERAL_NEWS_RSS_FEEDS:
            tasks.append(NewsService._fetch_rss_feed_articles(feed_url, source_name, 'general'))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                articles.extend(result)

        logger.info(f"Fetched {len(articles)} general articles from RSS feeds")
        return articles

    @staticmethod
    async def fetch_news_newsapi(keywords: list, category: str, per_keyword: int = 5) -> list:
        """Fetch curated news from NewsAPI.org for specific keywords."""
        articles = []
        if not NEWSAPI_KEY:
            return []
        if NewsService._newsapi_in_cooldown():
            return []

        try:
            async with aiohttp.ClientSession() as session:
                saw_rate_limit = False
                for keyword in keywords:
                    if saw_rate_limit:
                        break
                    from_time = NewsService._latest_from_time_iso()
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
                            if resp.status == 429:
                                saw_rate_limit = True
                                NewsService._activate_newsapi_cooldown()
        except Exception as e:
            logger.error(f"Failed to fetch from NewsAPI: {e}")

        return articles

    @staticmethod
    async def fetch_news_newsapi_compact() -> list:
        """Use two compact NewsAPI calls to reduce quota burn and 429 frequency."""
        if not NEWSAPI_KEY:
            return []
        if NewsService._newsapi_in_cooldown():
            return []

        query_sets = [
            ("(stock OR equities OR earnings OR guidance OR \"wall street\")", "market"),
            ("(oil OR gold OR silver OR copper OR \"natural gas\" OR opec OR commodities)", "commodities"),
        ]
        all_articles = []

        try:
            async with aiohttp.ClientSession() as session:
                from_time = (
                    NewsService._latest_from_time_iso()
                )

                for query, category in query_sets:
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q={quote(query)}"
                        "&language=en"
                        "&searchIn=title,description"
                        "&sortBy=publishedAt"
                        f"&from={quote(from_time)}"
                        "&pageSize=20"
                    )
                    headers = {"X-Api-Key": NEWSAPI_KEY}

                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for raw in data.get("articles", []):
                                normalized = NewsService._normalize_article(raw, category)
                                if not normalized['title'] or not normalized['url']:
                                    continue
                                if not NewsService._is_trusted_source(
                                    normalized['source']['name'],
                                    normalized['url'],
                                ):
                                    continue
                                all_articles.append(normalized)
                        else:
                            logger.warning(f"NewsAPI compact request returned {resp.status} for {category}")
                            if resp.status == 429:
                                NewsService._activate_newsapi_cooldown()
                                break
        except Exception as e:
            logger.error(f"Failed compact NewsAPI fetch: {e}")

        return all_articles

    @staticmethod
    async def fetch_news_alphavantage() -> list:
        """Fetch additional market news from Alpha Vantage."""
        articles = []
        if not ALPHAVANTAGE_KEY:
            return []

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
                            logger.info("Alpha Vantage returned no feed data")
                    else:
                        logger.warning(f"Alpha Vantage returned {resp.status}")
        except Exception as e:
            logger.error(f"Failed to fetch from Alpha Vantage: {e}")

        return articles

    @staticmethod
    async def fetch_all_news() -> list:
        """Aggregate and deduplicate market, crypto, war, and general news.

        Priority order:
        1. Primary financial sources (CNBC, RSS, AlphaVantage, NewsAPI)
        2. Crypto/blockchain news (fallback)
        3. War/conflict/world news (fallback)
        4. General news (last resort)

        If primary sources are empty or stale, falls back to secondary sources.
        Never returns empty unless ALL sources fail.
        """
        try:
            # ---- Step 1: Fetch primary financial sources ----
            cnbc_articles = await NewsService.fetch_news_cnbc_rss()
            public_rss_articles = await NewsService.fetch_news_public_rss()
            compact_newsapi = await NewsService.fetch_news_newsapi_compact()
            av_articles = await NewsService.fetch_news_alphavantage()

            market_articles = []
            commodity_articles = []
            if not compact_newsapi and not NewsService._newsapi_in_cooldown():
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
                market_articles = await NewsService.fetch_news_newsapi(
                    market_keywords, category='market', per_keyword=2
                )
                commodity_articles = await NewsService.fetch_news_newsapi(
                    commodity_keywords, category='commodities', per_keyword=2
                )

            primary_articles = (
                cnbc_articles
                + public_rss_articles
                + compact_newsapi
                + market_articles
                + commodity_articles
                + av_articles
            )
            logger.info(
                "Primary sources | CNBC=%d PublicRSS=%d NewsAPICompact=%d NewsAPI=%d AlphaVantage=%d",
                len(cnbc_articles),
                len(public_rss_articles),
                len(compact_newsapi),
                len(market_articles) + len(commodity_articles),
                len(av_articles),
            )

            # Deduplicate primary articles
            primary_deduped = NewsService._deduplicate_articles(primary_articles)

            # Filter by recency (3h window for primary financial sources)
            primary_fresh = [a for a in primary_deduped if NewsService._is_recent_article(a)]

            if primary_fresh:
                logger.info(f"Primary sources returned {len(primary_fresh)} fresh articles")
                return NewsService._select_and_balance_articles(primary_fresh)

            # ---- Step 2: Primary sources are dry - fetch fallback sources ----
            logger.warning("Primary financial sources returned no fresh articles. Fetching fallback sources.")

            crypto_articles = await NewsService.fetch_crypto_news()
            war_articles = await NewsService.fetch_war_conflict_news()
            general_articles = await NewsService.fetch_general_news()

            fallback_articles = crypto_articles + war_articles + general_articles
            logger.info(
                "Fallback sources | Crypto=%d World=%d General=%d",
                len(crypto_articles),
                len(war_articles),
                len(general_articles),
            )

            # Deduplicate fallback articles
            fallback_deduped = NewsService._deduplicate_articles(fallback_articles)

            # Use wider recency window for fallback sources
            if NewsService.FALLBACK_WINDOW_HOURS > NewsService.LATEST_NEWS_WINDOW_HOURS:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=NewsService.FALLBACK_WINDOW_HOURS)
                fallback_fresh = [
                    a for a in fallback_deduped
                    if NewsService._parse_published_time(a.get('publishedAt', '')) >= cutoff
                ]
            else:
                fallback_fresh = fallback_deduped

            if fallback_fresh:
                logger.info(f"Fallback sources returned {len(fallback_fresh)} articles after dedup+recency filter")
                return NewsService._select_and_balance_articles(fallback_fresh)

            logger.error("ALL news sources (primary + fallback) returned empty results")
            return []

        except Exception as e:
            logger.error(f"Failed to fetch all news: {e}")
            return []

    @staticmethod
    def _deduplicate_articles(articles: list) -> list:
        """Deduplicate articles by URL and semantic story signature."""
        clustered_by_story = {}
        seen_urls = set()

        for article in articles:
            url = NewsService._canonicalize_url(article.get('url', ''))
            if not url or url in seen_urls:
                continue
            article['url'] = url
            seen_urls.add(url)

            title_norm = NewsService._normalize_title_for_dedupe(article.get('title', ''))
            title_hash = hashlib.md5(title_norm.encode()).hexdigest()
            story_sig = NewsService._story_signature(article) or title_hash
            published = NewsService._parse_published_time(article.get('publishedAt', ''))
            candidate_rank = (published, NewsService._article_score(article))

            current = clustered_by_story.get(story_sig)
            if current is None:
                clustered_by_story[story_sig] = article
                continue

            current_published = NewsService._parse_published_time(current.get('publishedAt', ''))
            current_rank = (current_published, NewsService._article_score(current))
            if candidate_rank > current_rank:
                clustered_by_story[story_sig] = article

        return list(clustered_by_story.values())

    @staticmethod
    def _select_and_balance_articles(articles: list) -> list:
        """Select and balance articles across categories and sources."""
        def sort_key(article):
            published = NewsService._parse_published_time(article.get('publishedAt', ''))
            return (published, NewsService._article_score(article))

        # Sort by category
        category_sorted = {
            'market': sorted([a for a in articles if a.get('category') == 'market'], key=sort_key, reverse=True),
            'commodities': sorted([a for a in articles if a.get('category') == 'commodities'], key=sort_key, reverse=True),
            'crypto': sorted([a for a in articles if a.get('category') == 'crypto'], key=sort_key, reverse=True),
            'world': sorted([a for a in articles if a.get('category') == 'world'], key=sort_key, reverse=True),
            'general': sorted([a for a in articles if a.get('category') == 'general'], key=sort_key, reverse=True),
        }

        # Prefer financial content, then mix in other categories
        selected = (
            category_sorted['market'][:3]
            + category_sorted['commodities'][:2]
            + category_sorted['crypto'][:2]
            + category_sorted['world'][:2]
            + category_sorted['general'][:1]
        )

        # Fill remaining slots if needed
        if len(selected) < 6:
            seen_urls = {NewsService._canonicalize_url(a.get('url', '')) for a in selected}
            all_ranked = sorted(articles, key=sort_key, reverse=True)
            for candidate in all_ranked:
                url = NewsService._canonicalize_url(candidate.get('url', ''))
                if url and url not in seen_urls:
                    selected.append(candidate)
                    seen_urls.add(url)
                if len(selected) >= 7:
                    break

        # Enforce source diversity
        source_balanced = []
        source_counts = {}
        seen_urls_balanced = set()
        for candidate in sorted(selected, key=sort_key, reverse=True):
            source = (candidate.get('source', {}).get('name', 'unknown') or 'unknown').lower()
            url = NewsService._canonicalize_url(candidate.get('url', ''))
            if not url or url in seen_urls_balanced:
                continue
            if source_counts.get(source, 0) >= NewsService.SOURCE_MAX_PER_CYCLE:
                continue
            source_balanced.append(candidate)
            source_counts[source] = source_counts.get(source, 0) + 1
            seen_urls_balanced.add(url)
            if len(source_balanced) >= 7:
                break

        if len(source_balanced) < 3:
            # If balancing over-restricts, fall back to top-ranked.
            return selected[:7]
        return source_balanced[:7]

    @staticmethod
    def make_news_id(article: dict) -> str:
        """Generate stable cache key for an article."""
        published = NewsService._parse_published_time(article.get('publishedAt', ''))
        day_bucket = ""
        if published != datetime.min.replace(tzinfo=timezone.utc):
            day_bucket = published.strftime("%Y-%m-%d")

        signature = NewsService._story_signature(article)
        if not signature:
            signature = NewsService._normalize_title_for_dedupe(article.get('title', ''))

        category = (article.get('category') or 'market').lower()
        seed = f"{category}|{day_bucket}|{signature}".lower().strip()
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
