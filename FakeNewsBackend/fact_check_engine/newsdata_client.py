import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
import json

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class NewsDataArticle:
    """Data model for a news article from NewsData.io."""
    title: str
    link: str
    description: Optional[str]
    content: Optional[str]
    pub_date: datetime
    source_id: str
    source_name: str
    country: List[str]
    category: List[str]
    language: str
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    creator: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsDataArticle':
        """Create NewsDataArticle from API response data."""
        # Parse publication date
        pub_date = datetime.now()
        if data.get('pubDate'):
            try:
                pub_date = datetime.fromisoformat(
                    data['pubDate'].replace('Z', '+00:00')
                )
            except (ValueError, AttributeError):
                pass
        
        return cls(
            title=data.get('title', ''),
            link=data.get('link', ''),
            description=data.get('description'),
            content=data.get('content'),
            pub_date=pub_date,
            source_id=data.get('source_id', ''),
            source_name=data.get('source_name', 'Unknown'),
            country=data.get('country', []),
            category=data.get('category', []),
            language=data.get('language', 'en'),
            image_url=data.get('image_url'),
            video_url=data.get('video_url'),
            creator=data.get('creator'),
            keywords=data.get('keywords')
        )


@dataclass
class NewsDataSearchResult:
    """Container for NewsData.io search results."""
    articles: List[NewsDataArticle]
    total_results: int
    next_page: Optional[str]
    query: str
    
    @classmethod
    def from_response(cls, response_data: Dict[str, Any], query: str) -> 'NewsDataSearchResult':
        """Create NewsDataSearchResult from API response."""
        articles = [
            NewsDataArticle.from_dict(article_data)
            for article_data in response_data.get('results', [])
        ]
        
        return cls(
            articles=articles,
            total_results=response_data.get('totalResults', len(articles)),
            next_page=response_data.get('nextPage'),
            query=query
        )


class NewsDataError(Exception):
    """Custom exception for NewsData.io errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class NewsDataRateLimiter:
    """Rate limiter for NewsData.io API requests (daily limit)."""
    
    def __init__(self, max_requests: int = 200):
        self.max_requests = max_requests
        self.requests = []
    
    async def acquire(self) -> bool:
        """Check if request can be made within daily rate limits."""
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests for current day."""
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        recent_requests = [
            req_time for req_time in self.requests 
            if req_time > cutoff
        ]
        
        return max(0, self.max_requests - len(recent_requests))


class NewsDataClient:
    """Async client for NewsData.io API with rate limiting and error handling."""
    
    BASE_URL = "https://newsdata.io/api/1"
    
    def __init__(self, api_key: Optional[str] = None, session: Optional[ClientSession] = None):
        """
        Initialize NewsData.io client.
        
        Args:
            api_key: NewsData.io API key, defaults to config value
            session: Optional aiohttp session to reuse
        """
        self.api_key = api_key or config.api.newsdata_api_key
        self.session = session
        self._own_session = session is None
        
        self.rate_limiter = NewsDataRateLimiter(
            max_requests=config.api.newsdata_api_rate_limit
        )
        
        self.timeout = ClientTimeout(total=config.api.request_timeout)
        
        if not self.api_key:
            raise ValueError("NewsData.io API key is required")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self._own_session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._own_session and self.session:
            await self.session.close()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        retries: int = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to NewsData.io API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            retries: Number of retries, defaults to config value
            
        Returns:
            API response data
            
        Raises:
            NewsDataError: On API errors or rate limiting
        """
        if not await self.rate_limiter.acquire():
            remaining = self.rate_limiter.get_remaining_requests()
            raise NewsDataError(f"Daily rate limit exceeded. Remaining: {remaining}")
        
        retries = retries if retries is not None else config.api.max_retries
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Add API key to parameters
        params = params.copy()
        params['apikey'] = self.api_key
        
        headers = {
            "User-Agent": "FactCheckEngine/1.0"
        }
        
        for attempt in range(retries + 1):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession(timeout=self.timeout)
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        if response_data.get('status') == 'success':
                            logger.debug(f"NewsData.io request successful: {endpoint}")
                            return response_data
                        else:
                            error_msg = response_data.get('message', 'Unknown API error')
                            raise NewsDataError(
                                f"NewsData.io error: {error_msg}",
                                response.status,
                                response_data
                            )
                    
                    elif response.status == 429:  # Rate limited
                        if attempt < retries:
                            wait_time = config.api.retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise NewsDataError("Rate limit exceeded", response.status)
                    
                    elif response.status == 401:
                        raise NewsDataError("Invalid API key", response.status)
                    
                    elif response.status == 400:
                        error_msg = response_data.get('message', 'Bad request')
                        raise NewsDataError(f"Bad request: {error_msg}", response.status)
                    
                    elif response.status >= 500:
                        if attempt < retries:
                            wait_time = config.api.retry_delay * (2 ** attempt)
                            logger.warning(f"Server error, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise NewsDataError(f"Server error: {response.status}", response.status)
                    
                    else:
                        raise NewsDataError(f"HTTP error: {response.status}", response.status)
            
            except ClientError as e:
                if attempt < retries:
                    wait_time = config.api.retry_delay * (2 ** attempt)
                    logger.warning(f"Network error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise NewsDataError(f"Network error: {e}")
            
            except json.JSONDecodeError as e:
                raise NewsDataError(f"Invalid JSON response: {e}")
    
    async def search_news(
        self,
        query: str,
        language: str = "en",
        country: Optional[str] = None,
        category: Optional[str] = None,
        size: int = 10,
        page: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        domain: Optional[str] = None,
        exclude_domain: Optional[str] = None,
        sentiment: Optional[str] = None
    ) -> NewsDataSearchResult:
        """
        Search for news articles using NewsData.io.
        
        Args:
            query: Search query
            language: Language code (en, es, fr, etc.)
            country: Country code (us, uk, ca, etc.)
            category: News category (business, entertainment, etc.)
            size: Number of results (max 50 for free tier)
            page: Pagination token
            from_date: Start date for search
            to_date: End date for search
            domain: Specific domain to search
            exclude_domain: Domain to exclude
            sentiment: Sentiment filter (positive, negative, neutral)
            
        Returns:
            NewsDataSearchResult with articles and metadata
        """
        params = {
            "q": query,
            "language": language,
            "size": min(size, 50)  # API limit
        }
        
        if country:
            params["country"] = country
        if category:
            params["category"] = category
        if page:
            params["page"] = page
        if from_date:
            params["from_date"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["to_date"] = to_date.strftime("%Y-%m-%d")
        if domain:
            params["domain"] = domain
        if exclude_domain:
            params["excludedomain"] = exclude_domain
        if sentiment:
            params["sentiment"] = sentiment
        
        logger.info(f"Searching NewsData.io for: {query}")
        response_data = await self._make_request("news", params)
        
        return NewsDataSearchResult.from_response(response_data, query)
    
    async def get_latest_news(
        self,
        language: str = "en",
        country: Optional[str] = None,
        category: Optional[str] = None,
        size: int = 10,
        page: Optional[str] = None
    ) -> NewsDataSearchResult:
        """
        Get latest news articles from NewsData.io.
        
        Args:
            language: Language code
            country: Country code
            category: News category
            size: Number of results
            page: Pagination token
            
        Returns:
            NewsDataSearchResult with latest articles
        """
        params = {
            "language": language,
            "size": min(size, 50)
        }
        
        if country:
            params["country"] = country
        if category:
            params["category"] = category
        if page:
            params["page"] = page
        
        logger.info(f"Getting latest news from NewsData.io")
        response_data = await self._make_request("news", params)
        
        return NewsDataSearchResult.from_response(response_data, "latest-news")
    
    async def search_claim_related(
        self,
        claim: str,
        max_results: int = 30,
        languages: List[str] = None
    ) -> List[NewsDataArticle]:
        """
        Search for articles related to a specific claim across multiple languages.
        
        Args:
            claim: The claim to search for
            max_results: Maximum number of articles to return
            languages: List of language codes to search
            
        Returns:
            List of relevant news articles
        """
        if languages is None:
            languages = ["en"]  # Default to English
        
        all_articles = []
        seen_links = set()
        
        for language in languages:
            if len(all_articles) >= max_results:
                break
            
            # Search with different query variations
            search_queries = [
                claim,
                f'"{claim}"',  # Exact phrase
                # Extract key terms for broader search
                " ".join([word for word in claim.split() if len(word) > 3])[:100]
            ]
            
            for query in search_queries:
                if len(all_articles) >= max_results:
                    break
                
                try:
                    result = await self.search_news(
                        query=query,
                        language=language,
                        size=min(10, max_results - len(all_articles)),
                        from_date=datetime.now() - timedelta(days=365)  # Last year
                    )
                    
                    for article in result.articles:
                        if article.link not in seen_links and len(all_articles) < max_results:
                            all_articles.append(article)
                            seen_links.add(article.link)
                
                except NewsDataError as e:
                    logger.warning(f"Search failed for query '{query}' in {language}: {e}")
                    continue
        
        logger.info(f"Found {len(all_articles)} unique articles for claim across {len(languages)} languages")
        return all_articles[:max_results]
    
    async def get_sources(self, language: str = "en", country: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available news sources from NewsData.io.
        
        Args:
            language: Language code
            country: Country code
            
        Returns:
            List of available news sources
        """
        params = {
            "language": language
        }
        
        if country:
            params["country"] = country
        
        try:
            response_data = await self._make_request("sources", params)
            return response_data.get('results', [])
        except NewsDataError as e:
            logger.error(f"Failed to get sources: {e}")
            return []
    
    async def get_remaining_quota(self) -> int:
        """Get remaining API quota for current day."""
        return self.rate_limiter.get_remaining_requests()


## Suggestions for Upgrade:
# 1. Implement multi-language claim translation for global fact-checking coverage
# 2. Add source credibility scoring based on historical accuracy and bias analysis
# 3. Integrate with content extraction services to get full article text for analysis
# 4. Add real-time news monitoring with webhook support for breaking news alerts