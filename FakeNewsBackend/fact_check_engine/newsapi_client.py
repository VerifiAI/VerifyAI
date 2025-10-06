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
class NewsArticle:
    """Data model for a news article from NewsAPI."""
    title: str
    description: Optional[str]
    url: str
    source: str
    published_at: datetime
    author: Optional[str] = None
    content: Optional[str] = None
    url_to_image: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create NewsArticle from API response data."""
        return cls(
            title=data.get('title', ''),
            description=data.get('description'),
            url=data.get('url', ''),
            source=data.get('source', {}).get('name', 'Unknown'),
            published_at=datetime.fromisoformat(
                data.get('publishedAt', '').replace('Z', '+00:00')
            ) if data.get('publishedAt') else datetime.now(),
            author=data.get('author'),
            content=data.get('content'),
            url_to_image=data.get('urlToImage')
        )


@dataclass
class NewsSearchResult:
    """Container for news search results."""
    articles: List[NewsArticle]
    total_results: int
    status: str
    query: str
    
    @classmethod
    def from_response(cls, response_data: Dict[str, Any], query: str) -> 'NewsSearchResult':
        """Create NewsSearchResult from API response."""
        articles = [
            NewsArticle.from_dict(article_data) 
            for article_data in response_data.get('articles', [])
        ]
        
        return cls(
            articles=articles,
            total_results=response_data.get('totalResults', 0),
            status=response_data.get('status', 'unknown'),
            query=query
        )


class NewsAPIError(Exception):
    """Custom exception for NewsAPI errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_requests: int, time_window: int = 86400):  # 24 hours default
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self) -> bool:
        """Check if request can be made within rate limits."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True


class NewsAPIClient:
    """Async client for NewsAPI.org with rate limiting and error handling."""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: Optional[str] = None, session: Optional[ClientSession] = None):
        """
        Initialize NewsAPI client.
        
        Args:
            api_key: NewsAPI key, defaults to config value
            session: Optional aiohttp session to reuse
        """
        self.api_key = api_key or config.api.news_api_key
        self.session = session
        self._own_session = session is None
        
        self.rate_limiter = RateLimiter(
            max_requests=config.api.news_api_rate_limit,
            time_window=86400  # 24 hours
        )
        
        self.timeout = ClientTimeout(total=config.api.request_timeout)
        
        if not self.api_key:
            raise ValueError("NewsAPI key is required")
    
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
        Make an authenticated request to NewsAPI.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            retries: Number of retries, defaults to config value
            
        Returns:
            API response data
            
        Raises:
            NewsAPIError: On API errors or rate limiting
        """
        if not await self.rate_limiter.acquire():
            raise NewsAPIError("Rate limit exceeded for NewsAPI")
        
        retries = retries if retries is not None else config.api.max_retries
        url = f"{self.BASE_URL}/{endpoint}"
        
        headers = {
            "X-Api-Key": self.api_key,
            "User-Agent": "FactCheckEngine/1.0"
        }
        
        for attempt in range(retries + 1):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession(timeout=self.timeout)
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        if response_data.get('status') == 'ok':
                            logger.debug(f"NewsAPI request successful: {endpoint}")
                            return response_data
                        else:
                            error_msg = response_data.get('message', 'Unknown API error')
                            raise NewsAPIError(
                                f"NewsAPI error: {error_msg}",
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
                            raise NewsAPIError("Rate limit exceeded", response.status)
                    
                    elif response.status == 401:
                        raise NewsAPIError("Invalid API key", response.status)
                    
                    elif response.status >= 500:
                        if attempt < retries:
                            wait_time = config.api.retry_delay * (2 ** attempt)
                            logger.warning(f"Server error, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise NewsAPIError(f"Server error: {response.status}", response.status)
                    
                    else:
                        raise NewsAPIError(f"HTTP error: {response.status}", response.status)
            
            except ClientError as e:
                if attempt < retries:
                    wait_time = config.api.retry_delay * (2 ** attempt)
                    logger.warning(f"Network error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise NewsAPIError(f"Network error: {e}")
            
            except json.JSONDecodeError as e:
                raise NewsAPIError(f"Invalid JSON response: {e}")
    
    async def search_everything(
        self,
        query: str,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int = 20,
        page: int = 1,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> NewsSearchResult:
        """
        Search for articles using the everything endpoint.
        
        Args:
            query: Search query
            language: Language code (default: en)
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of results per page (max 100)
            page: Page number
            from_date: Earliest article date
            to_date: Latest article date
            domains: Domains to search within
            exclude_domains: Domains to exclude
            
        Returns:
            NewsSearchResult with articles and metadata
        """
        params = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "page": page
        }
        
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        if domains:
            params["domains"] = ",".join(domains)
        if exclude_domains:
            params["excludeDomains"] = ",".join(exclude_domains)
        
        logger.info(f"Searching NewsAPI for: {query}")
        response_data = await self._make_request("everything", params)
        
        return NewsSearchResult.from_response(response_data, query)
    
    async def get_top_headlines(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        country: str = "us",
        page_size: int = 20,
        page: int = 1
    ) -> NewsSearchResult:
        """
        Get top headlines from NewsAPI.
        
        Args:
            query: Search query
            category: News category
            country: Country code
            page_size: Number of results per page
            page: Page number
            
        Returns:
            NewsSearchResult with top headlines
        """
        params = {
            "country": country,
            "pageSize": min(page_size, 100),
            "page": page
        }
        
        if query:
            params["q"] = query
        if category:
            params["category"] = category
        
        logger.info(f"Getting top headlines: {query or 'general'}")
        response_data = await self._make_request("top-headlines", params)
        
        return NewsSearchResult.from_response(response_data, query or "top-headlines")
    
    async def search_claim_related(
        self,
        claim: str,
        max_results: int = 50
    ) -> List[NewsArticle]:
        """
        Search for articles related to a specific claim.
        
        Args:
            claim: The claim to search for
            max_results: Maximum number of articles to return
            
        Returns:
            List of relevant news articles
        """
        # Search with different strategies to get comprehensive results
        search_queries = [
            claim,  # Direct search
            f'"{claim}"',  # Exact phrase search
            # Extract key terms for broader search
            " ".join([word for word in claim.split() if len(word) > 3])[:100]
        ]
        
        all_articles = []
        seen_urls = set()
        
        for query in search_queries:
            if len(all_articles) >= max_results:
                break
                
            try:
                result = await self.search_everything(
                    query=query,
                    page_size=min(20, max_results - len(all_articles)),
                    sort_by="relevancy"
                )
                
                for article in result.articles:
                    if article.url not in seen_urls and len(all_articles) < max_results:
                        all_articles.append(article)
                        seen_urls.add(article.url)
                        
            except NewsAPIError as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        logger.info(f"Found {len(all_articles)} unique articles for claim")
        return all_articles[:max_results]


## Suggestions for Upgrade:
# 1. Implement intelligent query expansion using NLP techniques to find more relevant articles
# 2. Add article content extraction and full-text analysis capabilities for better fact-checking
# 3. Implement caching layer with Redis to reduce API calls and improve performance
# 4. Add support for real-time news monitoring and webhook notifications for breaking news