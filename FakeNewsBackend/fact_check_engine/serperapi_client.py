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
class SearchResult:
    """Data model for a search result from SerperAPI."""
    title: str
    link: str
    snippet: str
    position: int
    source: Optional[str] = None
    date: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], position: int) -> 'SearchResult':
        """Create SearchResult from API response data."""
        # Parse date if available
        date = None
        if 'date' in data:
            try:
                date = datetime.fromisoformat(data['date'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        return cls(
            title=data.get('title', ''),
            link=data.get('link', ''),
            snippet=data.get('snippet', ''),
            position=position,
            source=data.get('source'),
            date=date
        )


@dataclass
class NewsResult:
    """Data model for a news result from SerperAPI."""
    title: str
    link: str
    snippet: str
    date: Optional[datetime]
    source: str
    image_url: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsResult':
        """Create NewsResult from API response data."""
        # Parse date
        date = None
        if 'date' in data:
            try:
                date = datetime.fromisoformat(data['date'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        return cls(
            title=data.get('title', ''),
            link=data.get('link', ''),
            snippet=data.get('snippet', ''),
            date=date,
            source=data.get('source', 'Unknown'),
            image_url=data.get('imageUrl')
        )


@dataclass
class SerperSearchResponse:
    """Container for SerperAPI search response."""
    organic_results: List[SearchResult]
    news_results: List[NewsResult]
    total_results: Optional[int]
    query: str
    search_time: float
    
    @classmethod
    def from_response(cls, response_data: Dict[str, Any], query: str) -> 'SerperSearchResponse':
        """Create SerperSearchResponse from API response."""
        organic_results = [
            SearchResult.from_dict(result, idx + 1)
            for idx, result in enumerate(response_data.get('organic', []))
        ]
        
        news_results = [
            NewsResult.from_dict(result)
            for result in response_data.get('news', [])
        ]
        
        return cls(
            organic_results=organic_results,
            news_results=news_results,
            total_results=response_data.get('searchInformation', {}).get('totalResults'),
            query=query,
            search_time=response_data.get('searchInformation', {}).get('searchTime', 0.0)
        )


class SerperAPIError(Exception):
    """Custom exception for SerperAPI errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class SerperRateLimiter:
    """Rate limiter for SerperAPI requests (monthly limit)."""
    
    def __init__(self, max_requests: int = 2500):
        self.max_requests = max_requests
        self.requests = []
    
    async def acquire(self) -> bool:
        """Check if request can be made within monthly rate limits."""
        now = datetime.now()
        # Keep requests from current month
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        self.requests = [req_time for req_time in self.requests if req_time >= current_month_start]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests for current month."""
        now = datetime.now()
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        current_month_requests = [
            req_time for req_time in self.requests 
            if req_time >= current_month_start
        ]
        
        return max(0, self.max_requests - len(current_month_requests))


class SerperAPIClient:
    """Async client for SerperAPI Google Search with rate limiting and error handling."""
    
    BASE_URL = "https://google.serper.dev"
    
    def __init__(self, api_key: Optional[str] = None, session: Optional[ClientSession] = None):
        """
        Initialize SerperAPI client.
        
        Args:
            api_key: SerperAPI key, defaults to config value
            session: Optional aiohttp session to reuse
        """
        self.api_key = api_key or config.api.serper_api_key
        self.session = session
        self._own_session = session is None
        
        self.rate_limiter = SerperRateLimiter(
            max_requests=config.api.serper_api_rate_limit
        )
        
        self.timeout = ClientTimeout(total=config.api.request_timeout)
        
        if not self.api_key:
            raise ValueError("SerperAPI key is required")
    
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
        payload: Dict[str, Any],
        retries: int = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to SerperAPI.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            retries: Number of retries, defaults to config value
            
        Returns:
            API response data
            
        Raises:
            SerperAPIError: On API errors or rate limiting
        """
        if not await self.rate_limiter.acquire():
            remaining = self.rate_limiter.get_remaining_requests()
            raise SerperAPIError(f"Monthly rate limit exceeded. Remaining: {remaining}")
        
        retries = retries if retries is not None else config.api.max_retries
        url = f"{self.BASE_URL}/{endpoint}"
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "FactCheckEngine/1.0"
        }
        
        for attempt in range(retries + 1):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession(timeout=self.timeout)
                
                async with self.session.post(url, json=payload, headers=headers) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        logger.debug(f"SerperAPI request successful: {endpoint}")
                        return response_data
                    
                    elif response.status == 429:  # Rate limited
                        if attempt < retries:
                            wait_time = config.api.retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise SerperAPIError("Rate limit exceeded", response.status)
                    
                    elif response.status == 401:
                        raise SerperAPIError("Invalid API key", response.status)
                    
                    elif response.status == 400:
                        error_msg = response_data.get('message', 'Bad request')
                        raise SerperAPIError(f"Bad request: {error_msg}", response.status)
                    
                    elif response.status >= 500:
                        if attempt < retries:
                            wait_time = config.api.retry_delay * (2 ** attempt)
                            logger.warning(f"Server error, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise SerperAPIError(f"Server error: {response.status}", response.status)
                    
                    else:
                        raise SerperAPIError(f"HTTP error: {response.status}", response.status)
            
            except ClientError as e:
                if attempt < retries:
                    wait_time = config.api.retry_delay * (2 ** attempt)
                    logger.warning(f"Network error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise SerperAPIError(f"Network error: {e}")
            
            except json.JSONDecodeError as e:
                raise SerperAPIError(f"Invalid JSON response: {e}")
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        country: str = "us",
        language: str = "en",
        time_range: Optional[str] = None,
        safe_search: str = "moderate"
    ) -> SerperSearchResponse:
        """
        Perform a Google search using SerperAPI.
        
        Args:
            query: Search query
            num_results: Number of results to return (max 100)
            country: Country code for localized results
            language: Language code
            time_range: Time range filter (h, d, w, m, y)
            safe_search: Safe search setting (off, moderate, strict)
            
        Returns:
            SerperSearchResponse with search results
        """
        payload = {
            "q": query,
            "num": min(num_results, 100),
            "gl": country,
            "hl": language,
            "safe": safe_search
        }
        
        if time_range:
            payload["tbs"] = f"qdr:{time_range}"
        
        logger.info(f"Searching Google via SerperAPI for: {query}")
        response_data = await self._make_request("search", payload)
        
        return SerperSearchResponse.from_response(response_data, query)
    
    async def news_search(
        self,
        query: str,
        num_results: int = 10,
        country: str = "us",
        language: str = "en",
        time_range: Optional[str] = None
    ) -> List[NewsResult]:
        """
        Search for news articles using SerperAPI.
        
        Args:
            query: Search query
            num_results: Number of results to return
            country: Country code for localized results
            language: Language code
            time_range: Time range filter (h, d, w, m, y)
            
        Returns:
            List of news results
        """
        payload = {
            "q": query,
            "num": min(num_results, 100),
            "gl": country,
            "hl": language,
            "tbm": "nws"  # News search
        }
        
        if time_range:
            payload["tbs"] = f"qdr:{time_range}"
        
        logger.info(f"Searching Google News via SerperAPI for: {query}")
        response_data = await self._make_request("search", payload)
        
        news_results = [
            NewsResult.from_dict(result)
            for result in response_data.get('news', [])
        ]
        
        return news_results
    
    async def fact_check_search(
        self,
        claim: str,
        max_results: int = 20
    ) -> List[SearchResult]:
        """
        Search for fact-checking information about a claim.
        
        Args:
            claim: The claim to fact-check
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant search results
        """
        # Create fact-checking specific queries
        fact_check_queries = [
            f"{claim} fact check",
            f"{claim} debunked",
            f"{claim} verified",
            f"{claim} snopes",
            f"{claim} politifact",
            f'"{claim}" true false'
        ]
        
        all_results = []
        seen_links = set()
        
        for query in fact_check_queries:
            if len(all_results) >= max_results:
                break
            
            try:
                response = await self.search(
                    query=query,
                    num_results=min(10, max_results - len(all_results)),
                    time_range="y"  # Last year for recent fact-checks
                )
                
                for result in response.organic_results:
                    if result.link not in seen_links and len(all_results) < max_results:
                        # Prioritize known fact-checking sites
                        fact_check_domains = [
                            'snopes.com', 'politifact.com', 'factcheck.org',
                            'reuters.com', 'apnews.com', 'bbc.com',
                            'washingtonpost.com', 'nytimes.com'
                        ]
                        
                        is_fact_checker = any(domain in result.link.lower() for domain in fact_check_domains)
                        if is_fact_checker:
                            all_results.insert(0, result)  # Prioritize fact-checkers
                        else:
                            all_results.append(result)
                        
                        seen_links.add(result.link)
                
            except SerperAPIError as e:
                logger.warning(f"Fact-check search failed for query '{query}': {e}")
                continue
        
        logger.info(f"Found {len(all_results)} fact-check results for claim")
        return all_results[:max_results]
    
    async def get_remaining_quota(self) -> int:
        """Get remaining API quota for current month."""
        return self.rate_limiter.get_remaining_requests()


## Suggestions for Upgrade:
# 1. Implement intelligent query optimization using machine learning to improve search relevance
# 2. Add support for image search and reverse image lookup for visual fact-checking
# 3. Integrate with Google's Fact Check Tools API for enhanced fact-checking capabilities
# 4. Add domain reputation scoring to weight results from more credible sources higher