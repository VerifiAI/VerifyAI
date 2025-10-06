import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from aiohttp import ClientSession, ClientError

from fact_check_engine.serperapi_client import (
    SerperAPIClient, SearchResult, NewsResult, SerperSearchResponse, 
    SerperAPIError, SerperRateLimiter
)


class TestSearchResult:
    """Test cases for SearchResult data model."""
    
    def test_search_result_from_dict(self):
        """Test SearchResult creation from API response data."""
        data = {
            "title": "Test Search Result",
            "link": "https://example.com/result",
            "snippet": "This is a test snippet",
            "source": "Example Source",
            "date": "2024-01-15T10:00:00Z"
        }
        
        result = SearchResult.from_dict(data, position=1)
        
        assert result.title == "Test Search Result"
        assert result.link == "https://example.com/result"
        assert result.snippet == "This is a test snippet"
        assert result.position == 1
        assert result.source == "Example Source"
        assert isinstance(result.date, datetime)
    
    def test_search_result_from_dict_minimal(self):
        """Test SearchResult creation with minimal data."""
        data = {
            "title": "Minimal Result",
            "link": "https://example.com/minimal",
            "snippet": "Minimal snippet"
        }
        
        result = SearchResult.from_dict(data, position=5)
        
        assert result.title == "Minimal Result"
        assert result.link == "https://example.com/minimal"
        assert result.snippet == "Minimal snippet"
        assert result.position == 5
        assert result.source is None
        assert result.date is None
    
    def test_search_result_invalid_date(self):
        """Test SearchResult handles invalid date gracefully."""
        data = {
            "title": "Test Result",
            "link": "https://example.com/result",
            "snippet": "Test snippet",
            "date": "invalid-date"
        }
        
        result = SearchResult.from_dict(data, position=1)
        
        assert result.title == "Test Result"
        assert result.date is None


class TestNewsResult:
    """Test cases for NewsResult data model."""
    
    def test_news_result_from_dict(self):
        """Test NewsResult creation from API response data."""
        data = {
            "title": "Breaking News",
            "link": "https://news.example.com/breaking",
            "snippet": "This is breaking news",
            "source": "News Source",
            "date": "2024-01-15T12:00:00Z",
            "imageUrl": "https://example.com/image.jpg"
        }
        
        result = NewsResult.from_dict(data)
        
        assert result.title == "Breaking News"
        assert result.link == "https://news.example.com/breaking"
        assert result.snippet == "This is breaking news"
        assert result.source == "News Source"
        assert result.image_url == "https://example.com/image.jpg"
        assert isinstance(result.date, datetime)
    
    def test_news_result_from_dict_minimal(self):
        """Test NewsResult creation with minimal data."""
        data = {
            "title": "Simple News",
            "link": "https://news.example.com/simple",
            "snippet": "Simple news snippet"
        }
        
        result = NewsResult.from_dict(data)
        
        assert result.title == "Simple News"
        assert result.link == "https://news.example.com/simple"
        assert result.snippet == "Simple news snippet"
        assert result.source == "Unknown"
        assert result.date is None
        assert result.image_url is None


class TestSerperSearchResponse:
    """Test cases for SerperSearchResponse container."""
    
    def test_serper_search_response_from_response(self):
        """Test SerperSearchResponse creation from API response."""
        response_data = {
            "organic": [
                {
                    "title": "Organic Result 1",
                    "link": "https://example.com/1",
                    "snippet": "First organic result"
                },
                {
                    "title": "Organic Result 2", 
                    "link": "https://example.com/2",
                    "snippet": "Second organic result"
                }
            ],
            "news": [
                {
                    "title": "News Result 1",
                    "link": "https://news.example.com/1",
                    "snippet": "First news result",
                    "source": "News Source 1",
                    "date": "2024-01-15T10:00:00Z"
                }
            ],
            "searchInformation": {
                "totalResults": 1000,
                "searchTime": 0.45
            }
        }
        
        result = SerperSearchResponse.from_response(response_data, "test query")
        
        assert len(result.organic_results) == 2
        assert len(result.news_results) == 1
        assert result.total_results == 1000
        assert result.query == "test query"
        assert result.search_time == 0.45
        
        assert result.organic_results[0].title == "Organic Result 1"
        assert result.organic_results[0].position == 1
        assert result.organic_results[1].position == 2
        
        assert result.news_results[0].title == "News Result 1"
        assert result.news_results[0].source == "News Source 1"


class TestSerperRateLimiter:
    """Test cases for SerperRateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within monthly limit."""
        limiter = SerperRateLimiter(max_requests=5)
        
        # Should allow first 5 requests
        for _ in range(5):
            assert await limiter.acquire() is True
        
        # Should deny 6th request
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_monthly_reset(self):
        """Test rate limiter tracks monthly usage correctly."""
        limiter = SerperRateLimiter(max_requests=10)
        
        # Use some requests
        for _ in range(5):
            await limiter.acquire()
        
        remaining = limiter.get_remaining_requests()
        assert remaining == 5
        
        # Use remaining requests
        for _ in range(5):
            await limiter.acquire()
        
        remaining = limiter.get_remaining_requests()
        assert remaining == 0
    
    def test_rate_limiter_get_remaining_requests(self):
        """Test getting remaining requests count."""
        limiter = SerperRateLimiter(max_requests=100)
        
        # Initially should have full quota
        remaining = limiter.get_remaining_requests()
        assert remaining == 100
        
        # Add some mock requests
        now = datetime.now()
        limiter.requests = [now] * 30  # 30 requests this month
        
        remaining = limiter.get_remaining_requests()
        assert remaining == 70


class TestSerperAPIClient:
    """Test cases for SerperAPIClient."""
    
    def setup_method(self):
        """Setup test environment."""
        self.api_key = "test_serper_key"
        self.mock_session = AsyncMock(spec=ClientSession)
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = SerperAPIClient(api_key=self.api_key)
        
        assert client.api_key == self.api_key
        assert client.session is None
        assert client._own_session is True
    
    def test_client_initialization_no_key(self):
        """Test client initialization fails without API key."""
        with patch('fact_check_engine.serperapi_client.config') as mock_config:
            mock_config.api.serper_api_key = ""
            
            with pytest.raises(ValueError, match="SerperAPI key is required"):
                SerperAPIClient()
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            async with SerperAPIClient(api_key=self.api_key) as client:
                assert client.session == mock_session
            
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful API request."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "organic": [],
            "searchInformation": {"totalResults": 0}
        }
        
        self.mock_session.post.return_value.__aenter__.return_value = mock_response
        
        result = await client._make_request("search", {"q": "test"})
        
        assert "organic" in result
        self.mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit_exceeded(self):
        """Test rate limit exceeded error."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock rate limiter to deny requests
        with patch.object(client.rate_limiter, 'acquire', return_value=False):
            with patch.object(client.rate_limiter, 'get_remaining_requests', return_value=0):
                with pytest.raises(SerperAPIError, match="Monthly rate limit exceeded"):
                    await client._make_request("search", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_api_rate_limit(self):
        """Test API rate limit response handling."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        
        self.mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(SerperAPIError, match="Rate limit exceeded"):
            await client._make_request("search", {"q": "test"}, retries=0)
    
    @pytest.mark.asyncio
    async def test_make_request_unauthorized(self):
        """Test unauthorized error handling."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock unauthorized response
        mock_response = AsyncMock()
        mock_response.status = 401
        
        self.mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(SerperAPIError, match="Invalid API key"):
            await client._make_request("search", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_bad_request(self):
        """Test bad request error handling."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock bad request response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {"message": "Invalid query"}
        
        self.mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(SerperAPIError, match="Bad request: Invalid query"):
            await client._make_request("search", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_server_error_with_retry(self):
        """Test server error with retry logic."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock server error then success
        mock_responses = [
            AsyncMock(status=500),  # First attempt fails
            AsyncMock(status=200, json=AsyncMock(return_value={"organic": []}))  # Second succeeds
        ]
        
        self.mock_session.post.return_value.__aenter__.side_effect = mock_responses
        
        with patch('asyncio.sleep'):  # Speed up test
            result = await client._make_request("search", {"q": "test"}, retries=1)
        
        assert "organic" in result
        assert self.mock_session.post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_make_request_network_error(self):
        """Test network error handling."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock network error
        self.mock_session.post.side_effect = ClientError("Network error")
        
        with pytest.raises(SerperAPIError, match="Network error"):
            await client._make_request("search", {"q": "test"}, retries=0)
    
    @pytest.mark.asyncio
    async def test_search(self):
        """Test search method."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful search response
        mock_response_data = {
            "organic": [
                {
                    "title": "Search Result",
                    "link": "https://example.com/result",
                    "snippet": "Test search result"
                }
            ],
            "news": [
                {
                    "title": "News Result",
                    "link": "https://news.example.com/result",
                    "snippet": "Test news result",
                    "source": "News Source",
                    "date": "2024-01-15T10:00:00Z"
                }
            ],
            "searchInformation": {
                "totalResults": 1000,
                "searchTime": 0.35
            }
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data):
            result = await client.search("test query")
        
        assert isinstance(result, SerperSearchResponse)
        assert len(result.organic_results) == 1
        assert len(result.news_results) == 1
        assert result.organic_results[0].title == "Search Result"
        assert result.news_results[0].title == "News Result"
        assert result.query == "test query"
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with various filters."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        mock_response_data = {
            "organic": [],
            "searchInformation": {"totalResults": 0}
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data) as mock_request:
            await client.search(
                query="test query",
                num_results=50,
                country="uk",
                language="en",
                time_range="d",
                safe_search="strict"
            )
        
        # Verify parameters were passed correctly
        call_args = mock_request.call_args
        payload = call_args[0][1]  # Second argument is payload
        
        assert payload["q"] == "test query"
        assert payload["num"] == 50
        assert payload["gl"] == "uk"
        assert payload["hl"] == "en"
        assert payload["tbs"] == "qdr:d"
        assert payload["safe"] == "strict"
    
    @pytest.mark.asyncio
    async def test_news_search(self):
        """Test news_search method."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        mock_response_data = {
            "news": [
                {
                    "title": "Breaking News",
                    "link": "https://news.example.com/breaking",
                    "snippet": "This is breaking news",
                    "source": "News Source",
                    "date": "2024-01-15T12:00:00Z"
                },
                {
                    "title": "Another News",
                    "link": "https://news.example.com/another",
                    "snippet": "Another news story",
                    "source": "Another Source",
                    "date": "2024-01-15T13:00:00Z"
                }
            ]
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data):
            results = await client.news_search("breaking news")
        
        assert len(results) == 2
        assert all(isinstance(result, NewsResult) for result in results)
        assert results[0].title == "Breaking News"
        assert results[1].title == "Another News"
    
    @pytest.mark.asyncio
    async def test_news_search_with_filters(self):
        """Test news_search with filters."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        mock_response_data = {"news": []}
        
        with patch.object(client, '_make_request', return_value=mock_response_data) as mock_request:
            await client.news_search(
                query="test news",
                num_results=20,
                country="ca",
                language="fr",
                time_range="w"
            )
        
        # Verify parameters
        call_args = mock_request.call_args
        payload = call_args[0][1]
        
        assert payload["q"] == "test news"
        assert payload["num"] == 20
        assert payload["gl"] == "ca"
        assert payload["hl"] == "fr"
        assert payload["tbs"] == "qdr:w"
        assert payload["tbm"] == "nws"  # News search mode
    
    @pytest.mark.asyncio
    async def test_fact_check_search(self):
        """Test fact_check_search method."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock multiple search responses for different fact-check queries
        mock_responses = [
            SerperSearchResponse(
                organic_results=[
                    SearchResult(
                        title="Snopes Fact Check",
                        link="https://snopes.com/fact-check/test-claim",
                        snippet="This claim is false",
                        position=1
                    )
                ],
                news_results=[],
                total_results=1,
                query="test claim fact check",
                search_time=0.3
            ),
            SerperSearchResponse(
                organic_results=[
                    SearchResult(
                        title="PolitiFact Analysis",
                        link="https://politifact.com/test-claim",
                        snippet="We rate this claim as false",
                        position=1
                    )
                ],
                news_results=[],
                total_results=1,
                query="test claim debunked",
                search_time=0.25
            )
        ]
        
        with patch.object(client, 'search') as mock_search:
            mock_search.side_effect = mock_responses
            
            results = await client.fact_check_search("test claim", max_results=10)
        
        assert len(results) == 2
        assert results[0].title == "Snopes Fact Check"
        assert results[1].title == "PolitiFact Analysis"
        
        # Verify fact-checker domains are prioritized
        assert "snopes.com" in results[0].link
        assert "politifact.com" in results[1].link
    
    @pytest.mark.asyncio
    async def test_fact_check_search_prioritizes_fact_checkers(self):
        """Test that fact_check_search prioritizes known fact-checking sites."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock response with mixed results
        mock_response = SerperSearchResponse(
            organic_results=[
                SearchResult(
                    title="Random Blog",
                    link="https://randomblog.com/claim",
                    snippet="Some random opinion",
                    position=1
                ),
                SearchResult(
                    title="Reuters Fact Check",
                    link="https://reuters.com/fact-check/claim",
                    snippet="Reuters fact-check analysis",
                    position=2
                ),
                SearchResult(
                    title="Another Blog",
                    link="https://anotherblog.com/claim",
                    snippet="Another opinion",
                    position=3
                )
            ],
            news_results=[],
            total_results=3,
            query="test claim fact check",
            search_time=0.4
        )
        
        with patch.object(client, 'search', return_value=mock_response):
            results = await client.fact_check_search("test claim", max_results=10)
        
        # Reuters (fact-checker) should be first despite original position
        assert results[0].title == "Reuters Fact Check"
        assert results[1].title == "Random Blog"
        assert results[2].title == "Another Blog"
    
    @pytest.mark.asyncio
    async def test_fact_check_search_deduplication(self):
        """Test that fact_check_search removes duplicate URLs."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock responses with duplicate URLs
        duplicate_result = SearchResult(
            title="Duplicate Result",
            link="https://example.com/duplicate",
            snippet="Duplicate content",
            position=1
        )
        
        mock_responses = [
            SerperSearchResponse(
                organic_results=[duplicate_result],
                news_results=[],
                total_results=1,
                query="query1",
                search_time=0.3
            ),
            SerperSearchResponse(
                organic_results=[duplicate_result],  # Same URL
                news_results=[],
                total_results=1,
                query="query2",
                search_time=0.3
            )
        ]
        
        with patch.object(client, 'search') as mock_search:
            mock_search.side_effect = mock_responses
            
            results = await client.fact_check_search("test claim", max_results=10)
        
        # Should only have one result despite multiple searches returning same URL
        assert len(results) == 1
        assert results[0].title == "Duplicate Result"
    
    @pytest.mark.asyncio
    async def test_fact_check_search_handles_errors(self):
        """Test that fact_check_search handles search errors gracefully."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful response for first query, error for second
        successful_response = SerperSearchResponse(
            organic_results=[
                SearchResult(
                    title="Successful Result",
                    link="https://example.com/success",
                    snippet="Successful search",
                    position=1
                )
            ],
            news_results=[],
            total_results=1,
            query="successful query",
            search_time=0.3
        )
        
        with patch.object(client, 'search') as mock_search:
            mock_search.side_effect = [
                successful_response,
                SerperAPIError("Search failed")  # Second search fails
            ]
            
            results = await client.fact_check_search("test claim", max_results=10)
        
        # Should still return results from successful searches
        assert len(results) == 1
        assert results[0].title == "Successful Result"
    
    @pytest.mark.asyncio
    async def test_get_remaining_quota(self):
        """Test get_remaining_quota method."""
        client = SerperAPIClient(api_key=self.api_key, session=self.mock_session)
        
        with patch.object(client.rate_limiter, 'get_remaining_requests', return_value=2450):
            remaining = await client.get_remaining_quota()
        
        assert remaining == 2450


@pytest.mark.parametrize("status_code,expected_error", [
    (400, "Bad request"),
    (401, "Invalid API key"),
    (403, "HTTP error: 403"),
    (404, "HTTP error: 404"),
    (429, "Rate limit exceeded"),
    (500, "Server error: 500"),
    (502, "Server error: 502"),
    (503, "Server error: 503")
])
@pytest.mark.asyncio
async def test_various_http_errors(status_code, expected_error):
    """Test handling of various HTTP error codes."""
    client = SerperAPIClient(api_key="test_key", session=AsyncMock())
    
    mock_response = AsyncMock()
    mock_response.status = status_code
    mock_response.json.return_value = {"message": "Test error"}
    
    client.session.post.return_value.__aenter__.return_value = mock_response
    
    with pytest.raises(SerperAPIError, match=expected_error):
        await client._make_request("search", {"q": "test"}, retries=0)