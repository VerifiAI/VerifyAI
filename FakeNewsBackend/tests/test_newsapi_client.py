import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from aiohttp import ClientSession, ClientError

from fact_check_engine.newsapi_client import (
    NewsAPIClient, NewsArticle, NewsSearchResult, NewsAPIError, RateLimiter
)


class TestNewsArticle:
    """Test cases for NewsArticle data model."""
    
    def test_news_article_from_dict(self):
        """Test NewsArticle creation from API response data."""
        data = {
            "title": "Test Article",
            "description": "Test description",
            "url": "https://example.com/article",
            "source": {"name": "Test Source"},
            "publishedAt": "2024-01-15T10:00:00Z",
            "author": "Test Author",
            "content": "Test content",
            "urlToImage": "https://example.com/image.jpg"
        }
        
        article = NewsArticle.from_dict(data)
        
        assert article.title == "Test Article"
        assert article.description == "Test description"
        assert article.url == "https://example.com/article"
        assert article.source == "Test Source"
        assert article.author == "Test Author"
        assert article.content == "Test content"
        assert article.url_to_image == "https://example.com/image.jpg"
        assert isinstance(article.published_at, datetime)
    
    def test_news_article_from_dict_minimal(self):
        """Test NewsArticle creation with minimal data."""
        data = {
            "title": "Minimal Article",
            "url": "https://example.com/minimal"
        }
        
        article = NewsArticle.from_dict(data)
        
        assert article.title == "Minimal Article"
        assert article.url == "https://example.com/minimal"
        assert article.source == "Unknown"
        assert article.description is None
        assert isinstance(article.published_at, datetime)
    
    def test_news_article_invalid_date(self):
        """Test NewsArticle handles invalid date gracefully."""
        data = {
            "title": "Test Article",
            "url": "https://example.com/article",
            "publishedAt": "invalid-date"
        }
        
        article = NewsArticle.from_dict(data)
        
        assert article.title == "Test Article"
        assert isinstance(article.published_at, datetime)


class TestNewsSearchResult:
    """Test cases for NewsSearchResult container."""
    
    def test_news_search_result_from_response(self):
        """Test NewsSearchResult creation from API response."""
        response_data = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "title": "Article 1",
                    "url": "https://example.com/1",
                    "source": {"name": "Source 1"},
                    "publishedAt": "2024-01-15T10:00:00Z"
                },
                {
                    "title": "Article 2",
                    "url": "https://example.com/2",
                    "source": {"name": "Source 2"},
                    "publishedAt": "2024-01-15T11:00:00Z"
                }
            ]
        }
        
        result = NewsSearchResult.from_response(response_data, "test query")
        
        assert len(result.articles) == 2
        assert result.total_results == 2
        assert result.status == "ok"
        assert result.query == "test query"
        assert result.articles[0].title == "Article 1"
        assert result.articles[1].title == "Article 2"


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        # Should allow first 5 requests
        for _ in range(5):
            assert await limiter.acquire() is True
        
        # Should deny 6th request
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_resets_after_window(self):
        """Test rate limiter resets after time window."""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Use up the limit
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is False
        
        # Wait for window to reset
        await asyncio.sleep(1.1)
        
        # Should allow requests again
        assert await limiter.acquire() is True


class TestNewsAPIClient:
    """Test cases for NewsAPIClient."""
    
    def setup_method(self):
        """Setup test environment."""
        self.api_key = "test_api_key"
        self.mock_session = AsyncMock(spec=ClientSession)
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = NewsAPIClient(api_key=self.api_key)
        
        assert client.api_key == self.api_key
        assert client.session is None
        assert client._own_session is True
    
    def test_client_initialization_no_key(self):
        """Test client initialization fails without API key."""
        with patch('fact_check_engine.newsapi_client.config') as mock_config:
            mock_config.api.news_api_key = ""
            
            with pytest.raises(ValueError, match="NewsAPI key is required"):
                NewsAPIClient()
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            async with NewsAPIClient(api_key=self.api_key) as client:
                assert client.session == mock_session
            
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful API request."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "ok",
            "articles": []
        }
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        result = await client._make_request("test", {"q": "test"})
        
        assert result["status"] == "ok"
        self.mock_session.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_api_error(self):
        """Test API error handling."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock API error response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "error",
            "message": "Invalid API key"
        }
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(NewsAPIError, match="Invalid API key"):
            await client._make_request("test", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self):
        """Test rate limit handling."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(NewsAPIError, match="Rate limit exceeded"):
            await client._make_request("test", {"q": "test"}, retries=0)
    
    @pytest.mark.asyncio
    async def test_make_request_unauthorized(self):
        """Test unauthorized error handling."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock unauthorized response
        mock_response = AsyncMock()
        mock_response.status = 401
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(NewsAPIError, match="Invalid API key"):
            await client._make_request("test", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_server_error_with_retry(self):
        """Test server error with retry logic."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock server error then success
        mock_responses = [
            AsyncMock(status=500),  # First attempt fails
            AsyncMock(status=200, json=AsyncMock(return_value={"status": "ok"}))  # Second succeeds
        ]
        
        self.mock_session.get.return_value.__aenter__.side_effect = mock_responses
        
        with patch('asyncio.sleep'):  # Speed up test
            result = await client._make_request("test", {"q": "test"}, retries=1)
        
        assert result["status"] == "ok"
        assert self.mock_session.get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_make_request_network_error(self):
        """Test network error handling."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock network error
        self.mock_session.get.side_effect = ClientError("Network error")
        
        with pytest.raises(NewsAPIError, match="Network error"):
            await client._make_request("test", {"q": "test"}, retries=0)
    
    @pytest.mark.asyncio
    async def test_search_everything(self):
        """Test search_everything method."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful search response
        mock_response_data = {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "title": "Test Article",
                    "url": "https://example.com/article",
                    "source": {"name": "Test Source"},
                    "publishedAt": "2024-01-15T10:00:00Z"
                }
            ]
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data):
            result = await client.search_everything("test query")
        
        assert isinstance(result, NewsSearchResult)
        assert len(result.articles) == 1
        assert result.articles[0].title == "Test Article"
        assert result.query == "test query"
    
    @pytest.mark.asyncio
    async def test_search_everything_with_filters(self):
        """Test search_everything with various filters."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        from_date = datetime(2024, 1, 1)
        to_date = datetime(2024, 1, 15)
        domains = ["example.com", "test.com"]
        exclude_domains = ["spam.com"]
        
        mock_response_data = {
            "status": "ok",
            "totalResults": 0,
            "articles": []
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data) as mock_request:
            await client.search_everything(
                query="test",
                from_date=from_date,
                to_date=to_date,
                domains=domains,
                exclude_domains=exclude_domains,
                page_size=50,
                sort_by="popularity"
            )
        
        # Verify parameters were passed correctly
        call_args = mock_request.call_args
        params = call_args[0][1]  # Second argument is params
        
        assert params["q"] == "test"
        assert params["from"] == from_date.isoformat()
        assert params["to"] == to_date.isoformat()
        assert params["domains"] == "example.com,test.com"
        assert params["excludeDomains"] == "spam.com"
        assert params["pageSize"] == 50
        assert params["sortBy"] == "popularity"
    
    @pytest.mark.asyncio
    async def test_get_top_headlines(self):
        """Test get_top_headlines method."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        mock_response_data = {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "title": "Breaking News",
                    "url": "https://example.com/breaking",
                    "source": {"name": "News Source"},
                    "publishedAt": "2024-01-15T12:00:00Z"
                }
            ]
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data):
            result = await client.get_top_headlines(query="breaking", category="general")
        
        assert isinstance(result, NewsSearchResult)
        assert len(result.articles) == 1
        assert result.articles[0].title == "Breaking News"
    
    @pytest.mark.asyncio
    async def test_search_claim_related(self):
        """Test search_claim_related method."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock multiple search responses
        mock_responses = [
            {
                "status": "ok",
                "totalResults": 1,
                "articles": [
                    {
                        "title": "Claim Article 1",
                        "url": "https://example.com/claim1",
                        "source": {"name": "Source 1"},
                        "publishedAt": "2024-01-15T10:00:00Z"
                    }
                ]
            },
            {
                "status": "ok", 
                "totalResults": 1,
                "articles": [
                    {
                        "title": "Claim Article 2",
                        "url": "https://example.com/claim2",
                        "source": {"name": "Source 2"},
                        "publishedAt": "2024-01-15T11:00:00Z"
                    }
                ]
            }
        ]
        
        with patch.object(client, 'search_everything') as mock_search:
            mock_search.side_effect = [
                NewsSearchResult.from_response(resp, "query") for resp in mock_responses
            ]
            
            articles = await client.search_claim_related("test claim", max_results=10)
        
        assert len(articles) == 2
        assert articles[0].title == "Claim Article 1"
        assert articles[1].title == "Claim Article 2"
    
    @pytest.mark.asyncio
    async def test_search_claim_related_deduplication(self):
        """Test that search_claim_related removes duplicate URLs."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock response with duplicate URL
        duplicate_article = {
            "title": "Duplicate Article",
            "url": "https://example.com/duplicate",
            "source": {"name": "Source"},
            "publishedAt": "2024-01-15T10:00:00Z"
        }
        
        mock_response = {
            "status": "ok",
            "totalResults": 1,
            "articles": [duplicate_article]
        }
        
        with patch.object(client, 'search_everything') as mock_search:
            # Return same article multiple times
            mock_search.return_value = NewsSearchResult.from_response(mock_response, "query")
            
            articles = await client.search_claim_related("test claim", max_results=10)
        
        # Should only have one article despite multiple searches returning the same URL
        assert len(articles) == 1
        assert articles[0].title == "Duplicate Article"
    
    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self):
        """Test rate limiter integration with client."""
        client = NewsAPIClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock rate limiter to deny requests
        with patch.object(client.rate_limiter, 'acquire', return_value=False):
            with pytest.raises(NewsAPIError, match="Rate limit exceeded"):
                await client._make_request("test", {"q": "test"})


@pytest.mark.parametrize("status_code,expected_error", [
    (400, "HTTP error: 400"),
    (403, "HTTP error: 403"),
    (404, "HTTP error: 404"),
    (500, "Server error: 500"),
    (502, "Server error: 502"),
    (503, "Server error: 503")
])
@pytest.mark.asyncio
async def test_various_http_errors(status_code, expected_error):
    """Test handling of various HTTP error codes."""
    client = NewsAPIClient(api_key="test_key", session=AsyncMock())
    
    mock_response = AsyncMock()
    mock_response.status = status_code
    
    client.session.get.return_value.__aenter__.return_value = mock_response
    
    with pytest.raises(NewsAPIError, match=expected_error):
        await client._make_request("test", {"q": "test"}, retries=0)