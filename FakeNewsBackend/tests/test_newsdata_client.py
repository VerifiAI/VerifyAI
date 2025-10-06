import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from aiohttp import ClientSession, ClientError

from fact_check_engine.newsdata_client import (
    NewsDataClient, NewsDataArticle, NewsDataSearchResult, 
    NewsDataError, NewsDataRateLimiter
)


class TestNewsDataArticle:
    """Test cases for NewsDataArticle data model."""
    
    def test_newsdata_article_from_dict(self):
        """Test NewsDataArticle creation from API response data."""
        data = {
            "title": "Test News Article",
            "link": "https://example.com/article",
            "description": "Test description",
            "content": "Full article content",
            "pubDate": "2024-01-15T10:00:00Z",
            "source_id": "example_source",
            "source_name": "Example News",
            "country": ["US", "CA"],
            "category": ["politics", "world"],
            "language": "en",
            "image_url": "https://example.com/image.jpg",
            "video_url": "https://example.com/video.mp4",
            "creator": ["John Doe", "Jane Smith"],
            "keywords": ["politics", "election", "news"]
        }
        
        article = NewsDataArticle.from_dict(data)
        
        assert article.title == "Test News Article"
        assert article.link == "https://example.com/article"
        assert article.description == "Test description"
        assert article.content == "Full article content"
        assert article.source_id == "example_source"
        assert article.source_name == "Example News"
        assert article.country == ["US", "CA"]
        assert article.category == ["politics", "world"]
        assert article.language == "en"
        assert article.image_url == "https://example.com/image.jpg"
        assert article.video_url == "https://example.com/video.mp4"
        assert article.creator == ["John Doe", "Jane Smith"]
        assert article.keywords == ["politics", "election", "news"]
        assert isinstance(article.pub_date, datetime)
    
    def test_newsdata_article_from_dict_minimal(self):
        """Test NewsDataArticle creation with minimal data."""
        data = {
            "title": "Minimal Article",
            "link": "https://example.com/minimal",
            "source_id": "minimal_source",
            "language": "en"
        }
        
        article = NewsDataArticle.from_dict(data)
        
        assert article.title == "Minimal Article"
        assert article.link == "https://example.com/minimal"
        assert article.source_id == "minimal_source"
        assert article.source_name == "Unknown"
        assert article.language == "en"
        assert article.country == []
        assert article.category == []
        assert article.description is None
        assert article.content is None
        assert isinstance(article.pub_date, datetime)
    
    def test_newsdata_article_invalid_date(self):
        """Test NewsDataArticle handles invalid date gracefully."""
        data = {
            "title": "Test Article",
            "link": "https://example.com/article",
            "source_id": "test_source",
            "language": "en",
            "pubDate": "invalid-date"
        }
        
        article = NewsDataArticle.from_dict(data)
        
        assert article.title == "Test Article"
        assert isinstance(article.pub_date, datetime)


class TestNewsDataSearchResult:
    """Test cases for NewsDataSearchResult container."""
    
    def test_newsdata_search_result_from_response(self):
        """Test NewsDataSearchResult creation from API response."""
        response_data = {
            "status": "success",
            "totalResults": 2,
            "nextPage": "next_page_token",
            "results": [
                {
                    "title": "Article 1",
                    "link": "https://example.com/1",
                    "source_id": "source1",
                    "source_name": "Source 1",
                    "language": "en",
                    "pubDate": "2024-01-15T10:00:00Z"
                },
                {
                    "title": "Article 2",
                    "link": "https://example.com/2",
                    "source_id": "source2",
                    "source_name": "Source 2",
                    "language": "en",
                    "pubDate": "2024-01-15T11:00:00Z"
                }
            ]
        }
        
        result = NewsDataSearchResult.from_response(response_data, "test query")
        
        assert len(result.articles) == 2
        assert result.total_results == 2
        assert result.next_page == "next_page_token"
        assert result.query == "test query"
        assert result.articles[0].title == "Article 1"
        assert result.articles[1].title == "Article 2"
    
    def test_newsdata_search_result_empty(self):
        """Test NewsDataSearchResult with empty results."""
        response_data = {
            "status": "success",
            "results": []
        }
        
        result = NewsDataSearchResult.from_response(response_data, "empty query")
        
        assert len(result.articles) == 0
        assert result.total_results == 0
        assert result.next_page is None
        assert result.query == "empty query"


class TestNewsDataRateLimiter:
    """Test cases for NewsDataRateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within daily limit."""
        limiter = NewsDataRateLimiter(max_requests=5)
        
        # Should allow first 5 requests
        for _ in range(5):
            assert await limiter.acquire() is True
        
        # Should deny 6th request
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_daily_reset(self):
        """Test rate limiter tracks daily usage correctly."""
        limiter = NewsDataRateLimiter(max_requests=10)
        
        # Use some requests
        for _ in range(7):
            await limiter.acquire()
        
        remaining = limiter.get_remaining_requests()
        assert remaining == 3
        
        # Use remaining requests
        for _ in range(3):
            await limiter.acquire()
        
        remaining = limiter.get_remaining_requests()
        assert remaining == 0
    
    def test_rate_limiter_get_remaining_requests(self):
        """Test getting remaining requests count."""
        limiter = NewsDataRateLimiter(max_requests=200)
        
        # Initially should have full quota
        remaining = limiter.get_remaining_requests()
        assert remaining == 200
        
        # Add some mock requests
        now = datetime.now()
        limiter.requests = [now] * 50  # 50 requests today
        
        remaining = limiter.get_remaining_requests()
        assert remaining == 150


class TestNewsDataClient:
    """Test cases for NewsDataClient."""
    
    def setup_method(self):
        """Setup test environment."""
        self.api_key = "test_newsdata_key"
        self.mock_session = AsyncMock(spec=ClientSession)
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = NewsDataClient(api_key=self.api_key)
        
        assert client.api_key == self.api_key
        assert client.session is None
        assert client._own_session is True
    
    def test_client_initialization_no_key(self):
        """Test client initialization fails without API key."""
        with patch('fact_check_engine.newsdata_client.config') as mock_config:
            mock_config.api.newsdata_api_key = ""
            
            with pytest.raises(ValueError, match="NewsData.io API key is required"):
                NewsDataClient()
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            async with NewsDataClient(api_key=self.api_key) as client:
                assert client.session == mock_session
            
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful API request."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "success",
            "results": []
        }
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        result = await client._make_request("news", {"q": "test"})
        
        assert result["status"] == "success"
        self.mock_session.get.assert_called_once()
        
        # Verify API key was added to parameters
        call_args = self.mock_session.get.call_args
        params = call_args[1]["params"]
        assert params["apikey"] == self.api_key
    
    @pytest.mark.asyncio
    async def test_make_request_api_error(self):
        """Test API error handling."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock API error response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "error",
            "message": "Invalid API key"
        }
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(NewsDataError, match="Invalid API key"):
            await client._make_request("news", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit_exceeded(self):
        """Test rate limit exceeded error."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock rate limiter to deny requests
        with patch.object(client.rate_limiter, 'acquire', return_value=False):
            with patch.object(client.rate_limiter, 'get_remaining_requests', return_value=0):
                with pytest.raises(NewsDataError, match="Daily rate limit exceeded"):
                    await client._make_request("news", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_api_rate_limit(self):
        """Test API rate limit response handling."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(NewsDataError, match="Rate limit exceeded"):
            await client._make_request("news", {"q": "test"}, retries=0)
    
    @pytest.mark.asyncio
    async def test_make_request_unauthorized(self):
        """Test unauthorized error handling."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock unauthorized response
        mock_response = AsyncMock()
        mock_response.status = 401
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(NewsDataError, match="Invalid API key"):
            await client._make_request("news", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_bad_request(self):
        """Test bad request error handling."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock bad request response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {"message": "Invalid query"}
        
        self.mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with pytest.raises(NewsDataError, match="Bad request: Invalid query"):
            await client._make_request("news", {"q": "test"})
    
    @pytest.mark.asyncio
    async def test_make_request_server_error_with_retry(self):
        """Test server error with retry logic."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock server error then success
        mock_responses = [
            AsyncMock(status=500),  # First attempt fails
            AsyncMock(status=200, json=AsyncMock(return_value={"status": "success", "results": []}))
        ]
        
        self.mock_session.get.return_value.__aenter__.side_effect = mock_responses
        
        with patch('asyncio.sleep'):  # Speed up test
            result = await client._make_request("news", {"q": "test"}, retries=1)
        
        assert result["status"] == "success"
        assert self.mock_session.get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_make_request_network_error(self):
        """Test network error handling."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock network error
        self.mock_session.get.side_effect = ClientError("Network error")
        
        with pytest.raises(NewsDataError, match="Network error"):
            await client._make_request("news", {"q": "test"}, retries=0)
    
    @pytest.mark.asyncio
    async def test_search_news(self):
        """Test search_news method."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful search response
        mock_response_data = {
            "status": "success",
            "totalResults": 1,
            "results": [
                {
                    "title": "Test News Article",
                    "link": "https://example.com/article",
                    "description": "Test description",
                    "source_id": "test_source",
                    "source_name": "Test Source",
                    "language": "en",
                    "country": ["US"],
                    "category": ["politics"],
                    "pubDate": "2024-01-15T10:00:00Z"
                }
            ]
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data):
            result = await client.search_news("test query")
        
        assert isinstance(result, NewsDataSearchResult)
        assert len(result.articles) == 1
        assert result.articles[0].title == "Test News Article"
        assert result.query == "test query"
    
    @pytest.mark.asyncio
    async def test_search_news_with_filters(self):
        """Test search_news with various filters."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        from_date = datetime(2024, 1, 1)
        to_date = datetime(2024, 1, 15)
        
        mock_response_data = {
            "status": "success",
            "totalResults": 0,
            "results": []
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data) as mock_request:
            await client.search_news(
                query="test query",
                language="fr",
                country="ca",
                category="business",
                size=25,
                page="next_page_token",
                from_date=from_date,
                to_date=to_date,
                domain="example.com",
                exclude_domain="spam.com",
                sentiment="positive"
            )
        
        # Verify parameters were passed correctly
        call_args = mock_request.call_args
        params = call_args[0][1]  # Second argument is params
        
        assert params["q"] == "test query"
        assert params["language"] == "fr"
        assert params["country"] == "ca"
        assert params["category"] == "business"
        assert params["size"] == 25
        assert params["page"] == "next_page_token"
        assert params["from_date"] == "2024-01-01"
        assert params["to_date"] == "2024-01-15"
        assert params["domain"] == "example.com"
        assert params["excludedomain"] == "spam.com"
        assert params["sentiment"] == "positive"
    
    @pytest.mark.asyncio
    async def test_search_news_size_limit(self):
        """Test that search_news respects API size limits."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        mock_response_data = {
            "status": "success",
            "results": []
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data) as mock_request:
            await client.search_news("test", size=100)  # Request more than API limit
        
        # Should be capped at 50
        call_args = mock_request.call_args
        params = call_args[0][1]
        assert params["size"] == 50
    
    @pytest.mark.asyncio
    async def test_get_latest_news(self):
        """Test get_latest_news method."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        mock_response_data = {
            "status": "success",
            "totalResults": 1,
            "results": [
                {
                    "title": "Latest Breaking News",
                    "link": "https://example.com/latest",
                    "source_id": "news_source",
                    "source_name": "News Source",
                    "language": "en",
                    "pubDate": "2024-01-15T12:00:00Z"
                }
            ]
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data):
            result = await client.get_latest_news(
                language="en",
                country="us",
                category="general",
                size=20
            )
        
        assert isinstance(result, NewsDataSearchResult)
        assert len(result.articles) == 1
        assert result.articles[0].title == "Latest Breaking News"
        assert result.query == "latest-news"
    
    @pytest.mark.asyncio
    async def test_search_claim_related(self):
        """Test search_claim_related method."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock multiple search responses for different languages and queries
        mock_responses = [
            NewsDataSearchResult(
                articles=[
                    NewsDataArticle(
                        title="English Article 1",
                        link="https://example.com/en1",
                        description="English description",
                        content=None,
                        pub_date=datetime.now(),
                        source_id="en_source1",
                        source_name="English Source 1",
                        country=["US"],
                        category=["politics"],
                        language="en"
                    )
                ],
                total_results=1,
                next_page=None,
                query="test claim"
            ),
            NewsDataSearchResult(
                articles=[
                    NewsDataArticle(
                        title="English Article 2",
                        link="https://example.com/en2",
                        description="Another English description",
                        content=None,
                        pub_date=datetime.now(),
                        source_id="en_source2",
                        source_name="English Source 2",
                        country=["UK"],
                        category=["world"],
                        language="en"
                    )
                ],
                total_results=1,
                next_page=None,
                query='"test claim"'
            )
        ]
        
        with patch.object(client, 'search_news') as mock_search:
            mock_search.side_effect = mock_responses
            
            articles = await client.search_claim_related("test claim", max_results=10)
        
        assert len(articles) == 2
        assert articles[0].title == "English Article 1"
        assert articles[1].title == "English Article 2"
    
    @pytest.mark.asyncio
    async def test_search_claim_related_multiple_languages(self):
        """Test search_claim_related with multiple languages."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock responses for different languages
        en_response = NewsDataSearchResult(
            articles=[
                NewsDataArticle(
                    title="English Article",
                    link="https://example.com/en",
                    description=None,
                    content=None,
                    pub_date=datetime.now(),
                    source_id="en_source",
                    source_name="English Source",
                    country=["US"],
                    category=["politics"],
                    language="en"
                )
            ],
            total_results=1,
            next_page=None,
            query="test claim"
        )
        
        es_response = NewsDataSearchResult(
            articles=[
                NewsDataArticle(
                    title="Spanish Article",
                    link="https://example.com/es",
                    description=None,
                    content=None,
                    pub_date=datetime.now(),
                    source_id="es_source",
                    source_name="Spanish Source",
                    country=["ES"],
                    category=["politics"],
                    language="es"
                )
            ],
            total_results=1,
            next_page=None,
            query="test claim"
        )
        
        with patch.object(client, 'search_news') as mock_search:
            mock_search.side_effect = [en_response, es_response]
            
            articles = await client.search_claim_related(
                "test claim", 
                max_results=10, 
                languages=["en", "es"]
            )
        
        assert len(articles) == 2
        assert articles[0].language == "en"
        assert articles[1].language == "es"
    
    @pytest.mark.asyncio
    async def test_search_claim_related_deduplication(self):
        """Test that search_claim_related removes duplicate URLs."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock response with duplicate URL
        duplicate_article = NewsDataArticle(
            title="Duplicate Article",
            link="https://example.com/duplicate",
            description=None,
            content=None,
            pub_date=datetime.now(),
            source_id="source",
            source_name="Source",
            country=["US"],
            category=["politics"],
            language="en"
        )
        
        mock_response = NewsDataSearchResult(
            articles=[duplicate_article],
            total_results=1,
            next_page=None,
            query="test claim"
        )
        
        with patch.object(client, 'search_news') as mock_search:
            # Return same article multiple times
            mock_search.return_value = mock_response
            
            articles = await client.search_claim_related("test claim", max_results=10)
        
        # Should only have one article despite multiple searches returning the same URL
        assert len(articles) == 1
        assert articles[0].title == "Duplicate Article"
    
    @pytest.mark.asyncio
    async def test_search_claim_related_handles_errors(self):
        """Test that search_claim_related handles search errors gracefully."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        # Mock successful response for first query, error for second
        successful_response = NewsDataSearchResult(
            articles=[
                NewsDataArticle(
                    title="Successful Article",
                    link="https://example.com/success",
                    description=None,
                    content=None,
                    pub_date=datetime.now(),
                    source_id="source",
                    source_name="Source",
                    country=["US"],
                    category=["politics"],
                    language="en"
                )
            ],
            total_results=1,
            next_page=None,
            query="successful query"
        )
        
        with patch.object(client, 'search_news') as mock_search:
            mock_search.side_effect = [
                successful_response,
                NewsDataError("Search failed")  # Second search fails
            ]
            
            articles = await client.search_claim_related("test claim", max_results=10)
        
        # Should still return results from successful searches
        assert len(articles) == 1
        assert articles[0].title == "Successful Article"
    
    @pytest.mark.asyncio
    async def test_get_sources(self):
        """Test get_sources method."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        mock_response_data = {
            "status": "success",
            "results": [
                {
                    "id": "source1",
                    "name": "Source 1",
                    "url": "https://source1.com",
                    "country": ["US"],
                    "language": ["en"]
                },
                {
                    "id": "source2",
                    "name": "Source 2",
                    "url": "https://source2.com",
                    "country": ["UK"],
                    "language": ["en"]
                }
            ]
        }
        
        with patch.object(client, '_make_request', return_value=mock_response_data):
            sources = await client.get_sources(language="en", country="us")
        
        assert len(sources) == 2
        assert sources[0]["name"] == "Source 1"
        assert sources[1]["name"] == "Source 2"
    
    @pytest.mark.asyncio
    async def test_get_sources_error_handling(self):
        """Test get_sources error handling."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        with patch.object(client, '_make_request', side_effect=NewsDataError("API error")):
            sources = await client.get_sources()
        
        # Should return empty list on error
        assert sources == []
    
    @pytest.mark.asyncio
    async def test_get_remaining_quota(self):
        """Test get_remaining_quota method."""
        client = NewsDataClient(api_key=self.api_key, session=self.mock_session)
        
        with patch.object(client.rate_limiter, 'get_remaining_requests', return_value=150):
            remaining = await client.get_remaining_quota()
        
        assert remaining == 150


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
    client = NewsDataClient(api_key="test_key", session=AsyncMock())
    
    mock_response = AsyncMock()
    mock_response.status = status_code
    mock_response.json.return_value = {"message": "Test error"}
    
    client.session.get.return_value.__aenter__.return_value = mock_response
    
    with pytest.raises(NewsDataError, match=expected_error):
        await client._make_request("news", {"q": "test"}, retries=0)