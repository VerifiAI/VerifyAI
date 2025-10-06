import asyncio
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor

from .newsapi_client import NewsAPIClient, NewsArticle, NewsAPIError
from .serperapi_client import SerperAPIClient, SearchResult, NewsResult, SerperAPIError
from .newsdata_client import NewsDataClient, NewsDataArticle, NewsDataError
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class SearchSource:
    """Represents a search result source with metadata."""
    title: str
    url: str
    snippet: str
    source_name: str
    published_date: Optional[datetime]
    source_type: str  # 'news', 'web', 'fact_check'
    relevance_score: float
    credibility_score: float
    api_source: str  # 'newsapi', 'serper', 'newsdata'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_name": self.source_name,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "source_type": self.source_type,
            "relevance_score": self.relevance_score,
            "credibility_score": self.credibility_score,
            "api_source": self.api_source
        }


@dataclass
class SearchResults:
    """Container for aggregated search results."""
    query: str
    sources: List[SearchSource]
    total_sources: int
    search_time: float
    api_usage: Dict[str, int]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "sources": [source.to_dict() for source in self.sources],
            "total_sources": self.total_sources,
            "search_time": self.search_time,
            "api_usage": self.api_usage,
            "confidence": self.confidence
        }


class SearchOrchestratorError(Exception):
    """Custom exception for search orchestrator errors."""
    pass


class SearchOrchestrator:
    """Orchestrates searches across multiple APIs for comprehensive fact-checking."""
    
    def __init__(self):
        """Initialize search orchestrator with API clients."""
        self.newsapi_client = None
        self.serper_client = None
        self.newsdata_client = None
        
        # Initialize clients
        self._initialize_clients()
        
        # Source credibility mapping
        self.credible_sources = self._load_credible_sources()
        
        # Search strategy configuration
        self.search_config = {
            "max_results_per_api": 20,
            "timeout_seconds": 30,
            "parallel_searches": True,
            "enable_fact_check_prioritization": True,
            "min_relevance_score": 0.3
        }
        
        logger.info("SearchOrchestrator initialized")
    
    def _initialize_clients(self) -> None:
        """Initialize API clients with error handling."""
        try:
            self.newsapi_client = NewsAPIClient()
            logger.info("NewsAPI client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NewsAPI client: {e}")
        
        try:
            self.serper_client = SerperAPIClient()
            logger.info("SerperAPI client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SerperAPI client: {e}")
        
        try:
            self.newsdata_client = NewsDataClient()
            logger.info("NewsData client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NewsData client: {e}")
    
    def _load_credible_sources(self) -> Dict[str, float]:
        """Load credibility scores for news sources."""
        # This would ideally be loaded from a database or external service
        return {
            # Tier 1 - Highly credible
            "reuters.com": 0.95,
            "apnews.com": 0.95,
            "bbc.com": 0.90,
            "npr.org": 0.90,
            "pbs.org": 0.90,
            
            # Tier 2 - Generally credible
            "cnn.com": 0.80,
            "nytimes.com": 0.85,
            "washingtonpost.com": 0.85,
            "theguardian.com": 0.80,
            "wsj.com": 0.85,
            
            # Fact-checking sites - Very high credibility for fact-checking
            "snopes.com": 0.95,
            "politifact.com": 0.95,
            "factcheck.org": 0.95,
            "fullfact.org": 0.90,
            "checkyourfact.com": 0.85,
            
            # Default for unknown sources
            "default": 0.50
        }
    
    async def search_comprehensive(
        self,
        query: str,
        max_results: int = 50,
        include_fact_checks: bool = True,
        time_range: Optional[str] = None
    ) -> SearchResults:
        """
        Perform comprehensive search across all available APIs.
        
        Args:
            query: Search query
            max_results: Maximum total results to return
            include_fact_checks: Whether to include fact-checking specific searches
            time_range: Time range filter (h, d, w, m, y)
            
        Returns:
            Aggregated search results from all APIs
        """
        start_time = datetime.now()
        
        try:
            # Prepare search tasks
            search_tasks = []
            api_usage = {"newsapi": 0, "serper": 0, "newsdata": 0}
            
            # Calculate results per API
            results_per_api = min(
                self.search_config["max_results_per_api"],
                max_results // 3  # Distribute across 3 APIs
            )
            
            # NewsAPI search
            if self.newsapi_client:
                search_tasks.append(
                    self._search_newsapi(query, results_per_api, time_range)
                )
            
            # SerperAPI searches
            if self.serper_client:
                search_tasks.append(
                    self._search_serper_web(query, results_per_api, time_range)
                )
                search_tasks.append(
                    self._search_serper_news(query, results_per_api, time_range)
                )
                
                if include_fact_checks:
                    search_tasks.append(
                        self._search_serper_fact_checks(query, results_per_api)
                    )
            
            # NewsData search
            if self.newsdata_client:
                search_tasks.append(
                    self._search_newsdata(query, results_per_api, time_range)
                )
            
            # Execute searches
            if self.search_config["parallel_searches"]:
                search_results = await asyncio.gather(
                    *search_tasks, 
                    return_exceptions=True
                )
            else:
                search_results = []
                for task in search_tasks:
                    try:
                        result = await task
                        search_results.append(result)
                    except Exception as e:
                        search_results.append(e)
            
            # Aggregate results
            all_sources = []
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(f"Search task failed: {result}")
                    continue
                
                sources, api_name, usage_count = result
                all_sources.extend(sources)
                api_usage[api_name] = usage_count
            
            # Remove duplicates and rank results
            unique_sources = self._deduplicate_sources(all_sources)
            ranked_sources = self._rank_sources(unique_sources, query)
            
            # Limit to max_results
            final_sources = ranked_sources[:max_results]
            
            # Calculate overall confidence
            confidence = self._calculate_search_confidence(final_sources, api_usage)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            results = SearchResults(
                query=query,
                sources=final_sources,
                total_sources=len(final_sources),
                search_time=search_time,
                api_usage=api_usage,
                confidence=confidence
            )
            
            logger.info(f"Comprehensive search completed: {len(final_sources)} sources in {search_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive search failed: {e}")
            raise SearchOrchestratorError(f"Search failed: {e}")
    
    async def _search_newsapi(
        self, 
        query: str, 
        max_results: int, 
        time_range: Optional[str]
    ) -> Tuple[List[SearchSource], str, int]:
        """Search using NewsAPI."""
        sources = []
        
        try:
            async with self.newsapi_client as client:
                # Convert time_range to NewsAPI format
                from_date = None
                if time_range:
                    days_map = {"h": 1, "d": 1, "w": 7, "m": 30, "y": 365}
                    days = days_map.get(time_range, 7)
                    from_date = datetime.now() - timedelta(days=days)
                
                result = await client.search_everything(
                    query=query,
                    page_size=max_results,
                    from_date=from_date,
                    sort_by="relevancy"
                )
                
                for article in result.articles:
                    source = SearchSource(
                        title=article.title,
                        url=article.url,
                        snippet=article.description or "",
                        source_name=article.source,
                        published_date=article.published_at,
                        source_type="news",
                        relevance_score=0.7,  # Default relevance
                        credibility_score=self._get_source_credibility(article.url),
                        api_source="newsapi"
                    )
                    sources.append(source)
                
                return sources, "newsapi", len(sources)
                
        except NewsAPIError as e:
            logger.warning(f"NewsAPI search failed: {e}")
            return [], "newsapi", 0
    
    async def _search_serper_web(
        self, 
        query: str, 
        max_results: int, 
        time_range: Optional[str]
    ) -> Tuple[List[SearchSource], str, int]:
        """Search web using SerperAPI."""
        sources = []
        
        try:
            async with self.serper_client as client:
                result = await client.search(
                    query=query,
                    num_results=max_results,
                    time_range=time_range
                )
                
                for search_result in result.organic_results:
                    source = SearchSource(
                        title=search_result.title,
                        url=search_result.link,
                        snippet=search_result.snippet,
                        source_name=search_result.source or "Unknown",
                        published_date=search_result.date,
                        source_type="web",
                        relevance_score=max(0.1, 1.0 - (search_result.position * 0.05)),
                        credibility_score=self._get_source_credibility(search_result.link),
                        api_source="serper"
                    )
                    sources.append(source)
                
                return sources, "serper", len(sources)
                
        except SerperAPIError as e:
            logger.warning(f"SerperAPI web search failed: {e}")
            return [], "serper", 0
    
    async def _search_serper_news(
        self, 
        query: str, 
        max_results: int, 
        time_range: Optional[str]
    ) -> Tuple[List[SearchSource], str, int]:
        """Search news using SerperAPI."""
        sources = []
        
        try:
            async with self.serper_client as client:
                news_results = await client.news_search(
                    query=query,
                    num_results=max_results,
                    time_range=time_range
                )
                
                for news_result in news_results:
                    source = SearchSource(
                        title=news_result.title,
                        url=news_result.link,
                        snippet=news_result.snippet,
                        source_name=news_result.source,
                        published_date=news_result.date,
                        source_type="news",
                        relevance_score=0.8,  # News is generally more relevant
                        credibility_score=self._get_source_credibility(news_result.link),
                        api_source="serper"
                    )
                    sources.append(source)
                
                return sources, "serper", len(sources)
                
        except SerperAPIError as e:
            logger.warning(f"SerperAPI news search failed: {e}")
            return [], "serper", 0
    
    async def _search_serper_fact_checks(
        self, 
        query: str, 
        max_results: int
    ) -> Tuple[List[SearchSource], str, int]:
        """Search for fact-checks using SerperAPI."""
        sources = []
        
        try:
            async with self.serper_client as client:
                fact_check_results = await client.fact_check_search(
                    claim=query,
                    max_results=max_results
                )
                
                for result in fact_check_results:
                    source = SearchSource(
                        title=result.title,
                        url=result.link,
                        snippet=result.snippet,
                        source_name="Fact Check",
                        published_date=result.date,
                        source_type="fact_check",
                        relevance_score=0.9,  # Fact-checks are highly relevant
                        credibility_score=self._get_source_credibility(result.link),
                        api_source="serper"
                    )
                    sources.append(source)
                
                return sources, "serper", len(sources)
                
        except SerperAPIError as e:
            logger.warning(f"SerperAPI fact-check search failed: {e}")
            return [], "serper", 0
    
    async def _search_newsdata(
        self, 
        query: str, 
        max_results: int, 
        time_range: Optional[str]
    ) -> Tuple[List[SearchSource], str, int]:
        """Search using NewsData.io."""
        sources = []
        
        try:
            async with self.newsdata_client as client:
                # Convert time_range to date range
                from_date = None
                if time_range:
                    days_map = {"h": 1, "d": 1, "w": 7, "m": 30, "y": 365}
                    days = days_map.get(time_range, 7)
                    from_date = datetime.now() - timedelta(days=days)
                
                result = await client.search_news(
                    query=query,
                    size=max_results,
                    from_date=from_date
                )
                
                for article in result.articles:
                    source = SearchSource(
                        title=article.title,
                        url=article.link,
                        snippet=article.description or "",
                        source_name=article.source_name,
                        published_date=article.pub_date,
                        source_type="news",
                        relevance_score=0.7,
                        credibility_score=self._get_source_credibility(article.link),
                        api_source="newsdata"
                    )
                    sources.append(source)
                
                return sources, "newsdata", len(sources)
                
        except NewsDataError as e:
            logger.warning(f"NewsData search failed: {e}")
            return [], "newsdata", 0
    
    def _get_source_credibility(self, url: str) -> float:
        """Get credibility score for a source URL."""
        if not url:
            return self.credible_sources["default"]
        
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Check exact match first
            if domain in self.credible_sources:
                return self.credible_sources[domain]
            
            # Check for partial matches (subdomains)
            for credible_domain, score in self.credible_sources.items():
                if credible_domain != "default" and credible_domain in domain:
                    return score
            
            return self.credible_sources["default"]
            
        except Exception:
            return self.credible_sources["default"]
    
    def _deduplicate_sources(self, sources: List[SearchSource]) -> List[SearchSource]:
        """Remove duplicate sources based on URL."""
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            if source.url not in seen_urls:
                unique_sources.append(source)
                seen_urls.add(source.url)
        
        return unique_sources
    
    def _rank_sources(self, sources: List[SearchSource], query: str) -> List[SearchSource]:
        """Rank sources by relevance and credibility."""
        # Calculate combined score
        for source in sources:
            # Boost fact-checking sources
            type_boost = 1.0
            if source.source_type == "fact_check":
                type_boost = 1.3
            elif source.source_type == "news":
                type_boost = 1.1
            
            # Calculate final score
            source.relevance_score = (
                source.relevance_score * 0.6 +
                source.credibility_score * 0.4
            ) * type_boost
        
        # Sort by relevance score (highest first)
        sources.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return sources
    
    def _calculate_search_confidence(
        self, 
        sources: List[SearchSource], 
        api_usage: Dict[str, int]
    ) -> float:
        """Calculate overall confidence in search results."""
        if not sources:
            return 0.0
        
        # Base confidence from number of sources
        source_confidence = min(1.0, len(sources) / 20.0)
        
        # Credibility confidence from average source credibility
        avg_credibility = sum(s.credibility_score for s in sources) / len(sources)
        
        # API diversity confidence
        active_apis = sum(1 for count in api_usage.values() if count > 0)
        api_confidence = active_apis / 3.0  # 3 total APIs
        
        # Fact-check boost
        fact_check_count = sum(1 for s in sources if s.source_type == "fact_check")
        fact_check_boost = min(0.2, fact_check_count * 0.05)
        
        # Combined confidence
        confidence = (
            source_confidence * 0.3 +
            avg_credibility * 0.4 +
            api_confidence * 0.2 +
            fact_check_boost * 0.1
        )
        
        return min(1.0, confidence)
    
    async def search_targeted(
        self,
        claim: str,
        entities: List[str],
        keywords: List[str],
        max_results: int = 30
    ) -> SearchResults:
        """
        Perform targeted search optimized for specific claim with entities and keywords.
        
        Args:
            claim: The specific claim to fact-check
            entities: Named entities from the claim
            keywords: Important keywords from the claim
            max_results: Maximum results to return
            
        Returns:
            Targeted search results
        """
        # Generate optimized search queries
        search_queries = self._generate_search_queries(claim, entities, keywords)
        
        all_sources = []
        api_usage = {"newsapi": 0, "serper": 0, "newsdata": 0}
        
        start_time = datetime.now()
        
        try:
            # Search with each optimized query
            for query in search_queries[:3]:  # Limit to top 3 queries
                results = await self.search_comprehensive(
                    query=query,
                    max_results=max_results // len(search_queries[:3]),
                    include_fact_checks=True,
                    time_range="y"  # Last year for recent information
                )
                
                all_sources.extend(results.sources)
                for api, count in results.api_usage.items():
                    api_usage[api] += count
            
            # Deduplicate and rank
            unique_sources = self._deduplicate_sources(all_sources)
            ranked_sources = self._rank_sources(unique_sources, claim)
            
            # Limit results
            final_sources = ranked_sources[:max_results]
            
            search_time = (datetime.now() - start_time).total_seconds()
            confidence = self._calculate_search_confidence(final_sources, api_usage)
            
            return SearchResults(
                query=claim,
                sources=final_sources,
                total_sources=len(final_sources),
                search_time=search_time,
                api_usage=api_usage,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Targeted search failed: {e}")
            raise SearchOrchestratorError(f"Targeted search failed: {e}")
    
    def _generate_search_queries(
        self, 
        claim: str, 
        entities: List[str], 
        keywords: List[str]
    ) -> List[str]:
        """Generate optimized search queries for fact-checking."""
        queries = []
        
        # Original claim
        queries.append(claim)
        
        # Claim with fact-check keywords
        queries.append(f"{claim} fact check")
        queries.append(f"{claim} true false")
        queries.append(f"{claim} verified debunked")
        
        # Entity-based queries
        if entities:
            # Combine top entities
            top_entities = entities[:3]
            entity_query = " ".join(top_entities)
            queries.append(f"{entity_query} {' '.join(keywords[:3])}")
            
            # Individual entity queries
            for entity in top_entities:
                queries.append(f"{entity} {claim[:50]}")
        
        # Keyword-based queries
        if keywords:
            top_keywords = keywords[:5]
            keyword_query = " ".join(top_keywords)
            queries.append(keyword_query)
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen and len(query.strip()) > 5:
                unique_queries.append(query)
                seen.add(query)
        
        return unique_queries
    
    async def get_api_status(self) -> Dict[str, Any]:
        """Get status and quota information for all APIs."""
        status = {
            "newsapi": {"available": False, "remaining_quota": 0},
            "serper": {"available": False, "remaining_quota": 0},
            "newsdata": {"available": False, "remaining_quota": 0}
        }
        
        # Check NewsAPI
        if self.newsapi_client:
            try:
                # NewsAPI doesn't have a direct quota check, so we assume it's available
                status["newsapi"]["available"] = True
                status["newsapi"]["remaining_quota"] = "unknown"
            except Exception as e:
                logger.warning(f"NewsAPI status check failed: {e}")
        
        # Check SerperAPI
        if self.serper_client:
            try:
                remaining = await self.serper_client.get_remaining_quota()
                status["serper"]["available"] = True
                status["serper"]["remaining_quota"] = remaining
            except Exception as e:
                logger.warning(f"SerperAPI status check failed: {e}")
        
        # Check NewsData
        if self.newsdata_client:
            try:
                remaining = await self.newsdata_client.get_remaining_quota()
                status["newsdata"]["available"] = True
                status["newsdata"]["remaining_quota"] = remaining
            except Exception as e:
                logger.warning(f"NewsData status check failed: {e}")
        
        return status


## Suggestions for Upgrade:
# 1. Implement intelligent query expansion using word embeddings and semantic similarity
# 2. Add machine learning-based relevance scoring trained on fact-checking datasets
# 3. Integrate with specialized fact-checking APIs like Google Fact Check Tools API
# 4. Add real-time monitoring and alerting for API quota limits and failures