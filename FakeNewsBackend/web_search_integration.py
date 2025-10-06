#!/usr/bin/env python3
"""
Web Search Integration for Real-time Fact Checking
Implements multiple search APIs and cross-verification logic

Author: AI Assistant
Date: January 2025
Version: 1.0.0
"""

import os
import logging
import asyncio
import aiohttp
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlparse
import re
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0
    credibility_score: float = 0.0

@dataclass
class FactCheckResult:
    """Fact checking result"""
    verdict: str  # 'Real', 'Fake', 'Unverified'
    confidence: float
    supporting_sources: List[SearchResult]
    contradicting_sources: List[SearchResult]
    summary: str
    processing_time: float

class WebSearchIntegrator:
    """Comprehensive web search integration for fact checking"""
    
    def __init__(self):
        # API Keys from environment
        self.bing_api_key = os.getenv('BING_SEARCH_API_KEY', '')
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY', '')
        self.google_cx = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        
        # Credible news sources with reliability scores
        self.credible_sources = {
            'reuters.com': 0.95,
            'apnews.com': 0.95,
            'bbc.com': 0.90,
            'cnn.com': 0.85,
            'npr.org': 0.90,
            'theguardian.com': 0.85,
            'nytimes.com': 0.85,
            'washingtonpost.com': 0.85,
            'factcheck.org': 0.95,
            'snopes.com': 0.90,
            'politifact.com': 0.90,
            'abc.net.au': 0.85,
            'cbsnews.com': 0.80,
            'nbcnews.com': 0.80,
            'usatoday.com': 0.75
        }
        
        # Fact-checking specific sources
        self.fact_check_sources = {
            'factcheck.org': 0.95,
            'snopes.com': 0.90,
            'politifact.com': 0.90,
            'fullfact.org': 0.85,
            'checkyourfact.com': 0.80,
            'truthorfiction.com': 0.75
        }
        
        # Search endpoints
        self.search_endpoints = {
            'bing_news': 'https://api.bing.microsoft.com/v7.0/news/search',
            'bing_web': 'https://api.bing.microsoft.com/v7.0/search',
            'google_custom': 'https://www.googleapis.com/customsearch/v1',
            'newsapi': 'https://newsapi.org/v2/everything'
        }
        
        # Cache for search results (simple in-memory cache)
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("WebSearchIntegrator initialized")
    
    def _get_cache_key(self, query: str, search_type: str) -> str:
        """Generate cache key for search results"""
        return f"{search_type}:{hash(query.lower())}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    async def search_bing_news(self, query: str, count: int = 10) -> List[SearchResult]:
        """Search Bing News API"""
        if not self.bing_api_key:
            logger.warning("Bing API key not available")
            return []
        
        cache_key = self._get_cache_key(query, 'bing_news')
        if cache_key in self.search_cache and self._is_cache_valid(self.search_cache[cache_key]):
            return self.search_cache[cache_key]['results']
        
        try:
            headers = {'Ocp-Apim-Subscription-Key': self.bing_api_key}
            params = {
                'q': query,
                'count': count,
                'mkt': 'en-US',
                'sortBy': 'Relevance',
                'freshness': 'Month'  # Recent news
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.search_endpoints['bing_news'],
                    headers=headers,
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_bing_news_results(data)
                        
                        # Cache results
                        self.search_cache[cache_key] = {
                            'results': results,
                            'timestamp': time.time()
                        }
                        
                        return results
                    else:
                        logger.error(f"Bing News API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Bing News search error: {e}")
            return []
    
    def _parse_bing_news_results(self, data: Dict) -> List[SearchResult]:
        """Parse Bing News API response"""
        results = []
        
        for item in data.get('value', []):
            # Calculate credibility score based on source
            source_domain = urlparse(item.get('url', '')).netloc.lower()
            credibility = self.credible_sources.get(source_domain, 0.5)
            
            result = SearchResult(
                title=item.get('name', ''),
                url=item.get('url', ''),
                snippet=item.get('description', ''),
                source=item.get('provider', [{}])[0].get('name', 'Unknown'),
                published_date=item.get('datePublished', ''),
                credibility_score=credibility
            )
            results.append(result)
        
        return results
    
    async def search_google_custom(self, query: str, count: int = 10) -> List[SearchResult]:
        """Search Google Custom Search API"""
        if not self.google_api_key or not self.google_cx:
            logger.warning("Google Custom Search API credentials not available")
            return []
        
        cache_key = self._get_cache_key(query, 'google_custom')
        if cache_key in self.search_cache and self._is_cache_valid(self.search_cache[cache_key]):
            return self.search_cache[cache_key]['results']
        
        try:
            params = {
                'key': self.google_api_key,
                'cx': self.google_cx,
                'q': query,
                'num': min(count, 10),  # Google limits to 10 per request
                'dateRestrict': 'm6'  # Last 6 months
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.search_endpoints['google_custom'],
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_google_results(data)
                        
                        # Cache results
                        self.search_cache[cache_key] = {
                            'results': results,
                            'timestamp': time.time()
                        }
                        
                        return results
                    else:
                        logger.error(f"Google Custom Search API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Google Custom Search error: {e}")
            return []
    
    def _parse_google_results(self, data: Dict) -> List[SearchResult]:
        """Parse Google Custom Search API response"""
        results = []
        
        for item in data.get('items', []):
            # Calculate credibility score based on source
            source_domain = urlparse(item.get('link', '')).netloc.lower()
            credibility = self.credible_sources.get(source_domain, 0.5)
            
            result = SearchResult(
                title=item.get('title', ''),
                url=item.get('link', ''),
                snippet=item.get('snippet', ''),
                source=source_domain,
                credibility_score=credibility
            )
            results.append(result)
        
        return results
    
    async def search_newsapi(self, query: str, count: int = 10) -> List[SearchResult]:
        """Search NewsAPI.org"""
        if not self.news_api_key or self.news_api_key == 'demo_key_for_testing':
            logger.warning("NewsAPI key not available")
            return []
        
        cache_key = self._get_cache_key(query, 'newsapi')
        if cache_key in self.search_cache and self._is_cache_valid(self.search_cache[cache_key]):
            return self.search_cache[cache_key]['results']
        
        try:
            headers = {'X-API-Key': self.news_api_key}
            params = {
                'q': query,
                'pageSize': min(count, 100),
                'sortBy': 'relevancy',
                'language': 'en',
                'from': (datetime.now() - timedelta(days=30)).isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.search_endpoints['newsapi'],
                    headers=headers,
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_newsapi_results(data)
                        
                        # Cache results
                        self.search_cache[cache_key] = {
                            'results': results,
                            'timestamp': time.time()
                        }
                        
                        return results
                    else:
                        logger.error(f"NewsAPI error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"NewsAPI search error: {e}")
            return []
    
    def _parse_newsapi_results(self, data: Dict) -> List[SearchResult]:
        """Parse NewsAPI response"""
        results = []
        
        for item in data.get('articles', []):
            # Calculate credibility score based on source
            source_domain = urlparse(item.get('url', '')).netloc.lower()
            credibility = self.credible_sources.get(source_domain, 0.5)
            
            result = SearchResult(
                title=item.get('title', ''),
                url=item.get('url', ''),
                snippet=item.get('description', ''),
                source=item.get('source', {}).get('name', 'Unknown'),
                published_date=item.get('publishedAt', ''),
                credibility_score=credibility
            )
            results.append(result)
        
        return results
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text for search"""
        # Simple entity extraction - can be enhanced with NLP
        entities = []
        
        # Extract quoted text
        quoted_text = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted_text)
        
        # Extract capitalized words (potential names/places)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Extract numbers with context
        numbers_with_context = re.findall(r'\b\d+(?:[.,]\d+)*\s*(?:percent|%|million|billion|thousand|dollars?|years?|days?|months?)\b', text, re.IGNORECASE)
        entities.extend(numbers_with_context)
        
        # Remove duplicates and short entities
        entities = list(set([e.strip() for e in entities if len(e.strip()) > 2]))
        
        return entities[:10]  # Limit to top 10 entities
    
    def _calculate_relevance_score(self, search_result: SearchResult, query: str, entities: List[str]) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        query_lower = query.lower()
        title_lower = search_result.title.lower()
        snippet_lower = search_result.snippet.lower()
        
        # Exact query match in title (high weight)
        if query_lower in title_lower:
            score += 0.4
        
        # Exact query match in snippet
        if query_lower in snippet_lower:
            score += 0.2
        
        # Entity matches
        entity_matches = 0
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in title_lower or entity_lower in snippet_lower:
                entity_matches += 1
        
        if entities:
            score += (entity_matches / len(entities)) * 0.3
        
        # Credibility bonus
        score += search_result.credibility_score * 0.1
        
        return min(score, 1.0)
    
    async def comprehensive_search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """Perform comprehensive search across multiple APIs"""
        start_time = time.time()
        
        # Extract key entities for better search
        entities = self._extract_key_entities(query)
        
        # Create search tasks
        tasks = []
        
        # Bing News search
        tasks.append(self.search_bing_news(query, max_results // 3))
        
        # Google Custom Search
        tasks.append(self.search_google_custom(query, max_results // 3))
        
        # NewsAPI search
        tasks.append(self.search_newsapi(query, max_results // 3))
        
        # Execute searches in parallel
        try:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in comprehensive search: {e}")
            return []
        
        # Combine and deduplicate results
        all_results = []
        seen_urls = set()
        
        for result_list in search_results:
            if isinstance(result_list, list):
                for result in result_list:
                    if result.url not in seen_urls:
                        # Calculate relevance score
                        result.relevance_score = self._calculate_relevance_score(result, query, entities)
                        all_results.append(result)
                        seen_urls.add(result.url)
        
        # Sort by combined relevance and credibility score
        all_results.sort(
            key=lambda x: (x.relevance_score * 0.7 + x.credibility_score * 0.3),
            reverse=True
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Comprehensive search completed in {processing_time:.2f}s, found {len(all_results)} results")
        
        return all_results[:max_results]
    
    def _analyze_search_results(self, results: List[SearchResult], query: str) -> FactCheckResult:
        """Analyze search results to determine fact check verdict"""
        if not results:
            return FactCheckResult(
                verdict='Unverified',
                confidence=0.0,
                supporting_sources=[],
                contradicting_sources=[],
                summary='No reliable sources found to verify this claim.',
                processing_time=0.0
            )
        
        # Categorize results
        supporting_sources = []
        contradicting_sources = []
        neutral_sources = []
        
        # Keywords that suggest verification/debunking
        verification_keywords = ['true', 'confirmed', 'verified', 'accurate', 'correct', 'factual']
        debunking_keywords = ['false', 'fake', 'debunked', 'misleading', 'incorrect', 'hoax', 'myth']
        
        for result in results:
            text_to_analyze = (result.title + ' ' + result.snippet).lower()
            
            verification_score = sum(1 for keyword in verification_keywords if keyword in text_to_analyze)
            debunking_score = sum(1 for keyword in debunking_keywords if keyword in text_to_analyze)
            
            if verification_score > debunking_score and verification_score > 0:
                supporting_sources.append(result)
            elif debunking_score > verification_score and debunking_score > 0:
                contradicting_sources.append(result)
            else:
                neutral_sources.append(result)
        
        # Calculate confidence based on source quality and consensus
        total_credibility = sum(r.credibility_score for r in results)
        supporting_credibility = sum(r.credibility_score for r in supporting_sources)
        contradicting_credibility = sum(r.credibility_score for r in contradicting_sources)
        
        # Determine verdict
        if supporting_credibility > contradicting_credibility * 1.5:
            verdict = 'Real'
            confidence = min(supporting_credibility / total_credibility, 0.95)
        elif contradicting_credibility > supporting_credibility * 1.5:
            verdict = 'Fake'
            confidence = min(contradicting_credibility / total_credibility, 0.95)
        else:
            verdict = 'Unverified'
            confidence = 0.5
        
        # Generate summary
        summary_parts = []
        if supporting_sources:
            summary_parts.append(f"Found {len(supporting_sources)} sources supporting the claim")
        if contradicting_sources:
            summary_parts.append(f"Found {len(contradicting_sources)} sources contradicting the claim")
        if neutral_sources:
            summary_parts.append(f"Found {len(neutral_sources)} neutral sources")
        
        summary = '. '.join(summary_parts) if summary_parts else 'Limited information available for verification.'
        
        return FactCheckResult(
            verdict=verdict,
            confidence=confidence,
            supporting_sources=supporting_sources[:5],  # Top 5
            contradicting_sources=contradicting_sources[:5],  # Top 5
            summary=summary,
            processing_time=0.0  # Will be set by caller
        )
    
    async def fact_check_claim(self, claim: str) -> FactCheckResult:
        """Perform comprehensive fact checking of a claim"""
        start_time = time.time()
        
        try:
            # Perform comprehensive search
            search_results = await self.comprehensive_search(claim, max_results=30)
            
            # Analyze results
            fact_check_result = self._analyze_search_results(search_results, claim)
            
            # Set processing time
            fact_check_result.processing_time = time.time() - start_time
            
            logger.info(f"Fact check completed: {fact_check_result.verdict} (confidence: {fact_check_result.confidence:.2f})")
            
            return fact_check_result
        
        except Exception as e:
            logger.error(f"Error in fact checking: {e}")
            return FactCheckResult(
                verdict='Unverified',
                confidence=0.0,
                supporting_sources=[],
                contradicting_sources=[],
                summary=f'Error occurred during fact checking: {str(e)}',
                processing_time=time.time() - start_time
            )

# Global instance
web_search_integrator = WebSearchIntegrator()

# Convenience functions
async def search_news_for_verification(query: str) -> List[SearchResult]:
    """Search for news articles to verify a claim"""
    return await web_search_integrator.comprehensive_search(query)

async def verify_claim_with_web_search(claim: str) -> FactCheckResult:
    """Verify a claim using web search"""
    return await web_search_integrator.fact_check_claim(claim)

def run_fact_check_sync(claim: str) -> FactCheckResult:
    """Synchronous wrapper for fact checking"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(verify_claim_with_web_search(claim))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error in sync fact check: {e}")
        return FactCheckResult(
            verdict='Unverified',
            confidence=0.0,
            supporting_sources=[],
            contradicting_sources=[],
            summary=f'Error: {str(e)}',
            processing_time=0.0
        )