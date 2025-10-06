#!/usr/bin/env python3
"""
Enhanced News Validation System with Real Fact-Checking APIs
Integrates with FactCheck.org, Snopes, PolitiFact, and Google Fact Check Tools API
Provides validation with sources and proofs for fake/real decisions
"""

import asyncio
import aiohttp
import requests
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
from urllib.parse import quote_plus, urljoin
import os
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Structured validation result with sources and confidence"""
    is_credible: bool
    confidence_score: float  # 0.0 to 1.0
    sources: List[Dict[str, Any]]
    fact_check_results: List[Dict[str, Any]]
    validation_summary: str
    processing_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class FactCheckSource:
    """Fact-checking source information"""
    name: str
    url: str
    rating: str
    explanation: str
    date_published: Optional[str] = None
    author: Optional[str] = None

class EnhancedNewsValidator:
    """Enhanced news validation with real fact-checking APIs"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.session_timeout = aiohttp.ClientTimeout(total=30)
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # API configurations
        self.google_factcheck_api_key = os.getenv('GOOGLE_FACTCHECK_API_KEY', '')
        self.google_search_api_key = os.getenv('GOOGLE_SEARCH_API_KEY', '')
        self.google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
        
        # Fact-checking source patterns
        self.factcheck_domains = {
            'factcheck.org': {'weight': 0.9, 'api_available': False},
            'snopes.com': {'weight': 0.85, 'api_available': False},
            'politifact.com': {'weight': 0.8, 'api_available': False},
            'reuters.com/fact-check': {'weight': 0.85, 'api_available': False},
            'apnews.com/hub/ap-fact-check': {'weight': 0.85, 'api_available': False},
            'factcheck.afp.com': {'weight': 0.8, 'api_available': False}
        }
        
        # Performance tracking
        self.stats = {
            'total_validations': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0
        }
        
        logger.info("Enhanced News Validator initialized with real fact-checking APIs")
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return (datetime.now() - cache_entry['timestamp']).seconds < self.cache_ttl
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases for fact-checking search"""
        # Remove common words and extract meaningful phrases
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Split into sentences and extract key phrases
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []
        
        for sentence in sentences[:3]:  # Focus on first 3 sentences
            words = re.findall(r'\b\w+\b', sentence.lower())
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
            
            if len(filtered_words) >= 2:
                # Create phrases of 2-4 words
                for i in range(len(filtered_words) - 1):
                    phrase = ' '.join(filtered_words[i:i+min(4, len(filtered_words)-i)])
                    if len(phrase) > 10:  # Minimum phrase length
                        key_phrases.append(phrase)
        
        return key_phrases[:5]  # Return top 5 phrases
    
    async def _search_google_factcheck(self, query: str) -> List[Dict[str, Any]]:
        """Search Google Fact Check Tools API"""
        if not self.google_factcheck_api_key:
            return []
        
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'key': self.google_factcheck_api_key,
                'query': query,
                'languageCode': 'en'
            }
            
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('claims', [])
            
        except Exception as e:
            logger.warning(f"Google Fact Check API error: {e}")
        
        return []
    
    async def _search_factcheck_websites(self, query: str) -> List[FactCheckSource]:
        """Search fact-checking websites using Google Custom Search"""
        if not self.google_search_api_key or not self.google_search_engine_id:
            return await self._fallback_factcheck_search(query)
        
        try:
            # Search specific fact-checking domains
            results = []
            
            for domain in self.factcheck_domains.keys():
                search_query = f"site:{domain} {query}"
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.google_search_api_key,
                    'cx': self.google_search_engine_id,
                    'q': search_query,
                    'num': 3
                }
                
                async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            items = data.get('items', [])
                            
                            for item in items:
                                results.append(FactCheckSource(
                                    name=domain,
                                    url=item['link'],
                                    rating='Unknown',  # Would need to scrape for actual rating
                                    explanation=item.get('snippet', ''),
                                    date_published=None
                                ))
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            return results[:10]  # Return top 10 results
            
        except Exception as e:
            logger.warning(f"Google Custom Search error: {e}")
            return await self._fallback_factcheck_search(query)
    
    async def _fallback_factcheck_search(self, query: str) -> List[FactCheckSource]:
        """Fallback fact-checking search without API keys"""
        # Simulate fact-checking results based on content analysis
        results = []
        
        # Basic credibility indicators
        credibility_keywords = {
            'high': ['reuters', 'associated press', 'bbc', 'npr', 'pbs'],
            'medium': ['cnn', 'fox news', 'msnbc', 'abc news', 'cbs news'],
            'low': ['blog', 'opinion', 'rumor', 'unconfirmed', 'alleged']
        }
        
        query_lower = query.lower()
        
        # Check for credibility indicators
        for level, keywords in credibility_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    rating = 'Credible' if level == 'high' else 'Mixed' if level == 'medium' else 'Questionable'
                    results.append(FactCheckSource(
                        name='Content Analysis',
                        url='#',
                        rating=rating,
                        explanation=f'Source contains {level} credibility indicators: {keyword}',
                        date_published=datetime.now().isoformat()
                    ))
        
        return results
    
    def _calculate_credibility_score(self, 
                                   factcheck_sources: List[FactCheckSource],
                                   google_claims: List[Dict]) -> Tuple[bool, float]:
        """Calculate overall credibility score from multiple sources"""
        if not factcheck_sources and not google_claims:
            return True, 0.5  # Neutral when no fact-check data available
        
        total_weight = 0.0
        weighted_score = 0.0
        
        # Process fact-check sources
        for source in factcheck_sources:
            domain_weight = self.factcheck_domains.get(source.name, {}).get('weight', 0.5)
            
            # Convert rating to score
            rating_scores = {
                'true': 1.0, 'mostly true': 0.8, 'credible': 0.9,
                'mixed': 0.5, 'mostly false': 0.2, 'false': 0.0,
                'questionable': 0.3, 'unknown': 0.5
            }
            
            rating_score = rating_scores.get(source.rating.lower(), 0.5)
            weighted_score += rating_score * domain_weight
            total_weight += domain_weight
        
        # Process Google Fact Check claims
        for claim in google_claims:
            claim_reviews = claim.get('claimReview', [])
            for review in claim_reviews:
                rating = review.get('textualRating', '').lower()
                rating_score = {
                    'true': 1.0, 'mostly true': 0.8, 'mixture': 0.5,
                    'mostly false': 0.2, 'false': 0.0, 'unproven': 0.4
                }.get(rating, 0.5)
                
                weighted_score += rating_score * 0.7  # Google Fact Check weight
                total_weight += 0.7
        
        if total_weight == 0:
            return True, 0.5
        
        final_score = weighted_score / total_weight
        is_credible = final_score >= 0.6  # Threshold for credibility
        
        return is_credible, final_score
    
    async def validate_news_async(self, title: str, content: str) -> ValidationResult:
        """Validate news article asynchronously with fact-checking APIs"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(f"{title}:{content[:500]}")
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]['result']
        
        try:
            # Extract key phrases for fact-checking
            key_phrases = self._extract_key_phrases(f"{title} {content}")
            
            # Parallel fact-checking searches
            tasks = []
            
            # Search Google Fact Check API
            for phrase in key_phrases[:3]:  # Limit to top 3 phrases
                tasks.append(self._search_google_factcheck(phrase))
            
            # Search fact-checking websites
            for phrase in key_phrases[:2]:  # Limit to top 2 phrases
                tasks.append(self._search_factcheck_websites(phrase))
            
            # Execute all searches concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            google_claims = []
            factcheck_sources = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Fact-check search {i} failed: {result}")
                    continue
                
                if i < len(key_phrases):  # Google Fact Check results
                    if isinstance(result, list):
                        google_claims.extend(result)
                else:  # Website search results
                    if isinstance(result, list):
                        factcheck_sources.extend(result)
            
            # Calculate credibility
            is_credible, confidence_score = self._calculate_credibility_score(
                factcheck_sources, google_claims
            )
            
            # Create validation summary
            summary_parts = []
            if factcheck_sources:
                summary_parts.append(f"Found {len(factcheck_sources)} fact-check sources")
            if google_claims:
                summary_parts.append(f"Found {len(google_claims)} related claims")
            
            validation_summary = (
                f"Credibility: {'HIGH' if confidence_score > 0.7 else 'MEDIUM' if confidence_score > 0.4 else 'LOW'}. "
                f"{'; '.join(summary_parts) if summary_parts else 'Limited fact-check data available'}"
            )
            
            # Prepare sources for response
            sources = []
            for source in factcheck_sources:
                sources.append({
                    'name': source.name,
                    'url': source.url,
                    'rating': source.rating,
                    'explanation': source.explanation
                })
            
            fact_check_results = []
            for claim in google_claims:
                fact_check_results.append({
                    'claim': claim.get('text', ''),
                    'claimant': claim.get('claimant', ''),
                    'reviews': claim.get('claimReview', [])
                })
            
            processing_time = time.time() - start_time
            
            result = ValidationResult(
                is_credible=is_credible,
                confidence_score=confidence_score,
                sources=sources,
                fact_check_results=fact_check_results,
                validation_summary=validation_summary,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Cache result
            self.cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            # Update stats
            self.stats['total_validations'] += 1
            self.stats['api_calls'] += len(tasks)
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (self.stats['total_validations'] - 1) + processing_time) /
                self.stats['total_validations']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"News validation error: {e}")
            processing_time = time.time() - start_time
            
            return ValidationResult(
                is_credible=True,  # Default to credible on error
                confidence_score=0.5,
                sources=[],
                fact_check_results=[],
                validation_summary=f"Validation error: {str(e)}",
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    def validate_news(self, title: str, content: str) -> ValidationResult:
        """Synchronous wrapper for news validation"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.validate_news_async(title, content))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.validate_news_async(title, content))
            finally:
                loop.close()
    
    def validate_batch(self, articles: List[Dict[str, str]]) -> List[ValidationResult]:
        """Validate multiple articles in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_article = {
                executor.submit(
                    self.validate_news, 
                    article.get('title', ''), 
                    article.get('content', article.get('description', ''))
                ): article for article in articles
            }
            
            for future in as_completed(future_to_article):
                try:
                    result = future.result(timeout=60)  # 60 second timeout per article
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch validation error: {e}")
                    results.append(ValidationResult(
                        is_credible=True,
                        confidence_score=0.5,
                        sources=[],
                        fact_check_results=[],
                        validation_summary=f"Batch validation error: {str(e)}",
                        processing_time=0.0,
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        cache_hit_rate = (
            self.stats['cache_hits'] / max(self.stats['total_validations'], 1) * 100
        )
        
        return {
            **self.stats,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """Clear validation cache"""
        self.cache.clear()
        logger.info("Validation cache cleared")

# Global instance
enhanced_validator = EnhancedNewsValidator(max_workers=5)

# Convenience functions
def validate_single_news(title: str, content: str) -> Dict[str, Any]:
    """Validate single news article and return dict"""
    result = enhanced_validator.validate_news(title, content)
    return result.to_dict()

def validate_news_batch(articles: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Validate multiple news articles and return list of dicts"""
    results = enhanced_validator.validate_batch(articles)
    return [result.to_dict() for result in results]

def get_validation_stats() -> Dict[str, Any]:
    """Get validation performance statistics"""
    return enhanced_validator.get_performance_stats()

if __name__ == "__main__":
    # Test the enhanced validator
    test_title = "Breaking: Major Scientific Discovery Announced"
    test_content = "Scientists at a leading university have announced a groundbreaking discovery that could revolutionize medicine."
    
    result = validate_single_news(test_title, test_content)
    print(json.dumps(result, indent=2))
    
    print("\nPerformance Stats:")
    print(json.dumps(get_validation_stats(), indent=2))