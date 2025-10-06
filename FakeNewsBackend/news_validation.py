#!/usr/bin/env python3
"""
News Validation Module with FactCheck.org and Snopes Integration
Provides validation for fake/real news decisions with sources and proofs

Features:
- FactCheck.org API integration
- Snopes fact-checking integration
- Web search validation
- Source credibility scoring
- Proof aggregation and ranking
- Automated validation pipeline
"""

import requests
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote_plus, urlparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status enumeration"""
    VERIFIED_TRUE = "verified_true"
    VERIFIED_FALSE = "verified_false"
    PARTIALLY_TRUE = "partially_true"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"
    SATIRE = "satire"
    ERROR = "error"

@dataclass
class ValidationSource:
    """Data class for validation sources"""
    name: str
    url: str
    credibility_score: float
    verdict: str
    summary: str
    date_checked: str
    confidence: float

@dataclass
class ValidationResult:
    """Data class for validation results"""
    status: ValidationStatus
    confidence_score: float
    sources: List[ValidationSource]
    summary: str
    evidence_count: int
    last_updated: str
    search_queries_used: List[str]

class NewsValidationEngine:
    """Advanced news validation engine with multiple fact-checking sources"""
    
    def __init__(self):
        # API configurations
        self.factcheck_api_key = "YOUR_FACTCHECK_API_KEY"  # Replace with actual key
        self.snopes_api_key = "YOUR_SNOPES_API_KEY"  # Replace with actual key
        self.google_search_api_key = "YOUR_GOOGLE_API_KEY"  # Replace with actual key
        self.google_cx = "YOUR_GOOGLE_CX"  # Replace with actual CX
        
        # Fact-checking sources configuration
        self.fact_check_sources = {
            'factcheck_org': {
                'url': 'https://factcheck-org-api.p.rapidapi.com/search',
                'headers': {
                    'X-RapidAPI-Key': self.factcheck_api_key,
                    'X-RapidAPI-Host': 'factcheck-org-api.p.rapidapi.com'
                },
                'credibility': 0.95
            },
            'snopes': {
                'url': 'https://snopes-fact-check.p.rapidapi.com/search',
                'headers': {
                    'X-RapidAPI-Key': self.snopes_api_key,
                    'X-RapidAPI-Host': 'snopes-fact-check.p.rapidapi.com'
                },
                'credibility': 0.90
            },
            'politifact': {
                'url': 'https://politifact-fact-check.p.rapidapi.com/search',
                'headers': {
                    'X-RapidAPI-Key': self.factcheck_api_key,
                    'X-RapidAPI-Host': 'politifact-fact-check.p.rapidapi.com'
                },
                'credibility': 0.88
            }
        }
        
        # Google Custom Search for additional validation
        self.google_search_config = {
            'url': 'https://www.googleapis.com/customsearch/v1',
            'params': {
                'key': self.google_search_api_key,
                'cx': self.google_cx,
                'num': 10
            }
        }
        
        # Cache for validation results
        self.validation_cache = {}
        self.cache_duration = 3600  # 1 hour
        
        # Performance tracking
        self.validation_count = 0
        self.cache_hits = 0
        self.api_errors = 0
        
        # Credible news sources for cross-referencing
        self.credible_sources = {
            'reuters.com': 0.95,
            'apnews.com': 0.94,
            'bbc.com': 0.93,
            'npr.org': 0.92,
            'pbs.org': 0.91,
            'cnn.com': 0.85,
            'nytimes.com': 0.90,
            'washingtonpost.com': 0.89,
            'wsj.com': 0.88,
            'thehindu.com': 0.87,
            'ndtv.com': 0.82,
            'timesofindia.indiatimes.com': 0.80
        }
    
    def _generate_cache_key(self, text: str, url: str = "") -> str:
        """Generate cache key for validation request"""
        import hashlib
        content = f"{text}_{url}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        
        cache_time = cache_entry.get('timestamp', 0)
        return (time.time() - cache_time) < self.cache_duration
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases for fact-checking queries"""
        # Remove common words and extract meaningful phrases
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Extract sentences and key phrases
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []
        
        for sentence in sentences[:3]:  # Focus on first 3 sentences
            words = re.findall(r'\b\w+\b', sentence.lower())
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
            
            if len(filtered_words) >= 2:
                key_phrases.append(' '.join(filtered_words[:5]))  # Max 5 words per phrase
        
        # Add full title/headline if short enough
        if len(text) < 200:
            key_phrases.insert(0, text.strip())
        
        return key_phrases[:5]  # Return top 5 phrases
    
    def _search_factcheck_org(self, query: str) -> List[ValidationSource]:
        """Search FactCheck.org for validation"""
        sources = []
        
        try:
            # Use web search since direct API might not be available
            search_query = f"site:factcheck.org {query}"
            results = self._google_search(search_query, 5)
            
            for result in results:
                if 'factcheck.org' in result.get('link', ''):
                    # Extract verdict from snippet or title
                    snippet = result.get('snippet', '').lower()
                    title = result.get('title', '').lower()
                    
                    verdict = "unverified"
                    confidence = 0.5
                    
                    if any(word in snippet + title for word in ['false', 'fake', 'misleading', 'incorrect']):
                        verdict = "false"
                        confidence = 0.85
                    elif any(word in snippet + title for word in ['true', 'correct', 'accurate', 'verified']):
                        verdict = "true"
                        confidence = 0.85
                    elif any(word in snippet + title for word in ['mixed', 'partially', 'mostly']):
                        verdict = "partially_true"
                        confidence = 0.70
                    
                    source = ValidationSource(
                        name="FactCheck.org",
                        url=result.get('link', ''),
                        credibility_score=0.95,
                        verdict=verdict,
                        summary=result.get('snippet', '')[:200],
                        date_checked=datetime.now().isoformat(),
                        confidence=confidence
                    )
                    sources.append(source)
            
        except Exception as e:
            logger.error(f"Error searching FactCheck.org: {str(e)}")
            self.api_errors += 1
        
        return sources
    
    def _search_snopes(self, query: str) -> List[ValidationSource]:
        """Search Snopes for validation"""
        sources = []
        
        try:
            # Use web search for Snopes
            search_query = f"site:snopes.com {query}"
            results = self._google_search(search_query, 5)
            
            for result in results:
                if 'snopes.com' in result.get('link', ''):
                    snippet = result.get('snippet', '').lower()
                    title = result.get('title', '').lower()
                    
                    verdict = "unverified"
                    confidence = 0.5
                    
                    # Snopes-specific verdict detection
                    if any(word in snippet + title for word in ['false', 'fake', 'legend', 'hoax']):
                        verdict = "false"
                        confidence = 0.90
                    elif any(word in snippet + title for word in ['true', 'correct', 'real']):
                        verdict = "true"
                        confidence = 0.90
                    elif any(word in snippet + title for word in ['mixture', 'mixed', 'mostly']):
                        verdict = "partially_true"
                        confidence = 0.75
                    elif 'satire' in snippet + title:
                        verdict = "satire"
                        confidence = 0.95
                    
                    source = ValidationSource(
                        name="Snopes",
                        url=result.get('link', ''),
                        credibility_score=0.90,
                        verdict=verdict,
                        summary=result.get('snippet', '')[:200],
                        date_checked=datetime.now().isoformat(),
                        confidence=confidence
                    )
                    sources.append(source)
            
        except Exception as e:
            logger.error(f"Error searching Snopes: {str(e)}")
            self.api_errors += 1
        
        return sources
    
    def _google_search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform Google Custom Search"""
        try:
            params = self.google_search_config['params'].copy()
            params['q'] = query
            params['num'] = min(num_results, 10)
            
            response = requests.get(
                self.google_search_config['url'],
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            else:
                logger.warning(f"Google Search API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Google Search error: {str(e)}")
            return []
    
    def _search_credible_sources(self, query: str) -> List[ValidationSource]:
        """Search credible news sources for corroboration"""
        sources = []
        
        try:
            # Search across credible sources
            for domain, credibility in list(self.credible_sources.items())[:5]:
                search_query = f"site:{domain} {query}"
                results = self._google_search(search_query, 3)
                
                for result in results:
                    if domain in result.get('link', ''):
                        source = ValidationSource(
                            name=domain.replace('.com', '').replace('.org', '').title(),
                            url=result.get('link', ''),
                            credibility_score=credibility,
                            verdict="corroborated",
                            summary=result.get('snippet', '')[:200],
                            date_checked=datetime.now().isoformat(),
                            confidence=credibility
                        )
                        sources.append(source)
                        break  # One result per source
            
        except Exception as e:
            logger.error(f"Error searching credible sources: {str(e)}")
        
        return sources
    
    def validate_news(self, title: str, content: str = "", url: str = "") -> ValidationResult:
        """Comprehensive news validation with multiple sources"""
        # Check cache first
        cache_key = self._generate_cache_key(title + content, url)
        
        if cache_key in self.validation_cache and self._is_cache_valid(self.validation_cache[cache_key]):
            self.cache_hits += 1
            logger.info(f"Cache hit for validation: {title[:50]}...")
            return self.validation_cache[cache_key]['result']
        
        self.validation_count += 1
        logger.info(f"Starting validation for: {title[:50]}...")
        
        # Extract key phrases for searching
        search_text = f"{title} {content}"[:500]  # Limit to 500 chars
        key_phrases = self._extract_key_phrases(search_text)
        
        all_sources = []
        
        # Search fact-checking sources in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Submit fact-checking searches
            for phrase in key_phrases[:3]:  # Use top 3 phrases
                futures.append(executor.submit(self._search_factcheck_org, phrase))
                futures.append(executor.submit(self._search_snopes, phrase))
            
            # Submit credible source search
            if key_phrases:
                futures.append(executor.submit(self._search_credible_sources, key_phrases[0]))
            
            # Collect results
            for future in as_completed(futures, timeout=30):
                try:
                    sources = future.result()
                    all_sources.extend(sources)
                except Exception as e:
                    logger.error(f"Error in parallel validation: {str(e)}")
        
        # Analyze and aggregate results
        result = self._analyze_validation_results(all_sources, key_phrases)
        
        # Cache the result
        self.validation_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        logger.info(f"Validation completed: {result.status.value} (confidence: {result.confidence_score:.2f})")
        return result
    
    def _analyze_validation_results(self, sources: List[ValidationSource], queries: List[str]) -> ValidationResult:
        """Analyze validation sources and determine final verdict"""
        if not sources:
            return ValidationResult(
                status=ValidationStatus.UNVERIFIED,
                confidence_score=0.0,
                sources=[],
                summary="No validation sources found",
                evidence_count=0,
                last_updated=datetime.now().isoformat(),
                search_queries_used=queries
            )
        
        # Count verdicts weighted by credibility
        verdict_scores = {
            'false': 0.0,
            'true': 0.0,
            'partially_true': 0.0,
            'satire': 0.0,
            'corroborated': 0.0
        }
        
        total_weight = 0.0
        
        for source in sources:
            weight = source.credibility_score * source.confidence
            verdict_scores[source.verdict] += weight
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for verdict in verdict_scores:
                verdict_scores[verdict] /= total_weight
        
        # Determine final status
        max_score = max(verdict_scores.values())
        dominant_verdict = max(verdict_scores, key=verdict_scores.get)
        
        # Map to ValidationStatus
        status_mapping = {
            'false': ValidationStatus.VERIFIED_FALSE,
            'true': ValidationStatus.VERIFIED_TRUE,
            'partially_true': ValidationStatus.PARTIALLY_TRUE,
            'satire': ValidationStatus.SATIRE,
            'corroborated': ValidationStatus.VERIFIED_TRUE
        }
        
        final_status = status_mapping.get(dominant_verdict, ValidationStatus.UNVERIFIED)
        
        # Adjust confidence based on evidence quality
        confidence_score = min(max_score * len(sources) / 3, 1.0)  # Scale by evidence count
        
        # Generate summary
        fact_check_sources = [s for s in sources if s.name in ['FactCheck.org', 'Snopes']]
        credible_sources = [s for s in sources if s.verdict == 'corroborated']
        
        summary_parts = []
        if fact_check_sources:
            summary_parts.append(f"Fact-checked by {len(fact_check_sources)} sources")
        if credible_sources:
            summary_parts.append(f"Corroborated by {len(credible_sources)} credible news outlets")
        
        summary = "; ".join(summary_parts) if summary_parts else "Limited validation sources available"
        
        return ValidationResult(
            status=final_status,
            confidence_score=confidence_score,
            sources=sources[:10],  # Limit to top 10 sources
            summary=summary,
            evidence_count=len(sources),
            last_updated=datetime.now().isoformat(),
            search_queries_used=queries
        )
    
    def batch_validate(self, news_items: List[Dict]) -> List[Dict]:
        """Validate multiple news items in batch"""
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit validation tasks
            future_to_item = {
                executor.submit(
                    self.validate_news,
                    item.get('title', ''),
                    item.get('description', ''),
                    item.get('url', '')
                ): item for item in news_items
            }
            
            # Collect results
            for future in as_completed(future_to_item, timeout=60):
                item = future_to_item[future]
                try:
                    validation_result = future.result()
                    
                    # Add validation info to original item
                    enhanced_item = item.copy()
                    enhanced_item['validation'] = {
                        'status': validation_result.status.value,
                        'confidence': validation_result.confidence_score,
                        'sources_count': validation_result.evidence_count,
                        'summary': validation_result.summary,
                        'last_checked': validation_result.last_updated,
                        'sources': [{
                            'name': s.name,
                            'url': s.url,
                            'verdict': s.verdict,
                            'credibility': s.credibility_score
                        } for s in validation_result.sources[:3]]  # Top 3 sources
                    }
                    
                    results.append(enhanced_item)
                    
                except Exception as e:
                    logger.error(f"Error validating item {item.get('title', '')}: {str(e)}")
                    # Add error validation info
                    enhanced_item = item.copy()
                    enhanced_item['validation'] = {
                        'status': 'error',
                        'confidence': 0.0,
                        'sources_count': 0,
                        'summary': f"Validation error: {str(e)}",
                        'last_checked': datetime.now().isoformat(),
                        'sources': []
                    }
                    results.append(enhanced_item)
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        cache_hit_rate = (self.cache_hits / max(self.validation_count, 1)) * 100
        
        return {
            'total_validations': self.validation_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'api_errors': self.api_errors,
            'cached_results': len(self.validation_cache),
            'error_rate': f"{(self.api_errors / max(self.validation_count, 1)) * 100:.1f}%"
        }
    
    def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()
        logger.info("Validation cache cleared")

class NewsValidator:
    """Simplified news validator interface for easy integration"""
    
    def __init__(self):
        self.engine = NewsValidationEngine()
        self.batch_size = 10
        self.timeout = 30
    
    def validate(self, title: str, content: str = "", url: str = "") -> Dict[str, Any]:
        """Validate a single news article"""
        try:
            result = self.engine.validate_news(title, content, url)
            
            return {
                'is_valid': result.status in [ValidationStatus.VERIFIED_TRUE, ValidationStatus.PARTIALLY_TRUE],
                'status': result.status.value,
                'confidence': result.confidence_score,
                'credibility_score': result.confidence_score,
                'sources': [{
                    'name': s.name,
                    'url': s.url,
                    'verdict': s.verdict,
                    'credibility': s.credibility_score
                } for s in result.sources[:3]],
                'summary': result.summary,
                'evidence_count': result.evidence_count,
                'last_updated': result.last_updated
            }
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                'is_valid': False,
                'status': 'error',
                'confidence': 0.0,
                'credibility_score': 0.0,
                'sources': [],
                'summary': f"Validation failed: {str(e)}",
                'evidence_count': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def validate_batch(self, articles: List[Dict]) -> List[Dict]:
        """Validate multiple articles in batch"""
        try:
            return self.engine.batch_validate(articles)
        except Exception as e:
            logger.error(f"Batch validation error: {str(e)}")
            # Return error results for all articles
            return [{
                **article,
                'validation': {
                    'status': 'error',
                    'confidence': 0.0,
                    'sources_count': 0,
                    'summary': f"Batch validation failed: {str(e)}",
                    'last_checked': datetime.now().isoformat(),
                    'sources': []
                }
            } for article in articles]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.engine.get_validation_stats()
    
    def clear_cache(self):
        """Clear validation cache"""
        self.engine.clear_cache()

# Global instances
validation_engine = NewsValidationEngine()
news_validator = NewsValidator()

# Convenience functions
def validate_single_news(title: str, content: str = "", url: str = "") -> Dict[str, Any]:
    """Validate a single news item"""
    result = validation_engine.validate_news(title, content, url)
    
    return {
        'status': result.status.value,
        'confidence': result.confidence_score,
        'sources_count': result.evidence_count,
        'summary': result.summary,
        'last_checked': result.last_updated,
        'sources': [{
            'name': s.name,
            'url': s.url,
            'verdict': s.verdict,
            'credibility': s.credibility_score,
            'summary': s.summary
        } for s in result.sources[:5]]  # Top 5 sources
    }

def validate_news_batch(news_items: List[Dict]) -> List[Dict]:
    """Validate multiple news items"""
    return validation_engine.batch_validate(news_items)

def get_validation_performance() -> Dict[str, Any]:
    """Get validation performance stats"""
    return validation_engine.get_validation_stats()

if __name__ == "__main__":
    # Test the validation system
    print("Testing News Validation System...")
    
    # Test single validation
    test_title = "Scientists discover new planet in solar system"
    test_content = "Researchers claim to have found a new planet beyond Pluto"
    
    result = validate_single_news(test_title, test_content)
    print(f"Validation result: {result['status']} (confidence: {result['confidence']:.2f})")
    print(f"Sources found: {result['sources_count']}")
    
    # Show performance stats
    stats = get_validation_performance()
    print(f"Performance: {stats}")