#!/usr/bin/env python3
"""
URL Scraping Module for Article Text Extraction
Implements robust web scraping with multiple fallback methods

Author: AI Assistant
Date: January 2025
Version: 1.0.0
"""

import os
import logging
import requests
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin
import re
from dataclasses import dataclass

try:
    from newspaper import Article, Config
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logging.warning("newspaper3k not available, using BeautifulSoup fallback")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.error("BeautifulSoup not available")

try:
    import readability
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ScrapedArticle:
    """Structured scraped article data"""
    title: str
    text: str
    authors: List[str]
    publish_date: Optional[str]
    url: str
    source_domain: str
    meta_description: str
    keywords: List[str]
    images: List[str]
    success: bool
    error_message: Optional[str] = None
    extraction_method: str = 'unknown'
    processing_time: float = 0.0

class URLScraper:
    """Comprehensive URL scraping with multiple extraction methods"""
    
    def __init__(self):
        # Configure newspaper3k if available
        if NEWSPAPER_AVAILABLE:
            self.newspaper_config = Config()
            self.newspaper_config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            self.newspaper_config.request_timeout = 10
            self.newspaper_config.number_threads = 1
            self.newspaper_config.fetch_images = False
            self.newspaper_config.memoize_articles = False
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Common article content selectors (CSS selectors)
        self.content_selectors = [
            'article',
            '[role="main"]',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            '.story-body',
            '.article-body',
            '.post-body',
            '.main-content',
            '#content',
            '#main-content',
            '.article-text',
            '.story-content',
            '.news-content'
        ]
        
        # Selectors to remove (ads, navigation, etc.)
        self.remove_selectors = [
            'script',
            'style',
            'nav',
            'header',
            'footer',
            '.advertisement',
            '.ad',
            '.ads',
            '.social-share',
            '.comments',
            '.related-articles',
            '.sidebar',
            '.navigation',
            '.menu'
        ]
        
        # Cache for scraped articles
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("URLScraper initialized")
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return f"url_scrape:{hash(url)}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    def _extract_with_newspaper(self, url: str) -> Optional[ScrapedArticle]:
        """Extract article using newspaper3k"""
        if not NEWSPAPER_AVAILABLE:
            return None
        
        try:
            article = Article(url, config=self.newspaper_config)
            article.download()
            article.parse()
            
            # Extract additional metadata
            article.nlp()
            
            return ScrapedArticle(
                title=article.title or '',
                text=article.text or '',
                authors=article.authors or [],
                publish_date=article.publish_date.isoformat() if article.publish_date else None,
                url=url,
                source_domain=urlparse(url).netloc,
                meta_description=article.meta_description or '',
                keywords=article.keywords or [],
                images=[img for img in article.images] if article.images else [],
                success=bool(article.text),
                extraction_method='newspaper3k',
                processing_time=0.0
            )
        
        except Exception as e:
            logger.warning(f"Newspaper3k extraction failed for {url}: {e}")
            return None
    
    def _extract_with_beautifulsoup(self, url: str) -> Optional[ScrapedArticle]:
        """Extract article using BeautifulSoup fallback"""
        if not BS4_AVAILABLE:
            return None
        
        try:
            # Fetch the page
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for selector in self.remove_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract title
            title = ''
            title_selectors = ['h1', 'title', '.article-title', '.post-title', '.entry-title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            # Extract main content
            content_text = ''
            for selector in self.content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Get all paragraph text
                    paragraphs = content_elem.find_all(['p', 'div'], recursive=True)
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 20:  # Filter out short snippets
                            content_parts.append(text)
                    
                    content_text = '\n\n'.join(content_parts)
                    if content_text:
                        break
            
            # Fallback: get all paragraph text
            if not content_text:
                paragraphs = soup.find_all('p')
                content_parts = []
                for p in paragraphs:
                    text = p.get_text().strip()
                    if len(text) > 20:
                        content_parts.append(text)
                content_text = '\n\n'.join(content_parts)
            
            # Extract meta description
            meta_desc = ''
            meta_elem = soup.find('meta', attrs={'name': 'description'})
            if meta_elem:
                meta_desc = meta_elem.get('content', '')
            
            # Extract authors (common patterns)
            authors = []
            author_selectors = ['.author', '.byline', '[rel="author"]', '.article-author']
            for selector in author_selectors:
                author_elems = soup.select(selector)
                for elem in author_elems:
                    author_text = elem.get_text().strip()
                    if author_text and len(author_text) < 100:
                        authors.append(author_text)
            
            # Extract images
            images = []
            img_elems = soup.find_all('img')
            for img in img_elems:
                src = img.get('src')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(url, src)
                    images.append(src)
            
            return ScrapedArticle(
                title=title,
                text=content_text,
                authors=list(set(authors))[:5],  # Deduplicate and limit
                publish_date=None,  # Could be enhanced with date extraction
                url=url,
                source_domain=urlparse(url).netloc,
                meta_description=meta_desc,
                keywords=[],  # Could be enhanced with keyword extraction
                images=images[:10],  # Limit images
                success=bool(content_text),
                extraction_method='beautifulsoup',
                processing_time=0.0
            )
        
        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed for {url}: {e}")
            return None
    
    def _extract_with_readability(self, url: str) -> Optional[ScrapedArticle]:
        """Extract article using readability-lxml"""
        if not READABILITY_AVAILABLE:
            return None
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            doc = readability.Document(response.content)
            
            return ScrapedArticle(
                title=doc.title() or '',
                text=BeautifulSoup(doc.summary(), 'html.parser').get_text() if BS4_AVAILABLE else doc.summary(),
                authors=[],
                publish_date=None,
                url=url,
                source_domain=urlparse(url).netloc,
                meta_description='',
                keywords=[],
                images=[],
                success=True,
                extraction_method='readability',
                processing_time=0.0
            )
        
        except Exception as e:
            logger.warning(f"Readability extraction failed for {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ''
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common boilerplate text
        boilerplate_patterns = [
            r'Subscribe to our newsletter.*',
            r'Follow us on.*',
            r'Share this article.*',
            r'Click here to.*',
            r'Advertisement.*',
            r'Related articles.*',
            r'More from.*'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def scrape_article(self, url: str) -> ScrapedArticle:
        """Scrape article from URL with multiple fallback methods"""
        start_time = time.time()
        
        # Validate URL
        if not self._is_valid_url(url):
            return ScrapedArticle(
                title='',
                text='',
                authors=[],
                publish_date=None,
                url=url,
                source_domain='',
                meta_description='',
                keywords=[],
                images=[],
                success=False,
                error_message='Invalid URL format',
                extraction_method='none',
                processing_time=time.time() - start_time
            )
        
        # Check cache
        cache_key = self._get_cache_key(url)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            cached_result = self.cache[cache_key]['result']
            logger.info(f"Returning cached result for {url}")
            return cached_result
        
        # Try extraction methods in order of preference
        extraction_methods = [
            ('newspaper3k', self._extract_with_newspaper),
            ('readability', self._extract_with_readability),
            ('beautifulsoup', self._extract_with_beautifulsoup)
        ]
        
        result = None
        for method_name, method_func in extraction_methods:
            try:
                logger.info(f"Trying {method_name} extraction for {url}")
                result = method_func(url)
                
                if result and result.success and len(result.text.strip()) > 100:
                    # Clean the extracted text
                    result.text = self._clean_text(result.text)
                    result.processing_time = time.time() - start_time
                    
                    # Cache successful result
                    self.cache[cache_key] = {
                        'result': result,
                        'timestamp': time.time()
                    }
                    
                    logger.info(f"Successfully extracted article using {method_name}: {len(result.text)} characters")
                    return result
                
            except Exception as e:
                logger.warning(f"{method_name} extraction failed for {url}: {e}")
                continue
        
        # If all methods failed
        error_result = ScrapedArticle(
            title='',
            text='',
            authors=[],
            publish_date=None,
            url=url,
            source_domain=urlparse(url).netloc,
            meta_description='',
            keywords=[],
            images=[],
            success=False,
            error_message='All extraction methods failed',
            extraction_method='none',
            processing_time=time.time() - start_time
        )
        
        logger.error(f"Failed to extract article from {url}")
        return error_result
    
    def extract_text_from_url(self, url: str) -> str:
        """Simple interface to extract just the text content"""
        result = self.scrape_article(url)
        return result.text if result.success else ''
    
    def batch_scrape(self, urls: List[str]) -> List[ScrapedArticle]:
        """Scrape multiple URLs"""
        results = []
        for url in urls:
            try:
                result = self.scrape_article(url)
                results.append(result)
                time.sleep(0.5)  # Be respectful to servers
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                results.append(ScrapedArticle(
                    title='',
                    text='',
                    authors=[],
                    publish_date=None,
                    url=url,
                    source_domain=urlparse(url).netloc,
                    meta_description='',
                    keywords=[],
                    images=[],
                    success=False,
                    error_message=str(e),
                    extraction_method='none',
                    processing_time=0.0
                ))
        
        return results

# Global instance
url_scraper = URLScraper()

# Convenience functions
def extract_text_from_url(url: str) -> str:
    """Extract text content from URL"""
    return url_scraper.extract_text_from_url(url)

def scrape_article_from_url(url: str) -> ScrapedArticle:
    """Scrape complete article data from URL"""
    return url_scraper.scrape_article(url)

def batch_extract_from_urls(urls: List[str]) -> List[ScrapedArticle]:
    """Extract from multiple URLs"""
    return url_scraper.batch_scrape(urls)