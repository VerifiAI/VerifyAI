import re
import string
import logging
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive RSS feeds from reliable sources
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://rss.cnn.com/rss/edition.rss", 
    "http://feeds.reuters.com/reuters/topNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.apnews.com/ApNews/apf-topnews",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "http://feeds.reuters.com/Reuters/worldNews"
]

class VerificationResult:
    """Result object for RSS fact checking"""
    def __init__(self, verdict: str, confidence: float, sources: List[str], explanation: str = "", matched_articles: List[Dict] = None, similarity_scores: List[float] = None):
        self.verdict = verdict
        self.confidence = confidence
        self.sources = sources or []
        self.explanation = explanation
        self.matched_articles = matched_articles or []
        self.similarity_scores = similarity_scores or []

def normalize_text(text: str) -> str:
    """Clean and normalize text for comparison"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_session() -> requests.Session:
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_rss_articles() -> List[Dict[str, str]]:
    """Fetch articles from multiple RSS feeds with error handling"""
    articles = []
    session = create_session()
    
    for feed_url in RSS_FEEDS:
        try:
            logger.info(f"Fetching RSS feed: {feed_url}")
            
            # Set timeout and headers
            response = session.get(feed_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; RSS-Fact-Checker/1.0)'
            })
            response.raise_for_status()
            
            # Parse RSS feed
            parsed = feedparser.parse(response.content)
            
            if parsed.bozo:
                logger.warning(f"RSS feed may be malformed: {feed_url}")
            
            # Extract articles
            for entry in parsed.entries:  # Removed [:10] limit for testing
                title = getattr(entry, 'title', '')
                description = getattr(entry, 'description', '') or getattr(entry, 'summary', '')
                link = getattr(entry, 'link', '')
                
                if title and link:
                    logger.info(f"Fetched Article: Title='{title}', Link='{link}'") # Added for debugging
                    # Combine title and description for better matching
                    full_text = f"{title}. {description}"
                    articles.append({
                        "title": title,
                        "description": description,
                        "full_text": normalize_text(full_text),
                        "link": link,
                        "source": feed_url
                    })
                    
        except requests.RequestException as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
            continue
    
    logger.info(f"Successfully fetched {len(articles)} articles from RSS feeds")
    return articles

def extract_text(input_type: str, input_value: str) -> Tuple[str, str]:
    """Extract text from different input types"""
    try:
        if input_type == "text":
            # Direct text input
            if isinstance(input_value, str):
                return normalize_text(input_value), ""
            else:
                return normalize_text(str(input_value)), ""
                
        elif input_type == "url":
            # URL scraping with newspaper3k
            try:
                from newspaper import Article
                
                article = Article(input_value)
                article.download()
                article.parse()
                
                # Combine title and text
                full_text = f"{article.title}. {article.text}"
                return normalize_text(full_text), ""
                
            except ImportError:
                return "", "newspaper3k not available for URL scraping"
            except Exception as e:
                return "", f"Failed to extract text from URL: {str(e)}"
                
        elif input_type == "image":
            # OCR with pytesseract
            try:
                import pytesseract
                from PIL import Image
                import requests
                from io import BytesIO
                
                # Handle both local file paths and URLs
                if input_value.startswith(('http://', 'https://')):
                    response = requests.get(input_value, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(input_value)
                
                # Extract text using OCR
                extracted_text = pytesseract.image_to_string(image)
                return normalize_text(extracted_text), ""
                
            except ImportError:
                return "", "pytesseract not available for OCR"
            except Exception as e:
                return "", f"Failed to extract text from image: {str(e)}"
        else:
            return "", f"Unsupported input type: {input_type}"
            
    except Exception as e:
        return "", f"Text extraction failed: {str(e)}"

def verify_claim(claim: str) -> Dict[str, Any]:
    """Verify claim against RSS articles and return structured JSON"""
    try:
        # Normalize the claim
        normalized_claim = normalize_text(claim)
        
        if not normalized_claim:
            return {
                "verdict": "Unverified",
                "confidence": 0.0,
                "sources": [],
                "explanation": "Empty or invalid claim provided"
            }
        
        # Fetch RSS articles
        articles = fetch_rss_articles()
        
        if not articles:
            return {
                "verdict": "Unverified",
                "confidence": 0.0,
                "sources": [],
                "explanation": "No RSS articles available for comparison due to network issues"
            }
        
        # Prepare texts for TF-IDF
        article_texts = [article["full_text"] for article in articles if article["full_text"]]
        
        if not article_texts:
            return {
                "verdict": "Unverified",
                "confidence": 0.0,
                "sources": [],
                "explanation": "No valid article content available for comparison"
            }
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Fit and transform
        all_texts = [normalized_claim] + article_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarities
        claim_vector = tfidf_matrix[0:1]
        article_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(claim_vector, article_vectors).flatten()
        
        # Find top 3 matches
        top_indices = similarities.argsort()[-3:][::-1]
        top_similarities = similarities[top_indices]
        top_articles = [articles[i] for i in top_indices]
        
        # Apply decision logic
        high_similarity_matches = [(articles[i], similarities[i]) for i in range(len(similarities)) if similarities[i] >= 0.6]
        
        if len(high_similarity_matches) >= 2:  # Multiple high-confidence matches
            verdict = "Real"
            confidence = float(max(top_similarities))
            sources = [article["link"] for article, _ in high_similarity_matches[:3]]
            explanation = f"Matched {len(high_similarity_matches)} reliable news sources with high similarity (≥0.6)"
            
        elif len(high_similarity_matches) == 1 and top_similarities[0] >= 0.7:  # Single very high confidence match
            verdict = "Real"
            confidence = float(top_similarities[0])
            sources = [top_articles[0]["link"]]
            explanation = f"Strong match found with similarity score of {top_similarities[0]:.2f}"
            
        elif top_similarities[0] >= 0.4:  # Moderate similarity
            verdict = "Unverified"
            confidence = float(top_similarities[0])
            sources = [article["link"] for article in top_articles]
            explanation = f"Some similarity found (max: {top_similarities[0]:.2f}) but below confidence threshold"
            
        else:  # Low similarity
            verdict = "Possibly Fake"
            confidence = 0.0
            sources = []
            explanation = f"No matching articles found in reliable news sources (max similarity: {top_similarities[0]:.2f})"
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "sources": sources,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.error(f"Error in verify_claim: {e}")
        return {
            "verdict": "Unverified",
            "confidence": 0.0,
            "sources": [],
            "explanation": f"Verification failed due to technical error: {str(e)}"
        }

def detect_fake(input_type: str, input_value: str) -> Dict[str, Any]:
    """Main function to detect fake news from various input types"""
    # Extract text from input
    extracted_text, error = extract_text(input_type, input_value)
    
    if error:
        return {
            "verdict": "Unverified",
            "confidence": 0.0,
            "sources": [],
            "explanation": f"Text extraction failed: {error}"
        }
    
    if not extracted_text:
        return {
            "verdict": "Unverified",
            "confidence": 0.0,
            "sources": [],
            "explanation": "No text content found to verify"
        }
    
    # Verify the claim
    return verify_claim(extracted_text)

# Compatibility class for existing Flask app
class RSSFactChecker:
    def __init__(self):
        pass
    
    def verify_claim(self, claim: str) -> VerificationResult:
        """Verify claim and return VerificationResult object for compatibility"""
        result_dict = verify_claim(claim)
        
        return VerificationResult(
            verdict=result_dict["verdict"],
            confidence=result_dict["confidence"],
            sources=result_dict["sources"],
            explanation=result_dict["explanation"]
        )
    
    def extract_text(self, input_type: str, input_value: str) -> Tuple[str, str]:
        """Extract text from various input types"""
        return extract_text(input_type, input_value)

# Example usage
if __name__ == "__main__":
    # Test with different input types
    
    # Test 1: Direct text
    result1 = detect_fake("text", "Breaking news: Scientists discover new planet in solar system")
    print("Text input result:", result1)
    
    # Test 2: URL (requires newspaper3k)
    # result2 = detect_fake("url", "https://example.com/news-article")
    # print("URL input result:", result2)
    
    # Test 3: Image (requires pytesseract)
    # result3 = detect_fake("image", "path/to/news-image.jpg")
    # print("Image input result:", result3)