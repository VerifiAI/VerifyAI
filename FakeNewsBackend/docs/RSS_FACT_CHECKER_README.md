# RSS-Based Fake News Verification System

This module provides a comprehensive RSS-based fake news verification system that cross-checks claims against live RSS feeds from credible news sources.

## Features

### Multi-Input Support
- **Plain Text**: Direct text input for verification
- **URL**: Automatic article extraction using newspaper3k
- **Image**: OCR text extraction using pytesseract

### RSS Feed Integration
- Fetches articles from multiple credible news sources:
  - BBC News
  - CNN
  - Al Jazeera
  - Reuters
  - Associated Press
  - NPR
  - The Guardian
  - ABC News

### Advanced Verification
- **TF-IDF Vectorization**: Converts text to numerical vectors
- **Cosine Similarity**: Measures similarity between claims and RSS articles
- **Intelligent Caching**: Reduces API calls and improves performance
- **Graceful Error Handling**: Network-resilient with fallback mechanisms

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_rss.txt
```

2. For OCR functionality, install Tesseract:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Standalone Usage

```python
from rss_fact_checker import RSSFactChecker

# Initialize the fact checker
fact_checker = RSSFactChecker()

# Verify a text claim
result = fact_checker.detect_fake('text', 'Your news claim here')
print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence}")
print(f"Explanation: {result.explanation}")

# Verify from URL
result = fact_checker.detect_fake('url', 'https://example.com/news-article')

# Verify from image
result = fact_checker.detect_fake('image', '/path/to/image.jpg')
```

### Flask API Integration

The system is integrated into the main Flask application with two endpoints:

#### 1. Dedicated RSS Endpoint
```bash
# Text verification
curl -X POST http://localhost:5000/api/rss-fact-check \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news claim here"}'

# URL verification
curl -X POST http://localhost:5000/api/rss-fact-check \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# Image upload
curl -X POST http://localhost:5000/api/rss-fact-check \
  -F "image=@/path/to/image.jpg"
```

#### 2. Enhanced Main Detection Endpoint
The `/api/detect` endpoint now includes RSS verification alongside existing methods:

```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news claim here"}'
```

## Response Format

### RSS-Only Endpoint Response
```json
{
  "status": "success",
  "verdict": "Real",
  "confidence": 0.85,
  "explanation": "Found 3 matching articles from credible sources",
  "sources": [
    {
      "title": "Article Title",
      "url": "https://source.com/article",
      "description": "Article description...",
      "similarity_score": 0.87
    }
  ],
  "processing_time_s": 2.34,
  "input_type": "text",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Enhanced Detection Response
```json
{
  "status": "success",
  "verdict": "REAL",
  "confidence": 0.89,
  "evidence": [
    {
      "source_title": "BBC Article",
      "source_url": "https://bbc.com/article",
      "verification_method": "rss_feeds",
      "match_score": 0.85
    },
    {
      "source_title": "Web Search Result",
      "source_url": "https://example.com",
      "verification_method": "web_search",
      "match_score": 0.78
    }
  ],
  "reasoning": "Web search analysis: Supporting evidence found | RSS feed analysis: Multiple credible sources confirm",
  "analysis_details": {
    "rss_verification_available": true,
    "cross_check_available": true
  },
  "rss_verification_summary": {
    "verdict": "Real",
    "confidence": 0.85,
    "matching_sources": 3,
    "explanation": "Found matching articles from credible sources"
  }
}
```

## Configuration

### RSS Feeds
The system uses predefined RSS feeds from credible sources. You can modify the `RSS_FEEDS` list in `rss_fact_checker.py`:

```python
RSS_FEEDS = [
    'http://feeds.bbci.co.uk/news/rss.xml',
    'http://rss.cnn.com/rss/edition.rss',
    'https://www.aljazeera.com/xml/rss/all.xml',
    # Add more feeds as needed
]
```

### Similarity Threshold
Adjust the similarity threshold for claim verification:

```python
# In RSSFactChecker.__init__()
self.similarity_threshold = 0.6  # Default: 0.6 (60% similarity)
```

### Cache Settings
Modify caching behavior:

```python
# In RSSFactChecker.__init__()
self.cache_ttl = 1800  # Cache TTL in seconds (default: 30 minutes)
self.cache_maxsize = 1000  # Maximum cache entries
```

## Error Handling

The system includes comprehensive error handling:

- **Network Errors**: Graceful fallback when RSS feeds are unavailable
- **Invalid URLs**: Proper error messages for malformed URLs
- **OCR Failures**: Fallback mechanisms for image processing
- **Empty Results**: Safe handling when no matches are found

## Performance Considerations

- **Caching**: RSS articles are cached for 30 minutes to reduce network calls
- **Parallel Processing**: Multiple RSS feeds are fetched concurrently
- **Timeout Handling**: Network requests have reasonable timeouts
- **Memory Management**: Cache size is limited to prevent memory issues

## Limitations

1. **Language Support**: Currently optimized for English content
2. **RSS Availability**: Dependent on RSS feed availability and format
3. **Real-time Updates**: Cache may delay detection of very recent news
4. **Similarity Threshold**: May require tuning for specific use cases

## Contributing

To add new RSS feeds or improve the verification algorithm:

1. Add RSS feed URLs to the `RSS_FEEDS` list
2. Test with various claim types
3. Adjust similarity thresholds if needed
4. Update documentation

## Troubleshooting

### Common Issues

1. **OCR Not Working**: Ensure Tesseract is properly installed
2. **RSS Feeds Failing**: Check network connectivity and feed URLs
3. **Low Accuracy**: Consider adjusting similarity threshold
4. **Performance Issues**: Check cache settings and network latency

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This module is part of the Hybrid Deep Learning Fake News Detection system.