#!/usr/bin/env python3

import requests
import json
from datetime import datetime

# Configuration
FLASK_BASE_URL = "http://localhost:5001"

def test_news_proxy_debug():
    """Test news proxy with detailed debugging"""
    print("🔍 Testing News Proxy (Debug Mode)")
    print("=" * 50)
    
    # Test different APIs
    test_cases = [
        {'query': 'climate change', 'api': 'newsapi'},
        {'query': 'artificial intelligence', 'api': 'serper'},
        {'query': 'technology news', 'api': 'newsdata'}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📰 Test {i}: {test_case['api'].upper()} API")
        print(f"Query: {test_case['query']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{FLASK_BASE_URL}/api/news-proxy",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Size: {len(response.text)} characters")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"Response Keys: {list(result.keys())}")
                    
                    if 'articles' in result:
                        articles = result['articles']
                        print(f"Articles Count: {len(articles)}")
                        
                        if articles:
                            print("\n📄 Sample Articles:")
                            for j, article in enumerate(articles[:3], 1):
                                title = article.get('title', 'No title')
                                source = article.get('source', {}).get('name', 'Unknown source')
                                print(f"  {j}. {title[:80]}... (Source: {source})")
                        else:
                            print("⚠️  No articles found in response")
                    else:
                        print("⚠️  No 'articles' key in response")
                        print(f"Response content: {json.dumps(result, indent=2)[:500]}...")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"Raw response: {response.text[:500]}...")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n🎉 News Proxy Debug test completed!")

if __name__ == "__main__":
    test_news_proxy_debug()