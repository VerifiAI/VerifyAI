#!/usr/bin/env python3

import requests
import json
import time

# Configuration
FLASK_BASE_URL = "http://localhost:5001"

def test_proofs_validation():
    print("🔍 Testing Proofs Validation System (Detailed)")
    print("=" * 50)
    
    # Test claims
    test_claims = [
        "The Earth is flat",
        "COVID-19 vaccines are dangerous",
        "Climate change is not real"
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n📝 Test {i}: {claim}")
        
        try:
            # Test /api/detect endpoint
            response = requests.post(
                f"{FLASK_BASE_URL}/api/detect",
                json={'text': claim},
                headers=headers,
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result.get('label', 'N/A')} (confidence: {result.get('confidence', 0.0):.2f})")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"Response Text: {response.text[:500]}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test news proxy
    print("\n📰 Testing News Proxy...")
    try:
        news_payload = {'query': 'climate change', 'api': 'newsapi'}
        response = requests.post(
            f"{FLASK_BASE_URL}/api/news-proxy",
            json=news_payload,
            headers=headers,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            news_data = response.json()
            # Check for both 'articles' and 'results' keys
            articles = news_data.get('articles', news_data.get('results', []))
            articles_count = len(articles)
            print(f"✅ News proxy working - found {articles_count} articles")
            if articles_count > 0:
                print(f"Sample article: {articles[0].get('title', 'No title')}")
                print(f"API Type: {news_data.get('api_type', 'unknown')}")
                print(f"Success: {news_data.get('success', 'unknown')}")
        else:
            print(f"⚠️  News proxy returned: {response.status_code}")
            print(f"Response Text: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ News proxy error: {e}")
    
    print("\n🎉 Detailed Proofs Validation test completed!")

if __name__ == "__main__":
    test_proofs_validation()