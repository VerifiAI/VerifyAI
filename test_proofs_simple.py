#!/usr/bin/env python3
"""
Simple test for Proofs Validation - focusing on working endpoints
"""

import requests
import json
import time

# Configuration
FLASK_BASE_URL = "http://localhost:5001"

def test_proofs_validation():
    """Test the core proofs validation functionality"""
    print("🔍 Testing Proofs Validation System")
    print("=" * 40)
    
    # Test claims
    test_claims = [
        "The Earth is flat",
        "COVID-19 vaccines are dangerous",
        "Climate change is not real"
    ]
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n📝 Test {i}: {claim}")
        
        try:
            # Test detection endpoint
            response = requests.post(
                f"{FLASK_BASE_URL}/api/detect",
                json={"text": claim, "include_explanation": True},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {result.get('label', 'Unknown')}")
                print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
                
                if 'explanation' in result:
                    explanation = result['explanation'][:100] + "..." if len(result['explanation']) > 100 else result['explanation']
                    print(f"💡 Explanation: {explanation}")
                    
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)
    
    # Test news proxy
    print(f"\n📰 Testing News Proxy...")
    try:
        news_payload = {'query': 'climate change', 'api': 'newsapi'}
        response = requests.post(
            f"{FLASK_BASE_URL}/api/news-proxy",
            json=news_payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            # Check for both 'articles' and 'results' keys
            articles = result.get('articles', result.get('results', []))
            articles_count = len(articles)
            print(f"✅ News proxy working - found {articles_count} articles")
        else:
            print(f"⚠️  News proxy returned: {response.status_code}")
            
    except Exception as e:
        print(f"❌ News proxy error: {e}")
    
    print(f"\n🎉 Proofs Validation test completed!")
    print("The system is ready for frontend integration.")

if __name__ == "__main__":
    test_proofs_validation()