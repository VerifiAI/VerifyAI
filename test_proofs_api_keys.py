#!/usr/bin/env python3
"""
Proofs Validation Test with Direct API Keys
Tests the proofs validation functionality using the 3 API keys:
- NewsAPI
- SerperAPI  
- NewsData API
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
FLASK_BASE_URL = "http://localhost:5001"
TEST_CLAIMS = [
    "The Earth is flat and NASA is hiding the truth",
    "COVID-19 vaccines contain microchips for tracking",
    "Climate change is a hoax created by scientists"
]

def test_flask_server_health():
    """Test if Flask server is running"""
    try:
        response = requests.get(f"{FLASK_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Flask server is healthy")
            return True
        else:
            print(f"❌ Flask server returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Flask server connection failed: {e}")
        return False

def test_rss_fact_check_endpoint(claim: str) -> Dict[str, Any]:
    """Test RSS fact-check endpoint for proofs validation"""
    print(f"\n🗞️ Testing Proofs Validation for: '{claim[:50]}...'")
    
    try:
        response = requests.post(
            f"{FLASK_BASE_URL}/api/rss-fact-check",
            json={"text": claim},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Proofs validation successful")
            print(f"   📰 RSS Verdict: {result.get('verdict', 'N/A')}")
            print(f"   🔍 RSS Confidence: {result.get('confidence', 'N/A')}")
            print(f"   📝 Explanation: {result.get('explanation', 'N/A')[:100]}...")
            print(f"   🔗 Sources: {len(result.get('sources', []))} found")
            return {"success": True, "result": result}
        else:
            error_msg = response.text
            print(f"❌ Proofs validation failed: {response.status_code}")
            print(f"   Error: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"success": False, "error": str(e)}

def test_news_proxy_endpoint(claim: str) -> Dict[str, Any]:
    """Test news proxy endpoint with API keys"""
    print(f"\n📡 Testing News Proxy API Keys for: '{claim[:50]}...'")
    
    # Test each API type
    api_types = ['newsapi', 'serper', 'newsdata']
    results = {}
    
    for api_type in api_types:
        try:
            response = requests.post(
                f"{FLASK_BASE_URL}/api/news-proxy",
                json={
                    "query": claim,
                    "api_type": api_type
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {api_type.upper()} API working")
                print(f"   📰 Articles found: {len(result.get('articles', []))}")
                results[api_type] = {"success": True, "articles": len(result.get('articles', []))}
            else:
                print(f"❌ {api_type.upper()} API failed: {response.status_code}")
                results[api_type] = {"success": False, "error": response.text}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {api_type.upper()} API request failed: {e}")
            results[api_type] = {"success": False, "error": str(e)}
    
    return results

def run_proofs_validation_test():
    """Run comprehensive proofs validation test"""
    print("🚀 Starting Proofs Validation Test with API Keys")
    print("=" * 60)
    
    # Test server health
    if not test_flask_server_health():
        print("\n❌ Flask server is not running. Please start it first.")
        print("Run: cd FakeNewsBackend && python app.py")
        return False
    
    # Test each claim
    results = []
    for i, claim in enumerate(TEST_CLAIMS, 1):
        print(f"\n{'='*20} Test {i}/{len(TEST_CLAIMS)} {'='*20}")
        
        # Test RSS fact-check (proofs validation)
        rss_result = test_rss_fact_check_endpoint(claim)
        
        # Test news proxy APIs
        news_result = test_news_proxy_endpoint(claim)
        
        results.append({
            'claim': claim,
            'rss_result': rss_result,
            'news_result': news_result
        })
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    successful_rss = sum(1 for r in results if r['rss_result']['success'])
    
    # Count successful API calls
    api_success_counts = {'newsapi': 0, 'serper': 0, 'newsdata': 0}
    for result in results:
        for api_type in api_success_counts.keys():
            if result['news_result'].get(api_type, {}).get('success', False):
                api_success_counts[api_type] += 1
    
    print(f"✓ Successful RSS fact-checks: {successful_rss}/{len(TEST_CLAIMS)}")
    print(f"✓ NewsAPI successful calls: {api_success_counts['newsapi']}/{len(TEST_CLAIMS)}")
    print(f"✓ SerperAPI successful calls: {api_success_counts['serper']}/{len(TEST_CLAIMS)}")
    print(f"✓ NewsData API successful calls: {api_success_counts['newsdata']}/{len(TEST_CLAIMS)}")
    
    total_api_success = sum(api_success_counts.values())
    total_api_calls = len(TEST_CLAIMS) * 3
    
    if successful_rss >= len(TEST_CLAIMS) // 2 and total_api_success >= total_api_calls // 2:
        print("\n🎉 Proofs Validation is working! API keys are configured correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check API key configuration.")
        return False

if __name__ == "__main__":
    success = run_proofs_validation_test()
    sys.exit(0 if success else 1)