#!/usr/bin/env python3
"""
Test script for Proofs Validation functionality
Tests the fake news verification system end-to-end
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
    "Climate change is a hoax created by scientists",
    "The 2020 US election was rigged",
    "5G towers cause coronavirus"
]

def test_flask_server_health():
    """Test if Flask server is running"""
    try:
        # Test the root endpoint or a simple endpoint
        response = requests.get(f"{FLASK_BASE_URL}/", timeout=5)
        print(f"✓ Flask server health check: {response.status_code}")
        return response.status_code in [200, 404]  # 404 is also OK, means server is running
    except requests.exceptions.RequestException as e:
        print(f"✗ Flask server not accessible: {e}")
        return False

def test_detect_endpoint(claim: str) -> Dict[str, Any]:
    """Test the /api/detect endpoint with a claim"""
    try:
        payload = {
            "text": claim,
            "include_explanation": True
        }
        
        print(f"\n🔍 Testing claim: '{claim[:50]}...'")
        
        response = requests.post(
            f"{FLASK_BASE_URL}/api/detect",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Detection successful")
            print(f"  Label: {result.get('label', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            print(f"  Processing time: {result.get('processing_time', 'N/A')}s")
            
            if 'explanation' in result:
                print(f"  Explanation available: {len(result['explanation'])} characters")
            
            return result
        else:
            print(f"✗ Detection failed: {response.status_code} - {response.text}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return {}

def test_ensemble_predict_endpoint(claim: str) -> Dict[str, Any]:
    """Test the /api/ensemble-predict endpoint"""
    try:
        payload = {
            "text": claim,
            "include_sources": True,
            "include_explanation": True
        }
        
        print(f"\n🎯 Testing ensemble prediction for: '{claim[:50]}...'")
        
        response = requests.post(
            f"{FLASK_BASE_URL}/api/ensemble-predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Ensemble prediction successful")
            print(f"  Prediction: {result.get('prediction', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            
            if 'sources' in result and result['sources']:
                print(f"  Sources found: {len(result['sources'])}")
                for i, source in enumerate(result['sources'][:3]):
                    print(f"    {i+1}. {source.get('title', 'No title')[:60]}...")
                    print(f"       URL: {source.get('url', 'No URL')[:80]}...")
                    print(f"       Relevance: {source.get('relevance_score', 'N/A')}")
            
            return result
        else:
            print(f"✗ Ensemble prediction failed: {response.status_code} - {response.text}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return {}

def test_news_proxy_endpoint():
    """Test the /api/news-proxy endpoint"""
    try:
        payload = {
            "query": "climate change scientific consensus",
            "source": "newsapi"
        }
        
        print(f"\n📰 Testing news proxy endpoint...")
        
        response = requests.post(
            f"{FLASK_BASE_URL}/api/news-proxy",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ News proxy successful")
            if 'articles' in result:
                print(f"  Articles found: {len(result['articles'])}")
            return result
        else:
            print(f"✗ News proxy failed: {response.status_code} - {response.text}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return {}

def run_comprehensive_test():
    """Run comprehensive test of all endpoints"""
    print("🚀 Starting Proofs Validation Test Suite")
    print("=" * 50)
    
    # Test server health
    if not test_flask_server_health():
        print("\n❌ Flask server is not running. Please start it first.")
        print("Run: cd FakeNewsBackend && python app.py")
        return False
    
    # Test news proxy
    test_news_proxy_endpoint()
    
    # Test each claim
    results = []
    for i, claim in enumerate(TEST_CLAIMS, 1):
        print(f"\n{'='*20} Test {i}/{len(TEST_CLAIMS)} {'='*20}")
        
        # Test basic detection
        detect_result = test_detect_endpoint(claim)
        
        # Test ensemble prediction
        ensemble_result = test_ensemble_predict_endpoint(claim)
        
        results.append({
            'claim': claim,
            'detect_result': detect_result,
            'ensemble_result': ensemble_result
        })
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    successful_detections = sum(1 for r in results if r['detect_result'])
    successful_ensembles = sum(1 for r in results if r['ensemble_result'])
    
    print(f"✓ Successful detections: {successful_detections}/{len(TEST_CLAIMS)}")
    print(f"✓ Successful ensemble predictions: {successful_ensembles}/{len(TEST_CLAIMS)}")
    
    if successful_detections == len(TEST_CLAIMS) and successful_ensembles == len(TEST_CLAIMS):
        print("\n🎉 All tests passed! Proofs Validation is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)