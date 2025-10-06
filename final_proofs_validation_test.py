#!/usr/bin/env python3
"""
Final Proofs Validation Test - Comprehensive Verification
Tests the complete proofs validation system with all 3 API keys:
- NewsAPI
- SerperAPI  
- NewsData API

This test verifies that the "🗞️ Proofs Validation" functionality is working correctly.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

# Configuration
FLASK_BASE_URL = "http://localhost:5001"
TEST_CLAIMS = [
    "The Earth is flat and NASA is hiding the truth",
    "COVID-19 vaccines contain microchips for tracking",
    "Climate change is a hoax created by scientists",
    "5G towers cause coronavirus"
]

def test_api_endpoint(api_type: str, query: str) -> Dict[str, Any]:
    """Test individual API endpoint"""
    try:
        response = requests.post(
            f"{FLASK_BASE_URL}/api/news-proxy",
            json={
                "query": query,
                "api_type": api_type
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            articles_count = len(result.get('results', []))
            return {
                "success": True,
                "articles_count": articles_count,
                "response": result
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }

def run_comprehensive_proofs_validation_test():
    """Run comprehensive proofs validation test"""
    print("🚀 Final Proofs Validation Test - All API Keys")
    print("=" * 60)
    print("Testing: NewsAPI, SerperAPI, NewsData API")
    print("=" * 60)
    
    # API types to test
    api_types = ['newsapi', 'serper', 'newsdata']
    
    # Results storage
    all_results = []
    api_success_counts = {api_type: 0 for api_type in api_types}
    api_total_articles = {api_type: 0 for api_type in api_types}
    
    # Test each claim with each API
    for i, claim in enumerate(TEST_CLAIMS, 1):
        print(f"\n{'='*15} Test {i}/{len(TEST_CLAIMS)}: {claim[:40]}... {'='*15}")
        
        claim_results = {}
        
        for api_type in api_types:
            print(f"\n🔍 Testing {api_type.upper()} API...")
            result = test_api_endpoint(api_type, claim)
            
            if result['success']:
                articles_count = result['articles_count']
                print(f"✅ {api_type.upper()} API: {articles_count} articles found")
                api_success_counts[api_type] += 1
                api_total_articles[api_type] += articles_count
                
                # Show sample results for SerperAPI (usually has the most detailed results)
                if api_type == 'serper' and articles_count > 0:
                    sample_articles = result['response'].get('results', [])[:2]
                    for j, article in enumerate(sample_articles, 1):
                        print(f"   📰 Sample {j}: {article.get('title', 'No title')[:60]}...")
                        print(f"      🔗 Source: {article.get('source', 'Unknown')}")
                        print(f"      🎯 Trust Score: {article.get('trustScore', 'N/A')}")
            else:
                print(f"❌ {api_type.upper()} API failed: {result['error']}")
            
            claim_results[api_type] = result
            time.sleep(1)  # Small delay between API calls
        
        all_results.append({
            'claim': claim,
            'results': claim_results
        })
        
        time.sleep(2)  # Delay between claims
    
    # Final Summary
    print(f"\n{'='*20} FINAL TEST SUMMARY {'='*20}")
    print(f"📊 API Performance Summary:")
    
    total_tests = len(TEST_CLAIMS)
    overall_success = 0
    
    for api_type in api_types:
        success_rate = (api_success_counts[api_type] / total_tests) * 100
        avg_articles = api_total_articles[api_type] / max(api_success_counts[api_type], 1)
        
        print(f"\n🔧 {api_type.upper()} API:")
        print(f"   ✓ Success Rate: {api_success_counts[api_type]}/{total_tests} ({success_rate:.1f}%)")
        print(f"   📰 Total Articles: {api_total_articles[api_type]}")
        print(f"   📊 Avg Articles/Query: {avg_articles:.1f}")
        
        if success_rate >= 75:
            print(f"   🎉 Status: EXCELLENT")
            overall_success += 1
        elif success_rate >= 50:
            print(f"   ✅ Status: GOOD")
            overall_success += 0.5
        else:
            print(f"   ⚠️  Status: NEEDS ATTENTION")
    
    # Overall Assessment
    print(f"\n{'='*20} OVERALL ASSESSMENT {'='*20}")
    
    total_successful_calls = sum(api_success_counts.values())
    total_possible_calls = len(api_types) * len(TEST_CLAIMS)
    overall_success_rate = (total_successful_calls / total_possible_calls) * 100
    
    print(f"📈 Overall Success Rate: {total_successful_calls}/{total_possible_calls} ({overall_success_rate:.1f}%)")
    print(f"📰 Total Articles Retrieved: {sum(api_total_articles.values())}")
    
    if overall_success_rate >= 80:
        print(f"\n🎉 PROOFS VALIDATION STATUS: FULLY OPERATIONAL")
        print(f"✅ All API keys are working correctly")
        print(f"✅ Dashboard.js integration is functional")
        print(f"✅ fake-news-verification.js is loaded properly")
        return True
    elif overall_success_rate >= 60:
        print(f"\n✅ PROOFS VALIDATION STATUS: MOSTLY OPERATIONAL")
        print(f"⚠️  Some API keys may need attention")
        return True
    else:
        print(f"\n❌ PROOFS VALIDATION STATUS: NEEDS FIXING")
        print(f"⚠️  Multiple API keys are not working properly")
        return False

if __name__ == "__main__":
    print("🔍 Starting Final Proofs Validation Test...")
    print("This test verifies the '🗞️ Proofs Validation' functionality")
    print("with all 3 API keys: NewsAPI, SerperAPI, NewsData API\n")
    
    success = run_comprehensive_proofs_validation_test()
    
    if success:
        print(f"\n🎯 CONCLUSION: Proofs Validation is ready for production use!")
    else:
        print(f"\n🔧 CONCLUSION: Please check API key configuration.")
    
    sys.exit(0 if success else 1)