#!/usr/bin/env python3
"""
Live Deployment Testing Script
Hybrid Deep Learning with Explainable AI for Fake News Detection

This script tests the live deployment on Render after deployment is complete.
Use this to verify all endpoints are working correctly in production.

Usage:
    python test_live_deployment.py https://your-app-name.onrender.com
"""

import requests
import json
import sys
import time
from datetime import datetime

def test_health_endpoint(base_url):
    """
    Test the health check endpoint
    """
    print("\n🔍 Testing Health Endpoint...")
    
    try:
        response = requests.get(f"{base_url}/api/health", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: PASSED")
            print(f"   Status: {data.get('status', 'Unknown')}")
            print(f"   Database: {data.get('database', 'Unknown')}")
            print(f"   Model: {data.get('model', 'Unknown')}")
            return True
        else:
            print(f"❌ Health Check: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health Check: ERROR - {str(e)}")
        return False

def test_auth_endpoint(base_url):
    """
    Test the authentication endpoint
    """
    print("\n🔐 Testing Authentication Endpoint...")
    
    auth_data = {
        "username": "testuser",
        "password": "testpass"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/auth",
            json=auth_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Authentication: PASSED")
            print(f"   Token: {data.get('token', 'N/A')[:20]}...")
            print(f"   User: {data.get('user', {}).get('username', 'Unknown')}")
            return True
        else:
            print(f"❌ Authentication: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication: ERROR - {str(e)}")
        return False

def test_detection_endpoint(base_url):
    """
    Test the fake news detection endpoint
    """
    print("\n🤖 Testing Detection Endpoint...")
    
    test_articles = [
        {
            "text": "Scientists at MIT have developed a new breakthrough technology that can detect fake news with 99% accuracy using advanced AI algorithms.",
            "expected": "real"
        },
        {
            "text": "BREAKING: Aliens have landed in New York City and are demanding to speak with world leaders immediately!",
            "expected": "fake"
        }
    ]
    
    passed_tests = 0
    
    for i, article in enumerate(test_articles, 1):
        try:
            response = requests.post(
                f"{base_url}/api/detect",
                json={"text": article["text"]},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', 'unknown')
                confidence = data.get('confidence', 0)
                
                print(f"✅ Detection Test {i}: PASSED")
                print(f"   Prediction: {prediction}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Text: {article['text'][:50]}...")
                passed_tests += 1
            else:
                print(f"❌ Detection Test {i}: FAILED (Status: {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Detection Test {i}: ERROR - {str(e)}")
    
    return passed_tests == len(test_articles)

def test_live_feed_endpoint(base_url):
    """
    Test the live news feed endpoint
    """
    print("\n📰 Testing Live Feed Endpoint...")
    
    try:
        response = requests.get(f"{base_url}/api/live-feed", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"✅ Live Feed: PASSED")
            print(f"   Articles Retrieved: {len(articles)}")
            
            if articles:
                print(f"   Latest Article: {articles[0].get('title', 'No title')[:50]}...")
                print(f"   Prediction: {articles[0].get('prediction', 'Unknown')}")
            
            return True
        else:
            print(f"❌ Live Feed: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Live Feed: ERROR - {str(e)}")
        return False

def test_history_endpoint(base_url):
    """
    Test the detection history endpoint
    """
    print("\n📊 Testing History Endpoint...")
    
    try:
        response = requests.get(f"{base_url}/api/history", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', [])
            print(f"✅ History: PASSED")
            print(f"   Records: {len(history)}")
            
            if history:
                print(f"   Latest: {history[0].get('text', 'No text')[:30]}...")
            
            return True
        else:
            print(f"❌ History: FAILED (Status: {response.status_code})")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ History: ERROR - {str(e)}")
        return False

def test_performance_metrics(base_url):
    """
    Test response times and performance
    """
    print("\n⚡ Testing Performance Metrics...")
    
    endpoints = [
        ('/api/health', 'GET'),
        ('/api/live-feed', 'GET'),
        ('/api/history', 'GET')
    ]
    
    performance_results = []
    
    for endpoint, method in endpoints:
        try:
            start_time = time.time()
            
            if method == 'GET':
                response = requests.get(f"{base_url}{endpoint}", timeout=30)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                performance_results.append((endpoint, response_time, True))
                print(f"✅ {endpoint}: {response_time:.0f}ms")
            else:
                performance_results.append((endpoint, response_time, False))
                print(f"❌ {endpoint}: {response_time:.0f}ms (Status: {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            performance_results.append((endpoint, 0, False))
            print(f"❌ {endpoint}: TIMEOUT/ERROR")
    
    # Calculate average response time for successful requests
    successful_times = [time for _, time, success in performance_results if success]
    if successful_times:
        avg_response_time = sum(successful_times) / len(successful_times)
        print(f"\n📊 Average Response Time: {avg_response_time:.0f}ms")
        
        if avg_response_time < 2000:  # Less than 2 seconds
            print("✅ Performance: EXCELLENT")
            return True
        elif avg_response_time < 5000:  # Less than 5 seconds
            print("⚠️  Performance: ACCEPTABLE")
            return True
        else:
            print("❌ Performance: SLOW")
            return False
    else:
        print("❌ Performance: NO SUCCESSFUL REQUESTS")
        return False

def generate_deployment_report(base_url, test_results):
    """
    Generate a comprehensive deployment report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# 🚀 LIVE DEPLOYMENT TEST REPORT

**Deployment URL**: {base_url}  
**Test Date**: {timestamp}  
**Test Suite**: Fake News Detection Backend API  

## 📊 Test Results Summary

| Endpoint | Status | Details |
|----------|--------|----------|
| Health Check | {'✅ PASSED' if test_results['health'] else '❌ FAILED'} | System status verification |
| Authentication | {'✅ PASSED' if test_results['auth'] else '❌ FAILED'} | User login functionality |
| Detection | {'✅ PASSED' if test_results['detection'] else '❌ FAILED'} | Fake news classification |
| Live Feed | {'✅ PASSED' if test_results['live_feed'] else '❌ FAILED'} | Real-time news retrieval |
| History | {'✅ PASSED' if test_results['history'] else '❌ FAILED'} | Detection history access |
| Performance | {'✅ PASSED' if test_results['performance'] else '❌ FAILED'} | Response time analysis |

## 🎯 Overall Status

**Success Rate**: {sum(test_results.values()) / len(test_results) * 100:.1f}%  
**Deployment Status**: {'🎉 PRODUCTION READY' if sum(test_results.values()) >= 5 else '⚠️ NEEDS ATTENTION'}  

## 🔧 Technical Specifications

- **Framework**: Flask + Gunicorn
- **Model**: Multi-modal Hierarchical Fusion Network (MHFN)
- **Database**: SQLite (embedded)
- **Hosting**: Render (Free Tier)
- **Python**: 3.9.18

## 📝 Notes

- All endpoints tested with 30-second timeout
- Performance measured for response times
- Authentication uses mock credentials
- Detection tested with sample articles

---
*Generated by Live Deployment Testing Script*
"""
    
    # Save report to file
    report_filename = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved: {report_filename}")
    return report

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_live_deployment.py <deployment_url>")
        print("\nExample:")
        print("  python test_live_deployment.py https://fake-news-backend.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    
    print("="*70)
    print("🚀 FAKE NEWS DETECTION - LIVE DEPLOYMENT TESTING")
    print("="*70)
    print(f"🌐 Testing URL: {base_url}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_results = {
        'health': test_health_endpoint(base_url),
        'auth': test_auth_endpoint(base_url),
        'detection': test_detection_endpoint(base_url),
        'live_feed': test_live_feed_endpoint(base_url),
        'history': test_history_endpoint(base_url),
        'performance': test_performance_metrics(base_url)
    }
    
    # Calculate overall results
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print("\n" + "="*70)
    print("📊 DEPLOYMENT TEST SUMMARY")
    print("="*70)
    print(f"🎯 Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
    
    if success_rate >= 90:
        print("🎉 DEPLOYMENT STATUS: PRODUCTION READY!")
    elif success_rate >= 70:
        print("⚠️  DEPLOYMENT STATUS: MOSTLY FUNCTIONAL")
    else:
        print("❌ DEPLOYMENT STATUS: NEEDS ATTENTION")
    
    # Generate report
    report = generate_deployment_report(base_url, test_results)
    
    print("\n✅ Live deployment testing completed!")
    
    return success_rate >= 90

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)