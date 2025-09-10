#!/usr/bin/env python3
"""
Comprehensive System Test for Fake News Detection System
Tests all API endpoints and functionality
"""

import requests
import json
import time
import os
from datetime import datetime

# Configuration
BASE_URL = 'http://localhost:5001'
TEST_RESULTS = []

def log_test(test_name, success, message, response_time=None):
    """Log test results"""
    result = {
        'test': test_name,
        'success': success,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'response_time': response_time
    }
    TEST_RESULTS.append(result)
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}: {message}")
    if response_time:
        print(f"    Response time: {response_time:.3f}s")

def test_health_endpoint():
    """Test health check endpoint"""
    try:
        start_time = time.time()
        response = requests.get(f'{BASE_URL}/api/health', timeout=10)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                log_test('Health Check', True, 'API is healthy', response_time)
                return True
            else:
                log_test('Health Check', False, f'Unhealthy status: {data}')
                return False
        else:
            log_test('Health Check', False, f'HTTP {response.status_code}')
            return False
    except Exception as e:
        log_test('Health Check', False, f'Exception: {str(e)}')
        return False

def test_text_detection():
    """Test text-based fake news detection"""
    try:
        test_text = "Breaking news: Scientists discover new planet in our solar system with potential for life."
        
        start_time = time.time()
        response = requests.post(
            f'{BASE_URL}/api/detect',
            json={'text': test_text, 'input_type': 'text'},
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success' and 'prediction' in data:
                log_test('Text Detection', True, f'Prediction: {data["prediction"]} (confidence: {data.get("confidence", "N/A")})', response_time)
                return True
            else:
                log_test('Text Detection', False, f'Invalid response: {data}')
                return False
        else:
            log_test('Text Detection', False, f'HTTP {response.status_code}: {response.text}')
            return False
    except Exception as e:
        log_test('Text Detection', False, f'Exception: {str(e)}')
        return False

def test_url_detection():
    """Test URL-based fake news detection"""
    try:
        test_url = "https://www.bbc.com/news"
        
        start_time = time.time()
        response = requests.post(
            f'{BASE_URL}/api/detect',
            json={'text': test_url, 'input_type': 'url'},
            headers={'Content-Type': 'application/json'},
            timeout=20
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success' and 'prediction' in data:
                log_test('URL Detection', True, f'Prediction: {data["prediction"]} (confidence: {data.get("confidence", "N/A")})', response_time)
                return True
            else:
                log_test('URL Detection', False, f'Invalid response: {data}')
                return False
        else:
            log_test('URL Detection', False, f'HTTP {response.status_code}: {response.text}')
            return False
    except Exception as e:
        log_test('URL Detection', False, f'Exception: {str(e)}')
        return False

def test_live_feed():
    """Test live news feed endpoint"""
    sources = ['bbc', 'cnn', 'fox', 'reuters']
    
    for source in sources:
        try:
            start_time = time.time()
            response = requests.get(
                f'{BASE_URL}/api/live-feed?source={source}',
                timeout=15
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'data' in data:
                    article_count = len(data['data']) if data['data'] else 0
                    log_test(f'Live Feed ({source.upper()})', True, f'Retrieved {article_count} articles', response_time)
                else:
                    log_test(f'Live Feed ({source.upper()})', False, f'Invalid response: {data}')
            else:
                log_test(f'Live Feed ({source.upper()})', False, f'HTTP {response.status_code}')
        except Exception as e:
            log_test(f'Live Feed ({source.upper()})', False, f'Exception: {str(e)}')

def test_history_endpoint():
    """Test history retrieval endpoint"""
    try:
        start_time = time.time()
        response = requests.get(f'{BASE_URL}/api/history', timeout=10)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                record_count = len(data.get('data', [])) if data.get('data') else 0
                log_test('History Retrieval', True, f'Retrieved {record_count} history records', response_time)
                return True
            else:
                log_test('History Retrieval', False, f'Invalid response: {data}')
                return False
        else:
            log_test('History Retrieval', False, f'HTTP {response.status_code}')
            return False
    except Exception as e:
        log_test('History Retrieval', False, f'Exception: {str(e)}')
        return False

def test_authentication():
    """Test authentication endpoint"""
    try:
        start_time = time.time()
        response = requests.post(
            f'{BASE_URL}/api/auth',
            json={'username': 'test_user', 'password': 'test_pass'},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                log_test('Authentication', True, 'Authentication successful', response_time)
                return True
            else:
                log_test('Authentication', False, f'Auth failed: {data}')
                return False
        else:
            log_test('Authentication', False, f'HTTP {response.status_code}')
            return False
    except Exception as e:
        log_test('Authentication', False, f'Exception: {str(e)}')
        return False

def generate_report():
    """Generate test report"""
    total_tests = len(TEST_RESULTS)
    passed_tests = sum(1 for result in TEST_RESULTS if result['success'])
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "="*60)
    print("SYSTEM TEST REPORT")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print("="*60)
    
    # Save detailed report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate
        },
        'test_results': TEST_RESULTS
    }
    
    with open('test_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Detailed report saved to: test_report.json")
    return success_rate >= 90.0

def main():
    """Run all system tests"""
    print("Starting Comprehensive System Test...")
    print(f"Target URL: {BASE_URL}")
    print("-" * 60)
    
    # Run all tests
    test_health_endpoint()
    test_text_detection()
    test_url_detection()
    test_live_feed()
    test_history_endpoint()
    test_authentication()
    
    # Generate report
    success = generate_report()
    
    if success:
        print("\n🎉 SYSTEM TEST PASSED - 90%+ success rate achieved!")
        return 0
    else:
        print("\n⚠️  SYSTEM TEST FAILED - Less than 90% success rate")
        return 1

if __name__ == '__main__':
    exit(main())