#!/usr/bin/env python3
"""
Comprehensive Flask API Validation Script for Chunk 3
Tests all endpoints with 90%+ accuracy validation
Author: AI Assistant
Date: 2025-08-24
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5001"
TEST_RESULTS = []
TOTAL_TESTS = 0
PASSED_TESTS = 0

def log_test(test_name, passed, details=""):
    """Log test results"""
    global TOTAL_TESTS, PASSED_TESTS
    TOTAL_TESTS += 1
    if passed:
        PASSED_TESTS += 1
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"{status} - {test_name}")
    if details:
        print(f"    Details: {details}")
    
    TEST_RESULTS.append({
        "test_name": test_name,
        "passed": passed,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })

def test_health_endpoint():
    """Test /api/health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        
        # Test 1: Status code
        log_test("Health endpoint - Status code 200", response.status_code == 200, f"Got {response.status_code}")
        
        # Test 2: JSON response
        try:
            data = response.json()
            log_test("Health endpoint - Valid JSON response", True)
        except:
            log_test("Health endpoint - Valid JSON response", False, "Invalid JSON")
            return
        
        # Test 3: Required fields
        required_fields = ['status', 'message', 'database_connected', 'model_loaded']
        for field in required_fields:
            log_test(f"Health endpoint - Has {field} field", field in data, f"Missing {field}")
        
        # Test 4: Database and model status
        log_test("Health endpoint - Database connected", data.get('database_connected', False))
        log_test("Health endpoint - Model loaded", data.get('model_loaded', False))
        
    except requests.exceptions.RequestException as e:
        log_test("Health endpoint - Connection", False, str(e))

def test_auth_endpoint():
    """Test /api/auth endpoint"""
    try:
        # Test 1: Valid login
        payload = {"username": "testuser", "password": "testpass"}
        response = requests.post(f"{BASE_URL}/api/auth", json=payload, timeout=10)
        
        log_test("Auth endpoint - Status code 200", response.status_code == 200, f"Got {response.status_code}")
        
        try:
            data = response.json()
            log_test("Auth endpoint - Valid JSON response", True)
        except:
            log_test("Auth endpoint - Valid JSON response", False, "Invalid JSON")
            return
        
        # Test 2: Required fields in response
        required_fields = ['status', 'message', 'token', 'user']
        for field in required_fields:
            log_test(f"Auth endpoint - Has {field} field", field in data, f"Missing {field}")
        
        # Test 3: Success status
        log_test("Auth endpoint - Success status", data.get('status') == 'success')
        
        # Test 4: Token format
        token = data.get('token', '')
        log_test("Auth endpoint - Token format", token.startswith('mock_token_'), f"Token: {token}")
        
        # Test 5: Missing credentials
        response = requests.post(f"{BASE_URL}/api/auth", json={}, timeout=10)
        log_test("Auth endpoint - Missing credentials handling", response.status_code == 400, f"Got {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        log_test("Auth endpoint - Connection", False, str(e))

def test_detect_endpoint():
    """Test /api/detect endpoint with various inputs"""
    test_cases = [
        {
            "name": "Short real news",
            "text": "The weather today is sunny with temperatures reaching 25 degrees Celsius.",
            "expected_fields": ['status', 'result', 'confidence', 'message', 'text_length', 'threshold', 'timestamp']
        },
        {
            "name": "Long fake news",
            "text": "BREAKING: Scientists have discovered that drinking water backwards can cure all diseases and make you immortal. This revolutionary discovery was made by a team of researchers who have been studying the effects of reverse hydration for over 100 years. The government doesn't want you to know this secret!",
            "expected_fields": ['status', 'result', 'confidence', 'message', 'text_length', 'threshold', 'timestamp']
        },
        {
            "name": "Medium length news",
            "text": "Local community center opens new library wing with over 5000 books donated by residents. The mayor attended the opening ceremony and praised the community's efforts in promoting literacy and education.",
            "expected_fields": ['status', 'result', 'confidence', 'message', 'text_length', 'threshold', 'timestamp']
        }
    ]
    
    for test_case in test_cases:
        try:
            payload = {"text": test_case["text"]}
            response = requests.post(f"{BASE_URL}/api/detect", json=payload, timeout=15)
            
            # Test status code
            log_test(f"Detect endpoint - {test_case['name']} - Status code 200", 
                    response.status_code == 200, f"Got {response.status_code}")
            
            if response.status_code != 200:
                continue
                
            try:
                data = response.json()
                log_test(f"Detect endpoint - {test_case['name']} - Valid JSON", True)
            except:
                log_test(f"Detect endpoint - {test_case['name']} - Valid JSON", False, "Invalid JSON")
                continue
            
            # Test required fields
            for field in test_case['expected_fields']:
                log_test(f"Detect endpoint - {test_case['name']} - Has {field}", 
                        field in data, f"Missing {field}")
            
            # Test confidence range
            confidence = data.get('confidence', -1)
            log_test(f"Detect endpoint - {test_case['name']} - Confidence in range [0,1]", 
                    0 <= confidence <= 1, f"Confidence: {confidence}")
            
            # Test result values
            result = data.get('result', '')
            log_test(f"Detect endpoint - {test_case['name']} - Valid result", 
                    result in ['real', 'fake'], f"Result: {result}")
            
            # Test text length calculation
            expected_length = len(test_case['text'])
            actual_length = data.get('text_length', -1)
            log_test(f"Detect endpoint - {test_case['name']} - Correct text length", 
                    expected_length == actual_length, f"Expected: {expected_length}, Got: {actual_length}")
            
        except requests.exceptions.RequestException as e:
            log_test(f"Detect endpoint - {test_case['name']} - Connection", False, str(e))
    
    # Test error cases
    try:
        # Empty text
        response = requests.post(f"{BASE_URL}/api/detect", json={"text": ""}, timeout=10)
        log_test("Detect endpoint - Empty text handling", response.status_code == 400, f"Got {response.status_code}")
        
        # Missing text field
        response = requests.post(f"{BASE_URL}/api/detect", json={}, timeout=10)
        log_test("Detect endpoint - Missing text field handling", response.status_code == 400, f"Got {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        log_test("Detect endpoint - Error cases", False, str(e))

def test_history_endpoint():
    """Test /api/history endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/history", timeout=10)
        
        # Test status code
        log_test("History endpoint - Status code 200", response.status_code == 200, f"Got {response.status_code}")
        
        if response.status_code != 200:
            return
            
        try:
            data = response.json()
            log_test("History endpoint - Valid JSON response", True)
        except:
            log_test("History endpoint - Valid JSON response", False, "Invalid JSON")
            return
        
        # Test required fields
        required_fields = ['status', 'message', 'data', 'count']
        for field in required_fields:
            log_test(f"History endpoint - Has {field} field", field in data, f"Missing {field}")
        
        # Test data structure
        log_test("History endpoint - Data is list", isinstance(data.get('data', None), list))
        log_test("History endpoint - Count is integer", isinstance(data.get('count', None), int))
        
    except requests.exceptions.RequestException as e:
        log_test("History endpoint - Connection", False, str(e))

def test_cors_headers():
    """Test CORS headers"""
    try:
        response = requests.options(f"{BASE_URL}/api/health", timeout=10)
        headers = response.headers
        
        # Test CORS headers
        cors_headers = [
            'Access-Control-Allow-Origin',
            'Access-Control-Allow-Methods',
            'Access-Control-Allow-Headers'
        ]
        
        for header in cors_headers:
            log_test(f"CORS - {header} present", header in headers, f"Missing {header}")
        
    except requests.exceptions.RequestException as e:
        log_test("CORS - Headers test", False, str(e))

def test_server_availability():
    """Test if server is running and accessible"""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        log_test("Server - Accessibility", True, f"Server responding on {BASE_URL}")
        return True
    except requests.exceptions.RequestException as e:
        log_test("Server - Accessibility", False, f"Cannot connect to {BASE_URL}: {str(e)}")
        return False

def run_performance_tests():
    """Run performance tests"""
    try:
        # Test response time
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        end_time = time.time()
        
        response_time = end_time - start_time
        log_test("Performance - Health endpoint response time < 2s", 
                response_time < 2.0, f"Response time: {response_time:.3f}s")
        
        # Test detection endpoint performance
        test_text = "This is a test article for performance testing of the fake news detection system."
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/detect", json={"text": test_text}, timeout=15)
        end_time = time.time()
        
        response_time = end_time - start_time
        log_test("Performance - Detect endpoint response time < 5s", 
                response_time < 5.0, f"Response time: {response_time:.3f}s")
        
    except requests.exceptions.RequestException as e:
        log_test("Performance - Tests", False, str(e))

def main():
    """Main test runner"""
    print("🚀 Starting Flask API Validation Tests")
    print(f"📍 Testing server at: {BASE_URL}")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check server availability first
    if not test_server_availability():
        print("\n❌ Server is not accessible. Please ensure Flask app is running.")
        sys.exit(1)
    
    print("\n🔍 Running endpoint tests...")
    test_health_endpoint()
    test_auth_endpoint()
    test_detect_endpoint()
    test_history_endpoint()
    
    print("\n🌐 Running CORS tests...")
    test_cors_headers()
    
    print("\n⚡ Running performance tests...")
    run_performance_tests()
    
    # Calculate results
    accuracy = (PASSED_TESTS / TOTAL_TESTS) * 100 if TOTAL_TESTS > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {TOTAL_TESTS}")
    print(f"Passed: {PASSED_TESTS}")
    print(f"Failed: {TOTAL_TESTS - PASSED_TESTS}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 90.0:
        print("\n🎉 SUCCESS: Achieved 90%+ accuracy! Flask API validation passed.")
        exit_code = 0
    else:
        print(f"\n⚠️  WARNING: Accuracy {accuracy:.1f}% is below 90% threshold.")
        exit_code = 1
    
    print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save detailed results
    with open('flask_validation_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_tests': TOTAL_TESTS,
                'passed_tests': PASSED_TESTS,
                'failed_tests': TOTAL_TESTS - PASSED_TESTS,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': TEST_RESULTS
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: flask_validation_results.json")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()