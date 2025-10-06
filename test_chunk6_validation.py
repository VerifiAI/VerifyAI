#!/usr/bin/env python3
"""
Chunk 6 Validation Script - Full-Stack Integration Testing
Tests all API endpoints, frontend-backend integration, and system functionality
Target: 90%+ accuracy for complete system validation
"""

import requests
import json
import time
import os
from datetime import datetime

class Chunk6Validator:
    def __init__(self):
        self.api_base_url = 'http://127.0.0.1:5002'
        self.frontend_url = 'http://127.0.0.1:8080'
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name, passed, details=""):
        """Log test results"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        
        result = {
            'test': test_name,
            'status': status,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        print(f"{status}: {test_name} - {details}")
        
    def test_backend_health(self):
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/api/health", timeout=5)
            passed = response.status_code == 200 and 'status' in response.json()
            details = f"Status: {response.status_code}, Response: {response.json().get('status', 'N/A')}"
            self.log_test("Backend Health Check", passed, details)
        except Exception as e:
            self.log_test("Backend Health Check", False, f"Error: {str(e)}")
    
    def test_auth_endpoint(self):
        """Test authentication endpoint"""
        try:
            # Test valid credentials
            auth_data = {'username': 'testuser', 'password': 'testpass'}
            response = requests.post(f"{self.api_base_url}/api/auth", 
                                   json=auth_data, timeout=5)
            
            passed = (response.status_code == 200 and 
                     response.json().get('status') == 'success' and
                     'token' in response.json())
            
            details = f"Status: {response.status_code}, Auth: {response.json().get('status', 'N/A')}"
            self.log_test("Authentication Endpoint", passed, details)
            
            # Test invalid credentials
            invalid_data = {'username': '', 'password': ''}
            response = requests.post(f"{self.api_base_url}/api/auth", 
                                   json=invalid_data, timeout=5)
            
            passed = response.status_code == 401
            details = f"Invalid auth status: {response.status_code}"
            self.log_test("Authentication Rejection", passed, details)
            
        except Exception as e:
            self.log_test("Authentication Endpoint", False, f"Error: {str(e)}")
    
    def test_detection_endpoint(self):
        """Test fake news detection endpoint"""
        try:
            test_text = "This is a test news article for fake news detection."
            detection_data = {'text': test_text}
            
            response = requests.post(f"{self.api_base_url}/api/detect", 
                                   json=detection_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                passed = ('prediction' in data and 
                         'confidence' in data and
                         'processing_time' in data)
                details = f"Prediction: {data.get('prediction', 'N/A')}, Confidence: {data.get('confidence', 'N/A')}"
            else:
                passed = False
                details = f"Status: {response.status_code}, Error: {response.text}"
            
            self.log_test("Fake News Detection", passed, details)
            
        except Exception as e:
            self.log_test("Fake News Detection", False, f"Error: {str(e)}")
    
    def test_live_feed_endpoint(self):
        """Test live feed endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/api/live-feed?source=bbc", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                passed = ('status' in data and 
                         'data' in data and
                         isinstance(data['data'], list))
                details = f"Articles count: {len(data.get('data', []))}"
            else:
                passed = False
                details = f"Status: {response.status_code}"
            
            self.log_test("Live Feed Endpoint", passed, details)
            
        except Exception as e:
            self.log_test("Live Feed Endpoint", False, f"Error: {str(e)}")
    
    def test_history_endpoint(self):
        """Test history endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/api/history", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                passed = ('status' in data and 
                         'data' in data and
                         isinstance(data['data'], list))
                details = f"History records: {len(data.get('data', []))}"
            else:
                passed = False
                details = f"Status: {response.status_code}"
            
            self.log_test("History Endpoint", passed, details)
            
        except Exception as e:
            self.log_test("History Endpoint", False, f"Error: {str(e)}")
    
    def test_frontend_accessibility(self):
        """Test frontend accessibility"""
        try:
            response = requests.get(self.frontend_url, timeout=5)
            passed = response.status_code == 200 and 'html' in response.text.lower()
            details = f"Status: {response.status_code}, Content-Type: {response.headers.get('content-type', 'N/A')}"
            self.log_test("Frontend Accessibility", passed, details)
            
        except Exception as e:
            self.log_test("Frontend Accessibility", False, f"Error: {str(e)}")
    
    def test_cors_configuration(self):
        """Test CORS configuration"""
        try:
            # Test OPTIONS request
            response = requests.options(f"{self.api_base_url}/api/detect", timeout=5)
            
            cors_headers = {
                'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
            }
            
            passed = (response.status_code == 200 and 
                     cors_headers['Access-Control-Allow-Origin'] is not None)
            
            details = f"CORS Origin: {cors_headers['Access-Control-Allow-Origin']}"
            self.log_test("CORS Configuration", passed, details)
            
        except Exception as e:
            self.log_test("CORS Configuration", False, f"Error: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling capabilities"""
        try:
            # Test invalid endpoint
            response = requests.get(f"{self.api_base_url}/api/invalid", timeout=5)
            passed = response.status_code == 404
            details = f"404 handling: {response.status_code}"
            self.log_test("Error Handling (404)", passed, details)
            
            # Test malformed JSON
            response = requests.post(f"{self.api_base_url}/api/detect", 
                                   data="invalid json", 
                                   headers={'Content-Type': 'application/json'},
                                   timeout=5)
            passed = response.status_code == 400
            details = f"Bad request handling: {response.status_code}"
            self.log_test("Error Handling (400)", passed, details)
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {str(e)}")
    
    def test_file_structure(self):
        """Test required files exist"""
        required_files = [
            'index.html',
            'styles.css', 
            'script.js',
            'app.py',
            'model.py',
            'database.py'
        ]
        
        for file in required_files:
            exists = os.path.exists(file)
            self.log_test(f"File Exists: {file}", exists, f"Present: {exists}")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*60)
        print("CHUNK 6 FULL-STACK INTEGRATION VALIDATION")
        print("="*60)
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Backend URL: {self.api_base_url}")
        print(f"Frontend URL: {self.frontend_url}")
        print("\n")
        
        # Run all tests
        self.test_file_structure()
        self.test_backend_health()
        self.test_auth_endpoint()
        self.test_detection_endpoint()
        self.test_live_feed_endpoint()
        self.test_history_endpoint()
        self.test_frontend_accessibility()
        self.test_cors_configuration()
        self.test_error_handling()
        
        # Calculate results
        accuracy = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed Tests: {self.passed_tests}")
        print(f"Failed Tests: {self.total_tests - self.passed_tests}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Target: 90.00%")
        print(f"Status: {'✓ PASSED' if accuracy >= 90 else '✗ FAILED'}")
        
        # Save results to JSON
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'accuracy': accuracy,
            'target_accuracy': 90.0,
            'status': 'PASSED' if accuracy >= 90 else 'FAILED',
            'tests': self.results
        }
        
        with open('chunk6_validation_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nDetailed results saved to: chunk6_validation_results.json")
        print(f"Completed at: {datetime.now().isoformat()}")
        
        return accuracy >= 90

if __name__ == '__main__':
    validator = Chunk6Validator()
    success = validator.run_all_tests()
    exit(0 if success else 1)