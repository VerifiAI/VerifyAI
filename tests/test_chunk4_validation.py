#!/usr/bin/env python3
"""
Chunk 4 Validation Script for Fake News Detection Backend
Tests all new endpoints implemented in Chunk 4:
- /api/live-feed endpoint with RSS feeds
- Enhanced /api/history endpoint (last 5 entries)
- Data persistence verification

Requires: requests library
Usage: python test_chunk4_validation.py
"""

import requests
import json
import time
from datetime import datetime
import sys

class Chunk4Validator:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name, passed, details=""):
        """Log test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = {
            "test_name": test_name,
            "status": status,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status}: {test_name}")
        if details and not passed:
            print(f"   Details: {details}")
    
    def test_server_availability(self):
        """Test if Flask server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            self.log_test("Server Availability", response.status_code == 200, 
                         f"Status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            self.log_test("Server Availability", False, str(e))
            return False
    
    def test_live_feed_bbc(self):
        """Test /api/live-feed with BBC source"""
        try:
            response = requests.get(f"{self.base_url}/api/live-feed?source=bbc", timeout=10)
            data = response.json()
            
            passed = (
                response.status_code == 200 and
                data.get('status') == 'success' and
                data.get('source') == 'BBC' and
                isinstance(data.get('data'), list) and
                data.get('count', 0) > 0
            )
            
            details = f"Status: {response.status_code}, Count: {data.get('count', 0)}"
            self.log_test("Live Feed - BBC", passed, details)
            return passed
        except Exception as e:
            self.log_test("Live Feed - BBC", False, str(e))
            return False
    
    def test_live_feed_cnn(self):
        """Test /api/live-feed with CNN source"""
        try:
            response = requests.get(f"{self.base_url}/api/live-feed?source=cnn", timeout=10)
            data = response.json()
            
            passed = (
                response.status_code == 200 and
                data.get('status') == 'success' and
                data.get('source') == 'CNN'
            )
            
            details = f"Status: {response.status_code}, Count: {data.get('count', 0)}"
            self.log_test("Live Feed - CNN", passed, details)
            return passed
        except Exception as e:
            self.log_test("Live Feed - CNN", False, str(e))
            return False
    
    def test_live_feed_fox(self):
        """Test /api/live-feed with FOX source"""
        try:
            response = requests.get(f"{self.base_url}/api/live-feed?source=fox", timeout=10)
            data = response.json()
            
            passed = (
                response.status_code == 200 and
                data.get('status') == 'success' and
                data.get('source') == 'FOX'
            )
            
            details = f"Status: {response.status_code}, Count: {data.get('count', 0)}"
            self.log_test("Live Feed - FOX", passed, details)
            return passed
        except Exception as e:
            self.log_test("Live Feed - FOX", False, str(e))
            return False
    
    def test_live_feed_reuters(self):
        """Test /api/live-feed with Reuters source"""
        try:
            response = requests.get(f"{self.base_url}/api/live-feed?source=reuters", timeout=10)
            data = response.json()
            
            passed = (
                response.status_code == 200 and
                data.get('status') == 'success' and
                data.get('source') == 'REUTERS'
            )
            
            details = f"Status: {response.status_code}, Count: {data.get('count', 0)}"
            self.log_test("Live Feed - Reuters", passed, details)
            return passed
        except Exception as e:
            self.log_test("Live Feed - Reuters", False, str(e))
            return False
    
    def test_live_feed_invalid_source(self):
        """Test /api/live-feed with invalid source"""
        try:
            response = requests.get(f"{self.base_url}/api/live-feed?source=invalid", timeout=5)
            data = response.json()
            
            passed = (
                response.status_code == 400 and
                data.get('status') == 'error' and
                'Invalid' in data.get('message', '')
            )
            
            details = f"Status: {response.status_code}, Message: {data.get('message', '')}"
            self.log_test("Live Feed - Invalid Source", passed, details)
            return passed
        except Exception as e:
            self.log_test("Live Feed - Invalid Source", False, str(e))
            return False
    
    def test_detect_and_persistence(self):
        """Test /api/detect endpoint and verify data persistence"""
        try:
            # Send detection request
            test_text = "This is a comprehensive test for data persistence in the fake news detection system. The system should store this result in the database."
            payload = {"text": test_text}
            
            response = requests.post(f"{self.base_url}/api/detect", 
                                   json=payload, timeout=10)
            data = response.json()
            
            passed = (
                response.status_code == 200 and
                data.get('status') == 'success' and
                'confidence' in data and
                'result' in data and
                'timestamp' in data
            )
            
            details = f"Status: {response.status_code}, Result: {data.get('result')}, Confidence: {data.get('confidence')}"
            self.log_test("Detect Endpoint", passed, details)
            return passed, data.get('timestamp')
        except Exception as e:
            self.log_test("Detect Endpoint", False, str(e))
            return False, None
    
    def test_history_endpoint(self, expected_timestamp=None):
        """Test /api/history endpoint (last 5 entries)"""
        try:
            response = requests.get(f"{self.base_url}/api/history", timeout=5)
            data = response.json()
            
            passed = (
                response.status_code == 200 and
                data.get('status') == 'success' and
                isinstance(data.get('data'), list) and
                data.get('count', 0) >= 0 and
                len(data.get('data', [])) <= 5  # Should return max 5 entries
            )
            
            # Check if our test data was persisted
            persistence_verified = False
            if expected_timestamp and data.get('data'):
                for record in data.get('data', []):
                    if record.get('timestamp') == expected_timestamp:
                        persistence_verified = True
                        break
            
            details = f"Status: {response.status_code}, Count: {data.get('count', 0)}, Persistence: {persistence_verified}"
            self.log_test("History Endpoint", passed, details)
            
            if expected_timestamp:
                self.log_test("Data Persistence Verification", persistence_verified, 
                             f"Expected timestamp: {expected_timestamp}")
            
            return passed
        except Exception as e:
            self.log_test("History Endpoint", False, str(e))
            return False
    
    def test_live_feed_data_structure(self):
        """Test live feed data structure consistency"""
        try:
            response = requests.get(f"{self.base_url}/api/live-feed?source=bbc", timeout=10)
            data = response.json()
            
            if response.status_code != 200 or not data.get('data'):
                self.log_test("Live Feed Data Structure", False, "No data received")
                return False
            
            # Check data structure
            required_fields = ['title', 'link', 'description', 'published', 'source']
            structure_valid = True
            
            for item in data.get('data', [])[:3]:  # Check first 3 items
                for field in required_fields:
                    if field not in item:
                        structure_valid = False
                        break
                if not structure_valid:
                    break
            
            details = f"Required fields present: {structure_valid}"
            self.log_test("Live Feed Data Structure", structure_valid, details)
            return structure_valid
        except Exception as e:
            self.log_test("Live Feed Data Structure", False, str(e))
            return False
    
    def test_performance(self):
        """Test API response times"""
        endpoints = [
            ("/api/health", "GET", None),
            ("/api/live-feed?source=bbc", "GET", None),
            ("/api/history", "GET", None)
        ]
        
        performance_passed = True
        
        for endpoint, method, payload in endpoints:
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=10)
                
                response_time = time.time() - start_time
                
                # Consider response time acceptable if under 10 seconds
                time_acceptable = response_time < 10.0
                performance_passed = performance_passed and time_acceptable
                
                details = f"Response time: {response_time:.2f}s"
                self.log_test(f"Performance - {endpoint}", time_acceptable, details)
                
            except Exception as e:
                self.log_test(f"Performance - {endpoint}", False, str(e))
                performance_passed = False
        
        return performance_passed
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*60)
        print("CHUNK 4 VALIDATION TESTS - FAKE NEWS DETECTION BACKEND")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target URL: {self.base_url}")
        print("\n")
        
        # Test server availability first
        if not self.test_server_availability():
            print("\n❌ Server not available. Stopping tests.")
            return False
        
        print("\n--- Testing Live Feed Endpoints ---")
        self.test_live_feed_bbc()
        self.test_live_feed_cnn()
        self.test_live_feed_fox()
        self.test_live_feed_reuters()
        self.test_live_feed_invalid_source()
        self.test_live_feed_data_structure()
        
        print("\n--- Testing Detection and Persistence ---")
        detect_passed, timestamp = self.test_detect_and_persistence()
        time.sleep(1)  # Brief pause to ensure database write
        self.test_history_endpoint(timestamp)
        
        print("\n--- Performance Tests ---")
        self.test_performance()
        
        # Calculate results
        accuracy = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 90.0:
            print("\n🎉 VALIDATION SUCCESSFUL! (≥90% accuracy achieved)")
            success = True
        else:
            print("\n⚠️  VALIDATION NEEDS IMPROVEMENT (<90% accuracy)")
            success = False
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save results to file
        self.save_results(accuracy, success)
        
        return success
    
    def save_results(self, accuracy, success):
        """Save test results to JSON file"""
        results = {
            "test_suite": "Chunk 4 Validation",
            "timestamp": datetime.now().isoformat(),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.total_tests - self.passed_tests,
            "accuracy_percentage": round(accuracy, 1),
            "validation_successful": success,
            "target_url": self.base_url,
            "detailed_results": self.test_results
        }
        
        try:
            with open('chunk4_validation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: chunk4_validation_results.json")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")

def main():
    """Main function"""
    validator = Chunk4Validator()
    success = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()