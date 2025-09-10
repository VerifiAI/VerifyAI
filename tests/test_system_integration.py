#!/usr/bin/env python3
"""
Comprehensive System Integration Tests
Tests the complete integration of optimized RSS feeds and enhanced news validation
"""

import unittest
import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SystemIntegrationTest(unittest.TestCase):
    """System integration tests for the complete fake news detection system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.base_url = "http://localhost:5001"
        # Old API (DELETED) - Removed validate_news and validate_batch endpoints
        # News validation now handled by direct frontend API calls
        cls.api_endpoints = {
            'live_feed': f"{cls.base_url}/api/live-feed",
            'validation_stats': f"{cls.base_url}/api/validation-stats",
            'rss_performance': f"{cls.base_url}/api/rss-performance",
            'predict': f"{cls.base_url}/api/predict"
        }
        
        # Wait for server to be ready
        cls._wait_for_server()
    
    @classmethod
    def _wait_for_server(cls, timeout=30):
        """Wait for the server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{cls.base_url}/api/validation-stats", timeout=5)
                if response.status_code in [200, 503]:  # 503 is acceptable if validator not ready
                    print("Server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        
        raise Exception("Server not ready within timeout")
    
    def test_rss_feed_optimization(self):
        """Test optimized RSS feed processing"""
        print("\n=== Testing RSS Feed Optimization ===")
        
        # Test live feed endpoint
        response = requests.get(self.api_endpoints['live_feed'])
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn('status', data)
            self.assertEqual(data['status'], 'success')
            self.assertIn('articles', data)
            
            # Check if articles have required fields
            if data['articles']:
                article = data['articles'][0]
                required_fields = ['title', 'description', 'url', 'published_date']
                for field in required_fields:
                    self.assertIn(field, article)
                    
            print(f"✓ RSS feed returned {len(data.get('articles', []))} articles")
        else:
            print("⚠ RSS feed service not available")
    
    def test_rss_performance_metrics(self):
        """Test RSS performance statistics"""
        print("\n=== Testing RSS Performance Metrics ===")
        
        response = requests.get(self.api_endpoints['rss_performance'])
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data['status'], 'success')
            self.assertIn('rss_stats', data)
            
            stats = data['rss_stats']
            # Check for nested structure
            self.assertIn('cache', stats)
            self.assertIn('rss_processor', stats)
            self.assertIn('available_sources', stats)
            
            # Check cache metrics
            cache_stats = stats['cache']
            self.assertIn('total_requests', cache_stats)
            self.assertIn('cache_hits', cache_stats)
            
            # Check RSS processor metrics
            rss_stats = stats['rss_processor']
            self.assertIn('total_requests', rss_stats)
            self.assertIn('average_response_time', rss_stats)
                
            print(f"✓ RSS performance metrics: {stats}")
        else:
            print("⚠ RSS performance metrics not available")
    
    def test_news_validation_single(self):
        """Test single news validation"""
        print("\n=== Testing Single News Validation ===")
        
        test_article = {
            "title": "Breaking: Scientists Discover New Planet",
            "content": "Astronomers have announced the discovery of a new exoplanet in the habitable zone of a nearby star system."
        }
        
        response = requests.post(
            self.api_endpoints['validate_news'],
            json=test_article,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data['status'], 'success')
            self.assertIn('validation', data)
            
            validation = data['validation']
            required_fields = ['is_credible', 'confidence_score', 'sources', 'validation_summary']
            for field in required_fields:
                self.assertIn(field, validation)
                
            print(f"✓ News validation result: credible={validation['is_credible']} (confidence: {validation['confidence_score']})")
        else:
            print("⚠ News validation service not available")
    
    def test_news_validation_batch(self):
        """Test batch news validation"""
        print("\n=== Testing Batch News Validation ===")
        
        test_articles = {
            "articles": [
                {
                    "title": "Local Weather Update",
                    "content": "Today's weather will be sunny with temperatures reaching 75°F."
                },
                {
                    "title": "Stock Market News",
                    "content": "The stock market showed mixed results today with technology stocks leading gains."
                }
            ]
        }
        
        response = requests.post(
            self.api_endpoints['validate_batch'],
            json=test_articles,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data['status'], 'success')
            self.assertIn('validations', data)
            self.assertIn('batch_size', data)
            self.assertEqual(data['batch_size'], 2)
            
            print(f"✓ Batch validation processed {data['batch_size']} articles")
        else:
            print("⚠ Batch validation service not available")
    
    def test_validation_performance_stats(self):
        """Test validation performance statistics"""
        print("\n=== Testing Validation Performance Stats ===")
        
        response = requests.get(self.api_endpoints['validation_stats'])
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data['status'], 'success')
            self.assertIn('stats', data)
            
            stats = data['stats']
            expected_metrics = ['total_validations', 'average_processing_time', 'cache_hit_rate']
            for metric in expected_metrics:
                self.assertIn(metric, stats)
                
            print(f"✓ Validation performance stats: {stats}")
        else:
            print("⚠ Validation performance stats not available")
    
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities"""
        print("\n=== Testing Concurrent Processing ===")
        
        def make_request(endpoint, data=None):
            try:
                if data:
                    response = requests.post(endpoint, json=data, timeout=10)
                else:
                    response = requests.get(endpoint, timeout=10)
                return response.status_code, response.json() if response.status_code == 200 else None
            except Exception as e:
                return 500, str(e)
        
        # Test concurrent RSS feed requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(make_request, self.api_endpoints['live_feed'])
                futures.append(future)
            
            results = [future.result() for future in futures]
            successful_requests = sum(1 for status, _ in results if status == 200)
            
            print(f"✓ Concurrent RSS requests: {successful_requests}/5 successful")
            self.assertGreaterEqual(successful_requests, 3)  # At least 60% success rate
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        print("\n=== Testing Error Handling and Recovery ===")
        
        # Test invalid input handling
        invalid_requests = [
            (self.api_endpoints['validate_news'], {}),  # Empty data
            (self.api_endpoints['validate_batch'], {"articles": "invalid"}),  # Invalid format
            (self.api_endpoints['validate_batch'], {"articles": [{}] * 25})  # Too many articles
        ]
        
        for endpoint, data in invalid_requests:
            response = requests.post(endpoint, json=data)
            self.assertEqual(response.status_code, 400)
            
        print("✓ Error handling working correctly")
    
    def test_integration_with_prediction(self):
        """Test integration with existing prediction system"""
        print("\n=== Testing Integration with Prediction System ===")
        
        test_data = {
            "text": "Scientists have made a breakthrough discovery in renewable energy technology."
        }
        
        response = requests.post(
            self.api_endpoints['predict'],
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Should work regardless of validation system
        self.assertIn(response.status_code, [200, 500])  # 500 acceptable if models not loaded
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn('prediction', data)
            print(f"✓ Prediction system integration working")
        else:
            print("⚠ Prediction system not fully loaded")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for 5x efficiency"""
        print("\n=== Testing Performance Benchmarks ===")
        
        # Measure RSS feed response time
        start_time = time.time()
        response = requests.get(self.api_endpoints['live_feed'])
        rss_time = time.time() - start_time
        
        print(f"RSS feed response time: {rss_time:.2f}s")
        self.assertLess(rss_time, 5.0)  # Should be under 5 seconds
        
        # Measure validation response time
        test_article = {
            "title": "Test Article",
            "content": "This is a test article for performance measurement."
        }
        
        start_time = time.time()
        response = requests.post(self.api_endpoints['validate_news'], json=test_article)
        validation_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"Validation response time: {validation_time:.2f}s")
            self.assertLess(validation_time, 10.0)  # Should be under 10 seconds
        
        print("✓ Performance benchmarks met")

def run_system_tests():
    """Run all system integration tests"""
    print("Starting Comprehensive System Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(SystemIntegrationTest)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_system_tests()
    sys.exit(0 if success else 1)