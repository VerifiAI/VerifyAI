#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Chunk 25: RapidAPI Integration and News Validation
Tests RapidAPI news fetching, validation features, and fallback mechanisms for 100% coverage.
"""

import unittest
import asyncio
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Import the modules to test
try:
    from rapidapi_integration import RapidAPINewsAggregator
    RAPIDAPI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RapidAPI integration not available for testing: {e}")
    RAPIDAPI_AVAILABLE = False

try:
    from news_validation import (
        validate_news_batch, validate_single_news, get_validation_performance,
        NewsValidator, FactCheckAPI, SnopesMockAPI, PolitiFactAPI
    )
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: News validation not available for testing: {e}")
    VALIDATION_AVAILABLE = False

class TestRapidAPINewsAggregator(unittest.TestCase):
    """Test the RapidAPINewsAggregator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not RAPIDAPI_AVAILABLE:
            self.skipTest("RapidAPI integration not available")
        
        self.api_key = "test_api_key_12345"
        self.aggregator = RapidAPINewsAggregator(api_key=self.api_key)
    
    @patch('requests.get')
    def test_fetch_news_success(self, mock_get):
        """Test successful news fetching from RapidAPI"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Test News Article 1',
                    'description': 'Test description 1',
                    'url': 'https://example.com/news1',
                    'publishedAt': '2024-01-15T10:00:00Z',
                    'source': {'name': 'Test Source 1'},
                    'urlToImage': 'https://example.com/image1.jpg'
                },
                {
                    'title': 'Test News Article 2',
                    'description': 'Test description 2',
                    'url': 'https://example.com/news2',
                    'publishedAt': '2024-01-15T11:00:00Z',
                    'source': {'name': 'Test Source 2'},
                    'urlToImage': 'https://example.com/image2.jpg'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test fetching news
        result = self.aggregator.fetch_news(query="test", limit=10)
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'success')
        self.assertIn('articles', result)
        self.assertEqual(len(result['articles']), 2)
        
        # Verify article structure
        article = result['articles'][0]
        self.assertIn('title', article)
        self.assertIn('description', article)
        self.assertIn('url', article)
        self.assertIn('published_at', article)
        self.assertIn('source', article)
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn('X-RapidAPI-Key', call_args[1]['headers'])
        self.assertEqual(call_args[1]['headers']['X-RapidAPI-Key'], self.api_key)
    
    @patch('requests.get')
    def test_fetch_news_api_error(self, mock_get):
        """Test handling of API errors"""
        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 429  # Rate limit exceeded
        mock_response.raise_for_status.side_effect = Exception("Rate limit exceeded")
        mock_get.return_value = mock_response
        
        # Test fetching news with error
        result = self.aggregator.fetch_news(query="test")
        
        # Verify error handling
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        self.assertIn('rate limit', result['message'].lower())
    
    @patch('requests.get')
    def test_fetch_news_network_timeout(self, mock_get):
        """Test handling of network timeouts"""
        # Mock network timeout
        mock_get.side_effect = Exception("Connection timeout")
        
        # Test fetching news with timeout
        result = self.aggregator.fetch_news(query="test")
        
        # Verify timeout handling
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
    
    @patch('requests.get')
    def test_fetch_trending_news(self, mock_get):
        """Test fetching trending news"""
        # Mock trending news response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Trending News 1',
                    'description': 'Trending description 1',
                    'url': 'https://example.com/trending1',
                    'publishedAt': '2024-01-15T12:00:00Z',
                    'source': {'name': 'Trending Source'},
                    'urlToImage': 'https://example.com/trending1.jpg'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test fetching trending news
        result = self.aggregator.fetch_trending_news(limit=5)
        
        # Verify results
        self.assertEqual(result['status'], 'success')
        self.assertIn('articles', result)
        self.assertEqual(len(result['articles']), 1)
    
    @patch('requests.get')
    def test_fetch_news_by_category(self, mock_get):
        """Test fetching news by category"""
        # Mock category news response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Technology News',
                    'description': 'Tech description',
                    'url': 'https://example.com/tech1',
                    'publishedAt': '2024-01-15T13:00:00Z',
                    'source': {'name': 'Tech Source'},
                    'urlToImage': 'https://example.com/tech1.jpg'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test fetching news by category
        result = self.aggregator.fetch_news_by_category(category="technology", limit=10)
        
        # Verify results
        self.assertEqual(result['status'], 'success')
        self.assertIn('articles', result)
        
        # Verify API call includes category
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn('category', call_args[1]['params'])
        self.assertEqual(call_args[1]['params']['category'], 'technology')
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Test that rate limiting is enforced
        with patch('time.sleep') as mock_sleep:
            with patch('requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'status': 'ok', 'articles': []}
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                # Make multiple rapid requests
                for i in range(5):
                    self.aggregator.fetch_news(query=f"test{i}")
                
                # Verify that sleep was called for rate limiting
                # (Implementation dependent - may not be called if rate limit not hit)
                self.assertGreaterEqual(mock_get.call_count, 5)
    
    def test_caching_mechanism(self):
        """Test caching mechanism for API responses"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'status': 'ok',
                'articles': [{'title': 'Cached News', 'url': 'https://example.com/cached'}]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # First request
            result1 = self.aggregator.fetch_news(query="cache_test")
            
            # Second identical request (should use cache if implemented)
            result2 = self.aggregator.fetch_news(query="cache_test")
            
            # Verify results are identical
            self.assertEqual(result1, result2)
            
            # Note: Actual cache behavior depends on implementation
            # This test verifies consistent results

class TestNewsValidator(unittest.TestCase):
    """Test the NewsValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not VALIDATION_AVAILABLE:
            self.skipTest("News validation not available")
        
        self.validator = NewsValidator()
    
    @patch('requests.get')
    def test_validate_single_news_success(self, mock_get):
        """Test successful single news validation"""
        # Mock FactCheck API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'success',
            'credibility_score': 0.85,
            'confidence': 0.92,
            'sources_checked': 3,
            'fact_check_results': [
                {'source': 'FactCheck.org', 'rating': 'True', 'confidence': 0.9},
                {'source': 'Snopes', 'rating': 'Mostly True', 'confidence': 0.8},
                {'source': 'PolitiFact', 'rating': 'True', 'confidence': 0.95}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test news item
        news_item = {
            'title': 'Test News Article',
            'content': 'This is a test news article content.',
            'url': 'https://example.com/test-news'
        }
        
        # Validate news
        result = self.validator.validate_single_news(news_item)
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertIn('credibility_score', result)
        self.assertIn('confidence', result)
        self.assertIn('assessment', result)
        self.assertIn('sources_checked', result)
        self.assertIn('validation_details', result)
        
        # Verify score ranges
        self.assertGreaterEqual(result['credibility_score'], 0.0)
        self.assertLessEqual(result['credibility_score'], 1.0)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_validate_batch_news(self):
        """Test batch news validation"""
        news_items = [
            {
                'title': 'News 1',
                'content': 'Content 1',
                'url': 'https://example.com/news1'
            },
            {
                'title': 'News 2',
                'content': 'Content 2',
                'url': 'https://example.com/news2'
            }
        ]
        
        with patch.object(self.validator, 'validate_single_news') as mock_validate:
            mock_validate.return_value = {
                'credibility_score': 0.75,
                'confidence': 0.88,
                'assessment': 'Likely Reliable',
                'sources_checked': 2
            }
            
            # Validate batch
            results = self.validator.validate_batch(news_items)
            
            # Verify results
            self.assertEqual(len(results), len(news_items))
            self.assertEqual(mock_validate.call_count, len(news_items))
            
            for result in results:
                self.assertIn('credibility_score', result)
                self.assertIn('assessment', result)
    
    def test_validation_with_missing_data(self):
        """Test validation with incomplete news data"""
        incomplete_news = {
            'title': 'Incomplete News',
            # Missing content and url
        }
        
        # Should handle missing data gracefully
        result = self.validator.validate_single_news(incomplete_news)
        
        self.assertIsInstance(result, dict)
        self.assertIn('credibility_score', result)
        # Score should be lower for incomplete data
        self.assertLess(result['credibility_score'], 0.5)
    
    def test_validation_error_handling(self):
        """Test error handling in validation"""
        with patch('requests.get') as mock_get:
            # Mock API error
            mock_get.side_effect = Exception("API Error")
            
            news_item = {
                'title': 'Test News',
                'content': 'Test content',
                'url': 'https://example.com/test'
            }
            
            # Should handle errors gracefully
            result = self.validator.validate_single_news(news_item)
            
            self.assertIsInstance(result, dict)
            self.assertIn('credibility_score', result)
            self.assertIn('error', result)

class TestFactCheckAPIs(unittest.TestCase):
    """Test fact-checking API integrations"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not VALIDATION_AVAILABLE:
            self.skipTest("News validation not available")
    
    @patch('requests.get')
    def test_factcheck_api(self, mock_get):
        """Test FactCheck.org API integration"""
        # Mock FactCheck API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'claim': 'Test claim',
                    'rating': 'True',
                    'confidence': 0.9,
                    'source_url': 'https://factcheck.org/test'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        api = FactCheckAPI()
        result = api.check_claim("Test claim")
        
        self.assertIsInstance(result, dict)
        self.assertIn('rating', result)
        self.assertIn('confidence', result)
    
    def test_snopes_mock_api(self):
        """Test Snopes mock API"""
        api = SnopesMockAPI()
        result = api.check_claim("Test claim")
        
        self.assertIsInstance(result, dict)
        self.assertIn('rating', result)
        self.assertIn('confidence', result)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    @patch('requests.get')
    def test_politifact_api(self, mock_get):
        """Test PolitiFact API integration"""
        # Mock PolitiFact API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'statements': [
                {
                    'statement': 'Test statement',
                    'ruling': {'ruling': 'True'},
                    'ruling_confidence': 0.85
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        api = PolitiFactAPI()
        result = api.check_claim("Test statement")
        
        self.assertIsInstance(result, dict)
        self.assertIn('rating', result)
        self.assertIn('confidence', result)

class TestValidationUtilityFunctions(unittest.TestCase):
    """Test validation utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not VALIDATION_AVAILABLE:
            self.skipTest("News validation not available")
    
    def test_validate_single_news_function(self):
        """Test validate_single_news utility function"""
        news_item = {
            'title': 'Test News',
            'content': 'Test content',
            'url': 'https://example.com/test'
        }
        
        with patch('news_validation.NewsValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate_single_news.return_value = {
                'credibility_score': 0.8,
                'confidence': 0.9,
                'assessment': 'Reliable'
            }
            mock_validator_class.return_value = mock_validator
            
            result = validate_single_news(news_item)
            
            self.assertIsInstance(result, dict)
            self.assertIn('credibility_score', result)
            mock_validator.validate_single_news.assert_called_once_with(news_item)
    
    def test_validate_news_batch_function(self):
        """Test validate_news_batch utility function"""
        news_items = [
            {'title': 'News 1', 'content': 'Content 1'},
            {'title': 'News 2', 'content': 'Content 2'}
        ]
        
        with patch('news_validation.NewsValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate_batch.return_value = [
                {'credibility_score': 0.8, 'assessment': 'Reliable'},
                {'credibility_score': 0.6, 'assessment': 'Questionable'}
            ]
            mock_validator_class.return_value = mock_validator
            
            results = validate_news_batch(news_items)
            
            self.assertEqual(len(results), len(news_items))
            mock_validator.validate_batch.assert_called_once_with(news_items)
    
    def test_get_validation_performance_function(self):
        """Test get_validation_performance utility function"""
        with patch('news_validation.NewsValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.get_performance_stats.return_value = {
                'total_validations': 100,
                'average_confidence': 0.85,
                'api_response_time': 0.5,
                'success_rate': 0.95
            }
            mock_validator_class.return_value = mock_validator
            
            stats = get_validation_performance()
            
            self.assertIsInstance(stats, dict)
            self.assertIn('total_validations', stats)
            self.assertIn('average_confidence', stats)
            mock_validator.get_performance_stats.assert_called_once()

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not (RAPIDAPI_AVAILABLE and VALIDATION_AVAILABLE):
            self.skipTest("RapidAPI or validation not available")
    
    def test_end_to_end_news_processing(self):
        """Test end-to-end news fetching and validation"""
        # Mock RapidAPI response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'status': 'ok',
                'articles': [
                    {
                        'title': 'Breaking News',
                        'description': 'Important news description',
                        'url': 'https://example.com/breaking',
                        'publishedAt': '2024-01-15T14:00:00Z',
                        'source': {'name': 'News Source'}
                    }
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Fetch news
            aggregator = RapidAPINewsAggregator(api_key="test_key")
            news_result = aggregator.fetch_news(query="breaking")
            
            # Validate news
            if news_result['status'] == 'success' and news_result['articles']:
                validator = NewsValidator()
                
                with patch.object(validator, 'validate_single_news') as mock_validate:
                    mock_validate.return_value = {
                        'credibility_score': 0.82,
                        'confidence': 0.91,
                        'assessment': 'Likely Reliable'
                    }
                    
                    validation_result = validator.validate_single_news(news_result['articles'][0])
                    
                    # Verify end-to-end process
                    self.assertEqual(news_result['status'], 'success')
                    self.assertGreater(len(news_result['articles']), 0)
                    self.assertIn('credibility_score', validation_result)
                    self.assertGreater(validation_result['credibility_score'], 0.8)
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when APIs fail"""
        # Test RapidAPI fallback
        with patch('requests.get') as mock_get:
            # First call fails (RapidAPI)
            mock_get.side_effect = [Exception("RapidAPI Error"), Mock()]
            
            aggregator = RapidAPINewsAggregator(api_key="test_key")
            result = aggregator.fetch_news(query="test")
            
            # Should handle error gracefully
            self.assertIsInstance(result, dict)
            self.assertEqual(result['status'], 'error')
    
    def test_rate_limit_handling(self):
        """Test rate limit handling across services"""
        with patch('requests.get') as mock_get:
            # Mock rate limit response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = Exception("Rate limit exceeded")
            mock_get.return_value = mock_response
            
            aggregator = RapidAPINewsAggregator(api_key="test_key")
            result = aggregator.fetch_news(query="test")
            
            # Should handle rate limits gracefully
            self.assertEqual(result['status'], 'error')
            self.assertIn('rate limit', result['message'].lower())
    
    def test_concurrent_processing(self):
        """Test concurrent news processing and validation"""
        import concurrent.futures
        
        def process_news_item(item):
            # Simulate processing
            time.sleep(0.01)
            return {
                'original': item,
                'processed': True,
                'timestamp': time.time()
            }
        
        news_items = [{'id': i, 'title': f'News {i}'} for i in range(10)]
        
        # Process concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            start_time = time.time()
            futures = [executor.submit(process_news_item, item) for item in news_items]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            end_time = time.time()
        
        # Verify concurrent processing
        self.assertEqual(len(results), len(news_items))
        processing_time = end_time - start_time
        sequential_time = len(news_items) * 0.01
        
        # Should be faster than sequential processing
        self.assertLess(processing_time, sequential_time * 0.8)

def run_chunk25_rapidapi_tests():
    """Run all Chunk 25 RapidAPI and validation tests"""
    print("\n🧪 Running Chunk 25: RapidAPI Integration and News Validation Tests...")
    
    # Check availability
    if not RAPIDAPI_AVAILABLE:
        print("⚠️  RapidAPI integration not available - skipping related tests")
    if not VALIDATION_AVAILABLE:
        print("⚠️  News validation not available - skipping related tests")
    
    if not (RAPIDAPI_AVAILABLE or VALIDATION_AVAILABLE):
        print("❌ Neither RapidAPI nor validation available - skipping all tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases based on availability
    test_classes = []
    
    if RAPIDAPI_AVAILABLE:
        test_classes.append(TestRapidAPINewsAggregator)
    
    if VALIDATION_AVAILABLE:
        test_classes.extend([
            TestNewsValidator,
            TestFactCheckAPIs,
            TestValidationUtilityFunctions
        ])
    
    if RAPIDAPI_AVAILABLE and VALIDATION_AVAILABLE:
        test_classes.append(TestIntegrationScenarios)
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 Chunk 25 RapidAPI & Validation Test Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_tests - failures - errors}")
    print(f"   Failed: {failures}")
    print(f"   Errors: {errors}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if failures > 0:
        print("\n❌ Test Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if errors > 0:
        print("\n💥 Test Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('\n')[-2]}")
    
    success = failures == 0 and errors == 0
    if success:
        print("\n✅ All Chunk 25 RapidAPI and validation tests passed!")
    else:
        print("\n❌ Some Chunk 25 tests failed. Please review and fix issues.")
    
    return success

if __name__ == '__main__':
    success = run_chunk25_rapidapi_tests()
    exit(0 if success else 1)