#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Chunk 25: Caching Optimization System
Tests caching, parallel processing, and optimization features for 100% coverage.
"""

import unittest
import asyncio
import time
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the modules to test
try:
    from caching_optimization import (
        AdvancedCache, ParallelProcessor, NewsProcessingOptimizer,
        optimize_news_processing, optimize_news_validation, get_optimization_stats
    )
    CACHING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Caching optimization not available for testing: {e}")
    CACHING_AVAILABLE = False

class TestAdvancedCache(unittest.TestCase):
    """Test the AdvancedCache class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CACHING_AVAILABLE:
            self.skipTest("Caching optimization not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.cache = AdvancedCache(
            max_memory_size=10,
            cache_dir=self.temp_dir,
            enable_redis=False
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations"""
        test_key = "test_key"
        test_value = {"data": "test_value", "number": 42}
        
        # Set value
        self.cache.set(test_key, test_value)
        
        # Get value
        result = self.cache.get(test_key)
        self.assertEqual(result, test_value)
        
        # Test cache hit statistics
        stats = self.cache.get_stats()
        self.assertEqual(stats['memory_hits'], 1)
        self.assertEqual(stats['misses'], 0)
    
    def test_cache_miss(self):
        """Test cache miss behavior"""
        result = self.cache.get("nonexistent_key", default="default_value")
        self.assertEqual(result, "default_value")
        
        stats = self.cache.get_stats()
        self.assertEqual(stats['misses'], 1)
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        # Fill cache to capacity
        for i in range(15):  # More than max_memory_size (10)
            self.cache.set(f"key_{i}", f"value_{i}")
        
        # Check that oldest items were evicted
        self.assertIsNone(self.cache.get("key_0"))
        self.assertIsNone(self.cache.get("key_1"))
        
        # Check that newest items are still there
        self.assertEqual(self.cache.get("key_14"), "value_14")
        self.assertEqual(self.cache.get("key_13"), "value_13")
    
    def test_disk_cache_persistence(self):
        """Test disk cache persistence"""
        test_key = "disk_test_key"
        test_value = {"persistent": True, "data": [1, 2, 3]}
        
        # Set value (should be stored in disk)
        self.cache.set(test_key, test_value)
        
        # Clear memory cache
        self.cache.memory_cache.clear()
        
        # Get value (should come from disk)
        result = self.cache.get(test_key)
        self.assertEqual(result, test_value)
        
        # Check disk hit statistics
        stats = self.cache.get_stats()
        self.assertEqual(stats['disk_hits'], 1)
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        key1 = self.cache._generate_key("test_string")
        key2 = self.cache._generate_key("test_string")
        key3 = self.cache._generate_key("different_string")
        
        # Same input should generate same key
        self.assertEqual(key1, key2)
        
        # Different input should generate different key
        self.assertNotEqual(key1, key3)
        
        # Keys should be MD5 hashes (32 characters)
        self.assertEqual(len(key1), 32)
    
    def test_cache_statistics(self):
        """Test cache statistics calculation"""
        # Perform various cache operations
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Hit
        self.cache.get("key3")  # Miss
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['total_requests'], 3)
        self.assertEqual(stats['memory_hits'], 2)
        self.assertEqual(stats['misses'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 2/3, places=2)
        self.assertEqual(stats['memory_cache_size'], 2)

class TestParallelProcessor(unittest.TestCase):
    """Test the ParallelProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CACHING_AVAILABLE:
            self.skipTest("Caching optimization not available")
    
    def test_parallel_batch_processing(self):
        """Test parallel batch processing"""
        def process_item(item):
            time.sleep(0.01)  # Simulate processing time
            return f"processed_{item}"
        
        items = [f"item_{i}" for i in range(20)]
        
        with ParallelProcessor(max_workers=5) as processor:
            start_time = time.time()
            results = processor.process_batch(items, process_item, chunk_size=4)
            end_time = time.time()
        
        # Check results
        self.assertEqual(len(results), len(items))
        for i, result in enumerate(results):
            if result is not None:  # Some might be None due to chunking
                self.assertIn("processed_", str(result))
        
        # Check that parallel processing was faster than sequential
        # (This is approximate due to overhead)
        sequential_time = len(items) * 0.01
        self.assertLess(end_time - start_time, sequential_time * 0.8)
    
    def test_async_batch_processing(self):
        """Test asynchronous batch processing"""
        async def async_process_item(item):
            await asyncio.sleep(0.01)  # Simulate async processing
            return f"async_processed_{item}"
        
        async def run_test():
            items = [f"item_{i}" for i in range(10)]
            
            processor = ParallelProcessor(max_workers=5)
            start_time = time.time()
            results = await processor.process_async_batch(items, async_process_item, semaphore_limit=3)
            end_time = time.time()
            
            # Check results
            self.assertEqual(len(results), len(items))
            for result in results:
                if result is not None:
                    self.assertIn("async_processed_", str(result))
            
            # Check timing
            sequential_time = len(items) * 0.01
            self.assertLess(end_time - start_time, sequential_time * 0.8)
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_error_handling_in_parallel_processing(self):
        """Test error handling in parallel processing"""
        def process_item_with_error(item):
            if "error" in item:
                raise ValueError(f"Error processing {item}")
            return f"processed_{item}"
        
        items = ["item_1", "error_item", "item_3", "another_error", "item_5"]
        
        with ParallelProcessor(max_workers=3) as processor:
            results = processor.process_batch(items, process_item_with_error)
        
        # Check that errors are handled gracefully
        self.assertEqual(len(results), len(items))
        
        # Some results should be None (errors), others should be processed
        processed_count = sum(1 for r in results if r is not None and "processed_" in str(r))
        error_count = sum(1 for r in results if r is None)
        
        self.assertGreater(processed_count, 0)
        self.assertGreater(error_count, 0)
    
    def test_performance_statistics(self):
        """Test performance statistics collection"""
        def simple_process(item):
            time.sleep(0.001)
            return item * 2
        
        items = list(range(10))
        
        with ParallelProcessor(max_workers=3) as processor:
            processor.process_batch(items, simple_process)
            stats = processor.get_performance_stats()
        
        # Check that statistics are collected
        self.assertIn('batch_processing_times', stats)
        if 'batch_processing_times' in stats:
            batch_stats = stats['batch_processing_times']
            self.assertIn('count', batch_stats)
            self.assertIn('avg_time', batch_stats)
            self.assertGreater(batch_stats['count'], 0)
            self.assertGreater(batch_stats['avg_time'], 0)

class TestNewsProcessingOptimizer(unittest.TestCase):
    """Test the NewsProcessingOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CACHING_AVAILABLE:
            self.skipTest("Caching optimization not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.cache = AdvancedCache(max_memory_size=50, cache_dir=self.temp_dir, enable_redis=False)
        self.optimizer = NewsProcessingOptimizer(cache_instance=self.cache, max_workers=5)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('requests.get')
    def test_fetch_news_optimized(self, mock_get):
        """Test optimized news fetching with caching"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = "<html><body>Test news content</body></html>"
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        test_url = "https://example.com/news"
        
        # First fetch (should hit the network)
        result1 = self.optimizer.fetch_news_optimized(test_url)
        self.assertEqual(result1['url'], test_url)
        self.assertEqual(result1['status_code'], 200)
        self.assertIn('content', result1)
        
        # Second fetch (should hit the cache)
        result2 = self.optimizer.fetch_news_optimized(test_url)
        self.assertEqual(result1, result2)
        
        # Verify that the network was only called once (caching worked)
        self.assertEqual(mock_get.call_count, 1)
    
    def test_process_news_batch_optimized(self):
        """Test optimized batch news processing"""
        def mock_process_func(item):
            return {
                'original': item,
                'processed': f"processed_{item['id']}",
                'timestamp': time.time()
            }
        
        news_items = [
            {'id': 1, 'title': 'News 1', 'content': 'Content 1'},
            {'id': 2, 'title': 'News 2', 'content': 'Content 2'},
            {'id': 3, 'title': 'News 3', 'content': 'Content 3'},
            {'id': 1, 'title': 'News 1', 'content': 'Content 1'},  # Duplicate for cache test
        ]
        
        # Process batch
        results = self.optimizer.process_news_batch_optimized(news_items, mock_process_func)
        
        # Check results
        self.assertEqual(len(results), len(news_items))
        for result in results:
            self.assertIn('processed', result)
            self.assertIn('original', result)
        
        # Check cache statistics
        cache_stats = self.cache.get_stats()
        self.assertGreater(cache_stats['total_requests'], 0)
    
    def test_async_news_validation(self):
        """Test asynchronous news validation"""
        async def run_validation_test():
            news_items = [
                {'title': 'Test News 1', 'content': 'Test content 1'},
                {'title': 'Test News 2', 'content': 'Test content 2'},
            ]
            
            validation_sources = ['factcheck', 'snopes']
            
            # Mock the validation methods
            with patch.object(self.optimizer, '_validate_with_sources') as mock_validate:
                mock_validate.return_value = {
                    'item': news_items[0],
                    'validation': {
                        'credibility_score': 0.8,
                        'confidence': 0.9,
                        'sources_checked': 2
                    }
                }
                
                results = await self.optimizer.validate_news_batch_async(news_items, validation_sources)
                
                # Check results
                self.assertEqual(len(results), len(news_items))
                for result in results:
                    if result is not None:
                        self.assertIn('validation', result)
        
        # Run the async test
        asyncio.run(run_validation_test())
    
    def test_performance_report(self):
        """Test performance report generation"""
        # Perform some operations to generate metrics
        def dummy_process(item):
            time.sleep(0.001)
            return f"processed_{item}"
        
        items = [{'id': i, 'data': f'item_{i}'} for i in range(5)]
        self.optimizer.process_news_batch_optimized(items, dummy_process)
        
        # Get performance report
        report = self.optimizer.get_performance_report()
        
        # Check report structure
        self.assertIn('cache_performance', report)
        self.assertIn('parallel_processing', report)
        self.assertIn('efficiency_metrics', report)
        self.assertIn('custom_metrics', report)
        
        # Check cache performance
        cache_perf = report['cache_performance']
        self.assertIn('total_requests', cache_perf)
        self.assertIn('hit_rate', cache_perf)
        
        # Check efficiency metrics
        efficiency = report['efficiency_metrics']
        self.assertIn('cache_savings', efficiency)
        self.assertIn('estimated_speedup', efficiency)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CACHING_AVAILABLE:
            self.skipTest("Caching optimization not available")
    
    def test_optimize_news_processing_function(self):
        """Test the optimize_news_processing utility function"""
        def mock_process(item):
            return f"processed_{item['id']}"
        
        items = [{'id': 1, 'data': 'test1'}, {'id': 2, 'data': 'test2'}]
        
        results = optimize_news_processing(items, mock_process)
        
        self.assertEqual(len(results), len(items))
        for result in results:
            self.assertIn('processed_', str(result))
    
    def test_optimize_news_validation_function(self):
        """Test the optimize_news_validation utility function"""
        async def run_test():
            items = [{'title': 'Test', 'content': 'Content'}]
            
            # Mock the validation process
            with patch('caching_optimization.optimizer') as mock_optimizer:
                mock_optimizer.validate_news_batch_async.return_value = [
                    {
                        'item': items[0],
                        'validation': {'credibility_score': 0.7, 'confidence': 0.8}
                    }
                ]
                
                results = await optimize_news_validation(items)
                
                self.assertEqual(len(results), 1)
                mock_optimizer.validate_news_batch_async.assert_called_once()
        
        asyncio.run(run_test())
    
    def test_get_optimization_stats_function(self):
        """Test the get_optimization_stats utility function"""
        # Mock the optimizer to return stats
        with patch('caching_optimization.optimizer') as mock_optimizer:
            mock_optimizer.get_performance_report.return_value = {
                'cache_performance': {'hit_rate': 0.8},
                'efficiency_metrics': {'estimated_speedup': '4.2x'}
            }
            
            stats = get_optimization_stats()
            
            self.assertIn('cache_performance', stats)
            self.assertIn('efficiency_metrics', stats)
            mock_optimizer.get_performance_report.assert_called_once()

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CACHING_AVAILABLE:
            self.skipTest("Caching optimization not available")
    
    def test_high_load_scenario(self):
        """Test system behavior under high load"""
        cache = AdvancedCache(max_memory_size=100, enable_redis=False)
        optimizer = NewsProcessingOptimizer(cache_instance=cache, max_workers=10)
        
        def intensive_process(item):
            # Simulate CPU-intensive task
            result = 0
            for i in range(1000):
                result += i * item['value']
            return {'item_id': item['id'], 'result': result}
        
        # Generate large dataset
        items = [{'id': i, 'value': i % 100} for i in range(200)]
        
        start_time = time.time()
        results = optimizer.process_news_batch_optimized(items, intensive_process)
        end_time = time.time()
        
        # Check that all items were processed
        self.assertEqual(len(results), len(items))
        
        # Check performance metrics
        report = optimizer.get_performance_report()
        self.assertIn('cache_performance', report)
        
        processing_time = end_time - start_time
        self.assertLess(processing_time, 30)  # Should complete within 30 seconds
    
    def test_error_recovery_and_fallbacks(self):
        """Test error recovery and fallback mechanisms"""
        def unreliable_process(item):
            if item['id'] % 3 == 0:  # Fail every 3rd item
                raise Exception(f"Simulated error for item {item['id']}")
            return f"success_{item['id']}"
        
        items = [{'id': i} for i in range(15)]
        
        cache = AdvancedCache(max_memory_size=50, enable_redis=False)
        optimizer = NewsProcessingOptimizer(cache_instance=cache, max_workers=5)
        
        results = optimizer.process_news_batch_optimized(items, unreliable_process)
        
        # Check that system handled errors gracefully
        self.assertEqual(len(results), len(items))
        
        success_count = sum(1 for r in results if r is not None and 'success_' in str(r))
        error_count = sum(1 for r in results if r is None)
        
        # Should have both successes and errors
        self.assertGreater(success_count, 0)
        self.assertGreater(error_count, 0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        import sys
        
        cache = AdvancedCache(max_memory_size=1000, enable_redis=False)
        
        # Store large amount of data
        large_data = {'data': 'x' * 1000}  # 1KB per item
        
        initial_size = sys.getsizeof(cache.memory_cache)
        
        # Add many items
        for i in range(2000):  # More than cache capacity
            cache.set(f"key_{i}", large_data)
        
        final_size = sys.getsizeof(cache.memory_cache)
        
        # Memory should be bounded by LRU eviction
        self.assertLessEqual(len(cache.memory_cache), 1000)
        
        # Check that cache statistics are accurate
        stats = cache.get_stats()
        self.assertEqual(stats['memory_cache_size'], len(cache.memory_cache))

def run_chunk25_tests():
    """Run all Chunk 25 caching optimization tests"""
    if not CACHING_AVAILABLE:
        print("❌ Caching optimization not available - skipping tests")
        return False
    
    print("\n🧪 Running Chunk 25: Caching Optimization Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdvancedCache,
        TestParallelProcessor,
        TestNewsProcessingOptimizer,
        TestUtilityFunctions,
        TestIntegrationScenarios
    ]
    
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
    
    print(f"\n📊 Chunk 25 Test Results:")
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
        print("\n✅ All Chunk 25 caching optimization tests passed!")
    else:
        print("\n❌ Some Chunk 25 tests failed. Please review and fix issues.")
    
    return success

if __name__ == '__main__':
    success = run_chunk25_tests()
    exit(0 if success else 1)