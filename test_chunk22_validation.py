#!/usr/bin/env python3
"""
Chunk 22 Validation Test Script
Tests multimodal consistency, embedding optimization, and auto-batch processing

Author: FakeNewsBackend Team
Date: August 26, 2025
Chunk: 22
"""

import sys
import os
import time
import requests
import json
import numpy as np
import torch
from PIL import Image
import io
import base64
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
API_BASE_URL = "http://localhost:5001"
TEST_TIMEOUT = 30

class Chunk22Validator:
    """Comprehensive validator for Chunk 22 features"""
    
    def __init__(self):
        self.api_base = API_BASE_URL
        self.test_results = {
            'multimodal_consistency': {'passed': 0, 'failed': 0, 'errors': []},
            'embedding_optimization': {'passed': 0, 'failed': 0, 'errors': []},
            'auto_batch_processing': {'passed': 0, 'failed': 0, 'errors': []},
            'end_to_end_integration': {'passed': 0, 'failed': 0, 'errors': []}
        }
        self.start_time = time.time()
    
    def log_test_result(self, category: str, test_name: str, passed: bool, error_msg: str = None):
        """Log test result"""
        if passed:
            self.test_results[category]['passed'] += 1
            logger.info(f"✅ {test_name} - PASSED")
        else:
            self.test_results[category]['failed'] += 1
            self.test_results[category]['errors'].append(f"{test_name}: {error_msg}")
            logger.error(f"❌ {test_name} - FAILED: {error_msg}")
    
    def test_api_health(self) -> bool:
        """Test if API is running"""
        try:
            response = requests.get(f"{self.api_base}/api/health", timeout=TEST_TIMEOUT)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    def create_test_image(self) -> str:
        """Create a test image and return as base64"""
        try:
            # Create a simple test image
            img = Image.new('RGB', (100, 100), color='red')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error creating test image: {e}")
            return None
    
    def test_multimodal_consistency(self):
        """Test multimodal consistency with RoBERTa text embeddings and CLIP/BLIP image features"""
        logger.info("\n=== Testing Multimodal Consistency ===")
        
        test_cases = [
            {
                'text': 'Breaking: Scientists discover new planet with potential for life',
                'description': 'Consistent science news'
            },
            {
                'text': 'SHOCKING: Aliens landed in New York City yesterday!',
                'description': 'Inconsistent fake news'
            },
            {
                'text': 'Local weather forecast shows sunny skies ahead',
                'description': 'Neutral news content'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                # Create test image
                test_image = self.create_test_image()
                if not test_image:
                    self.log_test_result('multimodal_consistency', 
                                       f"Test {i+1}: Image creation", False, "Failed to create test image")
                    continue
                
                # Test multimodal consistency endpoint
                payload = {
                    'text': test_case['text'],
                    'image': test_image,
                    'threshold': 0.7
                }
                
                response = requests.post(
                    f"{self.api_base}/api/multimodal-consistency",
                    json=payload,
                    timeout=TEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate response structure
                    required_fields = ['consistent', 'similarity', 'threshold', 'fake_flag']
                    if all(field in result for field in required_fields):
                        # Validate data types and ranges
                        if (isinstance(result['consistent'], bool) and
                            isinstance(result['similarity'], (int, float)) and
                            0 <= result['similarity'] <= 1 and
                            isinstance(result['fake_flag'], bool)):
                            
                            self.log_test_result('multimodal_consistency', 
                                               f"Test {i+1}: {test_case['description']}", True)
                            logger.info(f"   Similarity: {result['similarity']:.3f}, Consistent: {result['consistent']}, Fake Flag: {result['fake_flag']}")
                        else:
                            self.log_test_result('multimodal_consistency', 
                                               f"Test {i+1}: Data validation", False, "Invalid data types or ranges")
                    else:
                        self.log_test_result('multimodal_consistency', 
                                           f"Test {i+1}: Response structure", False, f"Missing fields: {set(required_fields) - set(result.keys())}")
                else:
                    self.log_test_result('multimodal_consistency', 
                                       f"Test {i+1}: API response", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test_result('multimodal_consistency', 
                                   f"Test {i+1}: Exception handling", False, str(e))
    
    def test_embedding_optimization(self):
        """Test embedding optimization with PCA reduction"""
        logger.info("\n=== Testing Embedding Optimization ===")
        
        test_texts = [
            "This is a test article about technology and innovation in the modern world.",
            "Breaking news: Major political development affects global markets significantly.",
            "Sports update: Championship game results surprise fans and analysts alike.",
            "Health advisory: New research reveals important findings about nutrition.",
            "Entertainment news: Celebrity announcement creates buzz on social media platforms."
        ]
        
        try:
            # Test embedding extraction endpoint
            payload = {
                'texts': test_texts,
                'embedding_type': 'hybrid',
                'apply_pca': True,
                'target_dimension': 300
            }
            
            response = requests.post(
                f"{self.api_base}/api/extract-embeddings",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                if 'embeddings' in result and 'metadata' in result:
                    embeddings = result['embeddings']
                    metadata = result['metadata']
                    
                    # Check embedding dimensions
                    if (len(embeddings) == len(test_texts) and
                        all(len(emb) == 300 for emb in embeddings)):
                        
                        self.log_test_result('embedding_optimization', 
                                           "Embedding extraction", True)
                        
                        # Check PCA variance preservation
                        if 'pca_variance_explained' in metadata:
                            variance = metadata['pca_variance_explained']
                            if variance >= 0.85:  # At least 85% variance preserved
                                self.log_test_result('embedding_optimization', 
                                                   "PCA variance preservation", True)
                                logger.info(f"   PCA variance explained: {variance:.3f}")
                            else:
                                self.log_test_result('embedding_optimization', 
                                                   "PCA variance preservation", False, 
                                                   f"Low variance: {variance:.3f}")
                        else:
                            self.log_test_result('embedding_optimization', 
                                               "PCA metadata", False, "Missing variance info")
                    else:
                        self.log_test_result('embedding_optimization', 
                                           "Embedding dimensions", False, "Incorrect dimensions")
                else:
                    self.log_test_result('embedding_optimization', 
                                       "Response structure", False, "Missing required fields")
            else:
                self.log_test_result('embedding_optimization', 
                                   "API response", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result('embedding_optimization', 
                               "Exception handling", False, str(e))
    
    def test_auto_batch_processing(self):
        """Test auto-batch processing for 5x productivity boost"""
        logger.info("\n=== Testing Auto-Batch Processing ===")
        
        # Create larger batch for performance testing
        batch_texts = [
            f"Test article number {i}: This is sample content for batch processing validation with various topics and lengths to ensure proper handling."
            for i in range(50)  # 50 texts for batch testing
        ]
        
        try:
            # Test batch prediction endpoint
            start_time = time.time()
            
            payload = {
                'texts': batch_texts,
                'batch_size': 16,
                'use_hybrid_embeddings': True,
                'include_multimodal': False
            }
            
            response = requests.post(
                f"{self.api_base}/api/predict-batch",
                json=payload,
                timeout=TEST_TIMEOUT * 2  # Extended timeout for batch processing
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                if 'predictions' in result and 'metadata' in result:
                    predictions = result['predictions']
                    metadata = result['metadata']
                    
                    # Check prediction count and structure
                    if len(predictions) == len(batch_texts):
                        # Validate prediction structure
                        valid_predictions = all(
                            isinstance(pred, dict) and
                            'prediction' in pred and
                            'label' in pred and
                            'confidence' in pred and
                            0 <= pred['prediction'] <= 1
                            for pred in predictions
                        )
                        
                        if valid_predictions:
                            self.log_test_result('auto_batch_processing', 
                                               "Batch prediction structure", True)
                            
                            # Check processing efficiency
                            if 'batch_processed' in metadata and metadata['batch_processed']:
                                self.log_test_result('auto_batch_processing', 
                                                   "Batch processing mode", True)
                                logger.info(f"   Processing time: {processing_time:.2f}s for {len(batch_texts)} texts")
                                logger.info(f"   Average time per text: {processing_time/len(batch_texts):.3f}s")
                                
                                # Performance benchmark (should be faster than individual processing)
                                if processing_time < len(batch_texts) * 0.1:  # Less than 0.1s per text
                                    self.log_test_result('auto_batch_processing', 
                                                       "Performance benchmark", True)
                                else:
                                    self.log_test_result('auto_batch_processing', 
                                                       "Performance benchmark", False, 
                                                       f"Too slow: {processing_time:.2f}s")
                            else:
                                self.log_test_result('auto_batch_processing', 
                                                   "Batch processing mode", False, "Not using batch mode")
                        else:
                            self.log_test_result('auto_batch_processing', 
                                               "Prediction validation", False, "Invalid prediction structure")
                    else:
                        self.log_test_result('auto_batch_processing', 
                                           "Prediction count", False, 
                                           f"Expected {len(batch_texts)}, got {len(predictions)}")
                else:
                    self.log_test_result('auto_batch_processing', 
                                       "Response structure", False, "Missing required fields")
            else:
                self.log_test_result('auto_batch_processing', 
                                   "API response", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result('auto_batch_processing', 
                               "Exception handling", False, str(e))
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration with text/image pairs"""
        logger.info("\n=== Testing End-to-End Integration ===")
        
        test_cases = [
            {
                'text': 'Scientists announce breakthrough in renewable energy technology',
                'expected_label': 'Real',
                'description': 'Science news with image'
            },
            {
                'text': 'BREAKING: Aliens confirmed to exist by government officials',
                'expected_label': 'Fake',
                'description': 'Fake news with image'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                # Create test image
                test_image = self.create_test_image()
                if not test_image:
                    self.log_test_result('end_to_end_integration', 
                                       f"Test {i+1}: Image creation", False, "Failed to create test image")
                    continue
                
                # Test full pipeline endpoint
                payload = {
                    'text': test_case['text'],
                    'image': test_image,
                    'use_multimodal_consistency': True,
                    'use_hybrid_embeddings': True,
                    'consistency_threshold': 0.7
                }
                
                response = requests.post(
                    f"{self.api_base}/api/detect",
                    json=payload,
                    timeout=TEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate comprehensive response
                    required_fields = ['prediction', 'label', 'confidence', 'multimodal_consistency', 'embedding_info']
                    if all(field in result for field in required_fields):
                        # Check multimodal consistency info
                        mc_info = result['multimodal_consistency']
                        if ('consistent' in mc_info and 'similarity' in mc_info and 'fake_flag' in mc_info):
                            # Check embedding info
                            emb_info = result['embedding_info']
                            if ('type' in emb_info and 'dimension' in emb_info):
                                self.log_test_result('end_to_end_integration', 
                                                   f"Test {i+1}: {test_case['description']}", True)
                                
                                logger.info(f"   Prediction: {result['prediction']:.3f}")
                                logger.info(f"   Label: {result['label']}")
                                logger.info(f"   Confidence: {result['confidence']:.3f}")
                                logger.info(f"   Multimodal Consistent: {mc_info['consistent']}")
                                logger.info(f"   Embedding Type: {emb_info['type']}")
                            else:
                                self.log_test_result('end_to_end_integration', 
                                                   f"Test {i+1}: Embedding info", False, "Missing embedding details")
                        else:
                            self.log_test_result('end_to_end_integration', 
                                               f"Test {i+1}: Multimodal info", False, "Missing multimodal details")
                    else:
                        self.log_test_result('end_to_end_integration', 
                                           f"Test {i+1}: Response structure", False, 
                                           f"Missing fields: {set(required_fields) - set(result.keys())}")
                else:
                    self.log_test_result('end_to_end_integration', 
                                       f"Test {i+1}: API response", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test_result('end_to_end_integration', 
                                   f"Test {i+1}: Exception handling", False, str(e))
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_time = time.time() - self.start_time
        
        # Calculate overall statistics
        total_passed = sum(category['passed'] for category in self.test_results.values())
        total_failed = sum(category['failed'] for category in self.test_results.values())
        total_tests = total_passed + total_failed
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'chunk': 22,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_execution_time': f"{total_time:.2f}s",
            'overall_statistics': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': f"{success_rate:.1f}%"
            },
            'category_results': {},
            'completion_status': 'COMPLETED' if success_rate >= 80 else 'FAILED',
            'api_endpoint': self.api_base
        }
        
        # Add category-specific results
        for category, results in self.test_results.items():
            category_total = results['passed'] + results['failed']
            category_success = (results['passed'] / category_total * 100) if category_total > 0 else 0
            
            report['category_results'][category] = {
                'passed': results['passed'],
                'failed': results['failed'],
                'success_rate': f"{category_success:.1f}%",
                'errors': results['errors']
            }
        
        return report
    
    def run_all_tests(self):
        """Run all validation tests"""
        logger.info("🚀 Starting Chunk 22 Validation Tests")
        logger.info(f"API Base URL: {self.api_base}")
        
        # Check API health first
        if not self.test_api_health():
            logger.error("❌ API is not running. Please start the Flask application first.")
            return self.generate_report()
        
        logger.info("✅ API is running and healthy")
        
        # Run all test categories
        self.test_multimodal_consistency()
        self.test_embedding_optimization()
        self.test_auto_batch_processing()
        self.test_end_to_end_integration()
        
        # Generate and return report
        return self.generate_report()

def main():
    """Main validation function"""
    validator = Chunk22Validator()
    report = validator.run_all_tests()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 CHUNK 22 VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Tests: {report['overall_statistics']['total_tests']}")
    logger.info(f"Passed: {report['overall_statistics']['passed']}")
    logger.info(f"Failed: {report['overall_statistics']['failed']}")
    logger.info(f"Success Rate: {report['overall_statistics']['success_rate']}")
    logger.info(f"Status: {report['completion_status']}")
    logger.info(f"Execution Time: {report['total_execution_time']}")
    
    # Print category breakdown
    for category, results in report['category_results'].items():
        logger.info(f"\n{category.replace('_', ' ').title()}:")
        logger.info(f"  ✅ Passed: {results['passed']}")
        logger.info(f"  ❌ Failed: {results['failed']}")
        logger.info(f"  📈 Success Rate: {results['success_rate']}")
        
        if results['errors']:
            logger.info("  🔍 Errors:")
            for error in results['errors']:
                logger.info(f"    - {error}")
    
    # Save report to file
    try:
        with open('chunk22_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n📄 Detailed report saved to: chunk22_validation_report.json")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    return report['completion_status'] == 'COMPLETED'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)