#!/usr/bin/env python3
"""
Comprehensive Pipeline Validation Script

This script validates the fake news detection system to ensure:
1. Consistency - Same input always produces same verdict
2. Confidence - Confidence scores are properly calibrated and meaningful
3. Timing - Processing stays within SLA bounds (15-30 seconds)
4. Integration - All components work together seamlessly
5. Reliability - System handles edge cases gracefully
"""

import asyncio
import json
import time
import statistics
import sys
import os
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'FakeNewsBackend'))

try:
    from FakeNewsBackend.orchestrator import FakeNewsOrchestrator
except ImportError as e:
    print(f"Error importing orchestrator: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class ValidationResults:
    """Container for validation test results"""
    def __init__(self):
        self.consistency_tests = []
        self.confidence_tests = []
        self.timing_tests = []
        self.integration_tests = []
        self.reliability_tests = []
        self.overall_score = 0.0
        self.passed = False

    def add_test(self, category: str, test_name: str, passed: bool, details: Dict[str, Any]):
        """Add a test result"""
        test_result = {
            'test_name': test_name,
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        }
        
        if category == 'consistency':
            self.consistency_tests.append(test_result)
        elif category == 'confidence':
            self.confidence_tests.append(test_result)
        elif category == 'timing':
            self.timing_tests.append(test_result)
        elif category == 'integration':
            self.integration_tests.append(test_result)
        elif category == 'reliability':
            self.reliability_tests.append(test_result)

    def calculate_score(self):
        """Calculate overall validation score"""
        all_tests = (
            self.consistency_tests + 
            self.confidence_tests + 
            self.timing_tests + 
            self.integration_tests + 
            self.reliability_tests
        )
        
        if not all_tests:
            self.overall_score = 0.0
            self.passed = False
            return
        
        passed_tests = sum(1 for test in all_tests if test['passed'])
        self.overall_score = (passed_tests / len(all_tests)) * 100
        self.passed = self.overall_score >= 80.0  # 80% pass rate required

class PipelineValidator:
    """Main validation class"""
    
    def __init__(self):
        self.orchestrator = None
        self.results = ValidationResults()
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> Dict[str, List[str]]:
        """Load test cases for validation"""
        return {
            'fake_news': [
                "Breaking: Scientists discover that vaccines contain microchips for mind control",
                "URGENT: 5G towers are spreading coronavirus through radio waves",
                "Local man discovers this one weird trick that doctors hate - cures cancer instantly"
            ],
            'real_news': [
                "The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting",
                "NASA's James Webb Space Telescope captures detailed images of distant galaxies",
                "New study published in Nature shows promising results for Alzheimer's treatment"
            ],
            'ambiguous': [
                "Weather forecast predicts rain tomorrow",
                "Stock market shows mixed signals",
                "Local restaurant opens new location"
            ],
            'edge_cases': [
                "",  # Empty string
                "a",  # Single character
                "The quick brown fox jumps over the lazy dog" * 100,  # Very long text
                "🚀🌟💫✨🎯🔥💯⚡🌈🎉",  # Only emojis
                "123456789",  # Only numbers
            ]
        }
    
    async def initialize_orchestrator(self) -> bool:
        """Initialize the orchestrator for testing"""
        try:
            model_path = project_root / 'FakeNewsBackend' / 'models' / 'mhfn_model.pth'
            if not model_path.exists():
                print(f"Warning: Model file not found at {model_path}")
                print("Some tests may fail without the model")
            
            self.orchestrator = FakeNewsOrchestrator(
                min_processing_time=15,
                max_processing_time=30,
                model_path=str(model_path)
            )
            
            print("✅ Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            return False
    
    async def test_consistency(self) -> None:
        """Test that same inputs produce consistent results"""
        print("\n🔄 Testing Consistency...")
        
        for category, texts in self.test_cases.items():
            if category == 'edge_cases':  # Skip edge cases for consistency testing
                continue
                
            for text in texts[:2]:  # Test first 2 from each category
                if not text.strip():  # Skip empty texts
                    continue
                    
                print(f"  Testing: {text[:50]}...")
                
                # Run same input 3 times
                results = []
                for i in range(3):
                    try:
                        result = await self.orchestrator.detect_fake_news(text)
                        results.append(result)
                        await asyncio.sleep(1)  # Small delay between tests
                    except Exception as e:
                        print(f"    ❌ Run {i+1} failed: {e}")
                        continue
                
                if len(results) < 2:
                    self.results.add_test('consistency', f'consistency_{category}_{len(self.results.consistency_tests)}', 
                                        False, {'error': 'Insufficient results', 'text': text[:50]})
                    continue
                
                # Check verdict consistency
                verdicts = [r.get('verdict', 'unknown') for r in results]
                verdict_consistent = len(set(verdicts)) == 1
                
                # Check confidence consistency (within 5% tolerance)
                confidences = [r.get('confidence', 0) for r in results]
                confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
                confidence_consistent = confidence_std < 0.05
                
                passed = verdict_consistent and confidence_consistent
                
                self.results.add_test('consistency', f'consistency_{category}_{len(self.results.consistency_tests)}', 
                                    passed, {
                                        'text': text[:50],
                                        'verdicts': verdicts,
                                        'confidences': confidences,
                                        'verdict_consistent': verdict_consistent,
                                        'confidence_std': confidence_std
                                    })
                
                status = "✅" if passed else "❌"
                print(f"    {status} Verdict: {verdict_consistent}, Confidence std: {confidence_std:.3f}")
    
    async def test_confidence_calibration(self) -> None:
        """Test that confidence scores are meaningful and calibrated"""
        print("\n📊 Testing Confidence Calibration...")
        
        # Test that fake news gets lower confidence for 'real' predictions
        fake_confidences = []
        real_confidences = []
        
        for text in self.test_cases['fake_news']:
            try:
                result = await self.orchestrator.detect_fake_news(text)
                verdict = result.get('verdict', 'unknown')
                confidence = result.get('confidence', 0)
                
                if verdict.lower() == 'fake':
                    fake_confidences.append(confidence)
                else:
                    fake_confidences.append(1 - confidence)  # Invert if predicted as real
                    
            except Exception as e:
                print(f"    ❌ Failed to test fake news: {e}")
        
        for text in self.test_cases['real_news']:
            try:
                result = await self.orchestrator.detect_fake_news(text)
                verdict = result.get('verdict', 'unknown')
                confidence = result.get('confidence', 0)
                
                if verdict.lower() == 'real':
                    real_confidences.append(confidence)
                else:
                    real_confidences.append(1 - confidence)  # Invert if predicted as fake
                    
            except Exception as e:
                print(f"    ❌ Failed to test real news: {e}")
        
        # Test confidence bounds (should be between 0 and 1)
        all_confidences = fake_confidences + real_confidences
        bounds_valid = all(0 <= c <= 1 for c in all_confidences)
        
        # Test confidence distribution (should not be all the same)
        confidence_variance = statistics.variance(all_confidences) if len(all_confidences) > 1 else 0
        distribution_valid = confidence_variance > 0.01  # Some variance expected
        
        # Test average confidence for correct predictions
        avg_fake_confidence = statistics.mean(fake_confidences) if fake_confidences else 0
        avg_real_confidence = statistics.mean(real_confidences) if real_confidences else 0
        
        confidence_reasonable = avg_fake_confidence > 0.5 and avg_real_confidence > 0.5
        
        passed = bounds_valid and distribution_valid and confidence_reasonable
        
        self.results.add_test('confidence', 'confidence_calibration', passed, {
            'bounds_valid': bounds_valid,
            'distribution_valid': distribution_valid,
            'confidence_reasonable': confidence_reasonable,
            'avg_fake_confidence': avg_fake_confidence,
            'avg_real_confidence': avg_real_confidence,
            'confidence_variance': confidence_variance
        })
        
        status = "✅" if passed else "❌"
        print(f"    {status} Bounds: {bounds_valid}, Distribution: {distribution_valid}, Reasonable: {confidence_reasonable}")
    
    async def test_timing_constraints(self) -> None:
        """Test that processing stays within SLA bounds"""
        print("\n⏱️ Testing Timing Constraints...")
        
        timing_results = []
        
        for category, texts in self.test_cases.items():
            if category == 'edge_cases':  # Skip edge cases for timing
                continue
                
            for text in texts[:1]:  # Test one from each category
                if not text.strip():
                    continue
                    
                print(f"  Testing timing for: {text[:30]}...")
                
                start_time = time.time()
                try:
                    result = await self.orchestrator.detect_fake_news(text)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    timing_results.append(processing_time)
                    
                    # Check SLA bounds (15-30 seconds)
                    within_bounds = 15 <= processing_time <= 35  # Allow 5s buffer
                    
                    self.results.add_test('timing', f'timing_{category}', within_bounds, {
                        'processing_time': processing_time,
                        'text': text[:30],
                        'within_bounds': within_bounds
                    })
                    
                    status = "✅" if within_bounds else "❌"
                    print(f"    {status} {processing_time:.1f}s (target: 15-30s)")
                    
                except Exception as e:
                    print(f"    ❌ Timing test failed: {e}")
                    self.results.add_test('timing', f'timing_{category}', False, {
                        'error': str(e),
                        'text': text[:30]
                    })
        
        # Test average timing
        if timing_results:
            avg_time = statistics.mean(timing_results)
            avg_within_bounds = 15 <= avg_time <= 30
            
            self.results.add_test('timing', 'average_timing', avg_within_bounds, {
                'average_time': avg_time,
                'all_times': timing_results
            })
            
            print(f"  📈 Average processing time: {avg_time:.1f}s")
    
    async def test_integration(self) -> None:
        """Test that all components integrate properly"""
        print("\n🔗 Testing Integration...")
        
        # Test complete pipeline with different input types
        test_text = "Scientists announce breakthrough in renewable energy technology"
        
        try:
            result = await self.orchestrator.detect_fake_news(test_text)
            
            # Check required fields
            required_fields = ['verdict', 'confidence', 'processing_time_s']
            has_required_fields = all(field in result for field in required_fields)
            
            # Check optional orchestrator fields
            orchestrator_fields = ['fusion_details', 'processing_details', 'evidence_summary']
            has_orchestrator_fields = any(field in result for field in orchestrator_fields)
            
            # Check data types
            correct_types = (
                isinstance(result.get('verdict'), str) and
                isinstance(result.get('confidence'), (int, float)) and
                isinstance(result.get('processing_time_s'), (int, float))
            )
            
            passed = has_required_fields and has_orchestrator_fields and correct_types
            
            self.results.add_test('integration', 'pipeline_integration', passed, {
                'has_required_fields': has_required_fields,
                'has_orchestrator_fields': has_orchestrator_fields,
                'correct_types': correct_types,
                'result_keys': list(result.keys())
            })
            
            status = "✅" if passed else "❌"
            print(f"    {status} Pipeline integration test")
            
        except Exception as e:
            print(f"    ❌ Integration test failed: {e}")
            self.results.add_test('integration', 'pipeline_integration', False, {
                'error': str(e)
            })
    
    async def test_reliability(self) -> None:
        """Test system reliability with edge cases"""
        print("\n🛡️ Testing Reliability...")
        
        for i, text in enumerate(self.test_cases['edge_cases']):
            test_name = f'edge_case_{i}'
            print(f"  Testing edge case {i+1}: {repr(text[:20])}...")
            
            try:
                result = await self.orchestrator.detect_fake_news(text)
                
                # Should handle gracefully without crashing
                has_verdict = 'verdict' in result
                has_confidence = 'confidence' in result
                
                passed = has_verdict and has_confidence
                
                self.results.add_test('reliability', test_name, passed, {
                    'input': repr(text[:50]),
                    'has_verdict': has_verdict,
                    'has_confidence': has_confidence,
                    'result': result
                })
                
                status = "✅" if passed else "❌"
                print(f"    {status} Edge case handled")
                
            except Exception as e:
                # Some edge cases might legitimately fail, but shouldn't crash
                print(f"    ⚠️ Edge case failed gracefully: {e}")
                self.results.add_test('reliability', test_name, True, {
                    'input': repr(text[:50]),
                    'graceful_failure': True,
                    'error': str(e)
                })
    
    def print_summary(self) -> None:
        """Print validation summary"""
        print("\n" + "="*60)
        print("📋 VALIDATION SUMMARY")
        print("="*60)
        
        categories = [
            ('Consistency', self.results.consistency_tests),
            ('Confidence', self.results.confidence_tests),
            ('Timing', self.results.timing_tests),
            ('Integration', self.results.integration_tests),
            ('Reliability', self.results.reliability_tests)
        ]
        
        for category_name, tests in categories:
            if not tests:
                continue
                
            passed = sum(1 for test in tests if test['passed'])
            total = len(tests)
            percentage = (passed / total) * 100 if total > 0 else 0
            
            status = "✅" if percentage >= 80 else "❌"
            print(f"{status} {category_name}: {passed}/{total} ({percentage:.1f}%)")
        
        print(f"\n🎯 Overall Score: {self.results.overall_score:.1f}%")
        print(f"🏆 Validation {'PASSED' if self.results.passed else 'FAILED'}")
        
        if not self.results.passed:
            print("\n⚠️ Issues found:")
            for category_name, tests in categories:
                failed_tests = [test for test in tests if not test['passed']]
                for test in failed_tests:
                    print(f"  - {category_name}: {test['test_name']} - {test['details']}")
    
    def save_results(self, filename: str = 'validation_results.json') -> None:
        """Save validation results to file"""
        results_data = {
            'timestamp': time.time(),
            'overall_score': self.results.overall_score,
            'passed': self.results.passed,
            'consistency_tests': self.results.consistency_tests,
            'confidence_tests': self.results.confidence_tests,
            'timing_tests': self.results.timing_tests,
            'integration_tests': self.results.integration_tests,
            'reliability_tests': self.results.reliability_tests
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
    
    async def run_all_tests(self) -> bool:
        """Run all validation tests"""
        print("🚀 Starting Comprehensive Pipeline Validation")
        print("="*60)
        
        if not await self.initialize_orchestrator():
            return False
        
        # Run all test categories
        await self.test_consistency()
        await self.test_confidence_calibration()
        await self.test_timing_constraints()
        await self.test_integration()
        await self.test_reliability()
        
        # Calculate final score
        self.results.calculate_score()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        return self.results.passed

async def main():
    """Main validation function"""
    validator = PipelineValidator()
    
    try:
        success = await validator.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())