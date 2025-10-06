#!/usr/bin/env python3
"""
Diagnostic Model Testing Suite
Focuses on identifying specific model issues and providing detailed diagnostics
for the fake news detection system.
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiagnosticModelTester:
    """Diagnostic testing suite to identify model issues"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.test_results = []
        self.confidence_scores = []
        self.processing_times = []
        self.predictions = []
        
    def test_model_consistency(self) -> Dict:
        """Test if model produces consistent results for identical inputs"""
        logger.info("Testing model consistency...")
        
        test_text = "This is a test message for consistency checking."
        results = []
        
        # Run same input multiple times
        for i in range(5):
            try:
                response = requests.post(
                    f"{self.base_url}/api/detect",
                    json={'text': test_text},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        'run': i + 1,
                        'verdict': result.get('verdict'),
                        'confidence': result.get('confidence'),
                        'processing_time': time.time()
                    })
                    logger.info(f"Run {i+1}: {result.get('verdict')} ({result.get('confidence')})")
                else:
                    logger.error(f"Run {i+1} failed: {response.status_code}")
                    
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Run {i+1} exception: {e}")
        
        # Analyze consistency
        if results:
            verdicts = [r['verdict'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            verdict_consistency = len(set(verdicts)) == 1
            confidence_variance = np.var(confidences) if confidences else 0
            
            return {
                'test_type': 'consistency',
                'input': test_text,
                'runs': len(results),
                'verdict_consistency': verdict_consistency,
                'unique_verdicts': list(set(verdicts)),
                'confidence_variance': confidence_variance,
                'confidence_range': [min(confidences), max(confidences)] if confidences else [0, 0],
                'results': results,
                'status': 'success' if verdict_consistency and confidence_variance < 0.01 else 'inconsistent'
            }
        else:
            return {'test_type': 'consistency', 'status': 'failed', 'error': 'No successful runs'}
    
    def test_confidence_variation(self) -> Dict:
        """Test if model produces varied confidence scores for different inputs"""
        logger.info("Testing confidence score variation...")
        
        test_cases = [
            {"text": "The sky is blue and water is wet.", "expected_confidence": "high"},
            {"text": "BREAKING: Aliens invade Earth, government covers it up!", "expected_confidence": "high"},
            {"text": "Scientists may have discovered something interesting.", "expected_confidence": "medium"},
            {"text": "Some people think that maybe things could be different.", "expected_confidence": "low"},
            {"text": "URGENT SHOCKING BREAKING NEWS CLICK HERE NOW!!!", "expected_confidence": "high"}
        ]
        
        results = []
        confidences = []
        
        for i, case in enumerate(test_cases):
            try:
                response = requests.post(
                    f"{self.base_url}/api/detect",
                    json={'text': case['text']},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    confidence = result.get('confidence', 0)
                    confidences.append(confidence)
                    
                    results.append({
                        'case': i + 1,
                        'text': case['text'][:50] + '...',
                        'verdict': result.get('verdict'),
                        'confidence': confidence,
                        'expected_confidence': case['expected_confidence']
                    })
                    
                    logger.info(f"Case {i+1}: {result.get('verdict')} ({confidence})")
                else:
                    logger.error(f"Case {i+1} failed: {response.status_code}")
                    
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Case {i+1} exception: {e}")
        
        # Analyze confidence variation
        if confidences:
            confidence_variance = np.var(confidences)
            confidence_range = max(confidences) - min(confidences)
            unique_confidences = len(set(confidences))
            
            return {
                'test_type': 'confidence_variation',
                'total_cases': len(test_cases),
                'successful_cases': len(results),
                'confidence_variance': confidence_variance,
                'confidence_range': confidence_range,
                'unique_confidence_scores': unique_confidences,
                'all_confidences': confidences,
                'results': results,
                'status': 'varied' if confidence_variance > 0.01 and unique_confidences > 1 else 'static'
            }
        else:
            return {'test_type': 'confidence_variation', 'status': 'failed', 'error': 'No successful cases'}
    
    def test_extreme_cases(self) -> Dict:
        """Test model behavior with extreme or edge case inputs"""
        logger.info("Testing extreme cases...")
        
        extreme_cases = [
            {"text": "", "description": "Empty string"},
            {"text": "a", "description": "Single character"},
            {"text": "The " * 1000, "description": "Very long repetitive text"},
            {"text": "!@#$%^&*()_+{}|:<>?[]\\", "description": "Special characters only"},
            {"text": "123456789", "description": "Numbers only"},
            {"text": "AAAAAAAAAAAAAAAAAAAA", "description": "Repeated characters"},
            {"text": "This is a normal sentence with proper grammar and structure.", "description": "Well-formed sentence"},
            {"text": "thIs iS wEiRd CaPiTaLiZaTiOn", "description": "Mixed capitalization"}
        ]
        
        results = []
        
        for i, case in enumerate(extreme_cases):
            try:
                response = requests.post(
                    f"{self.base_url}/api/detect",
                    json={'text': case['text']},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        'case': i + 1,
                        'description': case['description'],
                        'text_length': len(case['text']),
                        'verdict': result.get('verdict'),
                        'confidence': result.get('confidence'),
                        'evidence_count': len(result.get('evidence', [])),
                        'status': 'success'
                    })
                    
                    logger.info(f"Extreme case {i+1} ({case['description']}): {result.get('verdict')} ({result.get('confidence')})")
                else:
                    results.append({
                        'case': i + 1,
                        'description': case['description'],
                        'text_length': len(case['text']),
                        'error': f"HTTP {response.status_code}",
                        'status': 'failed'
                    })
                    logger.error(f"Extreme case {i+1} failed: {response.status_code}")
                    
                time.sleep(1)
                
            except Exception as e:
                results.append({
                    'case': i + 1,
                    'description': case['description'],
                    'text_length': len(case['text']),
                    'error': str(e),
                    'status': 'failed'
                })
                logger.error(f"Extreme case {i+1} exception: {e}")
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        return {
            'test_type': 'extreme_cases',
            'total_cases': len(extreme_cases),
            'successful_cases': len(successful_results),
            'failed_cases': len(results) - len(successful_results),
            'results': results,
            'status': 'completed'
        }
    
    def test_input_types(self) -> Dict:
        """Test different input types (text, URL, image)"""
        logger.info("Testing different input types...")
        
        test_cases = [
            {
                'type': 'text',
                'input': {'text': 'This is a test text message.'},
                'description': 'Simple text input'
            },
            {
                'type': 'url',
                'input': {'url': 'https://www.example.com/news-article'},
                'description': 'URL input'
            },
            {
                'type': 'image',
                'input': {'image_path': 'test_image.jpg'},
                'description': 'Image path input'
            },
            {
                'type': 'image_base64',
                'input': {'image': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2ODApLCBxdWFsaXR5ID0gOTAK/9sAQwADAgIDAgIDAwMDBAMDBAUIBQUEBAUKBwcGCAwKDAwLCgsLDQ4SEA0OEQ4LCxAWEBETFBUVFQwPFxgWFBgSFBUU'},
                'description': 'Base64 image input'
            },
            {
                'type': 'multiple',
                'input': {'text': 'Test text', 'url': 'https://example.com'},
                'description': 'Multiple input types'
            },
            {
                'type': 'invalid',
                'input': {'invalid_field': 'test'},
                'description': 'Invalid input field'
            }
        ]
        
        results = []
        
        for i, case in enumerate(test_cases):
            try:
                response = requests.post(
                    f"{self.base_url}/api/detect",
                    json=case['input'],
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                result_data = {
                    'case': i + 1,
                    'type': case['type'],
                    'description': case['description'],
                    'status_code': response.status_code,
                    'input_keys': list(case['input'].keys())
                }
                
                if response.status_code == 200:
                    result = response.json()
                    result_data.update({
                        'verdict': result.get('verdict'),
                        'confidence': result.get('confidence'),
                        'evidence_count': len(result.get('evidence', [])),
                        'status': 'success'
                    })
                    logger.info(f"Input type {case['type']}: SUCCESS - {result.get('verdict')} ({result.get('confidence')})")
                else:
                    try:
                        error_data = response.json()
                        result_data.update({
                            'error_message': error_data.get('message', 'Unknown error'),
                            'status': 'failed'
                        })
                    except:
                        result_data.update({
                            'error_message': response.text,
                            'status': 'failed'
                        })
                    logger.error(f"Input type {case['type']}: FAILED - {response.status_code}")
                
                results.append(result_data)
                time.sleep(1)
                
            except Exception as e:
                results.append({
                    'case': i + 1,
                    'type': case['type'],
                    'description': case['description'],
                    'error': str(e),
                    'status': 'exception'
                })
                logger.error(f"Input type {case['type']}: EXCEPTION - {e}")
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        return {
            'test_type': 'input_types',
            'total_cases': len(test_cases),
            'successful_cases': len(successful_results),
            'failed_cases': len([r for r in results if r['status'] == 'failed']),
            'exception_cases': len([r for r in results if r['status'] == 'exception']),
            'results': results,
            'supported_input_types': [r['type'] for r in successful_results],
            'status': 'completed'
        }
    
    def test_api_endpoints(self) -> Dict:
        """Test various API endpoints for availability and functionality"""
        logger.info("Testing API endpoints...")
        
        endpoints = [
            {'path': '/api/health', 'method': 'GET', 'description': 'Health check'},
            {'path': '/api/detect', 'method': 'POST', 'description': 'Main detection endpoint', 'data': {'text': 'test'}},
            {'path': '/api/history', 'method': 'GET', 'description': 'Detection history'},
            {'path': '/api/live-feed', 'method': 'GET', 'description': 'Live news feed'},
            {'path': '/api/stats', 'method': 'GET', 'description': 'System statistics'},
            {'path': '/', 'method': 'GET', 'description': 'Frontend interface'}
        ]
        
        results = []
        
        for endpoint in endpoints:
            try:
                if endpoint['method'] == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint['path']}", timeout=10)
                else:
                    response = requests.post(
                        f"{self.base_url}{endpoint['path']}",
                        json=endpoint.get('data', {}),
                        headers={'Content-Type': 'application/json'},
                        timeout=10
                    )
                
                result_data = {
                    'endpoint': endpoint['path'],
                    'method': endpoint['method'],
                    'description': endpoint['description'],
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'status': 'success' if 200 <= response.status_code < 300 else 'failed'
                }
                
                # Try to parse JSON response
                try:
                    json_data = response.json()
                    result_data['response_type'] = 'json'
                    result_data['response_keys'] = list(json_data.keys()) if isinstance(json_data, dict) else None
                except:
                    result_data['response_type'] = 'text'
                    result_data['response_length'] = len(response.text)
                
                results.append(result_data)
                logger.info(f"Endpoint {endpoint['path']}: {response.status_code} ({response.elapsed.total_seconds():.2f}s)")
                
            except Exception as e:
                results.append({
                    'endpoint': endpoint['path'],
                    'method': endpoint['method'],
                    'description': endpoint['description'],
                    'error': str(e),
                    'status': 'exception'
                })
                logger.error(f"Endpoint {endpoint['path']}: EXCEPTION - {e}")
        
        successful_endpoints = [r for r in results if r['status'] == 'success']
        
        return {
            'test_type': 'api_endpoints',
            'total_endpoints': len(endpoints),
            'successful_endpoints': len(successful_endpoints),
            'failed_endpoints': len([r for r in results if r['status'] == 'failed']),
            'exception_endpoints': len([r for r in results if r['status'] == 'exception']),
            'results': results,
            'available_endpoints': [r['endpoint'] for r in successful_endpoints],
            'status': 'completed'
        }
    
    def run_diagnostic_suite(self) -> Dict:
        """Run complete diagnostic test suite"""
        logger.info("Starting comprehensive diagnostic testing...")
        start_time = time.time()
        
        diagnostic_tests = [
            ('Model Consistency', self.test_model_consistency),
            ('Confidence Variation', self.test_confidence_variation),
            ('Extreme Cases', self.test_extreme_cases),
            ('Input Types', self.test_input_types),
            ('API Endpoints', self.test_api_endpoints)
        ]
        
        results = {}
        
        for test_name, test_function in diagnostic_tests:
            logger.info(f"Running {test_name} diagnostic...")
            try:
                test_result = test_function()
                results[test_name] = test_result
                logger.info(f"Completed {test_name}: {test_result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"Error in {test_name}: {e}")
                results[test_name] = {'status': 'error', 'error': str(e)}
        
        total_time = time.time() - start_time
        
        # Generate diagnostic report
        report = {
            'diagnostic_timestamp': datetime.now().isoformat(),
            'total_diagnostic_time': total_time,
            'test_results': results,
            'summary': self._generate_diagnostic_summary(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save diagnostic report
        report_filename = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Diagnostic testing completed. Report saved to {report_filename}")
        return report
    
    def _generate_diagnostic_summary(self, results: Dict) -> Dict:
        """Generate summary of diagnostic findings"""
        summary = {
            'total_tests': len(results),
            'successful_tests': len([r for r in results.values() if r.get('status') not in ['error', 'failed']]),
            'critical_issues': [],
            'warnings': [],
            'system_health': 'unknown'
        }
        
        # Analyze each test result
        for test_name, result in results.items():
            if test_name == 'Model Consistency':
                if result.get('status') == 'inconsistent':
                    summary['critical_issues'].append('Model produces inconsistent results for identical inputs')
                elif result.get('confidence_variance', 0) > 0.1:
                    summary['warnings'].append('High confidence variance in consistency test')
            
            elif test_name == 'Confidence Variation':
                if result.get('status') == 'static':
                    summary['critical_issues'].append('Model produces static confidence scores - no variation')
                elif result.get('unique_confidence_scores', 0) < 3:
                    summary['warnings'].append('Limited confidence score variation')
            
            elif test_name == 'Input Types':
                if result.get('successful_cases', 0) < 2:
                    summary['critical_issues'].append('Multiple input types not supported')
                elif 'image' not in result.get('supported_input_types', []):
                    summary['warnings'].append('Image input not supported')
            
            elif test_name == 'API Endpoints':
                if result.get('successful_endpoints', 0) < 3:
                    summary['critical_issues'].append('Multiple API endpoints not functional')
        
        # Determine overall system health
        if len(summary['critical_issues']) == 0:
            summary['system_health'] = 'good' if len(summary['warnings']) < 2 else 'fair'
        else:
            summary['system_health'] = 'critical' if len(summary['critical_issues']) > 2 else 'poor'
        
        return summary
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate specific recommendations based on diagnostic results"""
        recommendations = []
        
        # Check consistency issues
        consistency_result = results.get('Model Consistency', {})
        if consistency_result.get('status') == 'inconsistent':
            recommendations.append("CRITICAL: Fix model consistency - implement deterministic inference")
        
        # Check confidence variation
        confidence_result = results.get('Confidence Variation', {})
        if confidence_result.get('status') == 'static':
            recommendations.append("CRITICAL: Fix confidence calculation - model returns identical scores")
        
        # Check input type support
        input_result = results.get('Input Types', {})
        if input_result.get('successful_cases', 0) < 3:
            recommendations.append("HIGH: Implement proper multimodal input handling")
        
        # Check API functionality
        api_result = results.get('API Endpoints', {})
        if api_result.get('failed_endpoints', 0) > 0:
            recommendations.append("MEDIUM: Fix failed API endpoints for complete functionality")
        
        # General recommendations
        recommendations.extend([
            "Implement proper model validation and testing pipeline",
            "Add comprehensive error handling and logging",
            "Establish performance monitoring and alerting",
            "Create automated regression testing suite"
        ])
        
        return recommendations
    
    def print_diagnostic_summary(self, report: Dict):
        """Print formatted diagnostic summary"""
        print("\n" + "="*80)
        print("DIAGNOSTIC MODEL TESTING RESULTS")
        print("="*80)
        
        summary = report['summary']
        
        print(f"\nSYSTEM HEALTH: {summary['system_health'].upper()}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        
        if summary['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES ({len(summary['critical_issues'])}):")
            for issue in summary['critical_issues']:
                print(f"  • {issue}")
        
        if summary['warnings']:
            print(f"\n⚠️  WARNINGS ({len(summary['warnings'])}):")
            for warning in summary['warnings']:
                print(f"  • {warning}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nDiagnostic testing completed in {report['total_diagnostic_time']:.2f} seconds")
        print("="*80)

def main():
    """Main function to run diagnostic testing"""
    tester = DiagnosticModelTester()
    
    try:
        # Run diagnostic suite
        report = tester.run_diagnostic_suite()
        
        # Print summary
        tester.print_diagnostic_summary(report)
        
        # Return appropriate exit code
        system_health = report['summary']['system_health']
        if system_health == 'good':
            print("\n✅ SYSTEM STATUS: HEALTHY")
            return 0
        elif system_health == 'fair':
            print("\n⚠️  SYSTEM STATUS: NEEDS ATTENTION")
            return 1
        elif system_health == 'poor':
            print("\n❌ SYSTEM STATUS: REQUIRES FIXES")
            return 2
        else:
            print("\n🚨 SYSTEM STATUS: CRITICAL ISSUES")
            return 3
            
    except Exception as e:
        logger.error(f"Diagnostic testing failed: {e}")
        print(f"\n💥 DIAGNOSTIC FAILED: {e}")
        return 4

if __name__ == "__main__":
    exit(main())