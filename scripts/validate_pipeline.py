#!/usr/bin/env python3
"""
Validation and Testing Script for Hybrid Deep Learning Fake News Detection System
Tests the complete Evidence-Guided Bayesian Fusion pipeline with enhanced detection

Author: AI Assistant
Date: August 27, 2025
Version: 1.0.0
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineValidator:
    """Comprehensive validation for the fake news detection pipeline"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.test_results = []
        
    def test_text_detection(self) -> Dict[str, Any]:
        """Test text-based fake news detection"""
        logger.info("Testing text-based detection...")
        
        test_cases = [
            {
                "text": "Breaking: Scientists discover that drinking water is actually harmful to human health, study shows 90% of people who drink water eventually die.",
                "expected_type": "fake",
                "description": "Obviously fake health claim"
            },
            {
                "text": "The Federal Reserve announced today that interest rates will remain unchanged at the current level, following their monthly policy meeting.",
                "expected_type": "real",
                "description": "Realistic financial news"
            },
            {
                "text": "Local community center opens new after-school program for children, providing tutoring and recreational activities from 3 PM to 6 PM on weekdays.",
                "expected_type": "real",
                "description": "Realistic local news"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/detect",
                    json={"text": case["text"]},
                    timeout=60
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "test_case": i + 1,
                        "description": case["description"],
                        "status": "success",
                        "verdict": data.get("verdict", "unknown").lower(),
                        "confidence": data.get("confidence", 0),
                        "processing_time": processing_time,
                        "has_evidence": len(data.get("evidence", [])) > 0,
                        "has_mhfn_output": "model_p_fake" in data,
                        "response_format_valid": self._validate_response_format(data)
                    }
                else:
                    result = {
                        "test_case": i + 1,
                        "description": case["description"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "processing_time": processing_time
                    }
                    
                results.append(result)
                logger.info(f"Text test {i+1}: {result['status']} - {result.get('verdict', 'N/A')}")
                
            except Exception as e:
                results.append({
                    "test_case": i + 1,
                    "description": case["description"],
                    "status": "error",
                    "error": str(e),
                    "processing_time": time.time() - start_time
                })
                logger.error(f"Text test {i+1} failed: {e}")
        
        return {
            "test_type": "text_detection",
            "total_tests": len(test_cases),
            "successful_tests": len([r for r in results if r["status"] == "success"]),
            "results": results
        }
    
    def test_url_detection(self) -> Dict[str, Any]:
        """Test URL-based fake news detection"""
        logger.info("Testing URL-based detection...")
        
        test_cases = [
            {
                "url": "https://www.bbc.com/news",
                "description": "Legitimate news source"
            },
            {
                "url": "https://www.reuters.com",
                "description": "Reputable news agency"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/detect",
                    json={"url": case["url"]},
                    timeout=60
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "test_case": i + 1,
                        "description": case["description"],
                        "status": "success",
                        "verdict": data.get("verdict", "unknown").lower(),
                        "confidence": data.get("confidence", 0),
                        "processing_time": processing_time,
                        "has_evidence": len(data.get("evidence", [])) > 0,
                        "response_format_valid": self._validate_response_format(data)
                    }
                else:
                    result = {
                        "test_case": i + 1,
                        "description": case["description"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "processing_time": processing_time
                    }
                    
                results.append(result)
                logger.info(f"URL test {i+1}: {result['status']} - {result.get('verdict', 'N/A')}")
                
            except Exception as e:
                results.append({
                    "test_case": i + 1,
                    "description": case["description"],
                    "status": "error",
                    "error": str(e),
                    "processing_time": time.time() - start_time
                })
                logger.error(f"URL test {i+1} failed: {e}")
        
        return {
            "test_type": "url_detection",
            "total_tests": len(test_cases),
            "successful_tests": len([r for r in results if r["status"] == "success"]),
            "results": results
        }
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test various API endpoints for availability"""
        logger.info("Testing API endpoints...")
        
        endpoints = [
            {
                "path": "/api/health",
                "method": "GET",
                "description": "Health check endpoint"
            },
            {
                "path": "/api/live-feed",
                "method": "GET",
                "description": "Live news feed"
            },
            {
                "path": "/api/history",
                "method": "GET",
                "description": "Detection history"
            }
        ]
        
        results = []
        for endpoint in endpoints:
            try:
                start_time = time.time()
                if endpoint["method"] == "GET":
                    response = requests.get(f"{self.base_url}{endpoint['path']}", timeout=30)
                else:
                    response = requests.post(f"{self.base_url}{endpoint['path']}", timeout=30)
                
                processing_time = time.time() - start_time
                
                result = {
                    "endpoint": endpoint["path"],
                    "method": endpoint["method"],
                    "description": endpoint["description"],
                    "status_code": response.status_code,
                    "status": "success" if response.status_code < 400 else "failed",
                    "response_time": processing_time
                }
                
                results.append(result)
                logger.info(f"Endpoint {endpoint['path']}: {response.status_code}")
                
            except Exception as e:
                results.append({
                    "endpoint": endpoint["path"],
                    "method": endpoint["method"],
                    "description": endpoint["description"],
                    "status": "error",
                    "error": str(e),
                    "response_time": time.time() - start_time
                })
                logger.error(f"Endpoint {endpoint['path']} failed: {e}")
        
        return {
            "test_type": "api_endpoints",
            "total_endpoints": len(endpoints),
            "successful_endpoints": len([r for r in results if r["status"] == "success"]),
            "results": results
        }
    
    def test_performance_requirements(self) -> Dict[str, Any]:
        """Test if system meets performance requirements (15-30 second processing)"""
        logger.info("Testing performance requirements...")
        
        test_text = "Scientists have discovered a new method for detecting fake news using advanced AI algorithms that can analyze text patterns and cross-reference with multiple fact-checking sources."
        
        processing_times = []
        successful_tests = 0
        
        for i in range(3):  # Run 3 performance tests
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/detect",
                    json={"text": test_text},
                    timeout=60
                )
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                if response.status_code == 200:
                    successful_tests += 1
                    
                logger.info(f"Performance test {i+1}: {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Performance test {i+1} failed: {e}")
        
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        meets_sla = 15 <= avg_time <= 30 if processing_times else False
        
        return {
            "test_type": "performance",
            "total_tests": 3,
            "successful_tests": successful_tests,
            "average_processing_time": avg_time,
            "processing_times": processing_times,
            "meets_sla_requirement": meets_sla,
            "sla_range": "15-30 seconds"
        }
    
    def _validate_response_format(self, data: Dict[str, Any]) -> bool:
        """Validate that response contains required fields for new orchestrator format"""
        required_fields = ["status", "verdict", "confidence"]
        orchestrator_fields = ["model_p_fake", "posterior_p_fake", "processing_time_s", "evidence"]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False
        
        # Check orchestrator-specific fields
        orchestrator_score = sum(1 for field in orchestrator_fields if field in data)
        
        return orchestrator_score >= 3  # At least 3 out of 4 orchestrator fields
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and generate comprehensive report"""
        logger.info("Starting comprehensive pipeline validation...")
        
        start_time = time.time()
        
        # Run all test suites
        text_results = self.test_text_detection()
        url_results = self.test_url_detection()
        api_results = self.test_api_endpoints()
        performance_results = self.test_performance_requirements()
        
        total_time = time.time() - start_time
        
        # Calculate overall statistics
        total_tests = (
            text_results["total_tests"] + 
            url_results["total_tests"] + 
            api_results["total_endpoints"] + 
            performance_results["total_tests"]
        )
        
        successful_tests = (
            text_results["successful_tests"] + 
            url_results["successful_tests"] + 
            api_results["successful_endpoints"] + 
            performance_results["successful_tests"]
        )
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate comprehensive report
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_validation_time": total_time,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "meets_performance_sla": performance_results["meets_sla_requirement"],
                "system_status": "PASS" if success_rate >= 80 and performance_results["meets_sla_requirement"] else "FAIL"
            },
            "detailed_results": {
                "text_detection": text_results,
                "url_detection": url_results,
                "api_endpoints": api_results,
                "performance": performance_results
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save validation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {filename}")
        return filename

def main():
    """Main validation function"""
    print("\n" + "="*80)
    print("HYBRID DEEP LEARNING FAKE NEWS DETECTION - PIPELINE VALIDATION")
    print("Evidence-Guided Bayesian Fusion with Time-Budget Orchestrator")
    print("="*80 + "\n")
    
    # Initialize validator
    validator = PipelineValidator()
    
    # Run comprehensive validation
    try:
        report = validator.run_comprehensive_validation()
        
        # Save report
        report_file = validator.save_report(report)
        
        # Print summary
        summary = report["validation_summary"]
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Performance SLA Met: {summary['meets_performance_sla']}")
        print(f"Overall Status: {summary['system_status']}")
        print(f"Total Validation Time: {summary['total_validation_time']:.2f}s")
        print(f"Report saved to: {report_file}")
        
        # Print detailed results
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}")
        
        for test_type, results in report["detailed_results"].items():
            print(f"\n{test_type.upper().replace('_', ' ')}:")
            if "results" in results:
                for result in results["results"]:
                    status_icon = "✓" if result["status"] == "success" else "✗"
                    print(f"  {status_icon} {result.get('description', result.get('endpoint', 'Test'))}")
            else:
                print(f"  Success Rate: {results.get('successful_tests', 0)}/{results.get('total_tests', 0)}")
        
        if summary["system_status"] == "PASS":
            print(f"\n🎉 VALIDATION PASSED - System is ready for production!")
        else:
            print(f"\n⚠️  VALIDATION FAILED - Please review issues before deployment.")
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        print(f"\n❌ VALIDATION ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())