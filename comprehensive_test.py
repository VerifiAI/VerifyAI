#!/usr/bin/env python3
"""
Comprehensive Test Suite for Fake News Detection System
Tests the complete workflow: Content Input -> Analysis -> Proofs Validation -> Results -> AI Explainability
"""

import requests
import json
import time
import os
import pandas as pd
from pathlib import Path
import base64
from typing import Dict, List, Any

class FakeNewsDetectionTester:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
        # Load test data
        self.data_path = "/Users/mullamabusubhani/Downloads/Lokesh proj/Hybrid-Deep-Learning-with-Explainable-AI-for-Fake-News-Detection/data"
        self.load_test_data()
        
    def load_test_data(self):
        """Load test data from processed parquet files"""
        try:
            test_data_path = os.path.join(self.data_path, "processed", "fakeddit_processed_test.parquet")
            self.df = pd.read_parquet(test_data_path)
            print(f"✅ Loaded {len(self.df)} test samples")
            
            # Select diverse test cases
            self.test_cases = [
                # Real news cases (label 0)
                *self.df[self.df['2_way_label'] == 0].head(2).to_dict('records'),
                # Fake news cases (label 1) 
                *self.df[self.df['2_way_label'] == 1].head(3).to_dict('records')
            ]
            
            print(f"✅ Selected {len(self.test_cases)} diverse test cases")
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.test_cases = []
    
    def get_test_image_path(self, image_id: str) -> str:
        """Get path to test image"""
        image_path = os.path.join(self.data_path, "fakeddit", "subset", "images", "test", f"{image_id}.jpg")
        return image_path if os.path.exists(image_path) else None
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for API upload"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image {image_path}: {e}")
            return None
    
    def test_server_health(self) -> bool:
        """Test if server is running and healthy"""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code == 200:
                print("✅ Server is healthy and running")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Server connection failed: {e}")
            return False
    
    def test_content_input_and_analysis(self, test_case: Dict) -> Dict[str, Any]:
        """Test content input (text + image) and analysis"""
        print(f"\n🧪 Testing Case: {test_case['id']} - {test_case['clean_title'][:50]}...")
        print(f"   Expected Label: {'Fake' if test_case['2_way_label'] == 1 else 'Real'}")
        
        # Get image path
        image_path = self.get_test_image_path(test_case['id'])
        
        # Prepare test data for /api/detect endpoint
        test_data = {
            "text": test_case['clean_title']
        }
        
        # Add image if available (as base64 or file upload)
        if image_path:
            image_b64 = self.encode_image_to_base64(image_path)
            if image_b64:
                test_data["image"] = image_b64
                print(f"   📷 Image included: {test_case['id']}.jpg")
        
        try:
            # Test main detection endpoint
            print("   🔍 Sending content for analysis...")
            response = self.session.post(
                f"{self.base_url}/api/detect",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Analysis completed successfully")
                print(f"   📊 Result: {result.get('result', 'N/A')}")
                print(f"   🎯 Confidence: {result.get('confidence', 'N/A')}")
                
                return {
                    "success": True,
                    "test_case": test_case,
                    "analysis_result": result,
                    "test_data": test_data
                }
            else:
                print(f"   ❌ Analysis failed: {response.status_code} - {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    def test_proofs_validation(self, analysis_result: Dict, test_data: Dict) -> Dict[str, Any]:
        """Test proofs validation section using RSS fact-check endpoint"""
        print("   🗞️ Testing Proofs Validation...")
        
        try:
            # Test RSS fact-check endpoint for proofs validation
            response = self.session.post(
                f"{self.base_url}/api/rss-fact-check",
                json={"text": test_data["text"]},
                timeout=15
            )
            
            if response.status_code == 200:
                proofs = response.json()
                print(f"   ✅ Proofs validation completed")
                print(f"   📰 RSS Verdict: {proofs.get('verdict', 'N/A')}")
                print(f"   🔍 RSS Confidence: {proofs.get('confidence', 'N/A')}")
                print(f"   📝 Explanation: {proofs.get('explanation', 'N/A')[:100]}...")
                
                return {
                    "success": True,
                    "proofs": proofs
                }
            else:
                print(f"   ❌ Proofs validation failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"   ❌ Proofs validation error: {e}")
            return {"success": False, "error": str(e)}
    
    def test_final_analysis_result(self, analysis_result: Dict) -> Dict[str, Any]:
        """Test final analysis result section using history endpoint"""
        print("   📊 Testing Final Analysis Result...")
        
        try:
            # Test history endpoint to get analysis results
            response = self.session.get(
                f"{self.base_url}/api/history",
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                print(f"   ✅ Final results retrieved from history")
                print(f"   📈 Status: {results.get('status', 'N/A')}")
                print(f"   📊 Data Count: {results.get('count', 'N/A')}")
                print(f"   💾 Message: {results.get('message', 'N/A')}")
                
                return {
                    "success": True,
                    "final_results": results
                }
            else:
                print(f"   ❌ Final results failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"   ❌ Final results error: {e}")
            return {"success": False, "error": str(e)}
    
    def test_ai_explainability(self, analysis_result: Dict, test_data: Dict) -> Dict[str, Any]:
        """Test AI explainability section using explain endpoint"""
        print("   🤖 Testing AI Explainability...")
        
        try:
            # Test explain endpoint
            response = self.session.post(
                f"{self.base_url}/api/explain",
                json={"text": test_data["text"]},
                timeout=15
            )
            
            if response.status_code == 200:
                explanations = response.json()
                print(f"   ✅ AI explanations retrieved")
                print(f"   🧠 SHAP Values Available: {bool(explanations.get('shap_values'))}")
                print(f"   🔍 LIME Explanations Available: {bool(explanations.get('lime_explanations'))}")
                print(f"   📊 Feature Importance Available: {bool(explanations.get('feature_importance'))}")
                
                # Display explanation summary
                if explanations.get('summary'):
                    print(f"   📝 Explanation Summary: {explanations['summary'][:100]}...")
                
                return {
                    "success": True,
                    "explanations": explanations
                }
            else:
                print(f"   ❌ AI explanations failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"   ❌ AI explanations error: {e}")
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive Fake News Detection System Test")
        print("=" * 60)
        
        # Test server health
        if not self.test_server_health():
            return {"success": False, "error": "Server not available"}
        
        if not self.test_cases:
            return {"success": False, "error": "No test data available"}
        
        test_summary = {
            "total_tests": len(self.test_cases),
            "successful_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
        # Run tests for each case
        for i, test_case in enumerate(self.test_cases):
            print(f"\n{'='*60}")
            print(f"TEST {i+1}/{len(self.test_cases)}")
            
            # Step 1: Content Input and Analysis
            analysis_result = self.test_content_input_and_analysis(test_case)
            
            if not analysis_result["success"]:
                test_summary["failed_tests"] += 1
                test_summary["test_details"].append({
                    "test_id": test_case['id'],
                    "success": False,
                    "error": analysis_result.get("error")
                })
                continue
            
            # Step 2: Proofs Validation
            proofs_result = self.test_proofs_validation(analysis_result["analysis_result"], analysis_result["test_data"])
            
            # Step 3: Final Analysis Result
            final_result = self.test_final_analysis_result(analysis_result["analysis_result"])
            
            # Step 4: AI Explainability
            explainability_result = self.test_ai_explainability(analysis_result["analysis_result"], analysis_result["test_data"])
            
            # Compile test results
            test_detail = {
                "test_id": test_case['id'],
                "title": test_case['clean_title'],
                "expected_label": "Fake" if test_case['2_way_label'] == 1 else "Real",
                "success": True,
                "analysis": analysis_result["success"],
                "proofs": proofs_result["success"],
                "final_results": final_result["success"],
                "explainability": explainability_result["success"],
                "prediction": analysis_result["analysis_result"].get("prediction"),
                "confidence": analysis_result["analysis_result"].get("confidence")
            }
            
            test_summary["successful_tests"] += 1
            test_summary["test_details"].append(test_detail)
            
            print(f"   ✅ Test {i+1} completed successfully")
            
            # Brief pause between tests
            time.sleep(1)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("📋 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {test_summary['total_tests']}")
        print(f"Successful: {test_summary['successful_tests']}")
        print(f"Failed: {test_summary['failed_tests']}")
        print(f"Success Rate: {(test_summary['successful_tests']/test_summary['total_tests']*100):.1f}%")
        
        # Detailed results
        print(f"\n📊 DETAILED RESULTS:")
        for detail in test_summary["test_details"]:
            status = "✅" if detail["success"] else "❌"
            print(f"{status} {detail['test_id']}: {detail['title'][:50]}...")
            if detail["success"]:
                print(f"   Expected: {detail['expected_label']} | Predicted: {detail.get('prediction', 'N/A')}")
                print(f"   Analysis: {'✅' if detail['analysis'] else '❌'} | "
                      f"Proofs: {'✅' if detail['proofs'] else '❌'} | "
                      f"Results: {'✅' if detail['final_results'] else '❌'} | "
                      f"Explainability: {'✅' if detail['explainability'] else '❌'}")
        
        return test_summary

def main():
    """Main test execution"""
    print("🔬 Fake News Detection System - Comprehensive Test Suite")
    print("Testing: Content Input → Analysis → Proofs → Results → AI Explainability")
    print()
    
    # Initialize tester
    tester = FakeNewsDetectionTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Save results to file
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {results_file}")
    
    # Return exit code based on success
    if results.get("success", False) or results.get("successful_tests", 0) > 0:
        print("\n🎉 Testing completed successfully!")
        return 0
    else:
        print("\n💥 Testing failed!")
        return 1

if __name__ == "__main__":
    exit(main())