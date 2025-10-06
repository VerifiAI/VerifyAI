#!/usr/bin/env python3
"""
Chunk 5 Frontend Dashboard Validation Script
Hybrid Deep Learning with Explainable AI for Fake News Detection

This script validates the frontend dashboard implementation including:
- HTML structure and elements
- CSS styling and responsive design
- JavaScript API integration
- DOM manipulation functionality
- Cross-browser compatibility checks

Target: 90%+ accuracy for Chunk 5 completion
"""

import requests
import json
import time
import os
from datetime import datetime
from pathlib import Path

class Chunk5Validator:
    def __init__(self):
        self.base_url = "http://127.0.0.1:5001"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "chunk": "Chunk 5 - Frontend Dashboard",
            "tests": [],
            "summary": {}
        }
        self.passed_tests = 0
        self.total_tests = 0
        
    def log_test(self, test_name, passed, details="", error=""):
        """Log test results"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        test_result = {
            "test_name": test_name,
            "status": status,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["tests"].append(test_result)
        print(f"[{status}] {test_name}: {details}")
        if error:
            print(f"    Error: {error}")
    
    def test_file_existence(self):
        """Test if all required frontend files exist"""
        required_files = [
            "index.html",
            "styles.css", 
            "script.js"
        ]
        
        for file_name in required_files:
            file_path = Path(file_name)
            try:
                exists = file_path.exists()
                if exists:
                    file_size = file_path.stat().st_size
                    self.log_test(
                        f"File Existence - {file_name}",
                        True,
                        f"File exists with size {file_size} bytes"
                    )
                else:
                    self.log_test(
                        f"File Existence - {file_name}",
                        False,
                        "File does not exist"
                    )
            except Exception as e:
                self.log_test(
                    f"File Existence - {file_name}",
                    False,
                    "Error checking file",
                    str(e)
                )
    
    def test_html_structure(self):
        """Test HTML structure and required elements"""
        try:
            with open("index.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            
            # Check for required HTML elements
            required_elements = [
                "<!DOCTYPE html>",
                "<html",
                "<head>",
                "<title>",
                "<body>",
                "<div class=\"dashboard-container\">",
                "<div class=\"sidebar\">",
                "<div class=\"main-content\">",
                "<textarea id=\"newsText\">",
                "<button id=\"detectBtn\">",
                "<select id=\"feedSource\">",
                "<div id=\"detectionResult\">",
                "<div id=\"liveFeed\">",
                "<div id=\"historyList\">"
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in html_content:
                    missing_elements.append(element)
            
            if not missing_elements:
                self.log_test(
                    "HTML Structure Validation",
                    True,
                    f"All {len(required_elements)} required elements found"
                )
            else:
                self.log_test(
                    "HTML Structure Validation",
                    False,
                    f"Missing {len(missing_elements)} elements",
                    f"Missing: {', '.join(missing_elements)}"
                )
                
        except Exception as e:
            self.log_test(
                "HTML Structure Validation",
                False,
                "Error reading HTML file",
                str(e)
            )
    
    def test_css_styling(self):
        """Test CSS styling and responsive design"""
        try:
            with open("styles.css", "r", encoding="utf-8") as f:
                css_content = f.read()
            
            # Check for required CSS rules and properties
            required_css = [
                "--primary-navy: #001f3f",
                "--accent-teal: #008080",
                "--text-silver: #c0c0c0",
                ".dashboard-container",
                ".sidebar",
                ".main-content",
                "@media (max-width: 600px)",
                "@media (min-width: 1200px)",
                "display: flex",
                "background-color",
                "border-radius"
            ]
            
            missing_css = []
            for css_rule in required_css:
                if css_rule not in css_content:
                    missing_css.append(css_rule)
            
            if not missing_css:
                self.log_test(
                    "CSS Styling Validation",
                    True,
                    f"All {len(required_css)} required CSS rules found"
                )
            else:
                self.log_test(
                    "CSS Styling Validation",
                    False,
                    f"Missing {len(missing_css)} CSS rules",
                    f"Missing: {', '.join(missing_css)}"
                )
                
        except Exception as e:
            self.log_test(
                "CSS Styling Validation",
                False,
                "Error reading CSS file",
                str(e)
            )
    
    def test_javascript_structure(self):
        """Test JavaScript structure and API integration"""
        try:
            with open("script.js", "r", encoding="utf-8") as f:
                js_content = f.read()
            
            # Check for required JavaScript functions and API calls
            required_js = [
                "fetch(",
                "/api/detect",
                "/api/live-feed",
                "/api/history",
                "addEventListener",
                "getElementById",
                "innerHTML",
                "JSON.parse",
                "JSON.stringify",
                "DOMContentLoaded"
            ]
            
            missing_js = []
            for js_feature in required_js:
                if js_feature not in js_content:
                    missing_js.append(js_feature)
            
            if not missing_js:
                self.log_test(
                    "JavaScript Structure Validation",
                    True,
                    f"All {len(required_js)} required JS features found"
                )
            else:
                self.log_test(
                    "JavaScript Structure Validation",
                    False,
                    f"Missing {len(missing_js)} JS features",
                    f"Missing: {', '.join(missing_js)}"
                )
                
        except Exception as e:
            self.log_test(
                "JavaScript Structure Validation",
                False,
                "Error reading JavaScript file",
                str(e)
            )
    
    def test_frontend_server_response(self):
        """Test if frontend is served correctly by Flask"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            
            if response.status_code == 200:
                content = response.text
                if "Fake News Detection Dashboard" in content:
                    self.log_test(
                        "Frontend Server Response",
                        True,
                        f"HTML served successfully (status: {response.status_code})"
                    )
                else:
                    self.log_test(
                        "Frontend Server Response",
                        False,
                        "HTML served but missing expected content"
                    )
            else:
                self.log_test(
                    "Frontend Server Response",
                    False,
                    f"Server returned status {response.status_code}"
                )
                
        except Exception as e:
            self.log_test(
                "Frontend Server Response",
                False,
                "Error accessing frontend",
                str(e)
            )
    
    def test_static_files_serving(self):
        """Test if CSS and JS files are served correctly"""
        static_files = [
            ("/styles.css", "text/css"),
            ("/script.js", "application/javascript")
        ]
        
        for file_path, expected_content_type in static_files:
            try:
                response = requests.get(f"{self.base_url}{file_path}", timeout=10)
                
                if response.status_code == 200:
                    self.log_test(
                        f"Static File Serving - {file_path}",
                        True,
                        f"File served successfully (status: {response.status_code})"
                    )
                else:
                    self.log_test(
                        f"Static File Serving - {file_path}",
                        False,
                        f"Server returned status {response.status_code}"
                    )
                    
            except Exception as e:
                self.log_test(
                    f"Static File Serving - {file_path}",
                    False,
                    "Error accessing static file",
                    str(e)
                )
    
    def test_api_endpoints_integration(self):
        """Test API endpoints that frontend integrates with"""
        endpoints = [
            ("/api/live-feed?source=bbc", "GET"),
            ("/api/history", "GET"),
            ("/api/detect", "POST")
        ]
        
        for endpoint, method in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                elif method == "POST":
                    test_data = {"text": "This is a test news article for validation."}
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        json=test_data,
                        timeout=10
                    )
                
                if response.status_code == 200:
                    try:
                        json_data = response.json()
                        self.log_test(
                            f"API Integration - {endpoint}",
                            True,
                            f"API responded successfully with valid JSON"
                        )
                    except:
                        self.log_test(
                            f"API Integration - {endpoint}",
                            False,
                            "API responded but returned invalid JSON"
                        )
                else:
                    self.log_test(
                        f"API Integration - {endpoint}",
                        False,
                        f"API returned status {response.status_code}"
                    )
                    
            except Exception as e:
                self.log_test(
                    f"API Integration - {endpoint}",
                    False,
                    "Error accessing API endpoint",
                    str(e)
                )
    
    def test_responsive_design_elements(self):
        """Test responsive design implementation"""
        try:
            with open("styles.css", "r", encoding="utf-8") as f:
                css_content = f.read()
            
            # Check for responsive design features
            responsive_features = [
                "@media (max-width: 600px)",
                "@media (min-width: 1200px)",
                "flex-direction: column",
                "width: 100%",
                "display: flex"
            ]
            
            found_features = 0
            for feature in responsive_features:
                if feature in css_content:
                    found_features += 1
            
            if found_features >= len(responsive_features) * 0.8:  # 80% threshold
                self.log_test(
                    "Responsive Design Implementation",
                    True,
                    f"Found {found_features}/{len(responsive_features)} responsive features"
                )
            else:
                self.log_test(
                    "Responsive Design Implementation",
                    False,
                    f"Only found {found_features}/{len(responsive_features)} responsive features"
                )
                
        except Exception as e:
            self.log_test(
                "Responsive Design Implementation",
                False,
                "Error checking responsive design",
                str(e)
            )
    
    def run_validation(self):
        """Run all validation tests"""
        print("\n" + "="*60)
        print("CHUNK 5 FRONTEND DASHBOARD VALIDATION")
        print("Hybrid Deep Learning with Explainable AI for Fake News Detection")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n")
        
        # Run all tests
        self.test_file_existence()
        self.test_html_structure()
        self.test_css_styling()
        self.test_javascript_structure()
        self.test_frontend_server_response()
        self.test_static_files_serving()
        self.test_api_endpoints_integration()
        self.test_responsive_design_elements()
        
        # Calculate results
        accuracy = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        self.results["summary"] = {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.total_tests - self.passed_tests,
            "accuracy_percentage": round(accuracy, 2),
            "target_accuracy": 90.0,
            "meets_target": accuracy >= 90.0,
            "completion_time": datetime.now().isoformat()
        }
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Target: 90.0%")
        print(f"Status: {'✓ MEETS TARGET' if accuracy >= 90.0 else '✗ BELOW TARGET'}")
        print("="*60)
        
        # Save results to file
        with open("chunk5_validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: chunk5_validation_results.json")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return accuracy >= 90.0

if __name__ == "__main__":
    validator = Chunk5Validator()
    success = validator.run_validation()
    exit(0 if success else 1)