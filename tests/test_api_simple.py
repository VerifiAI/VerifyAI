#!/usr/bin/env python3
"""
Simple API test to debug the 403 issue
"""

import requests
import json

def test_api():
    url = "http://localhost:5001/api/detect"
    
    # Test data
    data = {
        "text": "This is a simple test message for fake news detection."
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Testing API endpoint: {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            try:
                json_response = response.json()
                print(f"\nParsed JSON Response:")
                print(json.dumps(json_response, indent=2))
            except Exception as e:
                print(f"Failed to parse JSON: {e}")
        
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()