#!/usr/bin/env python3

import requests
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the exact API call that's causing the error
print("Testing /api/detect endpoint with local image path...")

api_url = "http://localhost:5001/api/detect"
test_data = {
    "image_url": "test_mars_image.jpg",
    "text": "Test image analysis"
}

print(f"\nCalling: {api_url}")
print(f"Data: {json.dumps(test_data, indent=2)}")

try:
    response = requests.post(api_url, json=test_data, timeout=10)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {response.text}")
    
    if response.status_code == 400:
        response_json = response.json()
        error_message = response_json.get('message', '')
        print(f"\nError Message Analysis:")
        print(f"- Full message: '{error_message}'")
        print(f"- Contains 'URL': {'URL' in error_message}")
        print(f"- Contains 'No scheme supplied': {'No scheme supplied' in error_message}")
        
except Exception as e:
    print(f"Error making API call: {e}")

# Also test the load_image_from_path_or_url function directly
print("\n" + "="*60)
print("Testing load_image_from_path_or_url function directly...")

try:
    from app import load_image_from_path_or_url
    result = load_image_from_path_or_url("test_mars_image.jpg")
    print(f"Direct function call result: {type(result) if result else 'None'}")
except Exception as e:
    print(f"Error in direct function call: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("ANALYSIS: Comparing API call vs direct function call")
print("If API fails but direct call works, there's a bug in the API endpoint")
print("If both fail the same way, the issue is in load_image_from_path_or_url")
print("="*60)