#!/usr/bin/env python3

import sys
import traceback
import requests
from unittest.mock import patch

# Patch requests.get to trace where it's being called from
original_get = requests.get

def traced_get(*args, **kwargs):
    print(f"\n=== REQUESTS.GET CALLED ===")
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")
    print("Call stack:")
    for line in traceback.format_stack():
        print(line.strip())
    print("=== END TRACE ===")
    return original_get(*args, **kwargs)

# Apply the patch
requests.get = traced_get

# Now test the API call
import json
import urllib.request
import urllib.parse

print("Testing /api/detect endpoint with tracing...")

url = "http://localhost:5001/api/detect"
data = {
    "image_url": "test_mars_image.jpg",
    "text": "Test image analysis"
}

try:
    # Convert data to JSON
    json_data = json.dumps(data).encode('utf-8')
    
    # Create request
    req = urllib.request.Request(url, data=json_data)
    req.add_header('Content-Type', 'application/json')
    
    # Make request
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(f"Success: {result}")
        
except urllib.error.HTTPError as e:
    error_response = e.read().decode('utf-8')
    print(f"HTTP Error {e.code}: {error_response}")
except Exception as e:
    print(f"Error: {e}")

print("\nTest completed.")