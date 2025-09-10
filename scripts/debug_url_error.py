#!/usr/bin/env python3

import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing requests.get with local file path...")

# Test with the same input that's causing the error
test_image_path = "test_mars_image.jpg"

print(f"Testing with: {test_image_path}")

try:
    # This should reproduce the exact error we're seeing
    response = requests.get(test_image_path)
    print(f"requests.get result: {response}")
except Exception as e:
    print(f"Error in requests.get: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test what happens when we try to use requests.get with a local file
try:
    # Try with file:// prefix
    file_url = f"file://{os.path.abspath(test_image_path)}"
    print(f"Testing with file URL: {file_url}")
    response = requests.get(file_url)
    print(f"File URL result: {response}")
except Exception as e:
    print(f"Error with file URL: {e}")
    print(f"Error type: {type(e)}")